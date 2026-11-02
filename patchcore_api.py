"""Small callable API around this PatchCore repository.

The three public entry points are:
- train_patchcore(): train a PatchCore memory bank and save it under models/.
- detect_patchcore(): load a saved model and run image-level + pixel-level inference.
- output_patchcore(): convenience wrapper that writes heatmaps/masks/CSV outputs.
"""

import csv
import json
import random
import shutil
import time
from pathlib import Path
import sys

PACKAGE_ROOT = Path(__file__).resolve().parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from typing import Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import patchcore.backbones
import patchcore.common
import patchcore.patchcore
import patchcore.sampler
from patchcore.datasets.mvtec import DatasetSplit, MVTecDataset


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DEFAULT_IMAGE_HEIGHT = 384
DEFAULT_IMAGE_WIDTH = 512
DEFAULT_BACKBONE = "wideresnet50"
DEFAULT_LAYERS = ["layer2", "layer3"]
DEFAULT_CORESET_RATIO = 0.1
IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def _default_augmented_root(dataset_root: Union[str, Path]):
    dataset_root = Path(dataset_root)
    if dataset_root.name.endswith("_patchcore_aug"):
        return dataset_root
    return dataset_root.with_name(f"{dataset_root.name}_patchcore_aug")


def _has_train_good(dataset_root: Union[str, Path], classname: str):
    train_good = Path(dataset_root) / classname / "train" / "good"
    return train_good.exists() and any(
        item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
        for item in train_good.iterdir()
    )


def _shift_image(image: Image.Image, shift_x: int, shift_y: int):
    fill = tuple(np.asarray(image).reshape(-1, 3).mean(axis=0).astype(np.uint8))
    return image.transform(
        image.size,
        Image.AFFINE,
        (1, 0, -shift_x, 0, 1, -shift_y),
        resample=Image.BICUBIC,
        fillcolor=fill,
    )


def augment_patchcore_dataset(
    dataset_root: Union[str, Path],
    output_root: Optional[Union[str, Path]] = None,
    classname: Optional[str] = None,
    rotation_angles: Sequence[float] = (-5.0, 5.0),
    shift_pixels: tuple = (12, 8),
    force: bool = False,
):
    """Create an augmented MVTec-style dataset for PatchCore training.

    Only train/good images are augmented. The full dataset tree is copied first,
    then each normal training image gets two rotated variants and one shifted
    variant. If output_root already exists and force is false, it is reused.
    """
    dataset_root = Path(dataset_root)
    output_root = Path(output_root) if output_root is not None else _default_augmented_root(dataset_root)

    if output_root.exists() and not force:
        return output_root

    if output_root.exists() and force:
        shutil.rmtree(output_root)
    shutil.copytree(dataset_root, output_root)

    class_dirs = [output_root / classname] if classname is not None else [
        item for item in output_root.iterdir() if item.is_dir()
    ]

    for class_dir in class_dirs:
        train_good = class_dir / "train" / "good"
        if not train_good.exists():
            continue
        image_paths = sorted(
            item for item in train_good.iterdir()
            if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
        )
        for image_path in image_paths:
            with Image.open(image_path).convert("RGB") as image:
                for idx, angle in enumerate(rotation_angles, start=1):
                    rotated = image.rotate(angle, resample=Image.BICUBIC, expand=False)
                    rotated.save(train_good / f"{image_path.stem}_aug_rot{idx}{image_path.suffix}")
                shifted = _shift_image(image, int(shift_pixels[0]), int(shift_pixels[1]))
                shifted.save(train_good / f"{image_path.stem}_aug_shift1{image_path.suffix}")

    return output_root


class ImagePathDataset(Dataset):
    def __init__(self, image_paths: Sequence[Union[str, Path]], resize, imagesize):
        self.image_paths = [Path(path) for path in image_paths]
        self.transform = transforms.Compose(
            [
                transforms.Resize(resize),
                transforms.CenterCrop(imagesize),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        return {"image": self.transform(image), "image_path": str(image_path)}


def _fix_seeds(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _device(device: Optional[Union[str, torch.device]] = None):
    if device is not None:
        return torch.device(device)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _sync_if_needed(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def _as_hw(image_height: int, image_width: int):
    return (int(image_height), int(image_width))


def _normalize_map(values):
    values = np.asarray(values, dtype=np.float32)
    min_value = float(values.min())
    max_value = float(values.max())
    if max_value <= min_value:
        return np.zeros_like(values, dtype=np.float32)
    return (values - min_value) / (max_value - min_value)


def _collect_images(path_or_paths: Union[str, Path, Sequence[Union[str, Path]]]):
    if isinstance(path_or_paths, (str, Path)):
        path = Path(path_or_paths)
        if path.is_file():
            return [path]
        return sorted(
            item for item in path.rglob("*")
            if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
        )
    return [Path(path) for path in path_or_paths]


def _build_patchcore(
    device,
    image_height: int = DEFAULT_IMAGE_HEIGHT,
    image_width: int = DEFAULT_IMAGE_WIDTH,
    backbone_name: str = DEFAULT_BACKBONE,
    layers: Sequence[str] = DEFAULT_LAYERS,
    coreset_sampling_ratio: float = DEFAULT_CORESET_RATIO,
):
    backbone = patchcore.backbones.load(backbone_name)
    backbone.name = backbone_name
    backbone.seed = None

    model = patchcore.patchcore.PatchCore(device)
    model.load(
        backbone=backbone,
        layers_to_extract_from=list(layers),
        device=device,
        input_shape=(3, int(image_height), int(image_width)),
        pretrain_embed_dimension=1024,
        target_embed_dimension=1024,
        patchsize=3,
        featuresampler=patchcore.sampler.ApproximateGreedyCoresetSampler(
            coreset_sampling_ratio, device
        ),
        anomaly_scorer_num_nn=1,
        nn_method=patchcore.common.FaissNN(False, 8),
    )
    return model


def _save_model_files(model, model_dir: Path, metadata: dict):
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_to_path(str(model_dir))
    model.anomaly_scorer.save(str(model_dir), save_features_separately=True)

    memory_bank = np.asarray(model.anomaly_scorer.detection_features)
    np.save(model_dir / "memory_bank.npy", memory_bank)
    with (model_dir / "memory_bank_shape.txt").open("w", encoding="utf-8") as handle:
        handle.write(f"shape: {memory_bank.shape}\n")
        handle.write(f"dtype: {memory_bank.dtype}\n")

    metadata = dict(metadata)
    metadata["memory_bank_shape"] = list(memory_bank.shape)
    metadata["memory_bank_dtype"] = str(memory_bank.dtype)
    with (model_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def _load_metadata(model_dir: Path):
    metadata_path = model_dir / "metadata.json"
    if not metadata_path.exists():
        return {}
    with metadata_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_model(model_dir: Union[str, Path], device=None):
    model_dir = Path(model_dir)
    device = _device(device)
    model = patchcore.patchcore.PatchCore(device)
    model.load_from_path(
        str(model_dir),
        device=device,
        nn_method=patchcore.common.FaissNN(False, 8),
    )
    return model, _load_metadata(model_dir), device


def _predict_batch(model, images, device):
    images = images.to(torch.float).to(device)
    model.forward_modules.eval()
    with torch.no_grad():
        features, patch_shapes = model._embed(images, provide_patch_shapes=True)

    features = np.asarray(features)
    patch_scores = image_scores = model.anomaly_scorer.predict([features])[0]

    batchsize = images.shape[0]
    image_scores = model.patch_maker.unpatch_scores(image_scores, batchsize=batchsize)
    image_scores = image_scores.reshape(*image_scores.shape[:2], -1)
    image_scores = model.patch_maker.score(image_scores)

    patch_scores = model.patch_maker.unpatch_scores(patch_scores, batchsize=batchsize)
    scales = patch_shapes[0]
    patch_scores = patch_scores.reshape(batchsize, scales[0], scales[1])
    masks = model.anomaly_segmentor.convert_to_segmentation(patch_scores)
    return image_scores, masks


def _save_heatmap_files(image_path, heatmap, output_dir: Path, score: float, mask_threshold: float):
    output_dir.mkdir(parents=True, exist_ok=True)
    heatmap_dir = output_dir / "heatmaps"
    overlay_dir = output_dir / "overlays"
    mask_dir = output_dir / "masks"
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    image_path = Path(image_path)
    heatmap_norm = _normalize_map(heatmap)
    defect_mask = (heatmap_norm >= float(mask_threshold)).astype(np.uint8) * 255

    cmap = plt.get_cmap("jet")
    original = Image.open(image_path).convert("RGB").resize(
        (heatmap_norm.shape[1], heatmap_norm.shape[0])
    )
    image = np.asarray(original).astype(np.float32) / 255.0
    image = np.clip(image ** 0.7, 0, 1)
    colored = cmap(heatmap_norm)[..., :3]
    # Keep the source image visible while still highlighting anomalous regions.
    overlay = np.clip(0.82 * image + 0.18 * colored, 0, 1)

    stem = f"{image_path.stem}_score_{float(score):.4f}"
    heatmap_path = heatmap_dir / f"{stem}_heatmap.png"
    overlay_path = overlay_dir / f"{stem}_overlay.png"
    mask_path = mask_dir / f"{stem}_mask.png"

    plt.imsave(
        heatmap_path,
        np.clip(0.82 * image + 0.18 * colored, 0, 1),
    )
    plt.imsave(overlay_path, overlay)
    Image.fromarray(defect_mask).save(mask_path)
    return {
        "heatmap_path": str(heatmap_path),
        "overlay_path": str(overlay_path),
        "mask_path": str(mask_path),
    }


def train_patchcore(
    dataset_root: Union[str, Path],
    classname: str,
    model_name: Optional[str] = None,
    models_root: Optional[Union[str, Path]] = None,
    image_height: int = DEFAULT_IMAGE_HEIGHT,
    image_width: int = DEFAULT_IMAGE_WIDTH,
    batch_size: int = 1,
    num_workers: int = 0,
    seed: int = 0,
    device: Optional[Union[str, torch.device]] = None,
    backbone_name: str = DEFAULT_BACKBONE,
    layers: Sequence[str] = DEFAULT_LAYERS,
    coreset_sampling_ratio: float = DEFAULT_CORESET_RATIO,
    use_augmentation: bool = True,
    augmented_dataset_root: Optional[Union[str, Path]] = None,
    force_augmentation: bool = False,
    print_model_dir: bool = True,
):
    """Train PatchCore on one MVTec-style class and save the model.

    dataset_root should contain classname/train/good images. By default this
    function trains on an augmented copy of dataset_root. The augmented copy is
    named <dataset_root>_patchcore_aug unless augmented_dataset_root is set.
    """
    _fix_seeds(seed)
    device = _device(device)
    resize = _as_hw(image_height, image_width)
    imagesize = _as_hw(image_height, image_width)

    source_dataset_root = Path(dataset_root)
    training_dataset_root = source_dataset_root
    if use_augmentation:
        target_augmented_root = Path(augmented_dataset_root) if augmented_dataset_root is not None else _default_augmented_root(source_dataset_root)
        if _has_train_good(target_augmented_root, classname) and not force_augmentation:
            training_dataset_root = target_augmented_root
        else:
            training_dataset_root = augment_patchcore_dataset(
                source_dataset_root,
                output_root=target_augmented_root,
                classname=classname,
                force=force_augmentation,
            )

    dataset = MVTecDataset(
        str(training_dataset_root),
        classname=classname,
        resize=resize,
        imagesize=imagesize,
        split=DatasetSplit.TRAIN,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = _build_patchcore(
        device=device,
        image_height=image_height,
        image_width=image_width,
        backbone_name=backbone_name,
        layers=layers,
        coreset_sampling_ratio=coreset_sampling_ratio,
    )

    _sync_if_needed(device)
    train_start = time.perf_counter()
    model.fit(loader)
    _sync_if_needed(device)
    train_time_ms = (time.perf_counter() - train_start) * 1000

    if model_name is None:
        model_name = f"patchcore_{classname}_{image_width}x{image_height}"
    if models_root is None:
        models_root = PACKAGE_ROOT / "models"
    model_dir = Path(models_root) / model_name
    metadata = {
        "classname": classname,
        "dataset_root": str(dataset_root),
        "training_dataset_root": str(training_dataset_root),
        "use_augmentation": bool(use_augmentation),
        "augmented_dataset_root": str(training_dataset_root) if use_augmentation else None,
        "augmentation_rotation_angles": [-5.0, 5.0],
        "augmentation_shift_pixels": [12, 8],
        "image_height": int(image_height),
        "image_width": int(image_width),
        "resize": list(resize),
        "imagesize": list(imagesize),
        "backbone_name": backbone_name,
        "layers": list(layers),
        "coreset_sampling_ratio": float(coreset_sampling_ratio),
        "train_images": len(dataset),
        "train_time_ms": train_time_ms,
        "threshold_source": "not_set",
        "image_threshold": None,
        "mask_threshold": 0.5,
    }
    _save_model_files(model, model_dir, metadata)
    if print_model_dir:
        print(f"Saved PatchCore model to: {model_dir}")
    return {
        "model_dir": str(model_dir),
        "train_images": len(dataset),
        "train_time_ms": train_time_ms,
        "metadata": metadata,
    }


def detect_patchcore(
    image_paths: Union[str, Path, Sequence[Union[str, Path]]],
    model_dir: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    save_heatmap: bool = True,
    image_threshold: Optional[float] = None,
    mask_threshold: Optional[float] = None,
    image_height: Optional[int] = None,
    image_width: Optional[int] = None,
    batch_size: int = 1,
    device: Optional[Union[str, torch.device]] = None,
):
    """Run inference with a saved PatchCore model.

    Returns one dict per input image. If output_dir is provided and save_heatmap
    is true, heatmap/overlay/mask PNG files are written there.
    """
    model, metadata, device = _load_model(model_dir, device=device)
    image_height = int(image_height or metadata.get("image_height", DEFAULT_IMAGE_HEIGHT))
    image_width = int(image_width or metadata.get("image_width", DEFAULT_IMAGE_WIDTH))
    image_threshold = metadata.get("image_threshold") if image_threshold is None else image_threshold
    mask_threshold = metadata.get("mask_threshold", 0.5) if mask_threshold is None else mask_threshold
    if image_threshold is None:
        image_threshold = float("inf")

    image_paths = _collect_images(image_paths)
    dataset = ImagePathDataset(image_paths, _as_hw(image_height, image_width), _as_hw(image_height, image_width))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    results = []
    output_dir = Path(output_dir) if output_dir is not None else None
    seen = 0
    _sync_if_needed(device)
    infer_start = time.perf_counter()
    for batch in loader:
        image_scores, heatmaps = _predict_batch(model, batch["image"], device)
        for score, heatmap, image_path in zip(image_scores, heatmaps, batch["image_path"]):
            score = float(score)
            heatmap = np.asarray(heatmap, dtype=np.float32)
            pred_is_anomaly = bool(score >= float(image_threshold))
            row = {
                "image": Path(image_path).name,
                "image_path": str(image_path),
                "image_score": score,
                "image_threshold": float(image_threshold),
                "pred_is_anomaly": int(pred_is_anomaly),
                "pred_label": "anomaly" if pred_is_anomaly else "normal",
                "heatmap": heatmap,
                "mask": (_normalize_map(heatmap) >= float(mask_threshold)).astype(np.uint8),
            }
            if output_dir is not None and save_heatmap:
                row.update(_save_heatmap_files(image_path, heatmap, output_dir, score, float(mask_threshold)))
            results.append(row)
            seen += 1
    _sync_if_needed(device)
    infer_time_ms = (time.perf_counter() - infer_start) * 1000

    for row in results:
        row["total_infer_time_ms"] = infer_time_ms
        row["avg_infer_time_ms"] = infer_time_ms / max(seen, 1)
    return results


def output_patchcore(
    image_paths: Union[str, Path, Sequence[Union[str, Path]]],
    model_dir: Union[str, Path],
    output_dir: Union[str, Path],
    csv_name: str = "patchcore_predictions.csv",
    save_heatmap: bool = True,
    image_threshold: Optional[float] = None,
    mask_threshold: Optional[float] = None,
    device: Optional[Union[str, torch.device]] = None,
):
    """Run detection and write CSV + optional heatmap/overlay/mask outputs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = detect_patchcore(
        image_paths=image_paths,
        model_dir=model_dir,
        output_dir=output_dir,
        save_heatmap=save_heatmap,
        image_threshold=image_threshold,
        mask_threshold=mask_threshold,
        device=device,
    )

    csv_path = output_dir / csv_name
    fieldnames = [
        "image",
        "image_path",
        "image_score",
        "image_threshold",
        "pred_is_anomaly",
        "pred_label",
        "heatmap_path",
        "overlay_path",
        "mask_path",
        "total_infer_time_ms",
        "avg_infer_time_ms",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow({key: result.get(key, "") for key in fieldnames})

    return {"csv_path": str(csv_path), "results": results}
