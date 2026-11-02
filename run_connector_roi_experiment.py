"""Evaluate PatchCore after removing the connector image background."""

import csv
import shutil
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from patchcore_api import detect_patchcore, train_patchcore


BASE_DIR = Path(__file__).resolve().parent
DATASET_ROOT = BASE_DIR / "data2" / "extracted_docx"
OUTPUT_ROOT = DATASET_ROOT / "connector_roi_experiment"
STAGING_ROOT = OUTPUT_ROOT / "staging"
MODELS_ROOT = OUTPUT_ROOT / "models"
SHOTS = (1, 3, 5)
IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def image_files(directory):
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def foreground_crop(source, destination):
    image = np.asarray(Image.open(source).convert("RGB"))
    border = np.concatenate(
        [image[:20].reshape(-1, 3), image[-20:].reshape(-1, 3),
         image[:, :20].reshape(-1, 3), image[:, -20:].reshape(-1, 3)]
    )
    background = np.median(border, axis=0)
    distance = np.sqrt(((image.astype(np.float32) - background) ** 2).sum(axis=2))
    mask = distance > max(18.0, float(np.percentile(distance, 72)))
    mask = ndimage.binary_closing(mask, iterations=10)
    mask = ndimage.binary_dilation(mask, iterations=8)
    labels, count = ndimage.label(mask)
    if count:
        sizes = ndimage.sum(mask, labels, range(1, count + 1))
        component = 1 + int(np.argmax(sizes))
        ys, xs = np.where(labels == component)
    else:
        ys, xs = np.where(mask)
    if len(xs) == 0:
        ys, xs = np.array([0, image.shape[0] - 1]), np.array([0, image.shape[1] - 1])

    left, right = int(xs.min()), int(xs.max())
    top, bottom = int(ys.min()), int(ys.max())
    width = right - left + 1
    height = bottom - top + 1
    target_ratio = 512 / 384
    crop_width = max(width * 1.25, height * 1.25 * target_ratio)
    crop_height = crop_width / target_ratio
    center_x = (left + right) / 2
    center_y = (top + bottom) / 2
    left = max(0, int(center_x - crop_width / 2))
    right = min(image.shape[1], int(center_x + crop_width / 2))
    top = max(0, int(center_y - crop_height / 2))
    bottom = min(image.shape[0], int(center_y + crop_height / 2))
    cropped = Image.fromarray(image[top:bottom, left:right]).convert("RGB")
    destination.parent.mkdir(parents=True, exist_ok=True)
    cropped.save(destination)


def prepare_data():
    if STAGING_ROOT.exists():
        shutil.rmtree(STAGING_ROOT)
    for split in ("normal", "defect"):
        for source in image_files(DATASET_ROOT / "connector" / split):
            foreground_crop(source, STAGING_ROOT / split / source.name)


def auc(labels, scores):
    positives = [s for label, s in zip(labels, scores) if label]
    negatives = [s for label, s in zip(labels, scores) if not label]
    return (
        sum(a > b for a in positives for b in negatives)
        + 0.5 * sum(a == b for a in positives for b in negatives)
    ) / (len(positives) * len(negatives))


def main():
    prepare_data()
    rows = []
    for shot in SHOTS:
        train_root = OUTPUT_ROOT / f"{shot}_shot_train"
        if train_root.exists():
            shutil.rmtree(train_root)
        for source in image_files(STAGING_ROOT / "normal")[:shot]:
            target = train_root / "connector" / "train" / "good" / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(source)

        start = time.perf_counter()
        training = train_patchcore(
            dataset_root=train_root,
            classname="connector",
            model_name=f"connector_{shot}shot_roi",
            models_root=MODELS_ROOT,
            use_augmentation=False,
            device="cuda:0",
        )
        train_ms = (time.perf_counter() - start) * 1000
        model_dir = Path(training["model_dir"])
        normal = detect_patchcore(
            STAGING_ROOT / "normal", model_dir, save_heatmap=False,
            image_threshold=10**18, device="cuda:0"
        )
        threshold = float(np.percentile([r["image_score"] for r in normal], 95))
        test_paths = image_files(STAGING_ROOT / "normal") + image_files(STAGING_ROOT / "defect")
        start = time.perf_counter()
        results = detect_patchcore(
            test_paths, model_dir, output_dir=OUTPUT_ROOT / f"{shot}_shot",
            save_heatmap=True, image_threshold=threshold, device="cuda:0"
        )
        infer_ms = (time.perf_counter() - start) * 1000
        labels = [not Path(r["image_path"]).parent.name == "normal" for r in results]
        scores = [r["image_score"] for r in results]
        rows.append({
            "shot": shot, "AUROC": auc(labels, scores),
            "train_ms": train_ms, "infer_ms": infer_ms,
            "total_ms": train_ms + infer_ms, "threshold": threshold,
        })
        print(rows[-1], flush=True)

    with (OUTPUT_ROOT / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
