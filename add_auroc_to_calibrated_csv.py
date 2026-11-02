from pathlib import Path
import csv

import numpy as np
from PIL import Image
from sklearn import metrics
from torchvision import transforms

from patchcore_api import detect_patchcore
from patchcore.datasets.mvtec import MVTecDataset, DatasetSplit

BASE = Path('/home/ubuntu/wmh/patchcore/patch_backage')
DATASET_ROOT = BASE / 'data_fortrain3'
MODELS_ROOT = BASE / 'models'
CSV_PATH = BASE / 'hot_pic_calibrated_p95' / 'part1_part8_calibrated_p95_results.csv'
CSV_ROOT_PATH = BASE / 'part1_part8_calibrated_p95_results.csv'
IMAGE_HEIGHT = 384
IMAGE_WIDTH = 512
RESIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)
IMAGE_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)


def model_dir_for(part):
    preferred = MODELS_ROOT / f'{part}_patchcore'
    if preferred.exists():
        return preferred
    return MODELS_ROOT / f'{part}_patchcore_retrained'


def load_thresholds():
    thresholds = {}
    with CSV_PATH.open('r', newline='', encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            thresholds.setdefault(row['part'], float(row['image_threshold']))
    return thresholds


def load_test_items(part):
    dataset = MVTecDataset(
        str(DATASET_ROOT),
        classname=part,
        resize=RESIZE,
        imagesize=IMAGE_SIZE,
        split=DatasetSplit.TEST,
    )
    image_paths = []
    labels = []
    masks = []
    mask_transform = transforms.Compose([
        transforms.Resize(RESIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
    ])
    for _, anomaly, image_path, mask_path in dataset.data_to_iterate:
        image_paths.append(image_path)
        labels.append(int(anomaly != 'good'))
        if mask_path is None:
            masks.append(np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8))
        else:
            mask = Image.open(mask_path)
            mask = mask_transform(mask).numpy()[0]
            masks.append((mask > 0.5).astype(np.uint8))
    return image_paths, np.asarray(labels, dtype=np.int32), np.asarray(masks, dtype=np.uint8)


def compute_part_metrics(part, threshold):
    image_paths, labels, masks_gt = load_test_items(part)
    results = detect_patchcore(
        image_paths=image_paths,
        model_dir=model_dir_for(part),
        save_heatmap=False,
        image_threshold=threshold,
    )
    scores = np.asarray([result['image_score'] for result in results], dtype=np.float32)
    heatmaps = np.asarray([result['heatmap'] for result in results], dtype=np.float32)
    image_auroc = float(metrics.roc_auc_score(labels, scores))
    pixel_auroc = float(metrics.roc_auc_score(masks_gt.ravel(), heatmaps.ravel()))
    return image_auroc, pixel_auroc


def main():
    thresholds = load_thresholds()
    part_metrics = {}
    for part in sorted(thresholds, key=lambda value: int(value.replace('part', ''))):
        print(f'Computing AUROC for {part}', flush=True)
        image_auroc, pixel_auroc = compute_part_metrics(part, thresholds[part])
        part_metrics[part] = {
            'image_auroc': image_auroc,
            'pixel_auroc': pixel_auroc,
        }
        print(f'{part}: image_auroc={image_auroc:.6f}, pixel_auroc={pixel_auroc:.6f}', flush=True)

    with CSV_PATH.open('r', newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0].keys())

    for key in ['image_auroc', 'pixel_auroc']:
        if key not in fieldnames:
            fieldnames.append(key)

    for row in rows:
        row.update(part_metrics[row['part']])

    with CSV_PATH.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    CSV_ROOT_PATH.write_text(CSV_PATH.read_text(encoding='utf-8'), encoding='utf-8')
    print(f'Updated {CSV_PATH}', flush=True)
    print(f'Updated {CSV_ROOT_PATH}', flush=True)


if __name__ == '__main__':
    main()
