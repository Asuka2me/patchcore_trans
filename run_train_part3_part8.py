from pathlib import Path
import csv
import json

from patchcore_api import train_patchcore, detect_patchcore


BASE = Path('/home/ubuntu/wmh/patchcore')
PACKAGE_DIR = BASE / 'patch_backage'
DATASET_ROOT = PACKAGE_DIR / 'data_fortrain3'
MODELS_ROOT = PACKAGE_DIR / 'models'
HOT_PIC_ROOT = PACKAGE_DIR / 'hot_pic'
SUMMARY_CSV = PACKAGE_DIR / 'part3_part8_train_detect_results.csv'
METRICS_CSV = (
    BASE
    / 'patchcore-main'
    / 'benchmark_outputs'
    / 'part1_part8_patchcore_r512x384_trainaug1_modelsaved'
    / 'part1_part8_metrics.csv'
)


def load_thresholds():
    thresholds = {}
    with METRICS_CSV.open('r', newline='', encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            if row['part'].startswith('part'):
                thresholds[row['part']] = float(row['image_f1_threshold'])
    return thresholds


def update_model_threshold(model_dir, threshold):
    metadata_path = Path(model_dir) / 'metadata.json'
    if not metadata_path.exists():
        return
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    metadata['threshold_source'] = str(METRICS_CSV)
    metadata['image_threshold'] = float(threshold)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')


def main():
    thresholds = load_thresholds()
    rows = []
    for idx in range(3, 9):
        part = f'part{idx}'
        model_name = f'{part}_patchcore_retrained'
        print(f'Training {part} -> {model_name}', flush=True)
        train_result = train_patchcore(
            dataset_root=DATASET_ROOT,
            classname=part,
            model_name=model_name,
            models_root=MODELS_ROOT,
        )
        model_dir = Path(train_result['model_dir'])
        threshold = thresholds[part]
        update_model_threshold(model_dir, threshold)

        print(f'Detecting {part}, threshold={threshold}', flush=True)
        detect_results = detect_patchcore(
            image_paths=DATASET_ROOT / part / 'test',
            model_dir=model_dir,
            output_dir=HOT_PIC_ROOT / part,
            save_heatmap=True,
            image_threshold=threshold,
        )
        for result in detect_results:
            rows.append({
                'part': part,
                'image': result['image'],
                'image_score': result['image_score'],
                'image_threshold': threshold,
                'pred_is_anomaly': result['pred_is_anomaly'],
                'pred_label': result['pred_label'],
                'train_time_ms': train_result['train_time_ms'],
                'total_infer_time_ms': result['total_infer_time_ms'],
                'avg_infer_time_ms': result['avg_infer_time_ms'],
            })

    fieldnames = [
        'part',
        'image',
        'image_score',
        'image_threshold',
        'pred_is_anomaly',
        'pred_label',
        'train_time_ms',
        'total_infer_time_ms',
        'avg_infer_time_ms',
    ]
    with SUMMARY_CSV.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wrote {SUMMARY_CSV}', flush=True)
    print(f'Rows {len(rows)}', flush=True)


if __name__ == '__main__':
    main()
