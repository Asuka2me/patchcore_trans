from pathlib import Path
import csv
import json

import numpy as np

from patchcore_api import detect_patchcore


BASE = Path('/home/ubuntu/wmh/patchcore/patch_backage')
DATASET_ROOT = BASE / 'data_fortrain3'
MODELS_ROOT = BASE / 'models'
OUTPUT_ROOT = BASE / 'hot_pic_calibrated_p95'
CSV_PATH = BASE / 'part1_part8_calibrated_p95_results.csv'
CSV_PATH_HOT = OUTPUT_ROOT / 'part1_part8_calibrated_p95_results.csv'
PERCENTILE = 95

FIELDNAMES = [
    'part',
    'image',
    'image_score',
    'image_threshold',
    'threshold_percentile',
    'calibration_images',
    'calibration_score_mean',
    'calibration_score_max',
    'pred_is_anomaly',
    'pred_label',
    'total_infer_time_ms',
    'avg_infer_time_ms',
]


def model_dir_for(part):
    preferred = MODELS_ROOT / f'{part}_patchcore'
    if preferred.exists():
        return preferred
    return MODELS_ROOT / f'{part}_patchcore_retrained'


def update_metadata(model_dir, threshold, part, stats):
    metadata_path = model_dir / 'metadata.json'
    if not metadata_path.exists():
        return
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    metadata['threshold_source'] = f'train/good normal scores p{PERCENTILE}'
    metadata['image_threshold'] = float(threshold)
    metadata['calibration_part'] = part
    metadata['calibration_images'] = int(stats['count'])
    metadata['calibration_score_mean'] = float(stats['mean'])
    metadata['calibration_score_max'] = float(stats['max'])
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx in range(1, 9):
        part = f'part{idx}'
        model_dir = model_dir_for(part)
        normal_dir = DATASET_ROOT / part / 'train' / 'good'
        test_dir = DATASET_ROOT / part / 'test'
        print(f'Calibrating {part} from {normal_dir}', flush=True)
        calibration = detect_patchcore(
            image_paths=normal_dir,
            model_dir=model_dir,
            save_heatmap=False,
            image_threshold=10**18,
        )
        normal_scores = np.asarray([item['image_score'] for item in calibration], dtype=np.float32)
        threshold = float(np.percentile(normal_scores, PERCENTILE))
        stats = {
            'count': len(normal_scores),
            'mean': float(np.mean(normal_scores)),
            'max': float(np.max(normal_scores)),
        }
        update_metadata(model_dir, threshold, part, stats)

        print(f'Detecting {part}, p{PERCENTILE} threshold={threshold}', flush=True)
        detections = detect_patchcore(
            image_paths=test_dir,
            model_dir=model_dir,
            output_dir=OUTPUT_ROOT / part,
            save_heatmap=True,
            image_threshold=threshold,
        )
        for result in detections:
            rows.append({
                'part': part,
                'image': result['image'],
                'image_score': result['image_score'],
                'image_threshold': threshold,
                'threshold_percentile': PERCENTILE,
                'calibration_images': stats['count'],
                'calibration_score_mean': stats['mean'],
                'calibration_score_max': stats['max'],
                'pred_is_anomaly': result['pred_is_anomaly'],
                'pred_label': result['pred_label'],
                'total_infer_time_ms': result['total_infer_time_ms'],
                'avg_infer_time_ms': result['avg_infer_time_ms'],
            })

    with CSV_PATH.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    CSV_PATH_HOT.write_text(CSV_PATH.read_text(encoding='utf-8'), encoding='utf-8')
    print(f'Wrote {CSV_PATH}', flush=True)
    print(f'Wrote {CSV_PATH_HOT}', flush=True)
    print(f'Rows {len(rows)}', flush=True)


if __name__ == '__main__':
    main()
