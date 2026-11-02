"""Train PatchCore on the updated dataset, excluding flange_scratch."""

import csv
import shutil
from pathlib import Path

import numpy as np

from patchcore_api import detect_patchcore, train_patchcore


BASE_DIR = Path(__file__).resolve().parent
DATASET_ROOT = BASE_DIR / "data2" / "extracted_docx"
OUTPUT_ROOT = DATASET_ROOT / "patchcore_updated_output"
STAGING_ROOT = OUTPUT_ROOT / "mvtec_staging"
MODELS_ROOT = OUTPUT_ROOT / "models"
CLASSES = (
    "ceramic",
    "composite",
    "connector_J599_20FC",
    "connector_J599_26FA",
    "flange_gasket",
)
IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def image_files(directory):
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def prepare_staging():
    if STAGING_ROOT.exists():
        shutil.rmtree(STAGING_ROOT)
    for classname in CLASSES:
        target = STAGING_ROOT / classname / "train" / "good"
        target.mkdir(parents=True, exist_ok=True)
        for source in image_files(DATASET_ROOT / classname / "normal"):
            (target / source.name).symlink_to(source)


def load_source_rows():
    with (DATASET_ROOT / "classification.csv").open(
        "r", newline="", encoding="utf-8-sig"
    ) as handle:
        return {row["图片名称"]: row for row in csv.DictReader(handle)}


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    prepare_staging()
    source_rows = load_source_rows()
    output_rows = []

    for classname in CLASSES:
        print(f"Training {classname}", flush=True)
        train_result = train_patchcore(
            dataset_root=STAGING_ROOT,
            classname=classname,
            model_name=f"{classname}_patchcore",
            models_root=MODELS_ROOT,
            use_augmentation=True,
            device="cuda:0",
        )
        model_dir = Path(train_result["model_dir"])

        normal_results = detect_patchcore(
            DATASET_ROOT / classname / "normal",
            model_dir,
            save_heatmap=False,
            image_threshold=10**18,
            device="cuda:0",
        )
        threshold = float(
            np.percentile(
                [result["image_score"] for result in normal_results], 95
            )
        )

        all_images = image_files(DATASET_ROOT / classname / "normal")
        all_images += image_files(DATASET_ROOT / classname / "defect")
        print(f"Detecting {classname}, threshold={threshold:.6f}", flush=True)
        results = detect_patchcore(
            all_images,
            model_dir,
            output_dir=OUTPUT_ROOT / classname,
            save_heatmap=True,
            image_threshold=threshold,
            device="cuda:0",
        )
        for result in results:
            output_rows.append(
                {
                    **source_rows[result["image"]],
                    "类别": classname,
                    "本地图片路径": result["image_path"],
                    "PatchCore分数": result["image_score"],
                    "PatchCore阈值": threshold,
                    "预测是否异常": result["pred_is_anomaly"],
                    "预测标签": result["pred_label"],
                    "热力图路径": result["heatmap_path"],
                    "叠加图路径": result["overlay_path"],
                    "掩膜路径": result["mask_path"],
                    "训练图片数": train_result["train_images"],
                    "训练耗时毫秒": train_result["train_time_ms"],
                    "推理总耗时毫秒": result["total_infer_time_ms"],
                    "平均推理耗时毫秒": result["avg_infer_time_ms"],
                }
            )

    fieldnames = [
        "图片路径",
        "图片名称",
        "缺陷原因",
        "类别",
        "本地图片路径",
        "PatchCore分数",
        "PatchCore阈值",
        "预测是否异常",
        "预测标签",
        "热力图路径",
        "叠加图路径",
        "掩膜路径",
        "训练图片数",
        "训练耗时毫秒",
        "推理总耗时毫秒",
        "平均推理耗时毫秒",
    ]
    output_csv = OUTPUT_ROOT / "classification_patchcore.csv"
    with output_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"Wrote {output_csv}", flush=True)
    print(f"Rows {len(output_rows)}", flush=True)


if __name__ == "__main__":
    main()
