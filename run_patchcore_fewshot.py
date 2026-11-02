"""Run 1-shot, 3-shot, and 5-shot PatchCore experiments."""

import csv
import shutil
import time
from pathlib import Path

import numpy as np

from patchcore_api import detect_patchcore, train_patchcore


BASE_DIR = Path(__file__).resolve().parent
DATASET_ROOT = BASE_DIR / "data2" / "extracted_docx"
OUTPUT_ROOT = DATASET_ROOT / "patchcore_fewshot_output"
STAGING_ROOT = OUTPUT_ROOT / "staging"
MODELS_ROOT = OUTPUT_ROOT / "models"
CLASSES = ("ceramic", "composite", "connector", "flange")
SHOTS = (1, 3, 5)
IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def image_files(directory: Path):
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def prepare_staging(shot: int):
    root = STAGING_ROOT / f"{shot}_shot"
    if root.exists():
        shutil.rmtree(root)
    for classname in CLASSES:
        train_good = root / classname / "train" / "good"
        train_good.mkdir(parents=True, exist_ok=True)
        normal_images = image_files(DATASET_ROOT / classname / "normal")
        for source in normal_images[:shot]:
            (train_good / source.name).symlink_to(source)
    return root


def load_source_rows():
    with (DATASET_ROOT / "classification.csv").open(
        "r", newline="", encoding="utf-8-sig"
    ) as handle:
        return {row["图片名称"]: row for row in csv.DictReader(handle)}


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    source_rows = load_source_rows()
    summary_rows = []

    for shot in SHOTS:
        staging_root = prepare_staging(shot)
        shot_root = OUTPUT_ROOT / f"{shot}_shot"
        shot_root.mkdir(parents=True, exist_ok=True)
        print(f"\n===== {shot}-shot =====", flush=True)

        for classname in CLASSES:
            model_start = time.perf_counter()
            train_result = train_patchcore(
                dataset_root=staging_root,
                classname=classname,
                model_name=f"{classname}_{shot}shot",
                models_root=MODELS_ROOT,
                use_augmentation=False,
                device="cuda:0",
            )
            train_wall_ms = (time.perf_counter() - model_start) * 1000
            model_dir = Path(train_result["model_dir"])

            normal_results = detect_patchcore(
                image_paths=DATASET_ROOT / classname / "normal",
                model_dir=model_dir,
                save_heatmap=False,
                image_threshold=10**18,
                device="cuda:0",
            )
            threshold = float(
                np.percentile(
                    np.asarray(
                        [result["image_score"] for result in normal_results],
                        dtype=np.float32,
                    ),
                    95,
                )
            )

            all_images = image_files(DATASET_ROOT / classname / "normal")
            all_images += image_files(DATASET_ROOT / classname / "defect")
            infer_start = time.perf_counter()
            results = detect_patchcore(
                image_paths=all_images,
                model_dir=model_dir,
                output_dir=shot_root / classname,
                save_heatmap=True,
                image_threshold=threshold,
                device="cuda:0",
            )
            infer_wall_ms = (time.perf_counter() - infer_start) * 1000

            rows = []
            for result in results:
                source_row = source_rows[result["image"]]
                rows.append(
                    {
                        **source_row,
                        "类别": classname,
                        "shot": shot,
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
                        "训练墙钟耗时毫秒": train_wall_ms,
                        "推理总耗时毫秒": result["total_infer_time_ms"],
                        "本次推理墙钟耗时毫秒": infer_wall_ms,
                        "平均推理耗时毫秒": result["avg_infer_time_ms"],
                    }
                )

            csv_path = shot_root / f"{classname}_classification.csv"
            fieldnames = list(rows[0].keys())
            with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            summary_rows.append(
                {
                    "shot": shot,
                    "类别": classname,
                    "训练图片数": train_result["train_images"],
                    "训练耗时毫秒": train_result["train_time_ms"],
                    "训练墙钟耗时毫秒": train_wall_ms,
                    "推理图片数": len(results),
                    "推理总耗时毫秒": infer_wall_ms,
                    "平均推理耗时毫秒": infer_wall_ms / len(results),
                    "总耗时毫秒": train_wall_ms + infer_wall_ms,
                    "PatchCore阈值": threshold,
                    "CSV路径": str(csv_path),
                }
            )
            print(
                f"{classname}: train={train_wall_ms / 1000:.2f}s, "
                f"infer={infer_wall_ms / 1000:.2f}s",
                flush=True,
            )

    summary_path = OUTPUT_ROOT / "fewshot_timing_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nWrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
