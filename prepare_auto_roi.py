"""Standalone foreground cropping for the five-class updated dataset.

Run beside run_patchcore_updated_dataset.py on the server:
    python prepare_auto_roi.py
Only numpy, Pillow and scipy are required; no GPU or PatchCore import.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy import ndimage


CLASSES = ("ceramic", "composite", "connector_J599_20FC",
           "connector_J599_26FA", "flange_gasket")
EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}
DEFAULT_ROOT = Path(__file__).resolve().parent / "data2" / "extracted_docx"


def locate(image, margin=0.15, min_difference=18.0):
    """Estimate background from borders. Coordinates refer to oriented input.

    Work at reduced resolution, retain significant disconnected components,
    and flag ambiguous images instead of silently accepting their crops.
    """
    small = image.copy()
    small.thumbnail((768, 768))
    pixels = np.asarray(small, dtype=np.float32)
    height, width = pixels.shape[:2]
    band = max(1, int(min(height, width) * 0.03))
    border = np.concatenate((pixels[:band].reshape(-1, 3),
                             pixels[-band:].reshape(-1, 3),
                             pixels[:, :band].reshape(-1, 3),
                             pixels[:, -band:].reshape(-1, 3)))
    background = np.median(border, axis=0)
    distances = np.linalg.norm(pixels - background, axis=2)
    border_distances = np.linalg.norm(border - background, axis=1)
    median = float(np.median(border_distances))
    mad = float(np.median(np.abs(border_distances - median)))
    threshold = max(min_difference, median + 4 * 1.4826 * mad)
    mask = distances > threshold
    mask = ndimage.binary_closing(mask, iterations=2)
    mask = ndimage.binary_fill_holes(mask)
    labels, count = ndimage.label(mask)
    sizes = np.bincount(labels.ravel())
    if count == 0:
        return None, {"status": "no_foreground", "threshold": threshold}
    sizes[0] = 0
    largest = int(sizes.max())
    # Keep other substantial pieces; a connector may contain separated pins.
    keep = np.flatnonzero(sizes >= max(16, height * width * 0.001, largest * 0.05))
    keep = keep[keep != 0]
    if not len(keep):
        return None, {"status": "too_small", "threshold": threshold}
    foreground = np.isin(labels, keep)
    ys, xs = np.where(foreground)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    fraction = float(foreground.mean())
    flags = []
    if fraction < 0.005:
        flags.append("small_foreground")
    if (x1 - x0) * (y1 - y0) / (width * height) > 0.9:
        flags.append("nearly_full_image")
    if x0 <= 2 or y0 <= 2 or x1 >= width - 2 or y1 >= height - 2:
        flags.append("touching_border")
    if len(keep) > 1:
        flags.append("multiple_components")
    dx, dy = (x1 - x0) * margin, (y1 - y0) * margin
    sx, sy = image.width / width, image.height / height
    box = (max(0, int(np.floor((x0 - dx) * sx))),
           max(0, int(np.floor((y0 - dy) * sy))),
           min(image.width, int(np.ceil((x1 + dx) * sx))),
           min(image.height, int(np.ceil((y1 + dy) * sy))))
    return box, {"status": "review" if flags else "ok", "flags": flags,
                 "foreground_fraction": fraction, "components": len(keep),
                 "threshold": threshold, "background_rgb": background.tolist()}


def crop_and_pad(image, box, size, background):
    crop = image.crop(box)
    scale = min(size[0] / crop.width, size[1] / crop.height)
    resized = crop.resize((max(1, round(crop.width * scale)),
                           max(1, round(crop.height * scale))), Image.Resampling.LANCZOS)
    offset = ((size[0] - resized.width) // 2, (size[1] - resized.height) // 2)
    canvas = Image.new("RGB", size, tuple(int(v) for v in background))
    canvas.paste(resized, offset)
    return canvas, {"resized_size": list(resized.size), "padding_left_top": offset,
                    "scale_xy": [resized.width / crop.width, resized.height / crop.height]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-root", type=Path, help="New directory; existing output is refused")
    parser.add_argument("--classes", nargs="+", default=list(CLASSES))
    parser.add_argument("--margin", type=float, default=0.15, help="Margin per side")
    parser.add_argument("--min-difference", type=float, default=18.0)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--limit", type=int, default=0, help="Maximum images per class/split; 0 = all")
    args = parser.parse_args()
    root = args.dataset_root.resolve()
    output = (args.output_root or root.with_name(root.name + "_auto_roi")).resolve()
    if output == root or root in output.parents:
        parser.error("Output must be outside the source dataset")
    if args.width < 1 or args.height < 1 or args.margin < 0 or args.min_difference <= 0 or args.limit < 0:
        parser.error("Invalid size, margin, difference or limit")
    sources = []
    for classname in args.classes:
        for split in ("normal", "defect"):
            directory = root / classname / split
            if not directory.is_dir():
                parser.error(f"Missing source directory: {directory}")
            files = sorted(p for p in directory.iterdir() if p.suffix.lower() in EXTENSIONS and p.is_file())
            if not files:
                parser.error(f"No images: {directory}")
            sources.extend((classname, split, p) for p in (files[:args.limit] if args.limit else files))
    if output.exists():
        parser.error(f"Output already exists; choose a new --output-root: {output}")
    output.mkdir(parents=True)
    records = []
    for classname, split, source in sources:
        relative = source.relative_to(root)
        record = {"source": str(source), "class": classname, "split": split}
        try:
            with Image.open(source) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB")
            box, info = locate(image, args.margin, args.min_difference)
            record.update(info)
            record["original_size"] = list(image.size)
            preview = image.copy()
            if box is not None:
                ImageDraw.Draw(preview).rectangle(box, outline="red", width=max(2, image.width // 250))
                crop, transform = crop_and_pad(image, box, (args.width, args.height), info["background_rgb"])
                record.update(transform)
                record["box_xyxy"] = list(box)
                # Ambiguous crops are quarantined until inspected.
                category = "dataset" if info["status"] == "ok" else "review"
                destination = output / category / relative.parent / (relative.name + ".png")
                destination.parent.mkdir(parents=True, exist_ok=True)
                crop.save(destination)
                record["output"] = str(destination)
            preview.thumbnail((1000, 750))
            preview_path = output / "previews" / relative.parent / (relative.name + ".jpg")
            preview_path.parent.mkdir(parents=True, exist_ok=True)
            preview.save(preview_path)
        except Exception as error:
            record.update(status="error", error=str(error))
        records.append(record)
        print(f"[{len(records)}/{len(sources)}] {record['status']}: {relative}", flush=True)
    with (output / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    with (output / "summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["class", "split", "source", "status", "output", "flags", "error"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    counts = {status: sum(r["status"] == status for r in records) for status in sorted({r["status"] for r in records})}
    (output / "config.json").write_text(json.dumps({**vars(args), "dataset_root": str(root), "output_root": str(output)}, default=str, indent=2), encoding="utf-8")
    print(f"\nOutput: {output}\nCounts: {counts}\nInspect previews and review/ before training.")


if __name__ == "__main__":
    main()
