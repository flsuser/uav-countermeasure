"""Merge the verified baseline dataset with per-model train/val datasets."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def copy_split(source: Path, output: Path, split: str, prefix: str) -> int:
    labels = sorted((source / "labels" / split).glob("*.txt"))
    copied = 0
    for label in labels:
        image = source / "images" / split / f"{label.stem}.jpg"
        if not image.exists():
            raise FileNotFoundError(image)
        text = label.read_text(encoding="ascii").strip()
        values = text.split()
        # An empty label is a valid YOLO background-negative image.  Non-empty
        # labels must remain one-class UAV annotations.
        if text and (len(values) != 5 or values[0] != "0"):
            raise ValueError(f"Invalid one-class YOLO label: {label}")
        stem = f"{prefix}_{label.stem}"
        shutil.copy2(image, output / "images" / split / f"{stem}.jpg")
        shutil.copy2(label, output / "labels" / split / f"{stem}.txt")
        copied += 1
    return copied


def main() -> int:
    parser = argparse.ArgumentParser()
    datasets = ROOT / "datasets"
    parser.add_argument("--baseline", type=Path, default=datasets / "rfly_uav_combined")
    parser.add_argument("--models-root", type=Path, default=datasets / "rfly_models")
    parser.add_argument(
        "--output", type=Path, default=datasets / "rfly_uav_multimodel_combined"
    )
    parser.add_argument(
        "--model-repeat",
        type=int,
        default=1,
        help="Repeat each model-specific training split for balanced sampling; validation is never repeated",
    )
    args = parser.parse_args()

    if args.model_repeat < 1:
        parser.error("--model-repeat must be at least 1")

    model_dirs = sorted(path for path in args.models_root.iterdir() if path.is_dir())
    if not model_dirs:
        raise FileNotFoundError(f"No model datasets found in {args.models_root}")
    if args.output.exists():
        shutil.rmtree(args.output)
    for split in ("train", "val"):
        (args.output / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.output / "labels" / split).mkdir(parents=True, exist_ok=True)

    manifest: dict[str, dict[str, int]] = {}
    manifest["baseline"] = {
        split: copy_split(args.baseline, args.output, split, "baseline")
        for split in ("train", "val")
    }
    for model_dir in model_dirs:
        train_count = sum(
            copy_split(
                model_dir,
                args.output,
                "train",
                f"{model_dir.name}_repeat{repeat_index + 1}",
            )
            for repeat_index in range(args.model_repeat)
        )
        counts = {
            "train": train_count,
            "val": copy_split(model_dir, args.output, "val", model_dir.name),
        }
        if not counts["train"] or not counts["val"]:
            raise RuntimeError(f"{model_dir.name} must contain both train and val images")
        manifest[model_dir.name] = counts

    (args.output / "sources.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    train_total = sum(item["train"] for item in manifest.values())
    val_total = sum(item["val"] for item in manifest.values())
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"Multi-model dataset: train={train_total}, val={val_total}, total={train_total + val_total}")
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
