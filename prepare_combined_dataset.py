"""Merge the long-range and close-range RflySim datasets for training."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_label(path: Path) -> tuple[float, float] | None:
    text = path.read_text(encoding="ascii").strip()
    if not text:
        return None
    values = [float(value) for value in text.split()]
    if len(values) != 5 or int(values[0]) != 0:
        raise ValueError(f"Invalid YOLO label: {path}")
    return values[3] * 640.0, values[4] * 480.0


def merge_source(
    source: Path,
    destination: Path,
    prefix: str,
    min_width: float = 0.0,
    min_height: float = 0.0,
) -> tuple[int, int]:
    copied = 0
    filtered = 0
    for split in ("train", "val"):
        for label in sorted((source / "labels" / split).glob("*.txt")):
            size = parse_label(label)
            if size is not None and (size[0] < min_width or size[1] < min_height):
                filtered += 1
                continue
            image = source / "images" / split / f"{label.stem}.jpg"
            if not image.exists():
                raise FileNotFoundError(image)
            new_stem = f"{prefix}_{label.stem}"
            shutil.copy2(image, destination / "images" / split / f"{new_stem}.jpg")
            shutil.copy2(label, destination / "labels" / split / f"{new_stem}.txt")
            copied += 1
    return copied, filtered


def main() -> int:
    parser = argparse.ArgumentParser()
    base = Path(__file__).resolve().parent / "datasets"
    parser.add_argument("--far", type=Path, default=base / "rfly_uav")
    parser.add_argument("--close", type=Path, default=base / "rfly_uav_close")
    parser.add_argument("--negatives", type=Path, default=base / "rfly_hard_negatives")
    parser.add_argument("--mountain", type=Path, default=base / "rfly_mountain_crossing")
    parser.add_argument("--output", type=Path, default=base / "rfly_uav_combined")
    args = parser.parse_args()

    if args.output.exists():
        shutil.rmtree(args.output)
    for split in ("train", "val"):
        (args.output / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.output / "labels" / split).mkdir(parents=True, exist_ok=True)

    far_copied, far_filtered = merge_source(args.far, args.output, "far")
    close_copied, close_filtered = merge_source(
        args.close,
        args.output,
        "close",
        min_width=20.0,
        min_height=10.0,
    )
    negative_copied = 0
    if args.negatives.exists():
        negative_copied, _ = merge_source(
            args.negatives,
            args.output,
            "negative",
        )
    mountain_copied = 0
    if args.mountain.exists():
        mountain_copied, _ = merge_source(
            args.mountain,
            args.output,
            "mountain",
        )
    print(
        f"Combined dataset: {far_copied + close_copied + negative_copied + mountain_copied} images "
        f"({mountain_copied} mountain hard positives, "
        f"{negative_copied} background negatives); filtered "
        f"{far_filtered + close_filtered} edge fragments"
    )
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
