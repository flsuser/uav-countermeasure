"""Measure false alarms on drone-free background validation images."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".runtime_cache"))

from ultralytics import YOLO


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        type=Path,
        default=ROOT / "runs" / "rfly_uav_yolo11n" / "weights" / "best.pt",
    )
    parser.add_argument(
        "--images",
        type=Path,
        default=ROOT / "datasets" / "rfly_hard_negatives" / "images" / "val",
    )
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    images = sorted(args.images.resolve().glob("*.jpg"))
    if not images:
        raise RuntimeError(f"No background validation images found in {args.images}")
    model = YOLO(str(args.weights.resolve()))
    false_images = 0
    false_boxes = 0
    maximum_confidence = 0.0
    results = model.predict(
        source=[str(path) for path in images],
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        half=args.device != "cpu",
        verbose=False,
        stream=True,
    )
    for result in results:
        count = 0 if result.boxes is None else len(result.boxes)
        if count:
            false_images += 1
            false_boxes += count
            maximum_confidence = max(
                maximum_confidence,
                float(result.boxes.conf.max().detach().cpu()),
            )

    rate = false_images / len(images)
    print(f"weights={args.weights.resolve()}")
    print(f"background_images={len(images)}")
    print(f"false_alarm_images={false_images}")
    print(f"false_alarm_image_rate={rate:.3%}")
    print(f"false_boxes={false_boxes}")
    print(f"max_false_confidence={maximum_confidence:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
