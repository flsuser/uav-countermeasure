"""Report detector recall separately for each seen and unseen UAV appearance."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".runtime_cache" / "ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".runtime_cache" / "matplotlib"))

from ultralytics import YOLO  # noqa: E402


def parse_truth(path: Path) -> np.ndarray:
    values = [float(value) for value in path.read_text(encoding="ascii").split()]
    if len(values) != 5 or int(values[0]) != 0:
        raise ValueError(path)
    _, cx, cy, width, height = values
    return np.array([cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2])


def iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = np.maximum(a[:2], b[:2])
    x2, y2 = np.minimum(a[2:], b[2:])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / (union + 1e-9)


def evaluate(
    model: YOLO,
    dataset: Path,
    split: str,
    conf: float,
    imgsz: int,
    iou_threshold: float,
) -> tuple[int, int, int, float]:
    images = sorted((dataset / "images" / split).glob("*.jpg"))
    if not images:
        return 0, 0, 0.0
    matched = 0
    center_matched = 0
    target_confidences: list[float] = []
    for image_path in images:
        truth = parse_truth(dataset / "labels" / split / f"{image_path.stem}.txt")
        # Give Ultralytics the path so it performs the same OpenCV/BGR loading
        # used during training and live RflySim inference.  Passing a PIL RGB
        # array here swaps red and blue because ndarray inputs are treated as
        # BGR images by the predictor.
        result = model.predict(str(image_path), imgsz=imgsz, conf=0.01, verbose=False)[0]
        best_confidence = 0.0
        best_center_confidence = 0.0
        for box, confidence in zip(
            result.boxes.xyxyn.cpu().numpy(), result.boxes.conf.cpu().numpy()
        ):
            if iou(truth, box) >= iou_threshold:
                best_confidence = max(best_confidence, float(confidence))
            center_x = (box[0] + box[2]) / 2
            center_y = (box[1] + box[3]) / 2
            if (
                truth[0] <= center_x <= truth[2]
                and truth[1] <= center_y <= truth[3]
            ):
                best_center_confidence = max(
                    best_center_confidence, float(confidence)
                )
        target_confidences.append(best_confidence)
        matched += int(best_confidence >= conf)
        center_matched += int(best_center_confidence >= conf)
    return matched, center_matched, len(images), float(np.median(target_confidences))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--models-root", type=Path, default=ROOT / "datasets" / "rfly_models")
    parser.add_argument("--unseen-root", type=Path, default=ROOT / "datasets" / "rfly_unseen")
    parser.add_argument("--conf", type=float, default=0.18)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="Minimum truth/prediction IoU counted as a matched target",
    )
    args = parser.parse_args()

    model = YOLO(str(args.weights))
    rows: list[tuple[str, str, int, int, float]] = []
    if args.models_root.exists():
        for dataset in sorted(path for path in args.models_root.iterdir() if path.is_dir()):
            matched, center_matched, total, median_conf = evaluate(
                model, dataset, "val", args.conf, args.imgsz, args.iou_threshold
            )
            rows.append(
                (dataset.name, "seen-val", matched, center_matched, total, median_conf)
            )
    if args.unseen_root.exists():
        for dataset in sorted(path for path in args.unseen_root.iterdir() if path.is_dir()):
            matched, center_matched, total, median_conf = evaluate(
                model, dataset, "test", args.conf, args.imgsz, args.iou_threshold
            )
            rows.append(
                (dataset.name, "unseen-test", matched, center_matched, total, median_conf)
            )
    if not rows:
        raise FileNotFoundError("No per-model validation or unseen test datasets found")

    print("model,split,matched,total,recall,center_recall,median_target_conf")
    for name, split, matched, center_matched, total, median_conf in rows:
        recall = matched / total if total else 0.0
        center_recall = center_matched / total if total else 0.0
        print(
            f"{name},{split},{matched},{total},{recall:.3%},"
            f"{center_recall:.3%},{median_conf:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
