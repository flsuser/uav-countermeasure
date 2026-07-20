"""Train a YOLO UAV detector on the merged RflySim dataset."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CACHE_ROOT = ROOT / ".runtime_cache"
os.environ.setdefault("YOLO_CONFIG_DIR", str(CACHE_ROOT / "ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "matplotlib"))

import torch  # noqa: E402
import yaml  # noqa: E402
from ultralytics import YOLO  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--name", default="rfly_uav_yolo11n")
    parser.add_argument("--device", default="0")
    return parser.parse_args()


def write_runtime_dataset_yaml() -> Path:
    dataset_root = (ROOT / "datasets" / "rfly_uav_combined").resolve()
    for relative in ("images/train", "images/val", "labels/train", "labels/val"):
        if not (dataset_root / relative).exists():
            raise FileNotFoundError(dataset_root / relative)
    runtime_dir = ROOT / "runs"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    path = runtime_dir / "dataset_runtime.yaml"
    config = {
        "path": dataset_root.as_posix(),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "drone"},
    }
    path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    return path


def main() -> int:
    args = parse_args()
    print(f"torch={torch.__version__} cuda={torch.cuda.is_available()}")
    if args.device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("CUDA training requested, but torch.cuda.is_available() is False")
    if torch.cuda.is_available():
        print(f"gpu={torch.cuda.get_device_name(0)}")

    data = write_runtime_dataset_yaml()
    model = YOLO(args.model)
    model.train(
        data=str(data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str((ROOT / "runs").resolve()),
        name=args.name,
        pretrained=True,
        seed=42,
        deterministic=True,
        plots=True,
        patience=20,
        close_mosaic=10,
        cache=False,
        exist_ok=True,
    )
    best = ROOT / "runs" / args.name / "weights" / "best.pt"
    print(f"best_weights={best.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

