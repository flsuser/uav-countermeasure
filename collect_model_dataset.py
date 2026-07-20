"""Collect truth-labelled images for one selectable RflySim3D UAV model.

Train and validation frames are collected from separate complete trajectories.
Use --test-only for a model that must remain unseen during training.
"""

from __future__ import annotations

import argparse
import copy
import math
import re
import shutil
import threading
import time
from pathlib import Path

import cv2
import numpy as np

import PX4MavCtrlV4 as PX4MavCtrl
import UE4CtrlAPI
import VisionCaptureApi

from collect_dataset import (
    OBSERVER_ID,
    TARGET_ID,
    clamp_box,
    project_target_box,
    wait_for_takeoff,
    write_jpeg,
    yolo_label,
)


ROOT = Path(__file__).resolve().parent
MODEL_NAMES = {
    "F450_Default",
    "Droneyee_QuadX330",
    "Droneyee_QuadX200",
    "DJI_Phantom_White",
    "DJI_Phantom_Red",
    "DJI_Phantom_Army",
    "AR_Drone_BlackGreen",
    "AR_Drone_Blue",
    "AR_Drone_Army",
    "Quad_Fold_Gray",
    "Quad_Fold_Army",
}
TRAIN_WAYPOINTS = [
    np.array([5.0, -2.0, -18.0]),
    np.array([8.0, 2.5, -22.0]),
    np.array([12.0, -4.0, -19.0]),
    np.array([18.0, 4.5, -23.0]),
    np.array([25.0, -5.0, -18.0]),
    np.array([30.0, 3.0, -24.0]),
    np.array([15.0, 0.0, -21.0]),
]
VAL_WAYPOINTS = [
    np.array([6.0, 3.5, -23.0]),
    np.array([11.0, -3.0, -17.0]),
    np.array([20.0, 1.0, -24.0]),
    np.array([28.0, -6.0, -20.0]),
    np.array([14.0, 5.0, -18.0]),
]
TEST_WAYPOINTS = [
    np.array([7.0, -4.0, -22.0]),
    np.array([13.0, 4.5, -18.0]),
    np.array([22.0, -2.0, -24.0]),
    np.array([31.0, 5.0, -19.0]),
    np.array([17.0, -5.5, -21.0]),
]


def slugify(model: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", model.lower()).strip("_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="RflySim3D model name")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--val-ratio", type=float, default=0.20)
    parser.add_argument("--frame-stride", type=int, default=4)
    parser.add_argument("--speed", type=float, default=2.0)
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "Config_gimbal.json")
    parser.add_argument("--clear", action="store_true")
    parser.add_argument("--no-preview", action="store_true")
    args = parser.parse_args()
    if args.frames <= 0 or args.frame_stride <= 0:
        parser.error("--frames and --frame-stride must be positive")
    if not 0.05 <= args.val_ratio <= 0.5:
        parser.error("--val-ratio must be between 0.05 and 0.5")
    if args.output is None:
        parent = "rfly_unseen" if args.test_only else "rfly_models"
        args.output = ROOT / "datasets" / parent / slugify(args.model)
    if args.model not in MODEL_NAMES:
        parser.error(
            "--model must be a ClassID 3 hot-swappable model: "
            + ", ".join(sorted(MODEL_NAMES))
        )
    return args


def trajectory(
    mav: PX4MavCtrl.PX4MavCtrler,
    waypoints: list[np.ndarray],
    stop: threading.Event,
    speed: float,
) -> None:
    index = 0
    mav.SendCopterSpeed(speed)
    while not stop.is_set():
        waypoint = waypoints[index]
        mav.SendPosNED(*waypoint, 0)
        position = np.asarray(mav.uavPosNED, dtype=np.float64)
        if position.shape == (3,) and np.linalg.norm(waypoint - position) < 1.2:
            index = (index + 1) % len(waypoints)
        stop.wait(0.08)


def ensure_split(root: Path, split: str) -> None:
    (root / "images" / split).mkdir(parents=True, exist_ok=True)
    (root / "labels" / split).mkdir(parents=True, exist_ok=True)


def collect_split(
    *,
    split: str,
    count: int,
    start_index: int,
    waypoints: list[np.ndarray],
    args: argparse.Namespace,
    vision: VisionCaptureApi.VisionCaptureApi,
    target: UE4CtrlAPI.CoptReqData,
    camera: UE4CtrlAPI.CameraData,
    mavs: list[PX4MavCtrl.PX4MavCtrler],
) -> int:
    ensure_split(args.output, split)
    stop = threading.Event()
    motion = threading.Thread(
        target=trajectory,
        args=(mavs[1], waypoints, stop, args.speed),
        daemon=True,
    )
    motion.start()
    saved = 0
    unique_frames = 0
    last_stamp = None
    model_slug = slugify(args.model)
    try:
        while saved < count:
            if mavs[0].uavPosNED[2] > -15.0 or mavs[1].uavPosNED[2] > -15.0:
                mavs[0].SendPosNED(0, 0, -20, 0)
                time.sleep(0.1)
                continue
            if not vision.hasData[0]:
                time.sleep(0.01)
                continue
            stamp = vision.timeStmp[0]
            if stamp == last_stamp:
                time.sleep(0.005)
                continue
            last_stamp = stamp
            unique_frames += 1
            if unique_frames % args.frame_stride:
                continue

            frame = copy.deepcopy(vision.Img[0])
            height, width = frame.shape[:2]
            box = clamp_box(project_target_box(target, camera), width, height)
            if box is None:
                continue
            x1, y1, x2, y2 = box
            if x1 <= 2 or y1 <= 2 or x2 >= width - 3 or y2 >= height - 3:
                continue

            index = start_index + saved
            stem = f"{model_slug}_{index:06d}"
            write_jpeg(args.output / "images" / split / f"{stem}.jpg", frame)
            (args.output / "labels" / split / f"{stem}.txt").write_text(
                yolo_label(box, width, height), encoding="ascii"
            )
            saved += 1

            if not args.no_preview:
                preview = frame.copy()
                cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    preview,
                    f"{args.model}  {split} {saved}/{count}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow("RflySim multi-model collector", preview)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    raise KeyboardInterrupt
            if saved % 25 == 0:
                print(f"{args.model}: saved {saved}/{count} ({split})")
    finally:
        stop.set()
        motion.join(timeout=1.0)
    return saved


def main() -> int:
    args = parse_args()
    if args.clear and args.output.exists():
        shutil.rmtree(args.output)
    if args.output.exists() and any(args.output.rglob("*.jpg")):
        raise FileExistsError(
            f"Dataset already contains images: {args.output}. Use --clear to replace it."
        )

    vision = VisionCaptureApi.VisionCaptureApi()
    ue = UE4CtrlAPI.UE4CtrlAPI()
    if not vision.jsonLoad(jsonPath=str(args.config.resolve())):
        raise RuntimeError(f"Failed to load vision config: {args.config}")
    if not vision.sendReqToUE4():
        raise RuntimeError("RflySim3D rejected the RGB camera request")
    vision.startImgCap()

    mavs = [
        PX4MavCtrl.PX4MavCtrler(OBSERVER_ID),
        PX4MavCtrl.PX4MavCtrler(TARGET_ID),
    ]
    try:
        for mav in mavs:
            mav.InitMavLoop()
            mav.InitTrueDataLoop()
        time.sleep(2)
        for mav in mavs:
            mav.initOffboard()
        time.sleep(2)
        print("Taking off before model-specific collection...")
        wait_for_takeoff(mavs)

        command = f"RflyChange3DModel {TARGET_ID} {args.model}".encode("ascii")
        for _ in range(3):
            ue.sendUE4Cmd(command)
            time.sleep(0.1)
        print(f"Target appearance changed to: {args.model}")
        time.sleep(2)

        ue.reqCamCoptObj(1, [OBSERVER_ID, TARGET_ID])
        ue.reqCamCoptObj(0, 0)
        time.sleep(2)
        ue.initUE4MsgRec()
        time.sleep(3)
        target = ue.getCamCoptObj(1, TARGET_ID)
        camera = ue.getCamCoptObj(0, 0)
        if not isinstance(target, UE4CtrlAPI.CoptReqData):
            raise RuntimeError("Failed to receive target truth data after model switch")
        if not isinstance(camera, UE4CtrlAPI.CameraData):
            raise RuntimeError("Failed to receive camera pose data")
        extent = np.asarray(target.BoxExtent, dtype=np.float64)
        if extent.shape != (3,) or not np.all(np.isfinite(extent)) or np.any(extent <= 0):
            raise RuntimeError(f"Invalid truth bounding extent for {args.model}: {extent}")
        print(f"Truth BoxExtent={extent.tolist()}")

        total = 0
        if args.test_only:
            total += collect_split(
                split="test",
                count=args.frames,
                start_index=0,
                waypoints=TEST_WAYPOINTS,
                args=args,
                vision=vision,
                target=target,
                camera=camera,
                mavs=mavs,
            )
        else:
            val_count = max(1, round(args.frames * args.val_ratio))
            train_count = args.frames - val_count
            total += collect_split(
                split="train",
                count=train_count,
                start_index=0,
                waypoints=TRAIN_WAYPOINTS,
                args=args,
                vision=vision,
                target=target,
                camera=camera,
                mavs=mavs,
            )
            print("Switching to the independent validation trajectory...")
            time.sleep(1)
            total += collect_split(
                split="val",
                count=val_count,
                start_index=train_count,
                waypoints=VAL_WAYPOINTS,
                args=args,
                vision=vision,
                target=target,
                camera=camera,
                mavs=mavs,
            )
        print(f"Dataset ready: {args.output.resolve()} ({total} labelled images)")
        return 0 if total == args.frames else 1
    except KeyboardInterrupt:
        print("Collection stopped by user; incomplete data should be recollected with --clear")
        return 130
    finally:
        cv2.destroyAllWindows()
        for mav in mavs:
            try:
                mav.endOffboard()
                mav.endMavLoop()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
