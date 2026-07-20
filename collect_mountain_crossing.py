"""Collect automatically labelled UAV mountain-crossing hard positives."""

from __future__ import annotations

import argparse
import copy
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
    prepare_dataset,
    project_target_box,
    wait_for_takeoff,
    write_jpeg,
    yolo_label,
)


ROOT = Path(__file__).resolve().parent


class CrossingState:
    def __init__(self) -> None:
        self.collect_rising = False
        self.crossing_index = 0


def crossing_trajectory(mav, stop: threading.Event, state: CrossingState, speed: float) -> None:
    """Reposition downward without saving, then save each upward crossing."""
    crossings = [
        (np.array([8.0, -1.0, -17.0]), np.array([8.0, -1.0, -23.0])),
        (np.array([10.0, 1.5, -16.5]), np.array([10.0, 1.5, -23.5])),
        (np.array([12.0, -2.0, -16.0]), np.array([12.0, -2.0, -24.0])),
        (np.array([14.0, 2.0, -16.0]), np.array([14.0, 2.0, -24.0])),
    ]
    mav.SendCopterSpeed(speed)
    phase = "reposition"
    target = crossings[0][0]
    while not stop.is_set():
        mav.SendPosNED(*target, 0.0)
        position = np.asarray(mav.uavPosNED, dtype=np.float64)
        if position.shape == (3,) and np.linalg.norm(target - position) < 0.7:
            if phase == "reposition":
                phase = "rising"
                state.collect_rising = True
                target = crossings[state.crossing_index][1]
            else:
                state.collect_rising = False
                state.crossing_index = (state.crossing_index + 1) % len(crossings)
                phase = "reposition"
                target = crossings[state.crossing_index][0]
        stop.wait(0.08)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect mountain-crossing UAV hard positives")
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--frame-stride", type=int, default=4)
    parser.add_argument("--val-crossing", type=int, default=4,
                        help="1-based crossing variant reserved for validation")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--min-center-y", type=float, default=0.20)
    parser.add_argument("--max-center-y", type=float, default=0.68)
    parser.add_argument("--clear", action="store_true")
    parser.add_argument("--config", type=Path, default=ROOT / "Config_gimbal.json")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "datasets" / "rfly_mountain_crossing",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.frames <= 0 or args.speed <= 0:
        raise ValueError("--frames and --speed must be positive")
    if not 1 <= args.val_crossing <= 4:
        raise ValueError("--val-crossing must be between 1 and 4")
    if args.clear and args.output.exists():
        import shutil
        shutil.rmtree(args.output)
    prepare_dataset(args.output)

    vision = VisionCaptureApi.VisionCaptureApi()
    ue = UE4CtrlAPI.UE4CtrlAPI()
    if not vision.jsonLoad(jsonPath=str(args.config.resolve())):
        raise RuntimeError(f"Failed to load vision config: {args.config}")
    if not vision.sendReqToUE4():
        raise RuntimeError("RflySim3D rejected the RGB camera request")
    vision.startImgCap()

    mavs = [PX4MavCtrl.PX4MavCtrler(OBSERVER_ID), PX4MavCtrl.PX4MavCtrler(TARGET_ID)]
    for mav in mavs:
        mav.InitMavLoop()
        mav.InitTrueDataLoop()
    time.sleep(2)
    for mav in mavs:
        mav.initOffboard()
    time.sleep(2)
    print("Taking off before mountain-crossing collection...")
    wait_for_takeoff(mavs)

    ue.reqCamCoptObj(1, [OBSERVER_ID, TARGET_ID])
    ue.reqCamCoptObj(0, 0)
    time.sleep(2)
    ue.initUE4MsgRec()
    time.sleep(3)
    target_truth = ue.getCamCoptObj(1, TARGET_ID)
    camera = ue.getCamCoptObj(0, 0)
    if not isinstance(target_truth, UE4CtrlAPI.CoptReqData):
        raise RuntimeError("Failed to receive target truth")
    if not isinstance(camera, UE4CtrlAPI.CameraData):
        raise RuntimeError("Failed to receive camera pose")

    stop = threading.Event()
    state = CrossingState()
    threading.Thread(
        target=crossing_trajectory,
        args=(mavs[1], stop, state, args.speed),
        daemon=True,
    ).start()

    # Small angle variations reproduce the gimbal viewpoints around the failure.
    sensor = vision.VisSensor[0]
    angle_variants = [(-3.0, -3.0), (-1.0, 0.0), (0.0, 2.0), (2.0, -2.0)]
    saved = 0
    unique_frames = 0
    last_stamp = None
    last_variant = -1
    print("Collecting only upward crossings over the mountain. Press q/Esc to stop.")
    try:
        while saved < args.frames:
            if not state.collect_rising:
                time.sleep(0.01)
                continue
            if state.crossing_index != last_variant:
                pitch, yaw = angle_variants[state.crossing_index % len(angle_variants)]
                sensor.SensorAngEular = [0.0, pitch, yaw]
                vision.sendUpdateUEImage(sensor)
                last_variant = state.crossing_index
            if not vision.hasData[0]:
                time.sleep(0.01)
                continue
            stamp = vision.timeStmp[0]
            if stamp == last_stamp:
                time.sleep(0.004)
                continue
            last_stamp = stamp
            unique_frames += 1
            if unique_frames % args.frame_stride:
                continue

            frame = copy.deepcopy(vision.Img[0])
            height, width = frame.shape[:2]
            box = clamp_box(project_target_box(target_truth, camera), width, height)
            if box is None:
                continue
            x1, y1, x2, y2 = box
            center_y = ((y1 + y2) / 2.0) / height
            if not args.min_center_y <= center_y <= args.max_center_y:
                continue

            split = "val" if state.crossing_index + 1 == args.val_crossing else "train"
            stem = f"mountain_{saved:06d}"
            image_path = args.output / "images" / split / f"{stem}.jpg"
            label_path = args.output / "labels" / split / f"{stem}.txt"
            write_jpeg(image_path, frame)
            label_path.write_text(yolo_label(box, width, height), encoding="ascii")
            saved += 1

            preview = frame.copy()
            cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                preview,
                f"mountain hard positive {saved}/{args.frames} crossing={state.crossing_index + 1}",
                (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("RflySim mountain-crossing collector", preview)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
            if saved % 50 == 0:
                print(f"Saved {saved}/{args.frames}")
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        cv2.destroyAllWindows()
        for mav in mavs:
            try:
                mav.endOffboard()
                mav.endMavLoop()
            except Exception:
                pass

    print(f"Mountain-crossing dataset ready: {args.output.resolve()} ({saved} labelled images)")
    return 0 if saved else 1


if __name__ == "__main__":
    raise SystemExit(main())
