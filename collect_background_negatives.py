"""Collect drone-free RflySim background frames for hard-negative training."""

from __future__ import annotations

import argparse
import copy
import math
import time
from pathlib import Path

from run_detect_track import (
    OBSERVER_ID,
    PROJECT_DIR,
    TARGET_ID,
    PX4MavCtrl,
    VisionCaptureApi,
    cv2,
    save_hard_negative,
    wait_for_takeoff,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect drone-free RflySim backgrounds")
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--frame-stride", type=int, default=3)
    parser.add_argument("--config", type=Path, default=PROJECT_DIR / "Config.json")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_DIR / "datasets" / "rfly_hard_negatives",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    vision = VisionCaptureApi.VisionCaptureApi()
    if not vision.jsonLoad(jsonPath=str(args.config.resolve())):
        raise RuntimeError(f"Failed to load vision config: {args.config}")
    if not vision.sendReqToUE4():
        raise RuntimeError("RflySim3D rejected the camera request")
    vision.startImgCap()

    mavs = [
        PX4MavCtrl.PX4MavCtrler(OBSERVER_ID),
        PX4MavCtrl.PX4MavCtrler(TARGET_ID),
    ]
    for mav in mavs:
        mav.InitMavLoop()
        mav.InitTrueDataLoop()
    time.sleep(2)
    for mav in mavs:
        mav.initOffboard()
    time.sleep(2)

    print("Taking off before background collection...")
    wait_for_takeoff(mavs)

    # Move the target well behind the observer.  The camera only sweeps the
    # forward +/-55 degree sector, so the target cannot enter a saved frame.
    target = mavs[1]
    target.SendCopterSpeed(20)
    deadline = time.time() + 20.0
    while time.time() < deadline:
        target.SendPosNED(-80.0, 0.0, -20.0, 0.0)
        if float(target.uavPosNED[0]) < -60.0:
            break
        time.sleep(0.1)
    if float(target.uavPosNED[0]) >= -60.0:
        raise RuntimeError("Target could not be parked behind the camera; no negatives saved")
    target.SendCopterSpeed(2)
    print("Target parked behind camera. Collecting drone-free mountain views...")

    saved = 0
    unique_frames = 0
    last_stamp = None
    observer = mavs[0]
    observer.SendCopterSpeed(2)
    try:
        while saved < args.frames:
            phase = saved / max(args.frames - 1, 1)
            yaw = math.radians(55.0) * math.sin(phase * 4.0 * math.pi)
            altitude = -20.0 + 4.0 * math.sin(phase * 2.0 * math.pi)
            observer.SendPosNED(0.0, 0.0, altitude, yaw)
            target.SendPosNED(-80.0, 0.0, -20.0, 0.0)

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
            save_hard_negative(frame, args.output.resolve())
            saved += 1
            preview = frame.copy()
            cv2.putText(preview, f"background {saved}/{args.frames}", (12, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                        cv2.LINE_AA)
            cv2.imshow("RflySim background-negative collector", preview)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
            if saved % 50 == 0:
                print(f"Saved {saved}/{args.frames}")
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        for mav in mavs:
            try:
                mav.endOffboard()
                mav.endMavLoop()
            except Exception:
                pass

    print(f"Background negatives ready: {args.output.resolve()} ({saved} images)")
    return 0 if saved else 1


if __name__ == "__main__":
    raise SystemExit(main())
