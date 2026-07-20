"""Collect automatically labelled UAV images from the RflySim e8 simulation.

Simulator truth is used only to generate training labels. The final detector
must not read target state at inference time.
"""

from __future__ import annotations

import argparse
import copy
import math
import shutil
import threading
import time
from pathlib import Path

import cv2
import numpy as np

import PX4MavCtrlV4 as PX4MavCtrl
import UE4CtrlAPI
import VisionCaptureApi


OBSERVER_ID = 1
TARGET_ID = 2


def rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Return a body-to-world ZYX Euler rotation matrix."""
    cr, cp, cy = math.cos(roll), math.cos(pitch), math.cos(yaw)
    sr, sp, sy = math.sin(roll), math.sin(pitch), math.sin(yaw)
    return np.array(
        [
            [cp * cy, sr * sp * cy - cr * sy, cr * sp * cy + sr * sy],
            [cp * sy, sr * sp * sy + cr * cy, cr * sp * sy - sr * cy],
            [-sp, sr * cp, cr * cp],
        ],
        dtype=np.float64,
    )


def project_target_box(
    target: UE4CtrlAPI.CoptReqData,
    camera: UE4CtrlAPI.CameraData,
) -> tuple[float, float, float, float] | None:
    """Project the target UE bounding box into the RGB image."""
    target_rpy = np.radians(np.asarray(target.angEuler, dtype=np.float64))
    target_rot = rotation_matrix(*target_rpy)
    target_pos = np.asarray(target.PosUE, dtype=np.float64)

    extent = np.asarray(copy.deepcopy(target.BoxExtent), dtype=np.float64)
    origin = np.asarray(copy.deepcopy(target.boxOrigin), dtype=np.float64)
    signs = np.array(
        [
            [1, -1, -1],
            [1, 1, -1],
            [1, 1, 1],
            [1, -1, 1],
            [-1, -1, -1],
            [-1, 1, -1],
            [-1, 1, 1],
            [-1, -1, 1],
        ],
        dtype=np.float64,
    )
    body_points = signs * extent + origin
    world_points = (target_rot @ body_points.T).T + target_pos

    camera_rpy = np.asarray(camera.angEuler, dtype=np.float64)
    camera_rot = rotation_matrix(*camera_rpy)
    camera_pos = np.asarray(camera.PosUE, dtype=np.float64)
    sensor_points = (camera_rot.T @ (world_points - camera_pos).T).T

    depth = sensor_points[:, 0]
    if np.any(depth <= 0.05):
        return None

    focal = camera.DataWidth / (
        2.0 * math.tan(math.radians(camera.CameraFOV / 2.0))
    )
    x = focal * sensor_points[:, 1] / depth + camera.DataWidth / 2.0
    y = focal * sensor_points[:, 2] / depth + camera.DataHeight / 2.0
    return float(x.min()), float(y.min()), float(x.max()), float(y.max())


def clamp_box(
    box: tuple[float, float, float, float] | None,
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    if box is None:
        return None
    x1, y1, x2, y2 = box
    x1 = int(np.clip(round(x1), 0, width - 1))
    y1 = int(np.clip(round(y1), 0, height - 1))
    x2 = int(np.clip(round(x2), 0, width - 1))
    y2 = int(np.clip(round(y2), 0, height - 1))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    return x1, y1, x2, y2


def yolo_label(box: tuple[int, int, int, int], width: int, height: int) -> str:
    x1, y1, x2, y2 = box
    cx = ((x1 + x2) / 2.0) / width
    cy = ((y1 + y2) / 2.0) / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    return f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n"


def write_jpeg(path: Path, frame: np.ndarray, quality: int = 95) -> None:
    """Write a JPEG through Python so Windows Unicode paths work reliably."""
    ok, encoded = cv2.imencode(
        ".jpg",
        frame,
        [cv2.IMWRITE_JPEG_QUALITY, quality],
    )
    if not ok:
        raise RuntimeError("OpenCV failed to encode a JPEG frame")
    path.write_bytes(encoded.tobytes())


def target_trajectory(
    mav: PX4MavCtrl.PX4MavCtrler,
    stop: threading.Event,
) -> None:
    """Move the target through different scales and screen positions."""
    waypoints = [
        np.array([3.5, 0.0, -20.0]),
        np.array([5.5, -1.0, -19.0]),
        np.array([8.0, 2.0, -22.0]),
        np.array([12.0, -3.0, -17.0]),
        np.array([18.0, 4.0, -23.0]),
        np.array([25.0, -5.0, -20.0]),
        np.array([30.0, 6.0, -24.0]),
        np.array([15.0, 0.0, -20.0]),
        np.array([6.0, 1.5, -21.0]),
    ]
    index = 0
    mav.SendCopterSpeed(3)
    while not stop.is_set():
        waypoint = waypoints[index]
        mav.SendPosNED(*waypoint, 0)
        position = np.asarray(mav.uavPosNED, dtype=np.float64)
        if position.shape == (3,) and np.linalg.norm(waypoint - position) < 1.5:
            index = (index + 1) % len(waypoints)
        stop.wait(0.1)


def prepare_dataset(root: Path) -> None:
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)


def wait_for_takeoff(
    mavs: list[PX4MavCtrl.PX4MavCtrler],
    timeout: float = 45.0,
) -> None:
    """Keep sending takeoff setpoints and fail instead of saving ground frames."""
    targets = [(0.0, 0.0, -20.0), (12.0, 0.0, -20.0)]
    deadline = time.time() + timeout
    last_report = 0.0
    while time.time() < deadline:
        for mav, target in zip(mavs, targets):
            mav.SendPosNED(*target, 0)
        positions = [np.asarray(mav.uavPosNED, dtype=np.float64) for mav in mavs]
        airborne = all(pos.shape == (3,) and pos[2] < -15.0 for pos in positions)
        if airborne:
            print(f"Takeoff confirmed: z1={positions[0][2]:.1f}, z2={positions[1][2]:.1f}")
            return
        if time.time() - last_report >= 2.0:
            armed = [bool(mav.isArmed) for mav in mavs]
            z_values = [float(pos[2]) if pos.shape == (3,) else math.nan for pos in positions]
            print(f"Waiting for takeoff: armed={armed}, z={z_values}")
            last_report = time.time()
        time.sleep(0.2)
    armed = [bool(mav.isArmed) for mav in mavs]
    z_values = [float(mav.uavPosNED[2]) for mav in mavs]
    raise RuntimeError(
        f"Takeoff was not confirmed within {timeout:.0f}s; armed={armed}, z={z_values}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=600)
    parser.add_argument("--frame-stride", type=int, default=3)
    parser.add_argument("--val-every", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "datasets" / "rfly_uav",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "Config.json",
    )
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument(
        "--clear",
        action="store_true",
        help="remove the previous generated dataset before collection",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.clear and args.output.exists():
        shutil.rmtree(args.output)
    prepare_dataset(args.output)

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
    for mav in mavs:
        mav.InitMavLoop()
        mav.InitTrueDataLoop()
    time.sleep(2)
    for mav in mavs:
        mav.initOffboard()
    time.sleep(2)
    print("Taking off; collection will not start until both aircraft exceed 15 m...")
    wait_for_takeoff(mavs)

    ue.reqCamCoptObj(1, [OBSERVER_ID, TARGET_ID])
    ue.reqCamCoptObj(0, 0)
    time.sleep(2)
    ue.initUE4MsgRec()
    time.sleep(3)

    target = ue.getCamCoptObj(1, TARGET_ID)
    camera = ue.getCamCoptObj(0, 0)
    if not isinstance(target, UE4CtrlAPI.CoptReqData):
        raise RuntimeError("Failed to receive target truth data")
    if not isinstance(camera, UE4CtrlAPI.CameraData):
        raise RuntimeError("Failed to receive camera pose data")

    stop = threading.Event()
    motion = threading.Thread(
        target=target_trajectory,
        args=(mavs[1], stop),
        daemon=True,
    )
    motion.start()

    saved = 0
    unique_frames = 0
    last_stamp = None
    print("Collecting. Press q in the preview window or Ctrl+C to stop.")
    try:
        while saved < args.frames:
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

            split = "val" if (saved + 1) % args.val_every == 0 else "train"
            stem = f"rfly_{saved:06d}"
            image_path = args.output / "images" / split / f"{stem}.jpg"
            label_path = args.output / "labels" / split / f"{stem}.txt"
            write_jpeg(image_path, frame)
            label_path.write_text(yolo_label(box, width, height), encoding="ascii")
            saved += 1

            if not args.no_preview:
                preview = frame.copy()
                x1, y1, x2, y2 = box
                cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    preview,
                    f"saved {saved}/{args.frames} ({split})",
                    (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow("RflySim UAV dataset collector", preview)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if saved % 50 == 0:
                print(f"Saved {saved}/{args.frames}")
    except KeyboardInterrupt:
        print("Collection interrupted by user")
    finally:
        stop.set()
        cv2.destroyAllWindows()
        for mav in mavs:
            try:
                mav.endOffboard()
                mav.endMavLoop()
            except Exception:
                pass

    print(f"Dataset ready: {args.output.resolve()} ({saved} labelled images)")
    return 0 if saved > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
