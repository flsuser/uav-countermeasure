"""Run truth-free UAV detection and single-target tracking in RflySim.

The script only consumes the RGB camera stream.  UE object truth is never read.
"""

from __future__ import annotations

import argparse
import copy
import csv
import math
import os
import sys
import threading
import time
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
RFLY_SDK = Path(r"D:\PX4PSP\RflySimAPIs\RflySimSDK")
RFLY_SITE_PACKAGES = Path(r"D:\PX4PSP\Python38\Lib\site-packages")
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_DIR / ".runtime_cache"))

# Import the venv packages first.  RflySim's site-packages is appended only for
# SDK-only dependencies such as open3d, so its CPU torch cannot shadow CUDA torch.
sys.path[:0] = [
    str(RFLY_SDK),
    str(RFLY_SDK / "vision"),
    str(RFLY_SDK / "ue"),
    str(RFLY_SDK / "ctrl"),
]

import cv2
import numpy as np
from ultralytics import YOLO

# Only append this after CUDA torch/Ultralytics/OpenCV have been imported.
sys.path.append(str(RFLY_SITE_PACKAGES))
import VisionCaptureApi
import PX4MavCtrlV4 as PX4MavCtrl


DEFAULT_WEIGHTS = PROJECT_DIR / "runs" / "rfly_uav_yolo11n" / "weights" / "best.pt"
DEFAULT_CONFIG = PROJECT_DIR / "Config.json"
OBSERVER_ID = 1
TARGET_ID = 2


def wait_for_takeoff(mavs: list, timeout: float = 45.0) -> None:
    targets = [(0.0, 0.0, -20.0), (8.0, 0.0, -20.0)]
    deadline = time.time() + timeout
    last_report = 0.0
    while time.time() < deadline:
        for mav, target in zip(mavs, targets):
            mav.SendPosNED(*target, 0)
        positions = [np.asarray(mav.uavPosNED, dtype=np.float64) for mav in mavs]
        if all(pos.shape == (3,) and pos[2] < -15.0 for pos in positions):
            print(f"Takeoff confirmed: z1={positions[0][2]:.1f}, z2={positions[1][2]:.1f}")
            return
        if time.time() - last_report >= 2.0:
            print(f"Waiting for takeoff: armed={[bool(m.isArmed) for m in mavs]}, "
                  f"z={[round(float(p[2]), 1) for p in positions]}")
            last_report = time.time()
        time.sleep(0.2)
    raise RuntimeError("Takeoff was not confirmed within 45 seconds")


def target_scenario(name: str) -> tuple[list[np.ndarray], float]:
    if name == "easy":
        return [
            np.array([7.0, 0.0, -20.0]),
            np.array([9.0, -1.5, -19.0]),
            np.array([12.0, 2.0, -21.5]),
            np.array([14.0, -2.0, -18.5]),
            np.array([10.0, 1.0, -20.5]),
            np.array([8.0, 0.0, -20.0]),
        ], 1.2
    if name == "medium":
        return [
            np.array([6.0, 0.0, -20.0]),
            np.array([10.0, -2.5, -18.0]),
            np.array([15.0, 3.0, -22.0]),
            np.array([20.0, -4.0, -17.5]),
            np.array([13.0, 2.0, -21.0]),
            np.array([8.0, -1.0, -19.0]),
        ], 2.0
    return [
        np.array([5.5, -1.0, -19.0]),
        np.array([8.0, 2.0, -22.0]),
        np.array([12.0, -3.0, -17.0]),
        np.array([18.0, 4.0, -23.0]),
        np.array([25.0, -5.0, -20.0]),
        np.array([15.0, 0.0, -20.0]),
        np.array([6.0, 1.5, -21.0]),
    ], 3.0


def fly_target(
    mav,
    stop: threading.Event,
    scenario_name: str,
    speed_override: float | None,
    waypoint_hold: float,
) -> None:
    waypoints, default_speed = target_scenario(scenario_name)
    index = 0
    arrived_at: float | None = None
    mav.SendCopterSpeed(speed_override or default_speed)
    while not stop.is_set():
        waypoint = waypoints[index]
        mav.SendPosNED(*waypoint, 0)
        position = np.asarray(mav.uavPosNED, dtype=np.float64)
        if position.shape == (3,) and np.linalg.norm(waypoint - position) < 1.5:
            if arrived_at is None:
                arrived_at = time.time()
            elif time.time() - arrived_at >= waypoint_hold:
                index = (index + 1) % len(waypoints)
                arrived_at = None
        else:
            arrived_at = None
        stop.wait(0.1)


class SingleTargetTrack:
    """Smooth detections and bridge a few missed frames with constant velocity."""

    def __init__(self, alpha: float = 0.65, max_missed: int = 8) -> None:
        self.alpha = alpha
        self.max_missed = max_missed
        self.box: np.ndarray | None = None
        self.velocity = np.zeros(4, dtype=np.float32)
        self.missed = 0

    def update(self, detection: np.ndarray | None) -> tuple[np.ndarray | None, bool]:
        if detection is not None:
            detection = detection.astype(np.float32)
            if self.box is None or self.missed > self.max_missed:
                self.box = detection
                self.velocity.fill(0)
            else:
                predicted = self.box + self.velocity
                new_box = self.alpha * detection + (1.0 - self.alpha) * predicted
                self.velocity = 0.65 * self.velocity + 0.35 * (new_box - self.box)
                self.box = new_box
            self.missed = 0
            return self.box.copy(), True

        self.missed += 1
        if self.box is None or self.missed > self.max_missed:
            self.box = None
            self.velocity.fill(0)
            return None, False
        self.box = self.box + self.velocity
        self.velocity *= 0.85
        return self.box.copy(), False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RflySim YOLO UAV detector/tracker")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--acquire-conf", type=float, default=0.35,
                        help="minimum confidence when starting a new track")
    parser.add_argument("--iou", type=float, default=0.6)
    parser.add_argument("--association-gate", type=float, default=0.12,
                        help="maximum centre jump as a fraction of image diagonal")
    parser.add_argument("--device", default="0", help="CUDA device, or cpu")
    parser.add_argument("--max-missed", type=int, default=8)
    parser.add_argument("--scenario", choices=("easy", "medium", "stress"),
                        default="easy", help="target motion and range profile")
    parser.add_argument("--target-speed", type=float,
                        help="override scenario target speed in m/s")
    parser.add_argument("--waypoint-hold", type=float, default=1.0,
                        help="seconds to pause at each target waypoint")
    parser.add_argument("--no-auto-flight", action="store_true",
                        help="do not arm/take off/control the two simulated aircraft")
    parser.add_argument("--closed-loop", action="store_true",
                        help="visually align observer yaw and altitude to the detected target")
    parser.add_argument("--gimbal", action="store_true",
                        help="control simulated camera yaw/pitch instead of aircraft motion")
    parser.add_argument("--yaw-kp", type=float, default=0.8)
    parser.add_argument("--vertical-kp", type=float, default=1.2)
    parser.add_argument("--max-yaw-rate", type=float, default=0.35)
    parser.add_argument("--max-vertical-speed", type=float, default=1.0)
    parser.add_argument("--deadband", type=float, default=0.06)
    parser.add_argument("--gimbal-yaw-kp", type=float, default=60.0,
                        help="yaw-rate gain in deg/s per normalized error")
    parser.add_argument("--gimbal-pitch-kp", type=float, default=45.0,
                        help="pitch-rate gain in deg/s per normalized error")
    parser.add_argument("--max-gimbal-yaw-rate", type=float, default=40.0)
    parser.add_argument("--max-gimbal-pitch-rate", type=float, default=30.0)
    parser.add_argument("--gimbal-yaw-limit", type=float, default=80.0)
    parser.add_argument("--gimbal-pitch-min", type=float, default=-60.0)
    parser.add_argument("--gimbal-pitch-max", type=float, default=30.0)
    parser.add_argument("--gimbal-predict-control-frames", type=int, default=5,
                        help="use low-gain predicted boxes for this many missed frames")
    parser.add_argument("--gimbal-predict-gain", type=float, default=0.35)
    parser.add_argument(
        "--hard-negative-dir",
        type=Path,
        default=PROJECT_DIR / "datasets" / "rfly_hard_negatives",
        help="press n to save a confirmed drone-free frame here",
    )
    parser.add_argument("--log-csv", type=Path,
                        help="write per-frame tracking telemetry to CSV")
    return parser.parse_args()


def deadband(value: float, width: float) -> float:
    if abs(value) <= width:
        return 0.0
    return value - np.sign(value) * width


def alignment_command(
    error_x: float,
    error_y: float,
    altitude_ned: float,
    args: argparse.Namespace,
) -> tuple[float, float]:
    """Return vertical speed (down positive) and yaw rate in body FRD."""
    yaw_rate = float(np.clip(
        args.yaw_kp * deadband(error_x, args.deadband),
        -args.max_yaw_rate,
        args.max_yaw_rate,
    ))
    vertical_speed = float(np.clip(
        args.vertical_kp * deadband(error_y, args.deadband),
        -args.max_vertical_speed,
        args.max_vertical_speed,
    ))

    # NED altitude is negative above the origin. Force the observer back into
    # the 10--30 m envelope before accepting image-based vertical commands.
    if altitude_ned > -10.0:
        vertical_speed = -min(args.max_vertical_speed, 0.5 + 0.2 * (altitude_ned + 10.0))
    elif altitude_ned < -30.0:
        vertical_speed = min(args.max_vertical_speed, 0.5 + 0.2 * (-30.0 - altitude_ned))
    return vertical_speed, yaw_rate


def gimbal_rate_command(
    error_x: float,
    error_y: float,
    args: argparse.Namespace,
) -> tuple[float, float]:
    """Return simulated gimbal pitch/yaw rates in degrees per second."""
    yaw_rate = float(np.clip(
        args.gimbal_yaw_kp * deadband(error_x, args.deadband),
        -args.max_gimbal_yaw_rate,
        args.max_gimbal_yaw_rate,
    ))
    # In RflySim a negative camera pitch looks downward.  Image y is positive
    # downward, hence the minus sign.
    pitch_rate = float(np.clip(
        -args.gimbal_pitch_kp * deadband(error_y, args.deadband),
        -args.max_gimbal_pitch_rate,
        args.max_gimbal_pitch_rate,
    ))
    return pitch_rate, yaw_rate


def best_detection(
    result,
    reference_box: np.ndarray | None,
    acquire_conf: float,
    association_gate: float,
) -> tuple[np.ndarray | None, float]:
    if result.boxes is None or len(result.boxes) == 0:
        return None, 0.0
    confidences = result.boxes.conf.detach().cpu().numpy()
    boxes = result.boxes.xyxy.detach().cpu().numpy()
    if reference_box is None:
        valid = np.flatnonzero(confidences >= acquire_conf)
        if len(valid) == 0:
            return None, 0.0
        index = int(valid[np.argmax(confidences[valid])])
    else:
        image_height, image_width = result.orig_shape
        diagonal = math.hypot(image_width, image_height)
        reference_center = np.array(
            [(reference_box[0] + reference_box[2]) / 2.0,
             (reference_box[1] + reference_box[3]) / 2.0]
        )
        centers = np.column_stack(
            ((boxes[:, 0] + boxes[:, 2]) / 2.0,
             (boxes[:, 1] + boxes[:, 3]) / 2.0)
        )
        distances = np.linalg.norm(centers - reference_center, axis=1) / diagonal
        valid = np.flatnonzero(distances <= association_gate)
        if len(valid) == 0:
            return None, 0.0
        scores = confidences[valid] - 0.25 * distances[valid] / association_gate
        index = int(valid[np.argmax(scores)])
    box = boxes[index]
    return box, float(confidences[index])


def clamp_box(box: np.ndarray, width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return (
        int(np.clip(round(x1), 0, width - 1)),
        int(np.clip(round(y1), 0, height - 1)),
        int(np.clip(round(x2), 0, width - 1)),
        int(np.clip(round(y2), 0, height - 1)),
    )


def save_hard_negative(frame: np.ndarray, root: Path) -> Path:
    """Save one user-confirmed drone-free image with an empty YOLO label."""
    existing = sum(1 for _ in root.glob("images/*/hardneg_*.jpg"))
    split = "val" if (existing + 1) % 5 == 0 else "train"
    image_dir = root / "images" / split
    label_dir = root / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    stem = f"hardneg_{existing:06d}"
    image_path = image_dir / f"{stem}.jpg"
    label_path = label_dir / f"{stem}.txt"
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise RuntimeError("Failed to encode hard-negative frame")
    image_path.write_bytes(encoded.tobytes())
    label_path.write_text("", encoding="ascii")
    return image_path


def main() -> int:
    args = parse_args()
    if args.closed_loop and args.gimbal:
        raise ValueError("Choose either --closed-loop aircraft alignment or --gimbal")
    if args.closed_loop and args.no_auto_flight:
        raise ValueError("--closed-loop requires observer control; remove --no-auto-flight")
    weights = args.weights.resolve()
    config = args.config.resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"Weights not found: {weights}")
    if not config.is_file():
        raise FileNotFoundError(f"Vision config not found: {config}")

    print(f"Loading detector: {weights}")
    model = YOLO(str(weights))
    tracker = SingleTargetTrack(max_missed=args.max_missed)

    vision = VisionCaptureApi.VisionCaptureApi()
    if not vision.jsonLoad(jsonPath=str(config)):
        raise RuntimeError(f"Failed to load vision config: {config}")
    if not vision.sendReqToUE4():
        raise RuntimeError("RflySim3D rejected the camera request")
    vision.startImgCap()

    mavs = []
    stop = threading.Event()
    if not args.no_auto_flight:
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
        print("Taking off; detection starts after both aircraft exceed 15 m...")
        wait_for_takeoff(mavs)
        threading.Thread(
            target=fly_target,
            args=(mavs[1], stop, args.scenario, args.target_speed, args.waypoint_hold),
            daemon=True,
        ).start()

    print("Waiting for RGB stream. Press q/Esc to stop.")
    print("Press n ONLY when no real drone is visible and a background false alarm appears.")
    last_frame_time = time.perf_counter()
    fps = 0.0
    negative_saved_at = 0.0
    sensor = vision.VisSensor[0]
    gimbal_pitch = float(sensor.SensorAngEular[1])
    gimbal_yaw = float(sensor.SensorAngEular[2])
    gimbal_pitch_rate = 0.0
    gimbal_yaw_rate = 0.0
    if args.gimbal:
        sensor.SensorAngEular = [0.0, gimbal_pitch, gimbal_yaw]
        vision.sendUpdateUEImage(sensor)
    log_file = None
    log_writer = None
    log_start = time.perf_counter()
    logged_rows = 0
    if args.log_csv:
        log_path = args.log_csv.resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = log_path.open("w", newline="", encoding="utf-8")
        log_writer = csv.DictWriter(
            log_file,
            fieldnames=[
                "time_s", "state", "confidence", "error_x", "error_y",
                "box_width_ratio", "fps", "gimbal_pitch_deg", "gimbal_yaw_deg",
                "gimbal_pitch_rate_dps", "gimbal_yaw_rate_dps",
                "aircraft_vertical_speed_mps", "aircraft_yaw_rate_rps",
            ],
        )
        log_writer.writeheader()
        print(f"Tracking telemetry: {log_path}")
    try:
        while True:
            if not vision.hasData[0]:
                time.sleep(0.005)
                continue

            raw_frame = copy.deepcopy(vision.Img[0])
            frame = raw_frame.copy()
            result = model.predict(
                source=frame,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                device=args.device,
                half=args.device != "cpu",
                verbose=False,
            )[0]
            detection, confidence = best_detection(
                result,
                tracker.box,
                args.acquire_conf,
                args.association_gate,
            )
            tracked, measured = tracker.update(detection)

            height, width = frame.shape[:2]
            now = time.perf_counter()
            frame_dt = min(max(now - last_frame_time, 1e-3), 0.1)
            instantaneous_fps = 1.0 / frame_dt
            fps = instantaneous_fps if fps == 0 else 0.9 * fps + 0.1 * instantaneous_fps
            last_frame_time = now

            cv2.drawMarker(
                frame,
                (width // 2, height // 2),
                (0, 255, 255),
                cv2.MARKER_CROSS,
                24,
                2,
            )
            vertical_speed = 0.0
            yaw_rate = 0.0
            error_x = math.nan
            error_y = math.nan
            box_width_ratio = math.nan
            state = "SEARCHING"
            if tracked is not None:
                x1, y1, x2, y2 = clamp_box(tracked, width, height)
                color = (0, 255, 0) if measured else (0, 165, 255)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                error_x = (cx - width / 2.0) / (width / 2.0)
                error_y = (cy - height / 2.0) / (height / 2.0)
                box_width_ratio = (x2 - x1) / float(width)
                if args.closed_loop and measured:
                    altitude_ned = float(mavs[0].uavPosNED[2])
                    vertical_speed, yaw_rate = alignment_command(
                        error_x, error_y, altitude_ned, args
                    )
                    mavs[0].SendVelFRD(0.0, 0.0, vertical_speed, yaw_rate)
                use_gimbal_prediction = (
                    args.gimbal
                    and not measured
                    and tracker.missed <= args.gimbal_predict_control_frames
                )
                if args.gimbal and (measured or use_gimbal_prediction):
                    gimbal_pitch_rate, gimbal_yaw_rate = gimbal_rate_command(
                        error_x, error_y, args
                    )
                    if use_gimbal_prediction:
                        gimbal_pitch_rate *= args.gimbal_predict_gain
                        gimbal_yaw_rate *= args.gimbal_predict_gain
                    gimbal_pitch = float(np.clip(
                        gimbal_pitch + gimbal_pitch_rate * frame_dt,
                        args.gimbal_pitch_min,
                        args.gimbal_pitch_max,
                    ))
                    gimbal_yaw = float(np.clip(
                        gimbal_yaw + gimbal_yaw_rate * frame_dt,
                        -args.gimbal_yaw_limit,
                        args.gimbal_yaw_limit,
                    ))
                    sensor.SensorAngEular = [0.0, gimbal_pitch, gimbal_yaw]
                    vision.sendUpdateUEImage(sensor)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(frame, (cx, cy), 4, color, -1)
                cv2.line(frame, (width // 2, height // 2), (cx, cy), color, 1)
                state = "DETECT" if measured else f"PREDICT {tracker.missed}/{tracker.max_missed}"
                label = f"ID 1  {state}  conf={confidence:.2f}  ex={error_x:+.2f} ey={error_y:+.2f}"
                cv2.putText(frame, label, (max(4, x1), max(22, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2, cv2.LINE_AA)
            else:
                cv2.putText(frame, "SEARCHING", (15, 55), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 0, 255), 2, cv2.LINE_AA)

            # Never fly on a predicted/orange box.  A missed detection commands
            # zero body velocity and yaw rate so the observer holds position.
            if args.closed_loop and not measured:
                mavs[0].SendVelFRD(0.0, 0.0, 0.0, 0.0)
            if args.gimbal and not measured and (
                tracked is None or tracker.missed > args.gimbal_predict_control_frames
            ):
                gimbal_pitch_rate = 0.0
                gimbal_yaw_rate = 0.0

            cv2.putText(frame, f"YOLO11n  FPS {fps:.1f}  scenario={args.scenario}", (15, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
            if args.closed_loop:
                cv2.putText(
                    frame,
                    f"ALIGN ON  vz={vertical_speed:+.2f} m/s  "
                    f"yawrate={yaw_rate:+.2f} rad/s",
                    (15, height - 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            if args.gimbal:
                cv2.putText(
                    frame,
                    f"GIMBAL ON pitch={gimbal_pitch:+.1f} yaw={gimbal_yaw:+.1f} "
                    f"prate={gimbal_pitch_rate:+.1f} yrate={gimbal_yaw_rate:+.1f} deg/s",
                    (15, height - 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (255, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
            cv2.putText(
                frame,
                "N: save hard negative (only if drone is absent)",
                (15, height - (45 if (args.closed_loop or args.gimbal) else 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
            if time.perf_counter() - negative_saved_at < 1.0:
                cv2.putText(frame, "HARD NEGATIVE SAVED", (15, 82),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
                            cv2.LINE_AA)
            if log_writer is not None:
                log_writer.writerow({
                    "time_s": f"{now - log_start:.6f}",
                    "state": state,
                    "confidence": f"{confidence:.6f}",
                    "error_x": "" if math.isnan(error_x) else f"{error_x:.6f}",
                    "error_y": "" if math.isnan(error_y) else f"{error_y:.6f}",
                    "box_width_ratio": "" if math.isnan(box_width_ratio) else f"{box_width_ratio:.6f}",
                    "fps": f"{fps:.3f}",
                    "gimbal_pitch_deg": f"{gimbal_pitch:.4f}",
                    "gimbal_yaw_deg": f"{gimbal_yaw:.4f}",
                    "gimbal_pitch_rate_dps": f"{gimbal_pitch_rate:.4f}",
                    "gimbal_yaw_rate_dps": f"{gimbal_yaw_rate:.4f}",
                    "aircraft_vertical_speed_mps": f"{vertical_speed:.4f}",
                    "aircraft_yaw_rate_rps": f"{yaw_rate:.4f}",
                })
                logged_rows += 1
                if logged_rows % 30 == 0:
                    log_file.flush()
            cv2.imshow("RflySim UAV detection + tracking", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("n"):
                saved_path = save_hard_negative(raw_frame, args.hard_negative_dir.resolve())
                negative_saved_at = time.perf_counter()
                print(f"Hard negative saved: {saved_path}")
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        cv2.destroyAllWindows()
        if log_file is not None:
            log_file.close()
        for mav in mavs:
            try:
                mav.endOffboard()
                mav.endMavLoop()
            except Exception:
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
