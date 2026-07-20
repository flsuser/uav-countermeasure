"""Interactively preview installed RflySim3D UAV appearances on Copter #2."""

from __future__ import annotations

import argparse
import time

import UE4CtrlAPI


TARGET_ID = 2
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
DEFAULT_MODELS = [
    "F450_Default",
    "DJI_Phantom_White",
    "AR_Drone_BlackGreen",
    "Quad_Fold_Gray",
    "Droneyee_QuadX330",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-id", type=int, default=TARGET_ID)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument(
        "--seconds",
        type=float,
        default=0.0,
        help="automatically advance after this many seconds; 0 waits for Enter",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ue = UE4CtrlAPI.UE4CtrlAPI()
    print("Keep RflySim3D open and watch Copter #2 in the main window.")
    print("Press Enter for the next model, or type q then Enter to stop.")
    for index, model in enumerate(args.models, start=1):
        command = f"RflyChange3DModel {args.target_id} {model}".encode("ascii")
        # UDP commands are repeated to avoid a silent one-packet loss.
        for _ in range(3):
            ue.sendUE4Cmd(command)
            time.sleep(0.1)
        time.sleep(0.7)
        print(f"[{index}/{len(args.models)}] {model}")
        if args.seconds > 0:
            time.sleep(args.seconds)
        elif input("> ").strip().lower() == "q":
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
