"""Summarize detection availability and centering performance from a run log."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    position = fraction * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--center-threshold", type=float, default=0.10)
    parser.add_argument(
        "--warmup-seconds",
        type=float,
        default=0.0,
        help="exclude this many seconds from the start of the log",
    )
    args = parser.parse_args()

    with args.log.resolve().open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) < 2:
        raise RuntimeError("Tracking log contains fewer than two frames")

    if args.warmup_seconds > 0:
        cutoff = float(rows[0]["time_s"]) + args.warmup_seconds
        rows = [row for row in rows if float(row["time_s"]) >= cutoff]
        if len(rows) < 2:
            raise RuntimeError("Warm-up exclusion leaves fewer than two frames")

    times = [float(row["time_s"]) for row in rows]
    states = [row["state"] for row in rows]
    detected = [state == "DETECT" for state in states]
    predicted = [state.startswith("PREDICT") for state in states]
    searching = [state == "SEARCHING" for state in states]
    fps_values = [float(row["fps"]) for row in rows]

    errors_x = []
    errors_y = []
    radial_errors = []
    centered = 0
    for row, is_detected in zip(rows, detected):
        if not is_detected or not row["error_x"] or not row["error_y"]:
            continue
        ex, ey = float(row["error_x"]), float(row["error_y"])
        errors_x.append(abs(ex))
        errors_y.append(abs(ey))
        radial_errors.append(math.hypot(ex, ey))
        if abs(ex) <= args.center_threshold and abs(ey) <= args.center_threshold:
            centered += 1

    loss_durations = []
    loss_start = None
    for index, is_detected in enumerate(detected):
        if not is_detected and loss_start is None:
            loss_start = times[index]
        elif is_detected and loss_start is not None:
            loss_durations.append(times[index] - loss_start)
            loss_start = None
    if loss_start is not None:
        loss_durations.append(times[-1] - loss_start)

    total = len(rows)
    duration = times[-1] - times[0]
    print(f"log={args.log.resolve()}")
    print(f"warmup_excluded_s={args.warmup_seconds:.2f}")
    print(f"duration_s={duration:.2f}")
    print(f"frames={total}")
    print(f"median_fps={statistics.median(fps_values):.2f}")
    print(f"detect_ratio={sum(detected) / total:.3%}")
    print(f"predict_ratio={sum(predicted) / total:.3%}")
    print(f"search_ratio={sum(searching) / total:.3%}")
    if radial_errors:
        print(f"mean_abs_error_x={statistics.mean(errors_x):.4f}")
        print(f"mean_abs_error_y={statistics.mean(errors_y):.4f}")
        print(f"x_within_threshold_ratio={sum(e <= args.center_threshold for e in errors_x) / len(errors_x):.3%}")
        print(f"y_within_threshold_ratio={sum(e <= args.center_threshold for e in errors_y) / len(errors_y):.3%}")
        print(f"radial_error_rmse={math.sqrt(statistics.mean(e * e for e in radial_errors)):.4f}")
        print(f"radial_error_p95={percentile(radial_errors, 0.95):.4f}")
        print(f"centered_ratio={centered / len(radial_errors):.3%}")
    print(f"loss_events={len(loss_durations)}")
    print(f"mean_loss_s={statistics.mean(loss_durations) if loss_durations else 0.0:.3f}")
    print(f"max_loss_s={max(loss_durations, default=0.0):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
