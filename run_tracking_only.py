"""Stable tracking-only entry point.

This version never commands forward or reverse motion.  In normal mode it only
outputs detections and normalized centre errors.  Use --gimbal for two-axis
camera yaw/pitch tracking, or --closed-loop for the earlier aircraft alignment.
"""

from run_detect_track import main


if __name__ == "__main__":
    raise SystemExit(main())
