"""
gather_data.py – Capture video frames / clips from a CCTV/USB/Pi camera
and annotate them with behaviour labels in real time.

Key-press controls during capture:
  F – feeding       D – drinking      W – walking
  R – resting       A – aggression    O – other
  S – save current frame with label
  Q – quit

Usage:
    python gather_data.py [--mode frame|clip] [--fps 5] [--duration 60]

The script saves:
  • Frames (JPEG) to  data/raw_frames/
  • Clips  (MP4)  to  data/clips/
  • A metadata CSV  data/annotations/metadata.csv
"""

import argparse
import csv
import logging
import os
import time
from datetime import datetime

import cv2
import numpy as np

import config
from camera_utils import CameraStream, FPSCounter
from logger import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CSV metadata writer
# ---------------------------------------------------------------------------

METADATA_CSV = os.path.join(config.ANNOTATIONS_DIR, "metadata.csv")
METADATA_FIELDS = ["file", "frame_id", "timestamp", "behaviour_label"]


def _ensure_metadata_csv():
    if not os.path.exists(METADATA_CSV) or os.path.getsize(METADATA_CSV) == 0:
        with open(METADATA_CSV, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=METADATA_FIELDS).writeheader()


def _append_metadata(file: str, frame_id: int, behaviour: str):
    with open(METADATA_CSV, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=METADATA_FIELDS).writerow({
            "file": file,
            "frame_id": frame_id,
            "timestamp": datetime.utcnow().isoformat(),
            "behaviour_label": behaviour,
        })


# ---------------------------------------------------------------------------
# Overlay helper
# ---------------------------------------------------------------------------

_LABEL_COLOR = {
    "feeding": (0, 200, 0),
    "drinking": (255, 165, 0),
    "walking": (0, 165, 255),
    "resting": (128, 0, 128),
    "aggression": (0, 0, 255),
    "other": (128, 128, 128),
    "none": (200, 200, 200),
}


def _draw_overlay(frame: np.ndarray, behaviour: str, fps: float,
                  n_saved: int) -> np.ndarray:
    vis = frame.copy()
    color = _LABEL_COLOR.get(behaviour, (200, 200, 200))
    cv2.putText(vis, f"Label: {behaviour.upper()}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(vis, f"FPS: {fps:.1f}  Saved: {n_saved}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(vis,
                "F:feed D:drink W:walk R:rest A:aggr O:other | S:save Q:quit",
                (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
    return vis


# ---------------------------------------------------------------------------
# Frame-capture mode
# ---------------------------------------------------------------------------

def run_frame_mode(cam, capture_fps: int = 5):
    """Interactive frame-by-frame capture with manual labelling.

    Press the behaviour key then S to save. Q to quit.
    """
    _ensure_metadata_csv()
    fps_counter = FPSCounter()
    current_behaviour = "none"
    n_saved = 0
    frame_id = 0
    last_capture = time.time()
    interval = 1.0 / capture_fps

    logger.info("Frame mode started. Window shows live feed.")
    print("\n=== Frame Capture Mode ===")
    print("Press a behaviour key, then S to save that frame.")

    while True:
        frame = cam.read()
        if frame is None:
            time.sleep(0.01)
            continue

        fps_val = fps_counter.tick()
        vis = _draw_overlay(frame, current_behaviour, fps_val, n_saved)
        cv2.imshow("Gather Data", vis)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif chr(key).lower() in config.BEHAVIOUR_KEYS:
            current_behaviour = config.BEHAVIOUR_KEYS[chr(key).lower()]
            print(f"  → Label set to: {current_behaviour}")
        elif key == ord("s"):
            if current_behaviour == "none":
                print("  ⚠ Set a behaviour label first!")
                continue
            fname = f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}_{current_behaviour}.jpg"
            fpath = os.path.join(config.RAW_FRAMES_DIR, fname)
            cv2.imwrite(fpath, frame)
            _append_metadata(fname, frame_id, current_behaviour)
            n_saved += 1
            frame_id += 1
            print(f"  ✓ Saved frame {n_saved}: {fname}")

    cv2.destroyAllWindows()
    logger.info("Frame capture complete. %d frames saved.", n_saved)


# ---------------------------------------------------------------------------
# Clip-capture mode
# ---------------------------------------------------------------------------

def run_clip_mode(cam, clip_duration_sec: int = 30, capture_fps: int = 15):
    """Record video clips and assign a per-clip behaviour label.

    One clip is recorded, then the user picks the label. Repeat.
    """
    _ensure_metadata_csv()
    clip_count = 0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    print("\n=== Clip Capture Mode ===")
    print(f"Each clip is {clip_duration_sec} seconds. Label it after recording.")

    while True:
        print(f"\nStarting clip {clip_count + 1} in 2 seconds… (CTRL-C to stop)")
        time.sleep(2)

        clip_name = f"clip_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.mp4"
        clip_path = os.path.join(config.CLIPS_DIR, clip_name)
        writer = cv2.VideoWriter(
            clip_path, fourcc, capture_fps,
            (config.CAMERA_WIDTH, config.CAMERA_HEIGHT)
        )
        if not writer.isOpened():
            logger.error("Could not open VideoWriter for %s", clip_path)
            break

        start = time.time()
        fps_counter = FPSCounter()
        print(f"Recording {clip_name} …")

        while time.time() - start < clip_duration_sec:
            frame = cam.read()
            if frame is None:
                continue
            fps_val = fps_counter.tick()
            writer.write(frame)
            vis = frame.copy()
            elapsed = time.time() - start
            cv2.putText(vis,
                        f"Recording {elapsed:.0f}/{clip_duration_sec}s  "
                        f"FPS: {fps_val:.1f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Gather Data – Clip Mode", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        writer.release()
        print(f"Clip saved: {clip_path}")

        # Label the clip
        print("Assign behaviour labels per second (comma-separated, e.g. F,F,W,W,R)")
        print("Or type a single label for the whole clip: F/D/W/R/A/O")
        raw = input("Label(s): ").strip().lower()
        if raw in config.BEHAVIOUR_KEYS:
            behaviour = config.BEHAVIOUR_KEYS[raw]
            _append_metadata(clip_name, 0, behaviour)
        else:
            keys = [k.strip() for k in raw.split(",")]
            for i, k in enumerate(keys):
                beh = config.BEHAVIOUR_KEYS.get(k, "other")
                _append_metadata(clip_name, i, beh)

        clip_count += 1
        cont = input("Record another clip? (Y/n): ").strip().lower()
        if cont == "n":
            break

    cv2.destroyAllWindows()
    logger.info("Clip capture complete. %d clips recorded.", clip_count)


# ---------------------------------------------------------------------------
# Auto-capture mode (headless, fixed interval)
# ---------------------------------------------------------------------------

def run_auto_mode(cam, capture_fps: int = 1, duration_sec: int = 3600):
    """Headless auto-capture: save frames at a fixed interval with no label.

    Use this to build an unlabelled dataset for later annotation with
    annotate.py.
    """
    _ensure_metadata_csv()
    n_saved = 0
    interval = 1.0 / capture_fps
    start = time.time()
    frame_id = 0
    next_capture = start

    logger.info("Auto-capture mode: %d FPS for %ds", capture_fps, duration_sec)

    while time.time() - start < duration_sec:
        frame = cam.read()
        if frame is None:
            time.sleep(0.01)
            continue

        now = time.time()
        if now >= next_capture:
            fname = f"auto_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
            fpath = os.path.join(config.RAW_FRAMES_DIR, fname)
            cv2.imwrite(fpath, frame)
            _append_metadata(fname, frame_id, "unlabelled")
            n_saved += 1
            frame_id += 1
            next_capture = now + interval
            if n_saved % 100 == 0:
                logger.info("Auto-capture: %d frames saved", n_saved)

    logger.info("Auto-capture complete. %d frames saved.", n_saved)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Chicken behaviour data gatherer")
    parser.add_argument("--mode", choices=["frame", "clip", "auto"],
                        default="frame",
                        help="Capture mode (default: frame)")
    parser.add_argument("--fps", type=int, default=5,
                        help="Capture rate for frame/auto modes (default: 5)")
    parser.add_argument("--duration", type=int, default=3600,
                        help="Duration in seconds for auto mode (default: 3600)")
    parser.add_argument("--clip-duration", type=int, default=30,
                        help="Clip length in seconds for clip mode (default: 30)")
    args = parser.parse_args()

    cam = CameraStream(
        source=config.CAMERA_SOURCE,
        width=config.CAMERA_WIDTH,
        height=config.CAMERA_HEIGHT,
        fps=config.CAMERA_FPS,
    )

    logger.info("Starting camera …")
    cam.start()
    time.sleep(1.0)  # Allow camera to warm up

    try:
        if args.mode == "frame":
            run_frame_mode(cam, capture_fps=args.fps)
        elif args.mode == "clip":
            run_clip_mode(cam, clip_duration_sec=args.clip_duration,
                          capture_fps=args.fps)
        elif args.mode == "auto":
            run_auto_mode(cam, capture_fps=args.fps, duration_sec=args.duration)
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        cam.stop()


if __name__ == "__main__":
    main()
