"""
gather_data.py — Kinect data acquisition for poultry weight estimation.

Connects to an Azure Kinect (via pyk4a) or falls back to reading from a
pre-recorded .mkv/.bag file for development without hardware.

Usage
-----
# Live camera:
    python gather_data.py --output data/raw

# From a recorded file:
    python gather_data.py --output data/raw --playback recording.mkv

Controls (OpenCV window)
------------------------
  s  — save the current frame and enter ground-truth weight
  q  — quit
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np

from utils import (
    DEFAULT_INTRINSICS,
    append_csv_row,
    colorize_depth,
    overlay_text,
    save_depth,
    save_intrinsics,
    save_metadata,
)

# ---------------------------------------------------------------------------
# Try to import pyk4a; fall back to a stub for offline development
# ---------------------------------------------------------------------------
try:
    import pyk4a
    from pyk4a import Config, PyK4A, PyK4APlayback, ColorResolution, DepthMode
    PYKAA_AVAILABLE = True
except ImportError:
    PYKAA_AVAILABLE = False
    print("[gather_data] pyk4a not found — running in DEMO mode with synthetic frames.")


# ---------------------------------------------------------------------------
# Demo / stub camera for development without hardware
# ---------------------------------------------------------------------------

class _DemoCamera:
    """Generates synthetic depth + colour frames for offline development."""

    def __init__(self):
        self.intrinsics = DEFAULT_INTRINSICS.copy()
        self._frame_idx = 0

    def start(self):
        pass

    def get_capture(self):
        """Return a mock capture object with .depth and .color attributes."""
        H, W = self.intrinsics["height"], self.intrinsics["width"]

        # Synthetic depth: flat surface at ~800 mm with a Gaussian bump (chicken)
        depth = np.full((H, W), 800, dtype=np.uint16)
        cy, cx = H // 2, W // 2
        Y, X = np.ogrid[:H, :W]
        r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        bump = (120 * np.exp(-(r ** 2) / (2 * 100 ** 2))).astype(np.uint16)
        depth = np.clip(depth.astype(int) - bump, 200, 1500).astype(np.uint16)

        # Add noise
        depth = (depth + np.random.randint(-5, 5, depth.shape)).astype(np.uint16)

        # Synthetic colour: green-ish blob
        color = np.zeros((H, W, 4), dtype=np.uint8)
        color[:, :, 1] = 80  # green channel
        color[:, :, 3] = 255
        mask_2d = r < 120
        color[mask_2d, 0] = 180
        color[mask_2d, 1] = 140
        color[mask_2d, 2] = 100

        self._frame_idx += 1

        class _Capture:
            pass

        cap = _Capture()
        cap.depth = depth
        cap.color = color
        return cap

    def stop(self):
        pass

    @property
    def calibration(self):
        return None


# ---------------------------------------------------------------------------
# Kinect wrapper
# ---------------------------------------------------------------------------

def open_camera(playback_path: str | None = None) -> tuple:
    """
    Open the Kinect camera or a playback file.

    Returns
    -------
    camera : camera object (PyK4A, PyK4APlayback, or _DemoCamera)
    intrinsics : dict
    """
    if not PYKAA_AVAILABLE:
        cam = _DemoCamera()
        cam.start()
        return cam, DEFAULT_INTRINSICS.copy()

    if playback_path:
        print(f"[gather_data] Opening playback: {playback_path}")
        cam = PyK4APlayback(playback_path)
        cam.open()
    else:
        print("[gather_data] Opening live Azure Kinect …")
        cam = PyK4A(
            Config(
                color_resolution=ColorResolution.RES_720P,
                depth_mode=DepthMode.NFOV_UNBINNED,
                synchronized_images_only=True,
            )
        )
        cam.start()

    # Extract intrinsics from calibration
    try:
        cal = cam.calibration
        depth_cal = cal.get_camera_matrix(pyk4a.CalibrationType.DEPTH)
        intrinsics = {
            "fx": float(depth_cal[0, 0]),
            "fy": float(depth_cal[1, 1]),
            "cx": float(depth_cal[0, 2]),
            "cy": float(depth_cal[1, 2]),
            "width": cam.configuration["depth_mode"].value[0],
            "height": cam.configuration["depth_mode"].value[1],
        }
    except Exception:
        print("[gather_data] Could not read intrinsics — using defaults.")
        intrinsics = DEFAULT_INTRINSICS.copy()

    return cam, intrinsics


# ---------------------------------------------------------------------------
# Main acquisition loop
# ---------------------------------------------------------------------------

def run_acquisition(output_dir: str, playback_path: str | None = None) -> None:
    """
    Run the live data-acquisition loop.

    Parameters
    ----------
    output_dir : str
        Root directory to save captured samples.
    playback_path : str, optional
        Path to an .mkv/.bag file for offline playback.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    csv_path = str(out / "weights.csv")
    intrinsics_path = str(out / "intrinsics.json")

    camera, intrinsics = open_camera(playback_path)
    save_intrinsics(intrinsics, intrinsics_path)
    print(f"[gather_data] Intrinsics saved → {intrinsics_path}")
    print("[gather_data] Press 's' to save a sample, 'q' to quit.\n")

    sample_id = _next_sample_id(out)

    while True:
        try:
            capture = camera.get_capture()
        except StopIteration:
            print("[gather_data] Playback ended.")
            break

        if capture is None or capture.depth is None:
            continue

        depth_frame = capture.depth  # shape (H, W), uint16, mm
        color_frame = capture.color  # shape (H, W, 4), BGRA uint8

        # Convert colour for display
        if color_frame is not None and color_frame.ndim == 3 and color_frame.shape[2] == 4:
            color_bgr = cv2.cvtColor(color_frame, cv2.COLOR_BGRA2BGR)
        elif color_frame is not None:
            color_bgr = color_frame
        else:
            color_bgr = np.zeros((*depth_frame.shape, 3), dtype=np.uint8)

        depth_vis = colorize_depth(depth_frame)
        combined = np.hstack([color_bgr, depth_vis])
        combined = overlay_text(combined, f"Sample #{sample_id} | s=save  q=quit")

        cv2.imshow("Poultry Weight Estimation — Data Gathering", combined)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord("s"):
            sample_dir = out / f"sample_{sample_id:04d}"
            sample_dir.mkdir(parents=True, exist_ok=True)

            # Paths
            depth_path = str(sample_dir / "depth.png")
            color_path = str(sample_dir / "color.jpg")
            meta_path = str(sample_dir / "meta.json")

            # Save depth & colour
            save_depth(depth_frame, depth_path)
            cv2.imwrite(color_path, color_bgr)

            # Metadata
            meta = {
                "sample_id": sample_id,
                "timestamp": time.time(),
                "depth_path": depth_path,
                "color_path": color_path,
                "intrinsics": intrinsics,
            }
            save_metadata(meta, meta_path)

            # Prompt for weight
            weight_g = _prompt_weight(sample_id)

            # Append to CSV
            append_csv_row(
                csv_path,
                {
                    "sample_id": sample_id,
                    "timestamp": meta["timestamp"],
                    "weight_g": weight_g,
                    "sample_dir": str(sample_dir),
                },
            )
            print(f"[gather_data] Saved sample {sample_id} — weight={weight_g} g\n")
            sample_id += 1

    camera.stop()
    cv2.destroyAllWindows()
    print(f"[gather_data] Done. {sample_id} samples recorded in {out}")


def _next_sample_id(out: Path) -> int:
    """Return the next available sample ID by scanning existing directories."""
    existing = sorted(out.glob("sample_*"))
    if not existing:
        return 0
    last = existing[-1].name.split("_")[-1]
    return int(last) + 1


def _prompt_weight(sample_id: int) -> float:
    """Prompt the user to enter the ground-truth weight for a sample."""
    while True:
        raw = input(f"  → Enter ground-truth weight (grams) for sample {sample_id}: ").strip()
        try:
            w = float(raw)
            if w > 0:
                return w
        except ValueError:
            pass
        print("  Invalid input. Please enter a positive number.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Acquire Kinect depth+RGB data with weight labels.")
    parser.add_argument("--output", default="data/raw", help="Directory to store captured samples.")
    parser.add_argument("--playback", default=None, help="Path to .mkv/.bag file for offline playback.")
    args = parser.parse_args()

    run_acquisition(output_dir=args.output, playback_path=args.playback)
