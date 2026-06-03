"""
utils.py — Shared utilities for the poultry weight-estimation pipeline.

Provides:
- Camera intrinsic parameter management
- Depth-to-point-cloud conversion
- Point cloud I/O (PLY)
- Visualization helpers
- General path / CSV helpers
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Default Azure Kinect intrinsics (1080p colour-to-depth aligned mode).
# Override via config JSON or environment variables.
# ---------------------------------------------------------------------------
DEFAULT_INTRINSICS = {
    "fx": 504.97,   # focal length x (pixels)
    "fy": 504.97,   # focal length y (pixels)
    "cx": 320.07,   # principal point x (pixels)
    "cy": 305.33,   # principal point y (pixels)
    "width": 640,
    "height": 576,
}


# ---------------------------------------------------------------------------
# Camera intrinsics
# ---------------------------------------------------------------------------

def load_intrinsics(path: Optional[str] = None) -> dict:
    """
    Load camera intrinsic parameters.

    Parameters
    ----------
    path : str, optional
        Path to a JSON file with keys fx, fy, cx, cy, width, height.
        If None, returns DEFAULT_INTRINSICS.

    Returns
    -------
    dict
        Camera intrinsics dictionary.
    """
    if path and Path(path).exists():
        with open(path, "r") as f:
            return json.load(f)
    return DEFAULT_INTRINSICS.copy()


def save_intrinsics(intrinsics: dict, path: str) -> None:
    """Persist intrinsics to a JSON file."""
    with open(path, "w") as f:
        json.dump(intrinsics, f, indent=2)


# ---------------------------------------------------------------------------
# Depth → Point Cloud
# ---------------------------------------------------------------------------

def depth_to_pointcloud(
    depth_mm: np.ndarray,
    intrinsics: dict,
    mask: Optional[np.ndarray] = None,
    max_depth_mm: float = 2000.0,
    min_depth_mm: float = 200.0,
) -> np.ndarray:
    """
    Convert a 16-bit depth image to a 3-D point cloud.

    Parameters
    ----------
    depth_mm : np.ndarray, shape (H, W)
        Depth image in millimetres (uint16 or float32).
    intrinsics : dict
        Camera intrinsics with keys fx, fy, cx, cy.
    mask : np.ndarray, optional
        Boolean mask (H, W). Only masked pixels are included.
    max_depth_mm : float
        Points deeper than this are discarded.
    min_depth_mm : float
        Points closer than this are discarded.

    Returns
    -------
    np.ndarray, shape (N, 3)
        Array of 3-D points (X, Y, Z) in millimetres.
    """
    fx, fy = intrinsics["fx"], intrinsics["fy"]
    cx, cy = intrinsics["cx"], intrinsics["cy"]

    H, W = depth_mm.shape
    u_grid, v_grid = np.meshgrid(np.arange(W), np.arange(H))

    z = depth_mm.astype(np.float32)

    # Apply depth range filter
    valid = (z > min_depth_mm) & (z < max_depth_mm)
    if mask is not None:
        valid &= mask.astype(bool)

    z = z[valid]
    u = u_grid[valid].astype(np.float32)
    v = v_grid[valid].astype(np.float32)

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    return np.stack([x, y, z], axis=-1)  # (N, 3)


# ---------------------------------------------------------------------------
# PLY I/O
# ---------------------------------------------------------------------------

def save_ply(points: np.ndarray, path: str, colors: Optional[np.ndarray] = None) -> None:
    """
    Write a point cloud to a PLY file.

    Parameters
    ----------
    points : np.ndarray, shape (N, 3)
        XYZ coordinates.
    path : str
        Output file path.
    colors : np.ndarray, shape (N, 3), optional
        RGB colours in [0, 255].
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(points)
    has_color = colors is not None

    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if has_color:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i in range(n):
            row = f"{points[i,0]:.4f} {points[i,1]:.4f} {points[i,2]:.4f}"
            if has_color:
                r, g, b = colors[i].astype(int)
                row += f" {r} {g} {b}"
            f.write(row + "\n")


def load_ply(path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Read a PLY file.

    Returns
    -------
    points : np.ndarray, shape (N, 3)
    colors : np.ndarray or None, shape (N, 3)
    """
    points, colors = [], []
    has_color = False
    in_header = True

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if in_header:
                if "property uchar red" in line:
                    has_color = True
                if line == "end_header":
                    in_header = False
                continue
            parts = line.split()
            points.append([float(parts[0]), float(parts[1]), float(parts[2])])
            if has_color:
                colors.append([int(parts[3]), int(parts[4]), int(parts[5])])

    pts = np.array(points, dtype=np.float32)
    clr = np.array(colors, dtype=np.uint8) if has_color and colors else None
    return pts, clr


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def colorize_depth(depth_mm: np.ndarray, min_mm: float = 200, max_mm: float = 2000) -> np.ndarray:
    """
    Map a depth image to a colour image for display.

    Returns
    -------
    np.ndarray, shape (H, W, 3), dtype uint8, BGR
    """
    clipped = np.clip(depth_mm.astype(np.float32), min_mm, max_mm)
    normalized = ((clipped - min_mm) / (max_mm - min_mm) * 255).astype(np.uint8)
    return cv2.applyColorMap(normalized, cv2.COLORMAP_JET)


def overlay_text(image: np.ndarray, text: str, pos: Tuple[int, int] = (10, 30)) -> np.ndarray:
    """Draw text on an image (in-place copy)."""
    img = image.copy()
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    return img


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def ensure_csv(path: str, columns: list) -> pd.DataFrame:
    """
    Load an existing CSV or create an empty one with given columns.

    Returns
    -------
    pd.DataFrame
    """
    p = Path(path)
    if p.exists():
        return pd.read_csv(p)
    df = pd.DataFrame(columns=columns)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    return df


def append_csv_row(path: str, row: dict) -> None:
    """Append a single row dict to a CSV file (creates file if needed)."""
    p = Path(path)
    df_row = pd.DataFrame([row])
    if p.exists():
        df_row.to_csv(p, mode="a", header=False, index=False)
    else:
        p.parent.mkdir(parents=True, exist_ok=True)
        df_row.to_csv(p, index=False)


# ---------------------------------------------------------------------------
# Depth image I/O (16-bit PNG ↔ numpy)
# ---------------------------------------------------------------------------

def save_depth(depth_mm: np.ndarray, path: str) -> None:
    """Save a uint16 depth image as a 16-bit PNG."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), depth_mm.astype(np.uint16))


def load_depth(path: str) -> np.ndarray:
    """Load a 16-bit PNG depth image as uint16 numpy array."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot load depth image: {path}")
    return img.astype(np.uint16)


def save_metadata(meta: dict, path: str) -> None:
    """Save metadata dict to JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)


def load_metadata(path: str) -> dict:
    """Load JSON metadata."""
    with open(path, "r") as f:
        return json.load(f)
