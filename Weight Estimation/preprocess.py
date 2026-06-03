"""
preprocess.py — Preprocessing pipeline for raw Kinect captures.

Steps
-----
1. Load raw depth + RGB images for each sample.
2. Apply depth noise filtering (bilateral / median).
3. Background removal via depth thresholding.
4. Optional: connected-components segmentation to isolate the largest foreground blob.
5. Crop to the chicken's bounding box.
6. Convert depth to a 3-D point cloud.
7. Save processed depth map, point cloud (.ply), and segmentation mask.
8. Compute basic 2-D/3-D features and append them to a features.csv.

Usage
-----
    python preprocess.py --data data/raw --output data/processed
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from utils import (
    append_csv_row,
    colorize_depth,
    depth_to_pointcloud,
    ensure_csv,
    load_depth,
    load_intrinsics,
    load_metadata,
    save_depth,
    save_metadata,
    save_ply,
)

# ---------------------------------------------------------------------------
# Preprocessing parameters (can be overridden via CLI)
# ---------------------------------------------------------------------------
DEFAULT_MIN_DEPTH = 300    # mm — anything closer is clipped
DEFAULT_MAX_DEPTH = 1500   # mm — anything farther is background
BILATERAL_D = 9
BILATERAL_SIGMA_COLOR = 75
BILATERAL_SIGMA_SPACE = 75
MIN_BLOB_AREA = 5000       # pixels — minimum blob to consider as chicken


# ---------------------------------------------------------------------------
# Individual processing steps
# ---------------------------------------------------------------------------

def filter_depth(depth_mm: np.ndarray) -> np.ndarray:
    """
    Apply a bilateral filter to smooth depth noise while preserving edges.

    Parameters
    ----------
    depth_mm : np.ndarray (H, W), uint16
        Raw depth image in millimetres.

    Returns
    -------
    np.ndarray (H, W), uint16
        Filtered depth image.
    """
    # Bilateral filter works on float32
    d_f32 = depth_mm.astype(np.float32)
    filtered = cv2.bilateralFilter(d_f32, BILATERAL_D, BILATERAL_SIGMA_COLOR, BILATERAL_SIGMA_SPACE)
    return filtered.astype(np.uint16)


def remove_background(
    depth_mm: np.ndarray,
    min_depth: int = DEFAULT_MIN_DEPTH,
    max_depth: int = DEFAULT_MAX_DEPTH,
) -> np.ndarray:
    """
    Create a foreground mask by thresholding the depth range.

    Returns
    -------
    mask : np.ndarray (H, W), uint8
        Binary mask: 255 = foreground, 0 = background.
    """
    mask = np.zeros_like(depth_mm, dtype=np.uint8)
    mask[(depth_mm >= min_depth) & (depth_mm <= max_depth)] = 255
    return mask


def segment_largest_blob(mask: np.ndarray) -> np.ndarray:
    """
    Keep only the largest connected component in the mask (assumed to be the chicken).

    Parameters
    ----------
    mask : np.ndarray (H, W), uint8

    Returns
    -------
    np.ndarray (H, W), uint8  — refined mask
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels < 2:
        return mask  # nothing to filter

    # Label 0 is background; find the largest foreground component
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = int(np.argmax(areas)) + 1

    if stats[largest_label, cv2.CC_STAT_AREA] < MIN_BLOB_AREA:
        return mask  # keep original if blob is too small

    refined = np.zeros_like(mask)
    refined[labels == largest_label] = 255
    return refined


def crop_to_mask(
    depth_mm: np.ndarray,
    color_bgr: np.ndarray,
    mask: np.ndarray,
    padding: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple]:
    """
    Crop depth, colour, and mask arrays to the bounding box of the foreground mask.

    Returns
    -------
    crop_depth, crop_color, crop_mask : cropped arrays
    bbox : (x, y, w, h) bounding box used
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        # No contour found — return originals
        return depth_mm, color_bgr, mask, (0, 0, depth_mm.shape[1], depth_mm.shape[0])

    all_pts = np.vstack(contours)
    x, y, w, h = cv2.boundingRect(all_pts)

    H, W = depth_mm.shape[:2]
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(W, x + w + padding)
    y2 = min(H, y + h + padding)

    return (
        depth_mm[y1:y2, x1:x2],
        color_bgr[y1:y2, x1:x2],
        mask[y1:y2, x1:x2],
        (x1, y1, x2 - x1, y2 - y1),
    )


def compute_basic_features(
    depth_mm: np.ndarray,
    mask: np.ndarray,
    bbox: tuple,
) -> dict:
    """
    Compute simple 2-D features from the cropped depth and mask.

    Features
    --------
    pixel_area : number of foreground pixels
    bbox_w, bbox_h : bounding box dimensions (pixels)
    mean_depth, std_depth : depth statistics within the mask (mm)
    depth_range : max - min depth within mask
    """
    fg = depth_mm[mask > 0]
    fg = fg[fg > 0]  # exclude invalid zeros
    features = {
        "pixel_area": int(np.sum(mask > 0)),
        "bbox_w": int(bbox[2]),
        "bbox_h": int(bbox[3]),
        "mean_depth": float(fg.mean()) if len(fg) else 0.0,
        "std_depth": float(fg.std()) if len(fg) else 0.0,
        "depth_range": float(fg.max() - fg.min()) if len(fg) else 0.0,
    }
    return features


# ---------------------------------------------------------------------------
# Per-sample processor
# ---------------------------------------------------------------------------

def process_sample(
    sample_dir: Path,
    out_dir: Path,
    intrinsics: dict,
    min_depth: int = DEFAULT_MIN_DEPTH,
    max_depth: int = DEFAULT_MAX_DEPTH,
) -> dict | None:
    """
    Process one raw sample directory.

    Parameters
    ----------
    sample_dir : Path
        Directory containing depth.png, color.jpg, meta.json.
    out_dir : Path
        Root output directory.
    intrinsics : dict
        Camera intrinsics.
    min_depth, max_depth : int
        Depth thresholds in mm.

    Returns
    -------
    features : dict  (returns None if processing fails)
    """
    depth_path = sample_dir / "depth.png"
    color_path = sample_dir / "color.jpg"
    meta_path = sample_dir / "meta.json"

    if not depth_path.exists() or not color_path.exists():
        print(f"  [WARN] Missing files in {sample_dir}. Skipping.")
        return None

    # Load
    depth_mm = load_depth(str(depth_path))
    color_bgr = cv2.imread(str(color_path))
    if color_bgr is None:
        print(f"  [WARN] Cannot load colour image: {color_path}. Skipping.")
        return None

    meta = load_metadata(str(meta_path)) if meta_path.exists() else {}
    sample_id = meta.get("sample_id", sample_dir.name)

    # 1. Filter noise
    depth_filtered = filter_depth(depth_mm)

    # 2. Background removal
    mask = remove_background(depth_filtered, min_depth, max_depth)

    # 3. Segment largest blob (the chicken)
    mask = segment_largest_blob(mask)

    # 4. Resize colour to match depth if needed
    dH, dW = depth_filtered.shape
    if color_bgr.shape[:2] != (dH, dW):
        color_bgr = cv2.resize(color_bgr, (dW, dH))

    # 5. Crop
    crop_depth, crop_color, crop_mask, bbox = crop_to_mask(depth_filtered, color_bgr, mask)

    # 6. Point cloud
    points = depth_to_pointcloud(
        crop_depth, intrinsics, mask=crop_mask, min_depth_mm=min_depth, max_depth_mm=max_depth
    )

    # 7. Basic features
    features = compute_basic_features(crop_depth, crop_mask, bbox)
    features["sample_id"] = sample_id
    features["n_points"] = len(points)

    # 8. Save outputs
    proc_dir = out_dir / f"sample_{str(sample_id).zfill(4)}"
    proc_dir.mkdir(parents=True, exist_ok=True)

    save_depth(crop_depth, str(proc_dir / "depth_cropped.png"))
    cv2.imwrite(str(proc_dir / "color_cropped.jpg"), crop_color)
    cv2.imwrite(str(proc_dir / "mask.png"), crop_mask)
    save_ply(points, str(proc_dir / "pointcloud.ply"))

    proc_meta = {**meta, "bbox": list(bbox), "n_points": len(points), "basic_features": features}
    save_metadata(proc_meta, str(proc_dir / "meta.json"))

    print(f"  ✓ {sample_dir.name}  →  {len(points)} points,  {features['pixel_area']} px")
    return features


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_preprocessing(
    data_dir: str,
    output_dir: str,
    min_depth: int = DEFAULT_MIN_DEPTH,
    max_depth: int = DEFAULT_MAX_DEPTH,
) -> None:
    """
    Process all raw samples found under data_dir.

    Parameters
    ----------
    data_dir : str
        Root directory containing sample_XXXX sub-directories and weights.csv.
    output_dir : str
        Root output directory for processed data.
    """
    data_root = Path(data_dir)
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    intrinsics = load_intrinsics(str(data_root / "intrinsics.json"))
    weights_csv = data_root / "weights.csv"

    if weights_csv.exists():
        weights_df = pd.read_csv(weights_csv)
        weight_map = dict(zip(weights_df["sample_id"].astype(str), weights_df["weight_g"]))
    else:
        print("[preprocess] weights.csv not found — weight column will be missing.")
        weight_map = {}

    sample_dirs = sorted(data_root.glob("sample_*"))
    if not sample_dirs:
        print(f"[preprocess] No sample directories found in {data_root}. Run gather_data.py first.")
        return

    print(f"[preprocess] Processing {len(sample_dirs)} samples …")
    all_features = []

    for sd in sample_dirs:
        feats = process_sample(sd, out_root, intrinsics, min_depth, max_depth)
        if feats is None:
            continue
        sid = str(feats["sample_id"])
        feats["weight_g"] = weight_map.get(sid, None)
        all_features.append(feats)

    if all_features:
        feat_df = pd.DataFrame(all_features)
        feat_csv_path = str(out_root / "basic_features.csv")
        feat_df.to_csv(feat_csv_path, index=False)
        print(f"\n[preprocess] Basic features → {feat_csv_path}")
    else:
        print("[preprocess] No features extracted.")

    print(f"[preprocess] Done. Processed data in {out_root}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess raw Kinect samples.")
    parser.add_argument("--data", default="data/raw", help="Raw data directory (output of gather_data.py).")
    parser.add_argument("--output", default="data/processed", help="Processed data output directory.")
    parser.add_argument("--min-depth", type=int, default=DEFAULT_MIN_DEPTH, help="Min depth threshold (mm).")
    parser.add_argument("--max-depth", type=int, default=DEFAULT_MAX_DEPTH, help="Max depth threshold (mm).")
    args = parser.parse_args()

    run_preprocessing(
        data_dir=args.data,
        output_dir=args.output,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
    )
