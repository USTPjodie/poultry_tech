"""
features.py — Volumetric and shape feature extraction from processed samples.

For each processed sample the script:
 1. Loads the cropped depth map and point cloud (.ply).
 2. Computes convex-hull volume (scipy.spatial.ConvexHull).
 3. Estimates voxel-column body volume by integrating depth pixels.
 4. Extracts shape descriptors: length, width, height, principal-axis ratios.
 5. Builds a per-sample feature vector and appends it to features.csv.

Usage
-----
    python features.py --data data/processed --output data/features/features.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy.spatial import ConvexHull, QhullError
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("[features] scipy not found — ConvexHull volume will be skipped.")

from utils import load_depth, load_metadata, load_ply


# ---------------------------------------------------------------------------
# Feature extraction functions
# ---------------------------------------------------------------------------

def convex_hull_volume(points: np.ndarray) -> float:
    """
    Compute the convex hull volume of a point cloud (in mm³).

    Parameters
    ----------
    points : np.ndarray (N, 3)

    Returns
    -------
    float : volume in mm³, or 0.0 if computation fails / scipy unavailable.
    """
    if not SCIPY_AVAILABLE or len(points) < 4:
        return 0.0
    try:
        hull = ConvexHull(points)
        return float(hull.volume)
    except (QhullError, Exception) as e:
        print(f"  [WARN] ConvexHull failed: {e}")
        return 0.0


def voxel_column_volume(
    depth_mm: np.ndarray,
    mask: np.ndarray,
    intrinsics: dict,
    background_depth_mm: float | None = None,
) -> float:
    """
    Estimate body volume by treating each depth pixel as a rectangular column.

    Each foreground pixel at depth Z covers a real-world area of
    (Z/fx) × (Z/fy) mm².  The height of the column is (background_z - Z).
    Volume = Σ (Z/fx) · (Z/fy) · max(background_z − Z, 0).

    Parameters
    ----------
    depth_mm : np.ndarray (H, W), uint16
    mask : np.ndarray (H, W), uint8  — foreground mask
    intrinsics : dict
    background_depth_mm : float, optional
        If None, uses the median depth of the frame border as background.

    Returns
    -------
    float : estimated volume in mm³
    """
    fx, fy = intrinsics["fx"], intrinsics["fy"]
    z = depth_mm.astype(np.float32)

    # Estimate background depth from image border if not provided
    if background_depth_mm is None:
        border_pixels = np.concatenate([
            z[0, :], z[-1, :], z[:, 0], z[:, -1]
        ])
        border_pixels = border_pixels[border_pixels > 0]
        background_depth_mm = float(np.median(border_pixels)) if len(border_pixels) else 1000.0

    fg = (mask > 0) & (z > 0)
    z_fg = z[fg]

    # Real-world pixel footprint at depth Z
    pixel_width = z_fg / fx   # mm
    pixel_height = z_fg / fy  # mm
    pixel_area = pixel_width * pixel_height  # mm²

    # Column height = background minus the chicken surface
    col_height = np.maximum(background_depth_mm - z_fg, 0)
    volume = float(np.sum(pixel_area * col_height))
    return volume


def shape_descriptors(points: np.ndarray) -> dict:
    """
    Compute shape descriptors from a 3-D point cloud.

    Uses PCA to find principal axes and returns length/width/height in mm.

    Returns
    -------
    dict with keys: length_mm, width_mm, height_mm, aspect_lw, aspect_lh
    """
    if len(points) < 3:
        return {k: 0.0 for k in ["length_mm", "width_mm", "height_mm", "aspect_lw", "aspect_lh"]}

    centred = points - points.mean(axis=0)
    cov = np.cov(centred.T)
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        return {k: 0.0 for k in ["length_mm", "width_mm", "height_mm", "aspect_lw", "aspect_lh"]}

    # Sort descending by eigenvalue
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]

    projected = centred @ eigenvectors
    extent = projected.max(axis=0) - projected.min(axis=0)  # shape (3,)

    length, width, height = float(extent[0]), float(extent[1]), float(extent[2])
    return {
        "length_mm": length,
        "width_mm": width,
        "height_mm": height,
        "aspect_lw": length / (width + 1e-6),
        "aspect_lh": length / (height + 1e-6),
    }


def depth_histogram_features(depth_mm: np.ndarray, mask: np.ndarray, n_bins: int = 8) -> dict:
    """
    Compute a normalised depth histogram over the foreground region.

    Returns
    -------
    dict with keys hist_0 … hist_{n_bins-1}
    """
    fg = depth_mm[mask > 0].astype(np.float32)
    fg = fg[fg > 0]
    if len(fg) == 0:
        return {f"hist_{i}": 0.0 for i in range(n_bins)}
    hist, _ = np.histogram(fg, bins=n_bins)
    hist = hist.astype(float) / (hist.sum() + 1e-9)
    return {f"hist_{i}": float(hist[i]) for i in range(n_bins)}


def extract_features_for_sample(
    proc_dir: Path,
    intrinsics: dict,
) -> dict | None:
    """
    Extract all features for a single processed sample directory.

    Expects: depth_cropped.png, pointcloud.ply, mask.png, meta.json.

    Returns
    -------
    dict of features (includes sample_id and weight_g)
    """
    import cv2

    depth_path = proc_dir / "depth_cropped.png"
    ply_path = proc_dir / "pointcloud.ply"
    mask_path = proc_dir / "mask.png"
    meta_path = proc_dir / "meta.json"

    if not depth_path.exists() or not ply_path.exists():
        print(f"  [WARN] Missing files in {proc_dir}. Skipping.")
        return None

    depth_mm = load_depth(str(depth_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        mask = np.ones(depth_mm.shape, dtype=np.uint8) * 255

    points, _ = load_ply(str(ply_path))
    meta = load_metadata(str(meta_path)) if meta_path.exists() else {}
    sample_id = meta.get("sample_id", proc_dir.name)
    weight_g = meta.get("weight_g", None)  # may be None

    # --- Feature computation ---
    feats: dict = {"sample_id": sample_id, "weight_g": weight_g}

    # Convex hull volume
    feats["convex_hull_volume_mm3"] = convex_hull_volume(points)

    # Voxel column volume
    feats["voxel_volume_mm3"] = voxel_column_volume(depth_mm, mask, intrinsics)

    # Shape descriptors
    feats.update(shape_descriptors(points))

    # Depth histogram
    feats.update(depth_histogram_features(depth_mm, mask))

    # Basic 2-D features (carry-forward from preprocessing)
    basic = meta.get("basic_features", {})
    for k in ["pixel_area", "bbox_w", "bbox_h", "mean_depth", "std_depth", "depth_range", "n_points"]:
        feats[k] = basic.get(k, meta.get(k, 0))

    print(f"  ✓ {proc_dir.name}  hull={feats['convex_hull_volume_mm3']:.0f} mm³  "
          f"voxel={feats['voxel_volume_mm3']:.0f} mm³")
    return feats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_feature_extraction(
    processed_dir: str,
    output_csv: str,
    weights_csv: str | None = None,
) -> None:
    """
    Extract features from all processed samples and save to a CSV file.

    Parameters
    ----------
    processed_dir : str
        Root directory of processed samples (output of preprocess.py).
    output_csv : str
        Path for the output features CSV.
    weights_csv : str, optional
        Path to the weights.csv from data gathering (to attach weight labels).
    """
    proc_root = Path(processed_dir)
    intrinsics_path = proc_root.parent / "raw" / "intrinsics.json"

    # Try to find intrinsics
    from utils import load_intrinsics
    intrinsics = load_intrinsics(str(intrinsics_path) if intrinsics_path.exists() else None)

    # Load weight labels
    weight_map: dict = {}
    if weights_csv and Path(weights_csv).exists():
        wdf = pd.read_csv(weights_csv)
        weight_map = dict(zip(wdf["sample_id"].astype(str), wdf["weight_g"]))

    sample_dirs = sorted(proc_root.glob("sample_*"))
    if not sample_dirs:
        print(f"[features] No processed samples found in {proc_root}.")
        return

    print(f"[features] Extracting features from {len(sample_dirs)} samples …")
    all_feats = []

    for sd in sample_dirs:
        f = extract_features_for_sample(sd, intrinsics)
        if f is None:
            continue
        sid = str(f["sample_id"])
        if sid in weight_map:
            f["weight_g"] = weight_map[sid]
        all_feats.append(f)

    if not all_feats:
        print("[features] No features extracted.")
        return

    df = pd.DataFrame(all_feats)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\n[features] Features saved → {output_csv}")
    print(df.describe())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract volumetric features from processed samples.")
    parser.add_argument("--data", default="data/processed", help="Processed samples directory.")
    parser.add_argument("--output", default="data/features/features.csv", help="Output features CSV path.")
    parser.add_argument("--weights", default="data/raw/weights.csv", help="Ground-truth weights CSV.")
    args = parser.parse_args()

    run_feature_extraction(
        processed_dir=args.data,
        output_csv=args.output,
        weights_csv=args.weights,
    )
