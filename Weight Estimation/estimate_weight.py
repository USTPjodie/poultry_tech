"""
estimate_weight.py — Real-time weight estimation from Kinect frames.

Loads the fine-tuned (or standard) model and runs inference either
on a live Kinect stream or from a playback file.

Usage
-----
    # Live camera, sklearn model:
    python estimate_weight.py --model models/model_rf_finetuned.pkl \
        --intrinsics data/raw/intrinsics.json

    # Playback, CNN model:
    python estimate_weight.py --model models/model_cnn_finetuned.pt \
        --model-type cnn --playback recording.mkv
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import cv2
import numpy as np

from features import (
    convex_hull_volume,
    depth_histogram_features,
    shape_descriptors,
    voxel_column_volume,
)
from preprocess import (
    crop_to_mask,
    filter_depth,
    remove_background,
    segment_largest_blob,
)
from utils import (
    colorize_depth,
    depth_to_pointcloud,
    load_intrinsics,
    overlay_text,
)

DEFAULT_MIN_DEPTH = 300
DEFAULT_MAX_DEPTH = 1500


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------

def load_sklearn_model(model_path: str, feature_names_path: str | None = None):
    """Load a pickled sklearn Pipeline and associated feature names."""
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    feat_names = None
    if feature_names_path and Path(feature_names_path).exists():
        with open(feature_names_path) as f:
            feat_names = json.load(f)
    elif Path(model_path).parent.joinpath("feature_names.json").exists():
        with open(str(Path(model_path).parent / "feature_names.json")) as f:
            feat_names = json.load(f)
    return model, feat_names


def load_cnn_model(model_path: str, device=None):
    """Load a PyTorch CNN model."""
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        raise ImportError("PyTorch is required for CNN inference.")

    class DepthCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 4)),
            )
            self.regressor = nn.Sequential(
                nn.Flatten(), nn.Linear(128 * 16, 256), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1),
            )
        def forward(self, x):
            return self.regressor(self.features(x)).squeeze(-1)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DepthCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def preprocess_frame(
    depth_mm: np.ndarray,
    intrinsics: dict,
    min_depth: int = DEFAULT_MIN_DEPTH,
    max_depth: int = DEFAULT_MAX_DEPTH,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply the full preprocessing pipeline to a single depth frame.

    Returns
    -------
    crop_depth, crop_mask, points, bbox
    """
    depth_filtered = filter_depth(depth_mm)
    mask = remove_background(depth_filtered, min_depth, max_depth)
    mask = segment_largest_blob(mask)

    dummy_color = np.zeros((*depth_filtered.shape, 3), dtype=np.uint8)
    crop_depth, _, crop_mask, bbox = crop_to_mask(depth_filtered, dummy_color, mask)

    points = depth_to_pointcloud(
        crop_depth, intrinsics, mask=crop_mask,
        min_depth_mm=min_depth, max_depth_mm=max_depth,
    )
    return crop_depth, crop_mask, points, bbox


def build_feature_vector(
    crop_depth: np.ndarray,
    crop_mask: np.ndarray,
    points: np.ndarray,
    bbox: tuple,
    intrinsics: dict,
    feature_names: list[str],
) -> np.ndarray:
    """Build the feature vector expected by the sklearn model."""
    features: dict = {}

    features["convex_hull_volume_mm3"] = convex_hull_volume(points)
    features["voxel_volume_mm3"] = voxel_column_volume(crop_depth, crop_mask, intrinsics)
    features.update(shape_descriptors(points))
    features.update(depth_histogram_features(crop_depth, crop_mask))

    H, W = crop_depth.shape
    fg = crop_depth[crop_mask > 0].astype(np.float32)
    fg = fg[fg > 0]
    features["pixel_area"] = int(np.sum(crop_mask > 0))
    features["bbox_w"] = int(bbox[2])
    features["bbox_h"] = int(bbox[3])
    features["mean_depth"] = float(fg.mean()) if len(fg) else 0.0
    features["std_depth"] = float(fg.std()) if len(fg) else 0.0
    features["depth_range"] = float(fg.max() - fg.min()) if len(fg) else 0.0
    features["n_points"] = len(points)

    vec = np.array([features.get(n, 0.0) for n in feature_names], dtype=np.float32)
    return vec.reshape(1, -1)


def infer_sklearn(model, feature_vec: np.ndarray) -> float:
    """Run sklearn model inference."""
    return float(model.predict(feature_vec)[0])


def infer_cnn(model, device, crop_depth: np.ndarray) -> float:
    """Run CNN inference on a cropped depth map."""
    import torch
    d = cv2.resize(crop_depth, (224, 224)).astype(np.float32) / 2000.0
    tensor = torch.from_numpy(d).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,H,W)
    with torch.no_grad():
        pred = model(tensor).item()
    return float(pred)


# ---------------------------------------------------------------------------
# Camera source (same as gather_data.py)
# ---------------------------------------------------------------------------

def open_camera(playback_path: str | None = None):
    try:
        import pyk4a
        from pyk4a import PyK4A, PyK4APlayback, Config, ColorResolution, DepthMode
        if playback_path:
            cam = PyK4APlayback(playback_path)
            cam.open()
        else:
            cam = PyK4A(Config(color_resolution=ColorResolution.RES_720P,
                               depth_mode=DepthMode.NFOV_UNBINNED,
                               synchronized_images_only=True))
            cam.start()
        return cam
    except ImportError:
        from gather_data import _DemoCamera
        cam = _DemoCamera()
        cam.start()
        return cam


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------

def run_inference(
    model_path: str,
    model_type: str = "rf",
    intrinsics_path: str | None = None,
    playback_path: str | None = None,
    min_depth: int = DEFAULT_MIN_DEPTH,
    max_depth: int = DEFAULT_MAX_DEPTH,
) -> None:
    """
    Run the live weight-estimation loop.

    Parameters
    ----------
    model_path : str
    model_type : str — 'rf', 'gb', 'mlp', or 'cnn'
    intrinsics_path : str, optional
    playback_path : str, optional
    """
    intrinsics = load_intrinsics(intrinsics_path)

    # Load model
    if model_type == "cnn":
        model, device = load_cnn_model(model_path)
        feature_names = None
    else:
        model, feature_names = load_sklearn_model(model_path)
        device = None

    camera = open_camera(playback_path)
    print("[estimate] Press 'q' to quit.")

    weight_estimate = None
    fps_time = time.time()
    fps = 0.0

    while True:
        try:
            capture = camera.get_capture()
        except StopIteration:
            print("[estimate] Playback ended.")
            break

        if capture is None or capture.depth is None:
            continue

        depth_frame = capture.depth
        color_frame = capture.color

        # Preprocessing
        crop_depth, crop_mask, points, bbox = preprocess_frame(
            depth_frame, intrinsics, min_depth, max_depth
        )

        # Inference
        if len(points) > 10:
            try:
                if model_type == "cnn":
                    weight_estimate = infer_cnn(model, device, crop_depth)
                else:
                    fvec = build_feature_vector(
                        crop_depth, crop_mask, points, bbox, intrinsics, feature_names
                    )
                    weight_estimate = infer_sklearn(model, fvec)
            except Exception as e:
                print(f"  [WARN] Inference error: {e}")

        # Build display
        if color_frame is not None and color_frame.ndim == 3:
            if color_frame.shape[2] == 4:
                display = cv2.cvtColor(color_frame, cv2.COLOR_BGRA2BGR)
            else:
                display = color_frame.copy()
            dH, dW = depth_frame.shape
            if display.shape[:2] != (dH, dW):
                display = cv2.resize(display, (dW, dH))
        else:
            display = np.zeros((*depth_frame.shape, 3), dtype=np.uint8)

        depth_vis = colorize_depth(depth_frame, min_depth, max_depth)

        # Overlay segmentation mask
        mask_full = np.zeros(depth_frame.shape, dtype=np.uint8)
        x, y, w, h = bbox
        if crop_mask.shape == mask_full[y:y+h, x:x+w].shape:
            mask_full[y:y+h, x:x+w] = crop_mask
        overlay = display.copy()
        overlay[mask_full > 0] = (0, 255, 100)
        display = cv2.addWeighted(display, 0.6, overlay, 0.4, 0)

        # FPS
        now = time.time()
        fps = 0.9 * fps + 0.1 / max(now - fps_time, 1e-9)
        fps_time = now

        # Weight text
        if weight_estimate is not None:
            wt_text = f"Weight: {weight_estimate:.0f} g  ({weight_estimate/1000:.3f} kg)"
        else:
            wt_text = "Weight: detecting..."

        display = overlay_text(display, wt_text, (10, 35))
        display = overlay_text(display, f"FPS: {fps:.1f}  pts={len(points)}", (10, 65))

        combined = np.hstack([display, depth_vis])
        cv2.imshow("Poultry Weight Estimator", combined)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.stop()
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live poultry weight estimation.")
    parser.add_argument("--model", required=True, help="Path to trained model file.")
    parser.add_argument("--model-type", default="rf",
                        choices=["rf", "gb", "mlp", "cnn"])
    parser.add_argument("--intrinsics", default=None,
                        help="Path to intrinsics.json (defaults to built-in Azure Kinect params).")
    parser.add_argument("--playback", default=None, help="Path to .mkv/.bag for offline testing.")
    parser.add_argument("--min-depth", type=int, default=DEFAULT_MIN_DEPTH)
    parser.add_argument("--max-depth", type=int, default=DEFAULT_MAX_DEPTH)
    args = parser.parse_args()

    run_inference(
        model_path=args.model,
        model_type=args.model_type,
        intrinsics_path=args.intrinsics,
        playback_path=args.playback,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
    )
