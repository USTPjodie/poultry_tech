"""
train_model.py — Train a weight-estimation model from extracted features.

Supported model types
---------------------
  rf   : Random Forest Regressor  (default, no GPU required)
  gb   : Gradient Boosting Regressor
  mlp  : Multi-Layer Perceptron (scikit-learn)
  cnn  : Lightweight CNN on depth maps (requires PyTorch)

Usage
-----
    # Classic feature-based:
    python train_model.py --features data/features/features.csv --model-type rf

    # Deep learning (depth maps):
    python train_model.py --features data/features/features.csv \
        --processed data/processed --model-type cnn --epochs 50
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Feature columns used for classical models
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    "convex_hull_volume_mm3",
    "voxel_volume_mm3",
    "length_mm",
    "width_mm",
    "height_mm",
    "aspect_lw",
    "aspect_lh",
    "pixel_area",
    "bbox_w",
    "bbox_h",
    "mean_depth",
    "std_depth",
    "depth_range",
    "n_points",
] + [f"hist_{i}" for i in range(8)]

TARGET_COL = "weight_g"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(features_csv: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load feature CSV and return X, y arrays.

    Returns
    -------
    X : np.ndarray (N, F)
    y : np.ndarray (N,)
    feature_names : List[str]
    """
    df = pd.read_csv(features_csv)

    # Drop rows without a weight label
    df = df.dropna(subset=[TARGET_COL])
    print(f"[train] Loaded {len(df)} labelled samples.")

    # Select only columns present in the dataframe
    available_cols = [c for c in FEATURE_COLS if c in df.columns]
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"[train] Missing feature columns (will be skipped): {missing}")

    df[available_cols] = df[available_cols].fillna(0)
    X = df[available_cols].values.astype(np.float32)
    y = df[TARGET_COL].values.astype(np.float32)
    return X, y, available_cols


# ---------------------------------------------------------------------------
# Classical model training
# ---------------------------------------------------------------------------

def build_classical_pipeline(model_type: str) -> Pipeline:
    """
    Build a scikit-learn Pipeline with scaler + regressor.

    Parameters
    ----------
    model_type : str — 'rf', 'gb', or 'mlp'
    """
    if model_type == "rf":
        regressor = RandomForestRegressor(n_estimators=200, max_features="sqrt", random_state=42)
    elif model_type == "gb":
        regressor = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                               max_depth=4, random_state=42)
    elif model_type == "mlp":
        regressor = MLPRegressor(hidden_layer_sizes=(128, 64, 32),
                                  activation="relu", max_iter=500, random_state=42)
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Choose rf, gb, or mlp.")

    return Pipeline([("scaler", StandardScaler()), ("regressor", regressor)])


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    """Compute regression metrics on test set."""
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = r2_score(y_test, y_pred)
    mape = float(np.mean(np.abs((y_test - y_pred) / (y_test + 1e-9))) * 100)
    return {"MAE_g": mae, "RMSE_g": rmse, "R2": r2, "MAPE_%": mape}


def train_classical(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str,
    output_dir: str,
    test_size: float = 0.2,
    cv_folds: int = 5,
) -> dict:
    """
    Train a classical (sklearn) regression pipeline.

    Parameters
    ----------
    X, y : feature matrix and targets
    model_type : 'rf', 'gb', or 'mlp'
    output_dir : directory to save the model and scaler
    test_size : fraction held out for final evaluation
    cv_folds : number of cross-validation folds

    Returns
    -------
    metrics : dict
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    model = build_classical_pipeline(model_type)

    # Cross-validation on training set
    print(f"[train] Cross-validating {model_type.upper()} ({cv_folds} folds) …")
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds,
                                 scoring="neg_mean_absolute_error")
    print(f"  CV MAE: {-cv_scores.mean():.2f} ± {cv_scores.std():.2f} g")

    # Final training
    model.fit(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    print(f"  Test MAE={metrics['MAE_g']:.2f} g  RMSE={metrics['RMSE_g']:.2f} g  "
          f"R²={metrics['R2']:.4f}  MAPE={metrics['MAPE_%']:.2f}%")

    # Save model
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model_path = str(out / f"model_{model_type}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"[train] Model saved → {model_path}")

    metrics["model_path"] = model_path
    metrics["model_type"] = model_type

    report_path = str(out / "training_report.json")
    with open(report_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[train] Report saved → {report_path}")

    return metrics


# ---------------------------------------------------------------------------
# Optional CNN training (PyTorch)
# ---------------------------------------------------------------------------

def train_cnn(
    features_csv: str,
    processed_dir: str,
    output_dir: str,
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 8,
) -> dict:
    """
    Train a lightweight CNN regressor on depth maps.

    Requires PyTorch. Each sample's depth_cropped.png is resized to 224×224
    and fed through a small ResNet-like network.

    Parameters
    ----------
    features_csv : str  (used for weight labels)
    processed_dir : str  (root with sample_XXXX directories)
    output_dir : str
    epochs : int
    lr : float  learning rate
    batch_size : int

    Returns
    -------
    metrics : dict
    """
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, Dataset
        import torchvision.transforms as T
        import cv2
    except ImportError:
        raise ImportError("PyTorch and torchvision are required for CNN training. "
                          "Install with: pip install torch torchvision")

    # --- Dataset ---
    class DepthDataset(Dataset):
        def __init__(self, records, transform=None):
            self.records = records
            self.transform = transform

        def __len__(self):
            return len(self.records)

        def __getitem__(self, idx):
            rec = self.records[idx]
            depth = cv2.imread(rec["depth_path"], cv2.IMREAD_UNCHANGED)
            if depth is None:
                depth = np.zeros((224, 224), dtype=np.uint16)
            depth = cv2.resize(depth, (224, 224)).astype(np.float32)
            # Normalise to [0, 1]
            depth = depth / 2000.0
            tensor = torch.from_numpy(depth).unsqueeze(0)  # (1, H, W)
            weight = torch.tensor(rec["weight_g"], dtype=torch.float32)
            return tensor, weight

    # --- Model ---
    class DepthCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
                nn.MaxPool2d(2),   # 112

                nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                nn.MaxPool2d(2),   # 56

                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                nn.MaxPool2d(2),   # 28

                nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 4)),  # 4×4
            )
            self.regressor = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128 * 16, 256), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(256, 64), nn.ReLU(),
                nn.Linear(64, 1),
            )

        def forward(self, x):
            return self.regressor(self.features(x)).squeeze(-1)

    # Build record list
    df = pd.read_csv(features_csv).dropna(subset=["weight_g"])
    proc_root = Path(processed_dir)
    records = []
    for _, row in df.iterrows():
        sid = str(int(row["sample_id"])).zfill(4)
        dp = proc_root / f"sample_{sid}" / "depth_cropped.png"
        if dp.exists():
            records.append({"depth_path": str(dp), "weight_g": float(row["weight_g"])})

    if len(records) < 4:
        raise ValueError(f"Only {len(records)} depth images found — need at least 4 for CNN training.")

    split = int(0.8 * len(records))
    train_ds = DepthDataset(records[:split])
    val_ds = DepthDataset(records[split:])
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] CNN training on {device}, {len(train_ds)} train / {len(val_ds)} val samples.")

    model = DepthCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    best_val_mae = float("inf")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model_path = str(out / "model_cnn.pt")

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_losses = []
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        scheduler.step()

        # Validate
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for x, y in val_dl:
                x = x.to(device)
                pred = model(x).cpu().numpy()
                val_preds.extend(pred.tolist())
                val_targets.extend(y.numpy().tolist())

        val_mae = mean_absolute_error(val_targets, val_preds)
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs}  train_loss={np.mean(train_losses):.2f}  val_MAE={val_mae:.2f} g")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), model_path)

    metrics = {"MAE_g": best_val_mae, "model_path": model_path, "model_type": "cnn"}
    report_path = str(out / "training_report.json")
    with open(report_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[train] Best CNN val MAE: {best_val_mae:.2f} g  →  {model_path}")
    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train a poultry weight estimation model.")
    parser.add_argument("--features", default="data/features/features.csv",
                        help="Features CSV produced by features.py.")
    parser.add_argument("--processed", default="data/processed",
                        help="Processed samples directory (only needed for CNN).")
    parser.add_argument("--model-type", default="rf",
                        choices=["rf", "gb", "mlp", "cnn"],
                        help="Model type to train.")
    parser.add_argument("--output", default="models",
                        help="Directory to save trained model and report.")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Fraction of data held out for final test.")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Training epochs (CNN only).")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate (CNN only).")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size (CNN only).")
    args = parser.parse_args()

    if args.model_type == "cnn":
        train_cnn(
            features_csv=args.features,
            processed_dir=args.processed,
            output_dir=args.output,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
        )
    else:
        X, y, feat_names = load_data(args.features)
        # Save feature names alongside model for inference
        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        with open(str(out / "feature_names.json"), "w") as f:
            json.dump(feat_names, f)
        train_classical(X, y, args.model_type, args.output, args.test_size)


if __name__ == "__main__":
    main()
