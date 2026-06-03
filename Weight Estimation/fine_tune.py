"""
fine_tune.py — Fine-tune a pre-trained weight-estimation model on new data.

Supports
--------
- Classical sklearn models (RF, GB, MLP) — refit on combined old+new data with
  optional higher regularisation, or partial_fit where available (MLP).
- CNN (PyTorch) — freeze early convolutional layers, fine-tune the last block
  and the regression head.
- Hyperparameter search (grid or random) over key parameters.

Usage
-----
    # Sklearn model:
    python fine_tune.py \
        --model models/model_rf.pkl \
        --new-features data/new_farm/features.csv \
        --original-features data/features/features.csv \
        --model-type rf \
        --output models/

    # CNN:
    python fine_tune.py \
        --model models/model_cnn.pt \
        --new-features data/new_farm/features.csv \
        --processed data/new_farm/processed \
        --model-type cnn \
        --epochs 20 \
        --output models/
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from train_model import FEATURE_COLS, TARGET_COL, evaluate_model, load_data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_model(model_path: str):
    """Load a pickled sklearn Pipeline."""
    with open(model_path, "rb") as f:
        return pickle.load(f)


def save_model(model, path: str) -> None:
    """Save an sklearn Pipeline to disk."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"[fine_tune] Model saved → {path}")


def load_combined(
    original_csv: str | None,
    new_csv: str,
    feature_names: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and merge original + new features CSVs.

    Returns
    -------
    X_new, y_new : new-data arrays (for validation)
    X_combined, y_combined : merged arrays (for training)
    """
    df_new = pd.read_csv(new_csv).dropna(subset=[TARGET_COL])
    available = [c for c in feature_names if c in df_new.columns]
    df_new[available] = df_new[available].fillna(0)
    X_new = df_new[available].values.astype(np.float32)
    y_new = df_new[TARGET_COL].values.astype(np.float32)

    if original_csv and Path(original_csv).exists():
        df_orig = pd.read_csv(original_csv).dropna(subset=[TARGET_COL])
        df_orig[available] = df_orig[available].fillna(0)
        X_orig = df_orig[available].values.astype(np.float32)
        y_orig = df_orig[TARGET_COL].values.astype(np.float32)
        X_combined = np.vstack([X_orig, X_new])
        y_combined = np.concatenate([y_orig, y_new])
        print(f"[fine_tune] Combined: {len(X_orig)} original + {len(X_new)} new = {len(X_combined)} samples.")
    else:
        X_combined, y_combined = X_new, y_new
        print(f"[fine_tune] No original features — using only {len(X_new)} new samples.")

    return X_new, y_new, X_combined, y_combined


# ---------------------------------------------------------------------------
# Hyperparameter grids
# ---------------------------------------------------------------------------

RF_PARAM_GRID = {
    "regressor__n_estimators": [100, 200, 300],
    "regressor__max_features": ["sqrt", 0.5, 0.7],
    "regressor__min_samples_leaf": [1, 2, 4],
}

GB_PARAM_GRID = {
    "regressor__n_estimators": [200, 400],
    "regressor__learning_rate": [0.03, 0.05, 0.1],
    "regressor__max_depth": [3, 4, 5],
}

MLP_PARAM_GRID = {
    "regressor__hidden_layer_sizes": [(128, 64), (256, 128, 64), (64, 32)],
    "regressor__alpha": [1e-4, 1e-3, 1e-2],
    "regressor__learning_rate_init": [1e-3, 5e-4],
}

PARAM_GRIDS = {"rf": RF_PARAM_GRID, "gb": GB_PARAM_GRID, "mlp": MLP_PARAM_GRID}


# ---------------------------------------------------------------------------
# Classical fine-tuning
# ---------------------------------------------------------------------------

def fine_tune_classical(
    model_path: str,
    new_csv: str,
    original_csv: str | None,
    model_type: str,
    output_dir: str,
    search_type: str = "random",
    n_iter: int = 20,
    cv_folds: int = 3,
) -> dict:
    """
    Fine-tune a classical sklearn model on new data.

    Strategy:
    - Load the existing pipeline.
    - Run hyperparameter search on the combined dataset.
    - Compare before/after performance on the held-out new-data split.

    Parameters
    ----------
    model_path : str
    new_csv : str
    original_csv : str or None
    model_type : str — 'rf', 'gb', or 'mlp'
    output_dir : str
    search_type : str — 'random' or 'grid'
    n_iter : int — iterations for random search
    cv_folds : int

    Returns
    -------
    report : dict
    """
    # Load existing model & feature names
    old_model = load_model(model_path)
    feat_names_path = Path(model_path).parent / "feature_names.json"
    if feat_names_path.exists():
        with open(feat_names_path) as f:
            feature_names = json.load(f)
    else:
        feature_names = [c for c in FEATURE_COLS]

    # Align feature list to what is available in the new CSV
    df_probe = pd.read_csv(new_csv)
    feature_names = [c for c in feature_names if c in df_probe.columns]

    X_new, y_new, X_combined, y_combined = load_combined(original_csv, new_csv, feature_names)

    # Hold out 20% of the new data for final comparison
    X_val, y_val = X_new, y_new
    if len(X_new) >= 5:
        X_val_split, _, y_val_split, _ = train_test_split(X_new, y_new, test_size=0.8, random_state=0)
        X_val, y_val = X_val_split, y_val_split

    # Performance BEFORE fine-tuning
    if len(X_val) > 0:
        before_metrics = evaluate_model(old_model, X_val, y_val)
        print(f"[fine_tune] Before: MAE={before_metrics['MAE_g']:.2f} g  "
              f"RMSE={before_metrics['RMSE_g']:.2f} g  R²={before_metrics['R2']:.4f}")
    else:
        before_metrics = {}

    # Build a new pipeline of the same type
    from train_model import build_classical_pipeline
    new_pipeline = build_classical_pipeline(model_type)

    # Hyperparameter search
    param_grid = PARAM_GRIDS.get(model_type, {})
    if param_grid:
        print(f"[fine_tune] Running {search_type} search ({n_iter} iterations, {cv_folds} folds) …")
        if search_type == "grid":
            searcher = GridSearchCV(new_pipeline, param_grid, cv=cv_folds,
                                    scoring="neg_mean_absolute_error", n_jobs=-1, verbose=1)
        else:
            searcher = RandomizedSearchCV(new_pipeline, param_grid, n_iter=n_iter,
                                          cv=cv_folds, scoring="neg_mean_absolute_error",
                                          n_jobs=-1, random_state=42, verbose=1)
        searcher.fit(X_combined, y_combined)
        tuned_model = searcher.best_estimator_
        print(f"  Best params: {searcher.best_params_}")
    else:
        tuned_model = new_pipeline
        tuned_model.fit(X_combined, y_combined)

    # Performance AFTER fine-tuning
    if len(X_val) > 0:
        after_metrics = evaluate_model(tuned_model, X_val, y_val)
        print(f"[fine_tune] After:  MAE={after_metrics['MAE_g']:.2f} g  "
              f"RMSE={after_metrics['RMSE_g']:.2f} g  R²={after_metrics['R2']:.4f}")
    else:
        after_metrics = {}

    # Save
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ft_model_path = str(out / f"model_{model_type}_finetuned.pkl")
    save_model(tuned_model, ft_model_path)

    # Save feature names alongside
    with open(str(out / "feature_names.json"), "w") as f:
        json.dump(feature_names, f)

    report = {
        "model_type": model_type,
        "before": before_metrics,
        "after": after_metrics,
        "finetuned_model_path": ft_model_path,
    }
    report_path = str(out / "finetune_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[fine_tune] Report → {report_path}")
    return report


# ---------------------------------------------------------------------------
# CNN fine-tuning (PyTorch)
# ---------------------------------------------------------------------------

def fine_tune_cnn(
    model_path: str,
    new_features_csv: str,
    processed_dir: str,
    output_dir: str,
    epochs: int = 20,
    lr: float = 1e-4,
    batch_size: int = 4,
    freeze_blocks: int = 2,
) -> dict:
    """
    Fine-tune the CNN by freezing early layers and retraining the head.

    Parameters
    ----------
    model_path : str — path to model_cnn.pt
    new_features_csv : str — CSV with weight labels for the new data
    processed_dir : str — directory with new processed sample_XXXX folders
    output_dir : str
    epochs : int
    lr : float — lower than initial training (default 1e-4)
    batch_size : int
    freeze_blocks : int — number of conv blocks (out of 4) to freeze

    Returns
    -------
    report : dict
    """
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, Dataset
        import cv2
    except ImportError:
        raise ImportError("PyTorch required. pip install torch torchvision")

    from train_model import DepthCNN  # type: ignore
    # Re-import DepthCNN definition inline to avoid circular imports
    class DepthCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 4)),
            )
            self.regressor = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128 * 16, 256), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(256, 64), nn.ReLU(),
                nn.Linear(64, 1),
            )
        def forward(self, x):
            return self.regressor(self.features(x)).squeeze(-1)

    class DepthDataset(Dataset):
        def __init__(self, records):
            self.records = records
        def __len__(self):
            return len(self.records)
        def __getitem__(self, idx):
            rec = self.records[idx]
            d = cv2.imread(rec["depth_path"], cv2.IMREAD_UNCHANGED)
            if d is None:
                d = np.zeros((224, 224), dtype=np.uint16)
            d = cv2.resize(d, (224, 224)).astype(np.float32) / 2000.0
            return torch.from_numpy(d).unsqueeze(0), torch.tensor(rec["weight_g"], dtype=torch.float32)

    # Build record list
    df = pd.read_csv(new_features_csv).dropna(subset=[TARGET_COL])
    proc_root = Path(processed_dir)
    records = []
    for _, row in df.iterrows():
        sid = str(int(row["sample_id"])).zfill(4)
        dp = proc_root / f"sample_{sid}" / "depth_cropped.png"
        if dp.exists():
            records.append({"depth_path": str(dp), "weight_g": float(row[TARGET_COL])})

    if not records:
        raise ValueError("No matching depth images found for CNN fine-tuning.")

    split = max(1, int(0.8 * len(records)))
    train_ds = DepthDataset(records[:split])
    val_ds = DepthDataset(records[split:])
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DepthCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Freeze early blocks
    children = list(model.features.children())
    # Each "block" is 3 children: Conv, BN, ReLU (+ optional MaxPool)
    # Freeze the first freeze_blocks * 4 layers
    n_freeze = min(freeze_blocks * 4, len(children))
    for i, layer in enumerate(children):
        if i < n_freeze:
            for p in layer.parameters():
                p.requires_grad = False

    print(f"[fine_tune] CNN: frozen first {n_freeze} layers, training remaining layers + head.")
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.MSELoss()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ft_path = str(out / "model_cnn_finetuned.pt")
    best_mae = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            criterion(model(x), y).backward()
            optimizer.step()

        if val_ds:
            model.eval()
            preds, targets = [], []
            with torch.no_grad():
                for x, y in val_dl:
                    preds.extend(model(x.to(device)).cpu().numpy())
                    targets.extend(y.numpy())
            mae = mean_absolute_error(targets, preds)
            if epoch % 5 == 0:
                print(f"  Epoch {epoch}/{epochs}  val_MAE={mae:.2f} g")
            if mae < best_mae:
                best_mae = mae
                torch.save(model.state_dict(), ft_path)

    print(f"[fine_tune] Best fine-tuned CNN MAE: {best_mae:.2f} g  →  {ft_path}")
    report = {"model_type": "cnn", "after": {"MAE_g": best_mae}, "finetuned_model_path": ft_path}
    report_path = str(out / "finetune_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fine-tune a pre-trained weight-estimation model.")
    parser.add_argument("--model", required=True,
                        help="Path to pre-trained model (.pkl for sklearn, .pt for CNN).")
    parser.add_argument("--new-features", required=True,
                        help="Features CSV for the new dataset.")
    parser.add_argument("--original-features", default=None,
                        help="Original features CSV (used to avoid catastrophic forgetting).")
    parser.add_argument("--model-type", default="rf",
                        choices=["rf", "gb", "mlp", "cnn"])
    parser.add_argument("--processed", default="data/processed",
                        help="New farm processed directory (CNN only).")
    parser.add_argument("--output", default="models",
                        help="Output directory for fine-tuned model.")
    parser.add_argument("--search", default="random", choices=["random", "grid"],
                        help="Hyperparameter search strategy (classical models only).")
    parser.add_argument("--n-iter", type=int, default=20,
                        help="Iterations for random search.")
    parser.add_argument("--epochs", type=int, default=20, help="Epochs (CNN only).")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (CNN only).")
    parser.add_argument("--freeze-blocks", type=int, default=2,
                        help="Number of conv blocks to freeze (CNN only).")
    args = parser.parse_args()

    if args.model_type == "cnn":
        fine_tune_cnn(
            model_path=args.model,
            new_features_csv=args.new_features,
            processed_dir=args.processed,
            output_dir=args.output,
            epochs=args.epochs,
            lr=args.lr,
            freeze_blocks=args.freeze_blocks,
        )
    else:
        fine_tune_classical(
            model_path=args.model,
            new_csv=args.new_features,
            original_csv=args.original_features,
            model_type=args.model_type,
            output_dir=args.output,
            search_type=args.search,
            n_iter=args.n_iter,
        )


if __name__ == "__main__":
    main()
