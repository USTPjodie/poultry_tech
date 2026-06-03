# Poultry Chicken Weight Estimation via Kinect 3D Camera

A complete Python pipeline for non-contact weight estimation of poultry chickens using an Azure Kinect (or Kinect v2) depth camera.

---

## Pipeline Overview

```
Kinect camera
     │
     ▼
gather_data.py  ──►  data/raw/
                         ├── sample_0000/  (depth.png, color.jpg, meta.json)
                         ├── sample_0001/
                         ├── …
                         ├── weights.csv
                         └── intrinsics.json
     │
     ▼
preprocess.py  ──►  data/processed/
                         ├── sample_0000/  (depth_cropped.png, mask.png, pointcloud.ply)
                         └── basic_features.csv
     │
     ▼
features.py    ──►  data/features/features.csv
                        (hull volume, voxel volume, shape descriptors, …)
     │
     ▼
train_model.py ──►  models/
                        ├── model_rf.pkl
                        ├── feature_names.json
                        └── training_report.json
     │
     ▼ (optional)
fine_tune.py   ──►  models/
                        ├── model_rf_finetuned.pkl
                        └── finetune_report.json
     │
     ▼
estimate_weight.py  (live Kinect stream → on-screen weight overlay)
```

---

## Requirements

### Hardware
- **Azure Kinect DK** (primary target) — uses `pyk4a` Python binding.
- **Kinect v2** — use `pykinect2` (Windows only); replace camera calls in `gather_data.py`.
- If no Kinect is available, the scripts run in **DEMO mode** with synthetic frames.

### Software

```bash
# 1. Install the Azure Kinect SDK (OS-level):
#    https://github.com/microsoft/Azure-Kinect-Sensor-SDK/blob/develop/docs/usage.md

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. (optional) Install pyk4a for live camera access
pip install pyk4a==1.5.0

# 5. (optional) Install PyTorch for CNN training
pip install torch torchvision
```

---

## Step-by-step Usage

### 1. Gather Data

```bash
# Live Kinect:
python gather_data.py --output data/raw

# From a recorded .mkv file (no hardware needed):
python gather_data.py --output data/raw --playback my_recording.mkv
```

**Controls:**  
- `s` — save the current frame; enter the chicken's true weight (grams) when prompted  
- `q` — quit

Each saved sample creates `data/raw/sample_XXXX/` and appends a row to `data/raw/weights.csv`.

---

### 2. Preprocess

```bash
python preprocess.py --data data/raw --output data/processed
```

This applies bilateral filtering, depth-range background removal, largest-blob segmentation, bounding-box cropping, and point-cloud generation for every sample.

---

### 3. Extract Features

```bash
python features.py \
    --data    data/processed \
    --output  data/features/features.csv \
    --weights data/raw/weights.csv
```

Adds convex-hull volume, voxel-column volume, PCA shape descriptors, and depth histograms to the feature CSV.

---

### 4. Train a Model

```bash
# Random Forest (recommended for small datasets, no GPU):
python train_model.py \
    --features   data/features/features.csv \
    --model-type rf \
    --output     models/

# Gradient Boosting:
python train_model.py --features data/features/features.csv --model-type gb

# MLP (scikit-learn):
python train_model.py --features data/features/features.csv --model-type mlp

# Lightweight CNN on depth images (requires PyTorch):
python train_model.py \
    --features   data/features/features.csv \
    --processed  data/processed \
    --model-type cnn \
    --epochs 80
```

---

### 5. Fine-tune on New Farm Data

```bash
# Collect and process new-farm samples first, then:
python fine_tune.py \
    --model             models/model_rf.pkl \
    --new-features      data/new_farm/features.csv \
    --original-features data/features/features.csv \
    --model-type        rf \
    --search            random \
    --output            models/

# CNN fine-tuning:
python fine_tune.py \
    --model        models/model_cnn.pt \
    --new-features data/new_farm/features.csv \
    --processed    data/new_farm/processed \
    --model-type   cnn \
    --epochs 20 --lr 1e-4 \
    --output       models/
```

---

### 6. Live Inference

```bash
# sklearn model (live camera):
python estimate_weight.py \
    --model      models/model_rf_finetuned.pkl \
    --model-type rf \
    --intrinsics data/raw/intrinsics.json

# CNN model (from playback):
python estimate_weight.py \
    --model      models/model_cnn_finetuned.pt \
    --model-type cnn \
    --playback   recording.mkv
```

The OpenCV window shows the RGB feed with a green foreground overlay and the estimated weight printed in the top-left corner.

---

## File Reference

| File | Purpose |
|---|---|
| `utils.py` | Shared helpers: intrinsics, depth→point cloud, PLY I/O, CSV, visualisation |
| `gather_data.py` | Kinect data acquisition + weight annotation |
| `preprocess.py` | Filtering, background removal, segmentation, point-cloud generation |
| `features.py` | Volumetric + shape feature extraction |
| `train_model.py` | RF / GB / MLP / CNN training |
| `fine_tune.py` | Hyperparameter search + domain adaptation |
| `estimate_weight.py` | Real-time inference with visual overlay |
| `requirements.txt` | Python package list |

---

## Tips

- **Small dataset (<50 samples)?** Use Random Forest (`rf`). It regularises well and is interpretable.  
- **Expected accuracy:** MAE < 10% of mean weight is achievable with ~100 labelled samples and clean depth data.  
- **Background control:** A flat, uniform-coloured surface (e.g. a weigh-platform) makes background removal much more reliable. Adjust `--min-depth` and `--max-depth` to bracket the chicken tightly.  
- **Camera height:** Mount the Kinect ~1–1.5 m above the chicken for best depth resolution.  
- **Fine-tuning frequency:** Re-run `fine_tune.py` whenever entering a new farm, season, or breed cohort.
