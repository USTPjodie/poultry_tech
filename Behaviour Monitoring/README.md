# 🐔 Chicken Behaviour Monitoring System

Real-time poultry behaviour monitoring using a Raspberry Pi 4 and a single CCTV/USB/Pi camera. Detects individual chickens, tracks their movements across frames, and classifies key behaviours (feeding, drinking, walking, resting, aggression) with alert notifications.

---

## System Architecture

```
Camera → Detector (TFLite) → Classifier (TFLite) → Tracker → Analyser → Alerts / Dashboard
```

| File | Role |
|------|------|
| `config.py` | Central configuration (paths, thresholds, camera) |
| `camera_utils.py` | OpenCV / PiCamera2 abstraction |
| `model_utils.py` | TFLite inference helpers |
| `logger.py` | CSV logging, email, MQTT alerts |
| `gather_data.py` | Frame/clip capture with behaviour labels |
| `annotate.py` | Bounding-box annotation (OpenCV GUI) |
| `train_detector.py` | Train EfficientDet-Lite0 or YOLOv8n |
| `train_classifier.py` | Train MobileNetV3-Small classifier |
| `tracker.py` | Centroid / SORT multi-object tracker |
| `analyser.py` | Behaviour analysis & alert generation |
| `monitor.py` | **Main real-time script** (run this on the Pi) |
| `fine_tune.py` | Adapt models to new farms |

---

## Hardware Requirements

| Component | Specification |
|-----------|---------------|
| Raspberry Pi | 4B – 4 GB or 8 GB RAM |
| Camera | USB webcam **or** Pi Camera Module v2/v3 |
| Storage | 32 GB+ microSD (Class 10) or USB SSD |
| OS | Raspberry Pi OS 64-bit (Bookworm) |
| Optional | Active cooling (fan/heatsink) |

---

## 1. Set Up the Raspberry Pi

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install system dependencies
sudo apt install -y python3-pip python3-venv libatlas-base-dev \
    libjpeg-dev libopenjp2-7 libhdf5-dev git \
    python3-picamera2          # Pi Camera Module support

# Clone or copy project to Pi
git clone https://github.com/your-repo/chicken-monitor.git
cd chicken-monitor

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python packages (uses piwheels for Pi-optimised wheels)
pip install --upgrade pip
pip install -r requirements.txt \
    --extra-index-url https://www.piwheels.org/simple
```

---

## 2. Configure the System

Edit **`config.py`** before running anything else:

```python
# Camera – 0 for first USB webcam, or "picamera2" for Pi Camera Module
CAMERA_SOURCE = 0

# Model paths (models trained on PC are copied here)
DETECTOR_MODEL_PATH   = "models/detector.tflite"
CLASSIFIER_MODEL_PATH = "models/classifier.tflite"

# Alert thresholds
INACTIVITY_THRESHOLD_MIN = 20   # minutes
AGGRESSION_RATE_PER_MIN  = 5

# Optional email alerts
EMAIL_ALERTS_ENABLED = True
EMAIL_SENDER         = "your_email@gmail.com"
EMAIL_PASSWORD       = "your_app_password"
EMAIL_RECIPIENTS     = ["farm_manager@example.com"]
```

---

## 3. Collect Training Data

```bash
# Interactive frame labelling (press behaviour key then S to save)
python gather_data.py --mode frame --fps 5

# Record 30-second clips and label each
python gather_data.py --mode clip --clip-duration 30

# Headless auto-capture for 1 hour (label later with annotate.py)
python gather_data.py --mode auto --fps 2 --duration 3600
```

Keyboard shortcuts during frame mode:

| Key | Behaviour |
|-----|-----------|
| F | feeding |
| D | drinking |
| W | walking |
| R | resting |
| A | aggression |
| O | other |
| S | save frame |
| Q | quit |

---

## 4. Annotate Bounding Boxes

```bash
# Draw bounding boxes interactively (SPACE to save & advance)
python annotate.py --images data/raw_frames --output data/annotations

# Also extract per-class crop images for classifier training
python annotate.py --images data/raw_frames --extract-crops

# Split into train / val / test sets
python annotate.py --split
```

---

## 5. Train Models

> **Run training on a PC or Google Colab with a GPU.** Copy the resulting `.tflite` files to the Pi's `models/` folder.

### 5a. Object Detector (YOLOv8n)

```bash
# On PC
python train_detector.py --backend yolov8 \
    --data data/split/data.yaml \
    --output models/ --epochs 50

# Copy to Pi
scp models/chicken_detector/weights/best.pt_saved_model/best_integer_quant.tflite \
    pi@<pi-ip>:~/chicken-monitor/models/detector.tflite
```

### 5b. Behaviour Classifier (MobileNetV3-Small)

```bash
# On PC
python train_classifier.py --crops data/crops --output models/ --epochs-unfrozen 30

# Copy to Pi
scp models/classifier.tflite pi@<pi-ip>:~/chicken-monitor/models/classifier.tflite
scp models/classifier_best.h5 pi@<pi-ip>:~/chicken-monitor/models/
```

### 5c. Evaluate Models

```bash
# Detector evaluation
python train_detector.py --eval --output models/

# Classifier evaluation
python train_classifier.py --eval --crops data/crops --output models/
```

---

## 6. Run Real-Time Monitoring

```bash
# With display window
python monitor.py

# Headless (no screen) – logs + alerts only
python monitor.py --headless

# With Flask web dashboard at http://<pi-ip>:5000
python monitor.py --dashboard

# Headless + dashboard
python monitor.py --headless --dashboard

# Use Pi Camera Module instead of USB webcam
python monitor.py --source picamera2
```

### Auto-start on boot (systemd)

```bash
sudo nano /etc/systemd/system/chicken-monitor.service
```

```ini
[Unit]
Description=Chicken Behaviour Monitor
After=network.target

[Service]
User=pi
WorkingDirectory=/home/pi/chicken-monitor
ExecStart=/home/pi/chicken-monitor/venv/bin/python monitor.py --headless --dashboard
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable chicken-monitor
sudo systemctl start chicken-monitor
```

---

## 7. Fine-Tune on New Farm Data

```bash
# 1. Collect a small dataset from the new environment
python gather_data.py --mode auto --fps 1 --duration 1800

# 2. Annotate bounding boxes & crops
python annotate.py --extract-crops --split

# 3. Fine-tune classifier (only a few epochs needed)
python fine_tune.py --mode classifier \
    --crops data/new_crops \
    --base-model models/classifier_best.h5 \
    --output models/ --epochs 10

# 4. Compare before / after
python fine_tune.py --mode classifier --eval \
    --base-model models/classifier.tflite \
    --new-model  models/classifier.tflite \
    --crops data/new_crops

# 5. Copy updated tflite to Pi
```

---

## 8. Behaviour Logs & Alerts

Logs are written to `logs/`:

```
logs/
  behaviour_log.csv   # One row per chicken per minute
  alerts.log          # Plain-text alert log with timestamps
```

CSV columns: `timestamp, track_id, behaviour, confidence, cx, cy, elapsed_sec`

### MQTT Integration

Set in `config.py`:

```python
MQTT_ALERTS_ENABLED = True
MQTT_BROKER         = "192.168.1.100"
MQTT_TOPIC_ALERTS   = "chicken_monitor/alerts"
MQTT_TOPIC_STATS    = "chicken_monitor/stats"
```

---

## Performance Tuning on Raspberry Pi 4

| Tip | Expected Gain |
|-----|--------------|
| Use int8-quantised TFLite models | ~2× speed |
| Reduce `CAMERA_WIDTH/HEIGHT` to 320×240 | +5–8 FPS |
| Increase `DETECTION_THRESHOLD` to 0.6 | Fewer NMS calls |
| Use `TRACKER_TYPE = "centroid"` | Lighter than SORT |
| Add active cooling | Prevents thermal throttling |
| Use `opencv-python-headless` on Pi | Smaller footprint |

Typical performance on Pi 4 (4 GB, int8 models, 640×480):

- **~12–18 FPS** with 2–4 chickens
- **~8–12 FPS** with 5–8 chickens

---

## Project Structure

```
chicken-monitor/
├── config.py
├── camera_utils.py
├── model_utils.py
├── logger.py
├── gather_data.py
├── annotate.py
├── train_detector.py
├── train_classifier.py
├── tracker.py
├── analyser.py
├── monitor.py
├── fine_tune.py
├── requirements.txt
├── README.md
├── data/
│   ├── raw_frames/
│   ├── clips/
│   ├── annotations/
│   ├── crops/
│   │   ├── feeding/
│   │   ├── drinking/
│   │   ├── walking/
│   │   ├── resting/
│   │   ├── aggression/
│   │   └── other/
│   └── split/
│       ├── train/  images/  labels/
│       ├── val/    images/  labels/
│       └── test/   images/  labels/
├── models/
│   ├── detector.tflite
│   ├── classifier.tflite
│   └── classifier_best.h5
└── logs/
    ├── behaviour_log.csv
    └── alerts.log
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Cannot open camera source: 0` | Check USB connection; try `--source 1` |
| `tflite_runtime not found` | `pip install tflite-runtime --extra-index-url https://www.piwheels.org/simple` |
| `picamera2 not found` | `sudo apt install python3-picamera2` |
| FPS < 5 | Reduce resolution in config; use int8 models |
| All chickens labelled "other" | Lower `CLASSIFIER_THRESHOLD` or retrain |
| SORT tracker needs filterpy | `pip install filterpy` |

---

## Google Colab Training Notebook

For GPU-accelerated training, create a Colab notebook with:

```python
!pip install ultralytics tflite-model-maker
!git clone https://github.com/your-repo/chicken-monitor && cd chicken-monitor
# Upload your dataset, then:
!python train_detector.py --backend yolov8 --data data/split/data.yaml
!python train_classifier.py --crops data/crops
# Download models/
from google.colab import files
files.download('models/classifier.tflite')
```

---

## Licence

MIT – see `LICENSE` file.
