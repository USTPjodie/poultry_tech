"""
config.py – Central configuration for the Chicken Behaviour Monitoring System.

Edit this file to match your hardware setup, model paths, and alert thresholds
before running any other script.
"""

import os

# ---------------------------------------------------------------------------
# Camera Settings
# ---------------------------------------------------------------------------
# 0 = first USB webcam, 1 = second USB webcam, or use "picamera2" to use the
# official Raspberry Pi camera module.
CAMERA_SOURCE = 0           # int for OpenCV index, or "picamera2"
CAMERA_WIDTH   = 640
CAMERA_HEIGHT  = 480
CAMERA_FPS     = 30         # Target capture FPS

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR          = os.path.dirname(os.path.abspath(__file__))
DATA_DIR          = os.path.join(BASE_DIR, "data")
RAW_FRAMES_DIR    = os.path.join(DATA_DIR, "raw_frames")
CLIPS_DIR         = os.path.join(DATA_DIR, "clips")
ANNOTATIONS_DIR   = os.path.join(DATA_DIR, "annotations")
CROPS_DIR         = os.path.join(DATA_DIR, "crops")          # Per-class subfolders
MODELS_DIR        = os.path.join(BASE_DIR, "models")
LOGS_DIR          = os.path.join(BASE_DIR, "logs")

DETECTOR_MODEL_PATH    = os.path.join(MODELS_DIR, "detector.tflite")
CLASSIFIER_MODEL_PATH  = os.path.join(MODELS_DIR, "classifier.tflite")

# Create directories if they don't exist
for _d in [RAW_FRAMES_DIR, CLIPS_DIR, ANNOTATIONS_DIR, CROPS_DIR,
           MODELS_DIR, LOGS_DIR]:
    os.makedirs(_d, exist_ok=True)

# ---------------------------------------------------------------------------
# Behaviour Classes
# Labels used both for annotation and for classifier output.
# ---------------------------------------------------------------------------
BEHAVIOUR_CLASSES = ["feeding", "drinking", "walking", "resting", "aggression", "other"]
BEHAVIOUR_KEYS    = {
    "f": "feeding",
    "d": "drinking",
    "w": "walking",
    "r": "resting",
    "a": "aggression",
    "o": "other",
}

# ---------------------------------------------------------------------------
# Object Detection
# ---------------------------------------------------------------------------
DETECTOR_INPUT_SIZE  = (320, 320)   # (width, height) expected by the TFLite model
DETECTION_THRESHOLD  = 0.45         # Min confidence score to keep a detection
NMS_IOU_THRESHOLD    = 0.45         # Non-maximum suppression IoU threshold
CHICKEN_CLASS_ID     = 0            # Class index for "chicken" in the detector

# ---------------------------------------------------------------------------
# Behaviour Classifier
# ---------------------------------------------------------------------------
CLASSIFIER_INPUT_SIZE   = (96, 96)  # (width, height) for the classifier crop
CLASSIFIER_THRESHOLD    = 0.50      # Min confidence to assign a behaviour label

# ---------------------------------------------------------------------------
# Tracker (Centroid / SORT)
# ---------------------------------------------------------------------------
TRACKER_TYPE            = "centroid"   # "centroid" or "sort"
MAX_DISAPPEARED         = 30           # Frames before a track is removed
MAX_TRACK_HISTORY       = 300          # Max positions stored per track (frames)
IOU_MATCH_THRESHOLD     = 0.3          # IoU threshold for SORT association

# ---------------------------------------------------------------------------
# Behaviour Analysis
# ---------------------------------------------------------------------------
BEHAVIOUR_WINDOW_FRAMES = 30           # Rolling window size for majority vote
INACTIVITY_THRESHOLD_MIN = 20          # Alert if resting > N consecutive minutes
AGGRESSION_RATE_PER_MIN  = 5           # Alert if aggression events exceed N/min
FEED_DROP_THRESHOLD      = 0.20        # Alert if feeding time drops below 20 % of normal

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_INTERVAL_SEC     = 60             # Write a behaviour summary row every N seconds
BEHAVIOUR_LOG_CSV    = os.path.join(LOGS_DIR, "behaviour_log.csv")
ALERT_LOG_FILE       = os.path.join(LOGS_DIR, "alerts.log")

# ---------------------------------------------------------------------------
# Alerts – Email (optional)
# ---------------------------------------------------------------------------
EMAIL_ALERTS_ENABLED = False
SMTP_SERVER          = "smtp.gmail.com"
SMTP_PORT            = 587
EMAIL_SENDER         = "your_email@gmail.com"
EMAIL_PASSWORD       = "your_app_password"    # Use an app-specific password
EMAIL_RECIPIENTS     = ["farm_manager@example.com"]

# ---------------------------------------------------------------------------
# Alerts – MQTT (optional)
# ---------------------------------------------------------------------------
MQTT_ALERTS_ENABLED  = False
MQTT_BROKER          = "localhost"
MQTT_PORT            = 1883
MQTT_TOPIC_ALERTS    = "chicken_monitor/alerts"
MQTT_TOPIC_STATS     = "chicken_monitor/stats"

# ---------------------------------------------------------------------------
# Web Dashboard (Flask – optional)
# ---------------------------------------------------------------------------
DASHBOARD_HOST       = "0.0.0.0"
DASHBOARD_PORT       = 5000

# ---------------------------------------------------------------------------
# Training (run on PC / Colab, then copy .tflite to Pi)
# ---------------------------------------------------------------------------
TRAIN_EPOCHS_DETECTOR    = 50
TRAIN_EPOCHS_CLASSIFIER  = 30
TRAIN_BATCH_SIZE         = 16
TRAIN_LEARNING_RATE      = 1e-4
TRAIN_VAL_SPLIT          = 0.15
TRAIN_TEST_SPLIT         = 0.10
FINETUNE_LEARNING_RATE   = 1e-5
FINETUNE_EPOCHS          = 10
