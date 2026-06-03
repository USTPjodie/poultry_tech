"""
train_detector.py – Train a lightweight chicken object detector.

Supported backends:
  A) TensorFlow Lite (EfficientDet-Lite0 via TensorFlow Model Maker)
     → Best for Raspberry Pi deployment
  B) YOLOv8n via Ultralytics
     → Train on PC/Colab, export to ONNX / TFLite, copy to Pi

Run on PC or Google Colab (GPU recommended for backend B).

Usage:
    # TFLite (backend A):
    python train_detector.py --backend tflite \
        --train  data/split/train/images \
        --val    data/split/val/images   \
        --output models/

    # YOLOv8 (backend B):
    python train_detector.py --backend yolov8 \
        --data   data/split/data.yaml    \
        --output models/
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import config
from logger import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Backend A: TensorFlow Model Maker → EfficientDet-Lite0
# ---------------------------------------------------------------------------

def train_tflite(
    train_images: str,
    val_images:   str,
    output_dir:   str,
    epochs:       int = config.TRAIN_EPOCHS_DETECTOR,
    batch_size:   int = config.TRAIN_BATCH_SIZE,
):
    """Train EfficientDet-Lite0 using TF Model Maker and export TFLite.

    Requirements::
        pip install tflite-model-maker

    Parameters
    ----------
    train_images : str
        Path to training images folder (YOLO-format labels expected in
        a sibling ``labels/`` folder).
    val_images : str
        Path to validation images folder.
    output_dir : str
        Folder where ``detector.tflite`` and ``labels.txt`` are saved.
    """
    try:
        from tflite_model_maker import model_spec, object_detector  # type: ignore
        from tflite_model_maker.object_detector import DataLoader    # type: ignore
    except ImportError:
        logger.error(
            "tflite-model-maker not installed.\n"
            "Install with: pip install tflite-model-maker"
        )
        sys.exit(1)

    logger.info("Loading training data from %s …", train_images)
    train_data = DataLoader.from_pascal_voc(
        train_images,
        os.path.join(os.path.dirname(train_images), "labels"),
        label_map={1: "chicken"},
    )
    val_data = DataLoader.from_pascal_voc(
        val_images,
        os.path.join(os.path.dirname(val_images), "labels"),
        label_map={1: "chicken"},
    )

    spec = model_spec.get("efficientdet_lite0")
    model = object_detector.create(
        train_data,
        model_spec=spec,
        batch_size=batch_size,
        train_whole_model=True,
        epochs=epochs,
        validation_data=val_data,
    )

    os.makedirs(output_dir, exist_ok=True)

    # Evaluate
    logger.info("Evaluating …")
    eval_result = model.evaluate(val_data)
    logger.info("Evaluation result: %s", eval_result)

    # Export TFLite
    out_path = os.path.join(output_dir, "detector.tflite")
    model.export(export_dir=output_dir, tflite_filename="detector.tflite")
    logger.info("Exported TFLite detector → %s", out_path)

    # Save labels
    labels_path = os.path.join(output_dir, "detector_labels.txt")
    with open(labels_path, "w") as f:
        f.write("chicken\n")
    logger.info("Labels saved → %s", labels_path)


# ---------------------------------------------------------------------------
# Backend B: YOLOv8 via Ultralytics
# ---------------------------------------------------------------------------

def train_yolov8(
    data_yaml:    str,
    output_dir:   str,
    model_name:   str = "yolov8n.pt",
    epochs:       int = config.TRAIN_EPOCHS_DETECTOR,
    batch_size:   int = config.TRAIN_BATCH_SIZE,
    imgsz:        int = 640,
):
    """Train YOLOv8n on chicken detections and export TFLite + ONNX.

    Requirements::
        pip install ultralytics

    Parameters
    ----------
    data_yaml : str
        Path to the YOLOv8 ``data.yaml`` file (created by annotate.py --split).
    output_dir : str
        Folder where exported models are saved.
    model_name : str
        Starting checkpoint (``yolov8n.pt`` = nano, lightest).
    """
    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError:
        logger.error("ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    logger.info("Training YOLOv8 model: %s", model_name)
    model = YOLO(model_name)

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        project=output_dir,
        name="chicken_detector",
        patience=15,
        save=True,
        val=True,
    )
    logger.info("Training complete. Metrics: %s", results)

    best_pt = os.path.join(output_dir, "chicken_detector", "weights", "best.pt")
    if not os.path.exists(best_pt):
        logger.error("Best weights not found at %s", best_pt)
        return

    # Export to TFLite (int8-quantised for Pi)
    logger.info("Exporting to TFLite …")
    export_model = YOLO(best_pt)
    export_model.export(format="tflite", int8=True, imgsz=imgsz)

    # Export to ONNX (alternative)
    logger.info("Exporting to ONNX …")
    export_model.export(format="onnx", imgsz=imgsz)

    logger.info(
        "Exports saved alongside best.pt in: %s",
        os.path.join(output_dir, "chicken_detector", "weights"),
    )
    logger.info(
        "Copy detector.tflite (or best.pt_saved_model/...) to models/detector.tflite"
    )


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_tflite_detector(
    model_path: str,
    test_images_dir: str,
    annotations_dir: str,
    iou_threshold: float = 0.5,
):
    """Compute approximate mAP@0.5 for a TFLite detector on a test set.

    Uses a simple IoU matching between ground-truth YOLO boxes and
    model predictions.
    """
    import cv2
    import numpy as np
    from model_utils import TFLiteDetector
    from pathlib import Path

    detector = TFLiteDetector(
        model_path=model_path,
        input_size=config.DETECTOR_INPUT_SIZE,
        score_threshold=config.DETECTION_THRESHOLD,
        chicken_class_id=config.CHICKEN_CLASS_ID,
    )

    img_paths = list(Path(test_images_dir).glob("*.jpg"))
    tp = fp = fn = 0

    for img_path in img_paths:
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        h, w = frame.shape[:2]

        # Load GT boxes
        txt_path = Path(annotations_dir) / (img_path.stem + ".txt")
        gt_boxes = []
        if txt_path.exists():
            with open(txt_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        _, cx, cy, bw, bh = map(float, parts)
                        x1 = int((cx - bw / 2) * w)
                        y1 = int((cy - bh / 2) * h)
                        x2 = int((cx + bw / 2) * w)
                        y2 = int((cy + bh / 2) * h)
                        gt_boxes.append((x1, y1, x2, y2))

        detections = detector.detect(frame)
        matched_gt = set()

        for det in detections:
            best_iou = 0.0
            best_j   = -1
            for j, (gx1, gy1, gx2, gy2) in enumerate(gt_boxes):
                if j in matched_gt:
                    continue
                # Compute IoU
                ix1 = max(det.x1, gx1); iy1 = max(det.y1, gy1)
                ix2 = min(det.x2, gx2); iy2 = min(det.y2, gy2)
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                area_d = det.area
                area_g = (gx2 - gx1) * (gy2 - gy1)
                union  = area_d + area_g - inter
                iou    = inter / union if union > 0 else 0
                if iou > best_iou:
                    best_iou = iou
                    best_j   = j
            if best_iou >= iou_threshold:
                tp += 1
                matched_gt.add(best_j)
            else:
                fp += 1

        fn += len(gt_boxes) - len(matched_gt)

    precision = tp / (tp + fp + 1e-9)
    recall    = tp / (tp + fn + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)
    logger.info(
        "Detection eval — TP:%d  FP:%d  FN:%d  "
        "Precision:%.3f  Recall:%.3f  F1:%.3f",
        tp, fp, fn, precision, recall, f1,
    )
    return {"precision": precision, "recall": recall, "f1": f1}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train chicken object detector")
    parser.add_argument("--backend", choices=["tflite", "yolov8"],
                        default="yolov8",
                        help="Training backend (default: yolov8)")
    # TFLite backend args
    parser.add_argument("--train",  default=os.path.join(config.DATA_DIR, "split/train/images"))
    parser.add_argument("--val",    default=os.path.join(config.DATA_DIR, "split/val/images"))
    # YOLOv8 backend args
    parser.add_argument("--data",   default=os.path.join(config.DATA_DIR, "split/data.yaml"))
    # Common
    parser.add_argument("--output", default=config.MODELS_DIR)
    parser.add_argument("--epochs", type=int, default=config.TRAIN_EPOCHS_DETECTOR)
    parser.add_argument("--batch",  type=int, default=config.TRAIN_BATCH_SIZE)
    parser.add_argument("--eval",   action="store_true",
                        help="Evaluate existing TFLite model after training")
    args = parser.parse_args()

    if args.backend == "tflite":
        train_tflite(
            train_images=args.train,
            val_images=args.val,
            output_dir=args.output,
            epochs=args.epochs,
            batch_size=args.batch,
        )
    else:
        train_yolov8(
            data_yaml=args.data,
            output_dir=args.output,
            epochs=args.epochs,
            batch_size=args.batch,
        )

    if args.eval:
        evaluate_tflite_detector(
            model_path=os.path.join(args.output, "detector.tflite"),
            test_images_dir=os.path.join(config.DATA_DIR, "split/test/images"),
            annotations_dir=os.path.join(config.DATA_DIR, "split/test/labels"),
        )


if __name__ == "__main__":
    main()
