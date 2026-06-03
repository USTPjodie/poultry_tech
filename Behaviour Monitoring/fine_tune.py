"""
fine_tune.py – Incrementally fine-tune the detector and/or classifier on
new farm data without retraining from scratch.

Two modes:
  --mode detector  : fine-tune the TFLite/YOLOv8 detector on new bounding boxes
  --mode classifier: fine-tune the behaviour classifier on new crop images

Workflow:
  1. Collect new images from the new environment (gather_data.py).
  2. Annotate bounding boxes (annotate.py).
  3. Run fine_tune.py --mode detector  (or --mode classifier).
  4. Evaluate with --eval flag.
  5. Copy updated .tflite files to the Pi.

Usage:
    python fine_tune.py --mode classifier \
        --crops     data/new_crops  \
        --base-model models/classifier.tflite \
        --output    models/

    python fine_tune.py --mode detector \
        --data data/split/data.yaml \
        --base-model models/chicken_detector/weights/best.pt \
        --output models/
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np

import config
from logger import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fine-tune classifier
# ---------------------------------------------------------------------------

def fine_tune_classifier(
    crops_dir:   str,
    base_model:  str,
    output_dir:  str,
    epochs:      int  = config.FINETUNE_EPOCHS,
    lr:          float = config.FINETUNE_LEARNING_RATE,
    batch_size:  int  = config.TRAIN_BATCH_SIZE,
    add_classes: list[str] | None = None,
):
    """Continue training the Keras classifier on new data.

    Loads the saved Keras ``.h5`` checkpoint (not TFLite), fine-tunes for
    a few epochs, then re-exports TFLite.

    Parameters
    ----------
    crops_dir : str
        Root folder of (possibly expanded) per-class crop images.
    base_model : str
        Path to saved Keras model (``classifier_best.h5``).
    output_dir : str
        Folder where updated TFLite models are saved.
    epochs : int
        Number of fine-tuning epochs.
    lr : float
        Learning rate (should be lower than initial training).
    add_classes : list[str] | None
        If provided, the classifier output head is resized to accommodate
        these new behaviour classes.
    """
    try:
        import tensorflow as tf  # type: ignore
    except ImportError:
        logger.error("TensorFlow not installed. Run: pip install tensorflow")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    # Determine input size from config
    img_h, img_w = config.CLASSIFIER_INPUT_SIZE
    img_size = (img_w, img_h)

    # Load datasets
    from train_classifier import build_datasets
    train_ds, val_ds, class_names = build_datasets(
        crops_dir=crops_dir,
        img_size=img_size,
        batch_size=batch_size,
    )
    num_classes = len(class_names)
    logger.info("Fine-tuning with %d classes: %s", num_classes, class_names)

    # Load base Keras model
    if not os.path.exists(base_model):
        logger.error("Base model not found: %s", base_model)
        sys.exit(1)

    model = tf.keras.models.load_model(base_model)
    logger.info("Loaded base model: %s", base_model)

    # If adding new classes, replace the final Dense layer
    if add_classes:
        old_num = model.layers[-1].units
        new_num = old_num + len(add_classes)
        logger.info(
            "Expanding output head from %d → %d classes", old_num, new_num
        )
        base_input  = model.input
        x           = model.layers[-2].output   # Pre-dense layer
        new_output  = tf.keras.layers.Dense(new_num, activation="softmax")(x)
        model       = tf.keras.Model(base_input, new_output)

        # Update labels file
        all_classes = class_names  # build_datasets uses the folder names
        labels_path = os.path.join(output_dir, "classifier_labels.txt")
        with open(labels_path, "w") as f:
            f.write("\n".join(all_classes))

    # Unfreeze last 30 layers for fine-tuning
    for layer in model.layers[-30:]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=5, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(output_dir, "classifier_finetuned.h5"),
            save_best_only=True, monitor="val_accuracy",
        ),
    ]

    logger.info("Fine-tuning classifier for %d epochs …", epochs)
    model.fit(
        train_ds, validation_data=val_ds,
        epochs=epochs, callbacks=callbacks,
    )

    # Evaluate
    val_loss, val_acc = model.evaluate(val_ds)
    logger.info("Fine-tuned val accuracy: %.3f  val loss: %.3f", val_acc, val_loss)

    # Export TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    tflite_path  = os.path.join(output_dir, "classifier.tflite")
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    logger.info("Updated TFLite classifier → %s", tflite_path)


# ---------------------------------------------------------------------------
# Fine-tune detector (YOLOv8)
# ---------------------------------------------------------------------------

def fine_tune_detector(
    data_yaml:   str,
    base_model:  str,
    output_dir:  str,
    epochs:      int  = config.FINETUNE_EPOCHS,
    lr:          float = config.FINETUNE_LEARNING_RATE,
    batch_size:  int  = config.TRAIN_BATCH_SIZE,
    freeze:      int  = 10,
):
    """Fine-tune a YOLOv8 detector checkpoint on new data.

    Parameters
    ----------
    data_yaml : str
        YOLOv8 data YAML path (from annotate.py --split).
    base_model : str
        Path to a previously trained ``best.pt``.
    output_dir : str
        Folder where new checkpoints and exports are saved.
    epochs : int
        Number of fine-tuning epochs.
    lr : float
        Initial learning rate.
    freeze : int
        Number of early backbone layers to freeze.
    """
    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError:
        logger.error("ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    logger.info("Fine-tuning YOLOv8 from %s …", base_model)
    model = YOLO(base_model)

    model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        lr0=lr,
        lrf=0.01,
        freeze=freeze,
        project=output_dir,
        name="chicken_detector_finetuned",
        patience=5,
        save=True,
        val=True,
    )

    # Export updated TFLite
    best_pt = os.path.join(
        output_dir, "chicken_detector_finetuned", "weights", "best.pt"
    )
    if os.path.exists(best_pt):
        export_model = YOLO(best_pt)
        export_model.export(format="tflite", int8=True)
        logger.info("Fine-tuned detector exported from %s", best_pt)
    else:
        logger.warning("best.pt not found after fine-tuning.")


# ---------------------------------------------------------------------------
# Before/after evaluation comparison
# ---------------------------------------------------------------------------

def compare_before_after(
    old_model:   str,
    new_model:   str,
    crops_dir:   str,
):
    """Print a side-by-side accuracy comparison for a classifier upgrade.

    Parameters
    ----------
    old_model : str
        Path to the original ``classifier.tflite``.
    new_model : str
        Path to the fine-tuned ``classifier.tflite``.
    crops_dir : str
        Crops directory to evaluate on.
    """
    from train_classifier import evaluate_tflite_classifier
    print("\n=== Before fine-tuning ===")
    evaluate_tflite_classifier(old_model, crops_dir)
    print("\n=== After fine-tuning ===")
    evaluate_tflite_classifier(new_model, crops_dir)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fine-tune chicken models")
    parser.add_argument("--mode", choices=["classifier", "detector"],
                        required=True,
                        help="Which model to fine-tune")
    # Classifier args
    parser.add_argument("--crops",      default=config.CROPS_DIR)
    # Detector args
    parser.add_argument("--data",       default=os.path.join(config.DATA_DIR, "split/data.yaml"))
    # Common
    parser.add_argument("--base-model", required=True,
                        help="Path to existing model checkpoint (.h5 or .pt)")
    parser.add_argument("--output",     default=config.MODELS_DIR)
    parser.add_argument("--epochs",     type=int, default=config.FINETUNE_EPOCHS)
    parser.add_argument("--lr",         type=float, default=config.FINETUNE_LEARNING_RATE)
    parser.add_argument("--batch",      type=int, default=config.TRAIN_BATCH_SIZE)
    parser.add_argument("--freeze",     type=int, default=10,
                        help="Layers to freeze during detector fine-tuning")
    parser.add_argument("--eval", action="store_true",
                        help="Compare before/after performance")
    parser.add_argument("--new-model",
                        help="Path to new model for --eval comparison")
    args = parser.parse_args()

    if args.eval and args.mode == "classifier":
        compare_before_after(
            old_model=args.base_model,
            new_model=args.new_model or os.path.join(args.output, "classifier.tflite"),
            crops_dir=args.crops,
        )
        return

    if args.mode == "classifier":
        fine_tune_classifier(
            crops_dir=args.crops,
            base_model=args.base_model,
            output_dir=args.output,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch,
        )
    else:
        fine_tune_detector(
            data_yaml=args.data,
            base_model=args.base_model,
            output_dir=args.output,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch,
            freeze=args.freeze,
        )


if __name__ == "__main__":
    main()
