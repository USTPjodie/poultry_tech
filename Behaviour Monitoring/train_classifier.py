"""
train_classifier.py – Train a lightweight behaviour classification model
(MobileNetV3-Small via Keras) on cropped chicken images.

Directory structure expected under ``crops_dir``::
    crops/
        feeding/    img_001.jpg  img_002.jpg …
        drinking/   …
        walking/    …
        resting/    …
        aggression/ …
        other/      …

The trained model is exported to TFLite (float32 + int8-quantised) and
optionally to ONNX.

Usage (on PC / Colab):
    python train_classifier.py --crops data/crops --output models/

Usage (evaluate only):
    python train_classifier.py --eval --model models/classifier.tflite \
        --crops data/crops
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
# Data loading
# ---------------------------------------------------------------------------

def build_datasets(
    crops_dir: str,
    img_size: tuple[int, int] = config.CLASSIFIER_INPUT_SIZE,
    batch_size: int = config.TRAIN_BATCH_SIZE,
    val_split: float = config.TRAIN_VAL_SPLIT,
    seed: int = 42,
):
    """Build Keras image dataset objects from the crops directory.

    Returns
    -------
    tuple
        (train_ds, val_ds, class_names)
    """
    try:
        import tensorflow as tf  # type: ignore
    except ImportError:
        logger.error("TensorFlow not installed. Run: pip install tensorflow")
        sys.exit(1)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        crops_dir,
        validation_split=val_split,
        subset="training",
        seed=seed,
        image_size=img_size,  # (H, W)
        batch_size=batch_size,
        label_mode="categorical",
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        crops_dir,
        validation_split=val_split,
        subset="validation",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="categorical",
    )

    class_names = train_ds.class_names
    logger.info("Classes found: %s", class_names)

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds   = val_ds.prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, class_names


# ---------------------------------------------------------------------------
# Model architecture: MobileNetV3-Small with fine-tuned head
# ---------------------------------------------------------------------------

def build_model(num_classes: int, img_size: tuple[int, int]) -> "tf.keras.Model":
    """Build a transfer-learning model on top of MobileNetV3-Small.

    Parameters
    ----------
    num_classes : int
        Number of behaviour output classes.
    img_size : tuple[int, int]
        (H, W) expected by the model input.

    Returns
    -------
    tf.keras.Model
    """
    import tensorflow as tf  # type: ignore

    base = tf.keras.applications.MobileNetV3Small(
        input_shape=(*img_size, 3),
        include_top=False,
        weights="imagenet",
        pooling="avg",
        minimalistic=True,
    )
    base.trainable = False  # Freeze during initial training

    inputs  = tf.keras.Input(shape=(*img_size, 3))
    # Rescale [0,255] → [-1, 1] as expected by MobileNetV3
    x = tf.keras.layers.Rescaling(scale=1.0 / 127.5, offset=-1.0)(inputs)
    x = base(x, training=False)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="ChickenBehaviourClassifier")
    return model


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_classifier(
    crops_dir: str,
    output_dir: str,
    epochs_frozen: int = 15,
    epochs_unfrozen: int = config.TRAIN_EPOCHS_CLASSIFIER,
    batch_size: int = config.TRAIN_BATCH_SIZE,
    lr_frozen: float = config.TRAIN_LEARNING_RATE,
    lr_unfrozen: float = config.FINETUNE_LEARNING_RATE,
):
    """Two-phase transfer learning: frozen base, then partial fine-tuning.

    Phase 1 – Train only the classification head (frozen base).
    Phase 2 – Unfreeze the last 20 layers and fine-tune at lower LR.

    Parameters
    ----------
    crops_dir : str
        Root folder of per-class crop images.
    output_dir : str
        Folder where models are saved.
    """
    import tensorflow as tf  # type: ignore

    os.makedirs(output_dir, exist_ok=True)

    img_h, img_w = config.CLASSIFIER_INPUT_SIZE  # stored as (W, H) in config
    img_size = (img_w, img_h)                     # Keras expects (H, W)

    train_ds, val_ds, class_names = build_datasets(
        crops_dir=crops_dir,
        img_size=img_size,
        batch_size=batch_size,
    )
    num_classes = len(class_names)

    model = build_model(num_classes, img_size)
    model.summary()

    # ── Phase 1: frozen base ────────────────────────────────────────────
    logger.info("Phase 1: Training classification head (%d epochs) …", epochs_frozen)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr_frozen),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=5, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7
        ),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(output_dir, "classifier_best.h5"),
            save_best_only=True, monitor="val_accuracy",
        ),
    ]

    history1 = model.fit(
        train_ds, validation_data=val_ds,
        epochs=epochs_frozen, callbacks=callbacks,
    )

    # ── Phase 2: partial unfreeze ───────────────────────────────────────
    logger.info("Phase 2: Fine-tuning last 20 layers (%d epochs) …", epochs_unfrozen)
    base_model = model.layers[2]  # MobileNetV3Small (index may vary)
    for layer in base_model.layers[-20:]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr_unfrozen),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    history2 = model.fit(
        train_ds, validation_data=val_ds,
        epochs=epochs_unfrozen, callbacks=callbacks,
    )

    # ── Save labels ─────────────────────────────────────────────────────
    labels_path = os.path.join(output_dir, "classifier_labels.txt")
    with open(labels_path, "w") as f:
        f.write("\n".join(class_names))
    logger.info("Saved class names → %s", labels_path)

    # ── Export TFLite (float32) ─────────────────────────────────────────
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    tflite_path = os.path.join(output_dir, "classifier.tflite")
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    logger.info("Exported float32 TFLite → %s", tflite_path)

    # ── Export TFLite (int8 quantised) ──────────────────────────────────
    def representative_dataset():
        for imgs, _ in train_ds.take(50):
            for img in imgs:
                yield [img.numpy()[np.newaxis, ...].astype(np.float32)]

    converter_q = tf.lite.TFLiteConverter.from_keras_model(model)
    converter_q.optimizations = [tf.lite.Optimize.DEFAULT]
    converter_q.representative_dataset = representative_dataset
    converter_q.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter_q.inference_input_type  = tf.uint8
    converter_q.inference_output_type = tf.uint8

    try:
        tflite_q = converter_q.convert()
        q_path = os.path.join(output_dir, "classifier_int8.tflite")
        with open(q_path, "wb") as f:
            f.write(tflite_q)
        logger.info("Exported int8 TFLite → %s", q_path)
    except Exception as e:
        logger.warning("int8 quantisation failed: %s. Using float32 only.", e)

    return model, class_names


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_tflite_classifier(
    model_path: str,
    crops_dir: str,
):
    """Compute accuracy and per-class F1-score on the crops test set."""
    import cv2
    from sklearn.metrics import classification_report  # type: ignore
    from model_utils import TFLiteClassifier

    classifier = TFLiteClassifier(
        model_path=model_path,
        input_size=config.CLASSIFIER_INPUT_SIZE,
        class_names=config.BEHAVIOUR_CLASSES,
        threshold=0.0,  # Use all predictions for eval
    )

    y_true, y_pred = [], []
    class_dirs = sorted(Path(crops_dir).iterdir())
    class_names = [d.name for d in class_dirs if d.is_dir()]

    for class_dir in class_dirs:
        if not class_dir.is_dir():
            continue
        for img_path in class_dir.glob("*.jpg"):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            label, _ = classifier.classify(img)
            y_true.append(class_dir.name)
            y_pred.append(label)

    report = classification_report(y_true, y_pred, target_names=class_names)
    print("\n=== Classifier Evaluation ===")
    print(report)
    return report


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train behaviour classifier")
    parser.add_argument("--crops",  default=config.CROPS_DIR)
    parser.add_argument("--output", default=config.MODELS_DIR)
    parser.add_argument("--epochs-frozen",   type=int, default=15)
    parser.add_argument("--epochs-unfrozen", type=int, default=config.TRAIN_EPOCHS_CLASSIFIER)
    parser.add_argument("--batch",  type=int, default=config.TRAIN_BATCH_SIZE)
    parser.add_argument("--eval",   action="store_true",
                        help="Evaluate existing TFLite classifier (skip training)")
    args = parser.parse_args()

    if args.eval:
        evaluate_tflite_classifier(
            model_path=os.path.join(args.output, "classifier.tflite"),
            crops_dir=args.crops,
        )
    else:
        train_classifier(
            crops_dir=args.crops,
            output_dir=args.output,
            epochs_frozen=args.epochs_frozen,
            epochs_unfrozen=args.epochs_unfrozen,
            batch_size=args.batch,
        )


if __name__ == "__main__":
    main()
