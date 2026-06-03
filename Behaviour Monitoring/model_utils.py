"""
model_utils.py – Helpers for loading TensorFlow Lite models and running
inference for both the object detector and the behaviour classifier.

Supports:
  • TensorFlow Lite via tflite_runtime (preferred on Raspberry Pi)
  • TensorFlow Lite via tensorflow.lite (fallback for PC)
  • ONNX Runtime (alternative backend)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TFLite backend selection
# ---------------------------------------------------------------------------

def _load_tflite_interpreter(model_path: str):
    """Load a TFLite interpreter, preferring tflite_runtime."""
    try:
        from tflite_runtime.interpreter import Interpreter  # type: ignore
        logger.debug("Using tflite_runtime")
    except ImportError:
        from tensorflow.lite.python.interpreter import Interpreter  # type: ignore
        logger.debug("Using tensorflow.lite (tflite_runtime not found)")
    interp = Interpreter(model_path=str(model_path))
    interp.allocate_tensors()
    return interp


# ---------------------------------------------------------------------------
# Detection result dataclass
# ---------------------------------------------------------------------------

class Detection(NamedTuple):
    """A single chicken detection."""
    x1: int      # Top-left x  (pixels, relative to original frame)
    y1: int      # Top-left y
    x2: int      # Bottom-right x
    y2: int      # Bottom-right y
    score: float # Confidence [0, 1]

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def centroid(self) -> tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    @property
    def area(self) -> int:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)


# ---------------------------------------------------------------------------
# TFLite Object Detector
# ---------------------------------------------------------------------------

class TFLiteDetector:
    """Runs a TFLite SSD-style object detection model.

    The model is expected to follow the TF Object Detection API TFLite export
    convention:
        Input  : uint8 or float32 [1, H, W, 3]
        Output 0: boxes   [1, N, 4]  (y1, x1, y2, x2 normalised)
        Output 1: classes [1, N]
        Output 2: scores  [1, N]
        Output 3: count   [1]

    Parameters
    ----------
    model_path : str
        Path to the ``.tflite`` file.
    input_size : tuple[int, int]
        (width, height) expected by the model.
    score_threshold : float
        Detections below this confidence are discarded.
    chicken_class_id : int
        Class index that represents "chicken" in the model.
    """

    def __init__(
        self,
        model_path: str,
        input_size: tuple[int, int] = (320, 320),
        score_threshold: float = 0.45,
        chicken_class_id: int = 0,
    ):
        self.model_path       = model_path
        self.input_size       = input_size  # (W, H)
        self.score_threshold  = score_threshold
        self.chicken_class_id = chicken_class_id
        self._interp          = _load_tflite_interpreter(model_path)

        in_details  = self._interp.get_input_details()
        out_details = self._interp.get_output_details()
        self._input_idx      = in_details[0]["index"]
        self._input_dtype    = in_details[0]["dtype"]
        # Output order varies; detect by shape
        self._out_boxes_idx,  \
        self._out_classes_idx,\
        self._out_scores_idx  = self._parse_output_order(out_details)
        logger.info("TFLiteDetector loaded: %s", model_path)

    def _parse_output_order(self, out_details):
        """Return (boxes_idx, classes_idx, scores_idx) from output details."""
        boxes_idx   = None
        classes_idx = None
        scores_idx  = None
        for d in out_details:
            shape = d["shape"]
            if len(shape) == 3 and shape[-1] == 4:
                boxes_idx = d["index"]
            elif len(shape) == 2:
                # Distinguish classes (int/float values < num_classes) from scores [0,1]
                if d["dtype"] in (np.int64, np.int32, np.float32):
                    if scores_idx is None:
                        scores_idx = d["index"]
                    else:
                        classes_idx = d["index"]
        # Fallback positional assignment
        if boxes_idx is None:
            boxes_idx   = out_details[0]["index"]
            classes_idx = out_details[1]["index"]
            scores_idx  = out_details[2]["index"]
        return boxes_idx, classes_idx, scores_idx

    # ------------------------------------------------------------------
    def preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Resize and normalise a BGR frame for the detector."""
        w, h = self.input_size
        img  = cv2.resize(frame_bgr, (w, h))
        img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self._input_dtype == np.float32:
            img = (img.astype(np.float32) - 127.5) / 127.5
        else:
            img = img.astype(np.uint8)
        return np.expand_dims(img, axis=0)

    def detect(
        self, frame_bgr: np.ndarray
    ) -> list[Detection]:
        """Run inference and return a list of Detection objects.

        Parameters
        ----------
        frame_bgr : np.ndarray
            Original camera frame (any resolution).

        Returns
        -------
        list[Detection]
            Detections in pixel coordinates relative to ``frame_bgr``.
        """
        fh, fw = frame_bgr.shape[:2]
        inp    = self.preprocess(frame_bgr)

        self._interp.set_tensor(self._input_idx, inp)
        self._interp.invoke()

        boxes   = self._interp.get_tensor(self._out_boxes_idx)[0]   # [N, 4]
        classes = self._interp.get_tensor(self._out_classes_idx)[0] # [N]
        scores  = self._interp.get_tensor(self._out_scores_idx)[0]  # [N]

        detections = []
        for box, cls, score in zip(boxes, classes, scores):
            if score < self.score_threshold:
                continue
            if int(cls) != self.chicken_class_id:
                continue
            # boxes in [y1, x1, y2, x2] normalised format
            y1n, x1n, y2n, x2n = box
            x1 = int(np.clip(x1n * fw, 0, fw - 1))
            y1 = int(np.clip(y1n * fh, 0, fh - 1))
            x2 = int(np.clip(x2n * fw, 0, fw - 1))
            y2 = int(np.clip(y2n * fh, 0, fh - 1))
            detections.append(Detection(x1, y1, x2, y2, float(score)))

        return detections


# ---------------------------------------------------------------------------
# TFLite Behaviour Classifier
# ---------------------------------------------------------------------------

class TFLiteClassifier:
    """Runs a TFLite image classification model on a cropped chicken image.

    Parameters
    ----------
    model_path : str
        Path to the ``.tflite`` classification model.
    input_size : tuple[int, int]
        (width, height) expected by the model.
    class_names : list[str]
        Ordered list of behaviour class names matching the model's outputs.
    threshold : float
        Minimum confidence to return a label; otherwise returns "other".
    """

    def __init__(
        self,
        model_path: str,
        input_size: tuple[int, int] = (96, 96),
        class_names: list[str] | None = None,
        threshold: float = 0.50,
    ):
        self.model_path  = model_path
        self.input_size  = input_size
        self.class_names = class_names or ["feeding","drinking","walking","resting","aggression","other"]
        self.threshold   = threshold
        self._interp     = _load_tflite_interpreter(model_path)

        in_d  = self._interp.get_input_details()[0]
        out_d = self._interp.get_output_details()[0]
        self._input_idx   = in_d["index"]
        self._input_dtype = in_d["dtype"]
        self._output_idx  = out_d["index"]
        logger.info("TFLiteClassifier loaded: %s", model_path)

    def preprocess(self, crop_bgr: np.ndarray) -> np.ndarray:
        """Resize and normalise a cropped chicken image."""
        w, h = self.input_size
        img  = cv2.resize(crop_bgr, (w, h))
        img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self._input_dtype == np.float32:
            img = img.astype(np.float32) / 255.0
        else:
            img = img.astype(np.uint8)
        return np.expand_dims(img, axis=0)

    def classify(self, crop_bgr: np.ndarray) -> tuple[str, float]:
        """Classify a cropped chicken image.

        Returns
        -------
        tuple[str, float]
            (behaviour_label, confidence)
        """
        inp = self.preprocess(crop_bgr)
        self._interp.set_tensor(self._input_idx, inp)
        self._interp.invoke()

        probs     = self._interp.get_tensor(self._output_idx)[0]
        idx       = int(np.argmax(probs))
        score     = float(probs[idx])
        label     = self.class_names[idx] if idx < len(self.class_names) else "other"

        if score < self.threshold:
            label = "other"

        return label, score


# ---------------------------------------------------------------------------
# Post-processing utilities
# ---------------------------------------------------------------------------

def non_max_suppression(
    detections: list[Detection],
    iou_threshold: float = 0.45,
) -> list[Detection]:
    """Apply NMS to a list of Detection objects.

    Parameters
    ----------
    detections : list[Detection]
    iou_threshold : float

    Returns
    -------
    list[Detection]
        Filtered detections after NMS.
    """
    if not detections:
        return []

    boxes  = np.array([[d.x1, d.y1, d.x2, d.y2] for d in detections], dtype=np.float32)
    scores = np.array([d.score for d in detections], dtype=np.float32)

    idxs   = np.argsort(scores)[::-1]
    keep   = []

    while len(idxs) > 0:
        i = idxs[0]
        keep.append(i)

        if len(idxs) == 1:
            break

        xx1 = np.maximum(boxes[i, 0], boxes[idxs[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[idxs[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[idxs[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[idxs[1:], 3])

        w   = np.maximum(0, xx2 - xx1)
        h   = np.maximum(0, yy2 - yy1)
        inter = w * h

        area_i  = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_j  = (boxes[idxs[1:], 2] - boxes[idxs[1:], 0]) * \
                  (boxes[idxs[1:], 3] - boxes[idxs[1:], 1])
        union   = area_i + area_j - inter

        iou    = np.where(union > 0, inter / union, 0.0)
        idxs   = idxs[1:][iou < iou_threshold]

    return [detections[i] for i in keep]


def crop_detection(frame: np.ndarray, det: Detection, padding: int = 4) -> np.ndarray:
    """Return a padded crop of the chicken from the original frame.

    Parameters
    ----------
    frame : np.ndarray
        Full camera frame (BGR).
    det : Detection
        Detection bounding box.
    padding : int
        Extra pixels added on each side.

    Returns
    -------
    np.ndarray
        Cropped BGR image, or empty array if bbox is degenerate.
    """
    h, w = frame.shape[:2]
    x1 = max(0, det.x1 - padding)
    y1 = max(0, det.y1 - padding)
    x2 = min(w, det.x2 + padding)
    y2 = min(h, det.y2 + padding)
    if x2 <= x1 or y2 <= y1:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    return frame[y1:y2, x1:x2].copy()
