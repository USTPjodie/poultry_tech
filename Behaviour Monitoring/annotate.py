"""
annotate.py – Interactive bounding-box annotation tool (OpenCV-based).

Features:
  • Draw bounding boxes with left-click + drag
  • Assign a behaviour label (F/D/W/R/A/O)
  • Save annotations in YOLO .txt format per image
  • Optionally extract and save cropped chicken images into per-class folders
    for classifier training
  • Automatically split dataset into train / val / test sets

Usage:
    python annotate.py --images data/raw_frames --output data/annotations
    python annotate.py --images data/raw_frames --extract-crops
    python annotate.py --split --annotations data/annotations
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

import config
from logger import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

# Class index map (0-indexed, matching BEHAVIOUR_CLASSES order for classifier)
CLASS_TO_IDX = {cls: i for i, cls in enumerate(config.BEHAVIOUR_CLASSES)}

# ---------------------------------------------------------------------------
# Annotation state
# ---------------------------------------------------------------------------

class AnnotationState:
    def __init__(self):
        self.boxes: list[tuple[int, int, int, int, str]] = []
        # Each box: (x1, y1, x2, y2, label)
        self.drawing = False
        self.start_pt = (0, 0)
        self.cur_pt   = (0, 0)
        self.current_label = "other"

    def begin(self, x: int, y: int):
        self.drawing  = True
        self.start_pt = (x, y)
        self.cur_pt   = (x, y)

    def update(self, x: int, y: int):
        self.cur_pt = (x, y)

    def finish(self):
        if not self.drawing:
            return
        self.drawing = False
        x1 = min(self.start_pt[0], self.cur_pt[0])
        y1 = min(self.start_pt[1], self.cur_pt[1])
        x2 = max(self.start_pt[0], self.cur_pt[0])
        y2 = max(self.start_pt[1], self.cur_pt[1])
        if (x2 - x1) > 5 and (y2 - y1) > 5:
            self.boxes.append((x1, y1, x2, y2, self.current_label))

    def undo(self):
        if self.boxes:
            self.boxes.pop()

    def clear(self):
        self.boxes.clear()


# ---------------------------------------------------------------------------
# Mouse callback
# ---------------------------------------------------------------------------

_state = AnnotationState()


def _mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        _state.begin(x, y)
    elif event == cv2.EVENT_MOUSEMOVE and _state.drawing:
        _state.update(x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        _state.finish()


# ---------------------------------------------------------------------------
# Drawing utilities
# ---------------------------------------------------------------------------

LABEL_COLORS = {
    "feeding":   (0, 200, 0),
    "drinking":  (255, 165, 0),
    "walking":   (0, 165, 255),
    "resting":   (128, 0, 128),
    "aggression":(0, 0, 255),
    "other":     (128, 128, 128),
}


def _render(frame: np.ndarray, state: AnnotationState) -> np.ndarray:
    vis = frame.copy()
    # Committed boxes
    for (x1, y1, x2, y2, lbl) in state.boxes:
        color = LABEL_COLORS.get(lbl, (200, 200, 200))
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(vis, lbl, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    # In-progress box
    if state.drawing:
        cv2.rectangle(vis, state.start_pt, state.cur_pt, (0, 255, 255), 1)

    # HUD
    cv2.putText(vis,
                f"Label: {state.current_label}  |  Boxes: {len(state.boxes)}",
                (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(vis,
                "Draw:LMB  F/D/W/R/A/O:label  Z:undo  C:clear  "
                "SPACE:save&next  Q:quit",
                (6, vis.shape[0] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)
    return vis


# ---------------------------------------------------------------------------
# YOLO format helpers
# ---------------------------------------------------------------------------

def _boxes_to_yolo(
    boxes: list[tuple[int, int, int, int, str]],
    img_w: int, img_h: int,
    class_id: int = 0,
) -> list[str]:
    """Convert pixel boxes to YOLO format lines (class cx cy w h normalised)."""
    lines = []
    for (x1, y1, x2, y2, lbl) in boxes:
        cx = ((x1 + x2) / 2) / img_w
        cy = ((y1 + y2) / 2) / img_h
        bw = (x2 - x1) / img_w
        bh = (y2 - y1) / img_h
        # For detection we always use class 0 (chicken); label goes in classifier crops
        lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines


def _save_yolo(annotation_dir: str, img_stem: str,
               boxes: list[tuple[int, int, int, int, str]],
               img_w: int, img_h: int):
    """Save a YOLO .txt annotation file."""
    lines = _boxes_to_yolo(boxes, img_w, img_h)
    txt_path = os.path.join(annotation_dir, img_stem + ".txt")
    with open(txt_path, "w") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# Crop extraction
# ---------------------------------------------------------------------------

def _extract_crops(
    frame: np.ndarray,
    img_stem: str,
    boxes: list[tuple[int, int, int, int, str]],
    crops_dir: str,
):
    """Save one JPEG crop per box into per-label subfolders."""
    for i, (x1, y1, x2, y2, lbl) in enumerate(boxes):
        dest_dir = os.path.join(crops_dir, lbl)
        os.makedirs(dest_dir, exist_ok=True)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        crop_name = f"{img_stem}_box{i:03d}.jpg"
        cv2.imwrite(os.path.join(dest_dir, crop_name), crop)


# ---------------------------------------------------------------------------
# Interactive annotation loop
# ---------------------------------------------------------------------------

def annotate_images(
    images_dir: str,
    output_dir: str,
    extract_crops: bool = False,
    crops_dir: str = config.CROPS_DIR,
):
    """Run interactive annotation over all JPEG images in ``images_dir``.

    Parameters
    ----------
    images_dir : str
        Folder of raw JPEG frames.
    output_dir : str
        Folder where YOLO .txt annotation files are written.
    extract_crops : bool
        If True, also extract and save per-label crop images.
    crops_dir : str
        Root folder for cropped images (sub-folder per class).
    """
    os.makedirs(output_dir, exist_ok=True)

    img_paths = sorted(
        p for p in Path(images_dir).glob("*.jpg")
    )
    if not img_paths:
        logger.error("No JPEG images found in %s", images_dir)
        return

    logger.info("Found %d images to annotate.", len(img_paths))

    window_name = "Annotate"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, _mouse_callback)

    i = 0
    while i < len(img_paths):
        img_path = img_paths[i]
        frame = cv2.imread(str(img_path))
        if frame is None:
            logger.warning("Cannot read %s – skipping", img_path)
            i += 1
            continue

        _state.clear()
        img_h, img_w = frame.shape[:2]
        stem = img_path.stem

        # Check for existing annotation – reload if present
        existing_txt = os.path.join(output_dir, stem + ".txt")
        # (Existing boxes loaded as detection-class only; no label info stored)
        # Skip reload for simplicity; restart annotation if needed.

        print(f"\n[{i + 1}/{len(img_paths)}] {img_path.name}")
        print("  Draw boxes, then SPACE to save. Q to quit.")

        while True:
            vis = _render(frame, _state)
            cv2.imshow(window_name, vis)
            key = cv2.waitKey(20) & 0xFF

            if key == ord("q"):
                cv2.destroyAllWindows()
                logger.info("Annotation session ended by user.")
                return
            elif chr(key).lower() in config.BEHAVIOUR_KEYS:
                _state.current_label = config.BEHAVIOUR_KEYS[chr(key).lower()]
                print(f"  Label → {_state.current_label}")
            elif key == ord("z"):
                _state.undo()
            elif key == ord("c"):
                _state.clear()
            elif key == ord(" ") or key == 13:  # SPACE or Enter
                # Save annotation
                _save_yolo(output_dir, stem, _state.boxes, img_w, img_h)
                if extract_crops:
                    _extract_crops(frame, stem, _state.boxes, crops_dir)
                logger.debug("Saved: %s (%d boxes)", stem, len(_state.boxes))
                i += 1
                break
            elif key == ord("b") and i > 0:
                i -= 1  # Go back
                break

    cv2.destroyAllWindows()
    logger.info("Annotation complete.")


# ---------------------------------------------------------------------------
# Dataset split
# ---------------------------------------------------------------------------

def split_dataset(
    annotations_dir: str,
    images_dir: str,
    output_dir: str,
    val_ratio: float = config.TRAIN_VAL_SPLIT,
    test_ratio: float = config.TRAIN_TEST_SPLIT,
    seed: int = 42,
):
    """Split image + annotation pairs into train / val / test folders.

    Creates:
        output_dir/
            train/  images/  labels/
            val/    images/  labels/
            test/   images/  labels/

    Parameters
    ----------
    annotations_dir : str
        Folder containing YOLO .txt files.
    images_dir : str
        Folder containing the corresponding JPEG files.
    output_dir : str
        Root output folder for the split dataset.
    """
    txt_files = sorted(Path(annotations_dir).glob("*.txt"))
    if not txt_files:
        logger.error("No annotation files found in %s", annotations_dir)
        return

    random.seed(seed)
    random.shuffle(txt_files)

    n     = len(txt_files)
    n_val = max(1, int(n * val_ratio))
    n_test= max(1, int(n * test_ratio))

    splits = {
        "val":   txt_files[:n_val],
        "test":  txt_files[n_val:n_val + n_test],
        "train": txt_files[n_val + n_test:],
    }

    for split_name, files in splits.items():
        img_out = os.path.join(output_dir, split_name, "images")
        lbl_out = os.path.join(output_dir, split_name, "labels")
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)
        for txt_path in files:
            stem = txt_path.stem
            img_src = os.path.join(images_dir, stem + ".jpg")
            if not os.path.exists(img_src):
                continue
            shutil.copy(img_src, os.path.join(img_out, stem + ".jpg"))
            shutil.copy(str(txt_path), os.path.join(lbl_out, stem + ".txt"))
        logger.info("Split '%s': %d samples", split_name, len(files))

    # Write data.yaml for YOLOv5/v8 training
    yaml_path = os.path.join(output_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {os.path.abspath(output_dir)}\n")
        f.write("train: train/images\nval: val/images\ntest: test/images\n")
        f.write(f"nc: 1\nnames: ['chicken']\n")
    logger.info("Wrote data.yaml → %s", yaml_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Chicken bounding-box annotator")
    parser.add_argument("--images", default=config.RAW_FRAMES_DIR,
                        help="Folder of raw frame JPEGs")
    parser.add_argument("--output", default=config.ANNOTATIONS_DIR,
                        help="Output folder for YOLO .txt annotations")
    parser.add_argument("--extract-crops", action="store_true",
                        help="Also extract and save per-label crop images")
    parser.add_argument("--crops-dir", default=config.CROPS_DIR,
                        help="Root folder for extracted crops")
    parser.add_argument("--split", action="store_true",
                        help="Run dataset split after annotation")
    parser.add_argument("--split-output", default=os.path.join(config.DATA_DIR, "split"),
                        help="Output folder for the train/val/test split")
    args = parser.parse_args()

    annotate_images(
        images_dir=args.images,
        output_dir=args.output,
        extract_crops=args.extract_crops,
        crops_dir=args.crops_dir,
    )

    if args.split:
        split_dataset(
            annotations_dir=args.output,
            images_dir=args.images,
            output_dir=args.split_output,
        )


if __name__ == "__main__":
    main()
