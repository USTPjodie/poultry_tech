"""
monitor.py – Real-time chicken behaviour monitoring on Raspberry Pi 4.

Architecture (multi-threaded):
  Thread 1 (main)   – inference: detect → classify → track → analyse → draw
  Thread 2           – camera I/O (background reader in CameraStream)
  Thread 3 (opt.)   – Flask web dashboard

Performance target: ≥ 10 FPS on Raspberry Pi 4 (4 GB) with 2–5 chickens.

Usage:
    # With display:
    python monitor.py

    # Headless (no display, logs + alerts only):
    python monitor.py --headless

    # Enable web dashboard on http://<pi-ip>:5000:
    python monitor.py --dashboard

    # Use a specific camera source:
    python monitor.py --source 0
    python monitor.py --source picamera2

Press Q in the display window to stop.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import threading
import time

import cv2
import numpy as np

import config
from analyser import BehaviourAnalyser
from camera_utils import CameraStream, FPSCounter
from logger import setup_logging
from model_utils import (
    TFLiteClassifier,
    TFLiteDetector,
    crop_detection,
    non_max_suppression,
)
from tracker import build_tracker

setup_logging()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Colour palette for behaviour labels
# ---------------------------------------------------------------------------

BEHAVIOUR_COLOURS = {
    "feeding":    (0, 200, 0),
    "drinking":   (255, 165, 0),
    "walking":    (0, 165, 255),
    "resting":    (128, 0, 128),
    "aggression": (0, 0, 255),
    "other":      (150, 150, 150),
}


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_tracks(frame: np.ndarray, tracks: dict, fps: float) -> np.ndarray:
    """Overlay bounding boxes, IDs, behaviours, and motion trails on a frame.

    Parameters
    ----------
    frame : np.ndarray
        Original BGR frame.
    tracks : dict[int, Track]
        Active tracks from the tracker.
    fps : float
        Current processing FPS (displayed in the HUD).

    Returns
    -------
    np.ndarray
        Annotated copy of ``frame``.
    """
    vis = frame.copy()

    for tid, track in tracks.items():
        if track.disappeared > 0:
            continue

        colour = BEHAVIOUR_COLOURS.get(track.behaviour, (150, 150, 150))
        x1, y1, x2, y2 = track.bbox

        # Bounding box
        cv2.rectangle(vis, (x1, y1), (x2, y2), colour, 2)

        # Label badge
        label = f"#{tid} {track.behaviour} ({track.behaviour_confidence:.0%})"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw + 4, y1), colour, -1)
        cv2.putText(vis, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Motion trail
        pts = list(track.history)[-20:]
        for i in range(1, len(pts)):
            alpha = i / len(pts)
            c = tuple(int(ch * alpha) for ch in colour)
            cv2.line(vis, pts[i - 1], pts[i], c, 1)

    # HUD
    h_frame = vis.shape[0]
    cv2.putText(vis, f"FPS: {fps:.1f}  Chickens: {sum(1 for t in tracks.values() if t.disappeared == 0)}",
                (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
    cv2.putText(vis, time.strftime("%Y-%m-%d %H:%M:%S"),
                (8, h_frame - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    return vis


# ---------------------------------------------------------------------------
# Flask web dashboard (optional)
# ---------------------------------------------------------------------------

def start_dashboard(frame_queue: queue.Queue, stats_queue: queue.Queue):
    """Launch a Flask MJPEG stream + stats JSON endpoint in a daemon thread.

    Endpoints:
      GET /           → HTML dashboard
      GET /video_feed → MJPEG stream
      GET /stats      → JSON flock summary
    """
    try:
        from flask import Flask, Response, jsonify  # type: ignore
    except ImportError:
        logger.warning("Flask not installed – dashboard disabled. "
                       "Install with: pip install flask")
        return

    app = Flask(__name__)

    def _gen_frames():
        while True:
            try:
                frame = frame_queue.get(timeout=1.0)
                ret, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if not ret:
                    continue
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                       + buf.tobytes() + b"\r\n")
            except queue.Empty:
                continue

    @app.route("/video_feed")
    def video_feed():
        return Response(_gen_frames(),
                        mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/stats")
    def stats():
        try:
            data = stats_queue.get_nowait()
        except queue.Empty:
            data = {}
        return jsonify(data)

    @app.route("/")
    def index():
        return """
<!DOCTYPE html><html><head>
<title>Chicken Monitor</title>
<style>body{background:#111;color:#eee;font-family:monospace;text-align:center}
img{max-width:100%;border:2px solid #444;margin-top:12px}
#stats{margin-top:16px;text-align:left;display:inline-block;background:#1a1a1a;
padding:12px;border-radius:6px;min-width:320px}
</style></head><body>
<h2>🐔 Chicken Behaviour Monitor</h2>
<img src="/video_feed" /><br>
<div id="stats">Loading stats…</div>
<script>
setInterval(()=>{
  fetch('/stats').then(r=>r.json()).then(d=>{
    document.getElementById('stats').innerHTML =
      '<pre>'+JSON.stringify(d,null,2)+'</pre>';
  });
}, 2000);
</script></body></html>
"""

    t = threading.Thread(
        target=lambda: app.run(
            host=config.DASHBOARD_HOST,
            port=config.DASHBOARD_PORT,
            debug=False,
            use_reloader=False,
        ),
        daemon=True,
    )
    t.start()
    logger.info("Dashboard started at http://%s:%d",
                config.DASHBOARD_HOST, config.DASHBOARD_PORT)


# ---------------------------------------------------------------------------
# Main monitoring loop
# ---------------------------------------------------------------------------

def run_monitor(
    headless: bool = False,
    enable_dashboard: bool = False,
    source=None,
):
    """Main inference loop.

    Parameters
    ----------
    headless : bool
        If True, suppress all OpenCV display windows.
    enable_dashboard : bool
        If True, start the Flask web dashboard.
    source : int | str | None
        Override camera source (default: ``config.CAMERA_SOURCE``).
    """
    # ── Model loading ────────────────────────────────────────────────────
    if not os.path.exists(config.DETECTOR_MODEL_PATH):
        logger.critical(
            "Detector model not found: %s\n"
            "Train it first with train_detector.py or copy a .tflite model "
            "to the models/ directory.",
            config.DETECTOR_MODEL_PATH,
        )
        return
    if not os.path.exists(config.CLASSIFIER_MODEL_PATH):
        logger.critical(
            "Classifier model not found: %s\n"
            "Train it first with train_classifier.py.",
            config.CLASSIFIER_MODEL_PATH,
        )
        return

    logger.info("Loading models …")
    detector = TFLiteDetector(
        model_path=config.DETECTOR_MODEL_PATH,
        input_size=config.DETECTOR_INPUT_SIZE,
        score_threshold=config.DETECTION_THRESHOLD,
        chicken_class_id=config.CHICKEN_CLASS_ID,
    )
    classifier = TFLiteClassifier(
        model_path=config.CLASSIFIER_MODEL_PATH,
        input_size=config.CLASSIFIER_INPUT_SIZE,
        class_names=config.BEHAVIOUR_CLASSES,
        threshold=config.CLASSIFIER_THRESHOLD,
    )
    tracker    = build_tracker(config.TRACKER_TYPE)
    analyser   = BehaviourAnalyser()
    fps_counter = FPSCounter()

    # ── Camera ────────────────────────────────────────────────────────────
    cam_source = source if source is not None else config.CAMERA_SOURCE
    cam = CameraStream(
        source=cam_source,
        width=config.CAMERA_WIDTH,
        height=config.CAMERA_HEIGHT,
        fps=config.CAMERA_FPS,
    )
    cam.start()
    time.sleep(0.5)  # Warm-up
    logger.info("Camera ready.")

    # ── Optional web dashboard queues ─────────────────────────────────────
    frame_q = queue.Queue(maxsize=2)
    stats_q = queue.Queue(maxsize=2)

    if enable_dashboard:
        start_dashboard(frame_q, stats_q)

    # ── Main loop ─────────────────────────────────────────────────────────
    logger.info("Monitoring started. Press Q to quit.")
    skip = 0  # Frame-skip counter for adaptive processing

    try:
        while True:
            frame = cam.read()
            if frame is None:
                time.sleep(0.01)
                continue

            # ── Detection ─────────────────────────────────────────────
            raw_dets = detector.detect(frame)
            dets     = non_max_suppression(raw_dets, config.NMS_IOU_THRESHOLD)

            # ── Behaviour classification per detection ─────────────────
            for det in dets:
                crop = crop_detection(frame, det)
                if crop.size > 0:
                    label, conf = classifier.classify(crop)
                else:
                    label, conf = "other", 0.0
                # Store temporarily on the Detection tuple via a wrapper dict
                det._label = label  # type: ignore[attr-defined]
                det._conf  = conf   # type: ignore[attr-defined]

            # ── Tracker update ────────────────────────────────────────
            tracks = tracker.update(dets)

            # Push classifier labels into tracks
            for det in dets:
                if not hasattr(det, "_label"):
                    continue
                # Find the closest track centroid to this detection centroid
                det_cx, det_cy = det.centroid
                best_tid  = None
                best_dist = float("inf")
                for tid, track in tracks.items():
                    if track.disappeared > 0:
                        continue
                    dx = track.centroid[0] - det_cx
                    dy = track.centroid[1] - det_cy
                    d  = (dx * dx + dy * dy) ** 0.5
                    if d < best_dist:
                        best_dist = d
                        best_tid  = tid
                if best_tid is not None and best_dist < 60:
                    tracks[best_tid].update_behaviour(det._label, det._conf)  # type: ignore

            # ── Behaviour analysis ────────────────────────────────────
            fps_val = fps_counter.tick()
            analyser.update(tracks, fps=fps_val)

            # ── Render & display ──────────────────────────────────────
            vis = draw_tracks(frame, tracks, fps_val)

            if not headless:
                cv2.imshow("Chicken Monitor – press Q to quit", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            # Push to dashboard if enabled (non-blocking)
            if enable_dashboard:
                try:
                    frame_q.put_nowait(vis)
                except queue.Full:
                    pass
                try:
                    stats_q.put_nowait(analyser.get_flock_summary())
                except queue.Full:
                    pass

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        cam.stop()
        if not headless:
            cv2.destroyAllWindows()
        logger.info("Monitor stopped.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Real-time chicken behaviour monitor")
    parser.add_argument("--headless",   action="store_true",
                        help="Run without display (log + alerts only)")
    parser.add_argument("--dashboard",  action="store_true",
                        help="Enable Flask web dashboard")
    parser.add_argument("--source",
                        help="Camera source (int index or 'picamera2')")
    args = parser.parse_args()

    # Convert integer string to int
    source = args.source
    if source is not None and source.isdigit():
        source = int(source)

    run_monitor(
        headless=args.headless,
        enable_dashboard=args.dashboard,
        source=source,
    )


if __name__ == "__main__":
    main()
