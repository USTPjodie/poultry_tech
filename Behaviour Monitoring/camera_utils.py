"""
camera_utils.py – Camera abstraction for Raspberry Pi Camera Module (Picamera2)
and USB/RTSP cameras via OpenCV.

Usage:
    cam = CameraStream(source=config.CAMERA_SOURCE,
                       width=config.CAMERA_WIDTH,
                       height=config.CAMERA_HEIGHT)
    cam.start()
    frame = cam.read()
    cam.stop()
"""

import time
import threading
import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resize_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize a frame to (width × height) using bilinear interpolation."""
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)


def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    """Convert OpenCV BGR frame to RGB (required by TFLite models)."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# FPS Counter
# ---------------------------------------------------------------------------

class FPSCounter:
    """Rolling-average FPS counter.

    Parameters
    ----------
    window : int
        Number of most-recent frame intervals to average.
    """

    def __init__(self, window: int = 30):
        self._window = window
        self._times: list[float] = []
        self._last = time.perf_counter()

    def tick(self) -> float:
        """Call once per processed frame; returns current FPS estimate."""
        now = time.perf_counter()
        self._times.append(now - self._last)
        self._last = now
        if len(self._times) > self._window:
            self._times.pop(0)
        return self.fps

    @property
    def fps(self) -> float:
        if not self._times:
            return 0.0
        return 1.0 / (sum(self._times) / len(self._times))


# ---------------------------------------------------------------------------
# OpenCV Camera (USB, built-in, RTSP)
# ---------------------------------------------------------------------------

class OpenCVCamera:
    """Thread-safe OpenCV camera wrapper with background capture thread.

    Parameters
    ----------
    source : int | str
        Camera index (0, 1 …) or RTSP/HTTP URL string.
    width, height : int
        Capture resolution.
    fps : int
        Capture frame-rate hint (not enforced by all cameras).
    """

    def __init__(self, source=0, width=640, height=480, fps=30):
        self.source = source
        self.width  = width
        self.height = height
        self.fps    = fps

        self._cap    = None
        self._frame  = None
        self._lock   = threading.Lock()
        self._thread = None
        self._running = False

    # ------------------------------------------------------------------
    def start(self):
        """Open the capture device and start the background reader thread."""
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera source: {self.source}")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)

        self._running = True
        self._thread  = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()
        logger.info("OpenCVCamera started (source=%s, %dx%d @ %dfps)",
                    self.source, self.width, self.height, self.fps)

    def _reader(self):
        """Background thread: continuously grab the latest frame."""
        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                logger.warning("OpenCVCamera: failed to grab frame – retrying")
                time.sleep(0.05)
                continue
            with self._lock:
                self._frame = frame

    def read(self) -> np.ndarray | None:
        """Return the most recent frame (BGR), or None if not yet available."""
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def stop(self):
        """Stop the background thread and release the capture device."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._cap:
            self._cap.release()
        logger.info("OpenCVCamera stopped.")


# ---------------------------------------------------------------------------
# Picamera2 Camera (Raspberry Pi)
# ---------------------------------------------------------------------------

class PiCamera2Stream:
    """Wrapper around Picamera2 for the official Raspberry Pi camera module.

    Falls back gracefully if picamera2 is not installed.

    Parameters
    ----------
    width, height : int
        Capture resolution.
    fps : int
        Target capture frame-rate.
    """

    def __init__(self, width=640, height=480, fps=30):
        self.width  = width
        self.height = height
        self.fps    = fps
        self._picam = None
        self._frame  = None
        self._lock   = threading.Lock()
        self._thread = None
        self._running = False

    def start(self):
        try:
            from picamera2 import Picamera2  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "picamera2 not installed. Run: sudo apt install python3-picamera2"
            ) from exc

        self._picam = Picamera2()
        config = self._picam.create_preview_configuration(
            main={"size": (self.width, self.height), "format": "BGR888"}
        )
        self._picam.configure(config)
        self._picam.set_controls({"FrameRate": self.fps})
        self._picam.start()

        self._running = True
        self._thread  = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()
        logger.info("PiCamera2Stream started (%dx%d @ %dfps)",
                    self.width, self.height, self.fps)

    def _reader(self):
        while self._running:
            frame = self._picam.capture_array()
            with self._lock:
                self._frame = frame

    def read(self) -> np.ndarray | None:
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._picam:
            self._picam.stop()
        logger.info("PiCamera2Stream stopped.")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def CameraStream(source=0, width=640, height=480, fps=30):
    """Factory that returns the appropriate camera object.

    Parameters
    ----------
    source : int | str
        Integer for USB/built-in camera, ``"picamera2"`` for Pi Camera Module,
        or an RTSP URL string.

    Returns
    -------
    OpenCVCamera | PiCamera2Stream
    """
    if source == "picamera2":
        return PiCamera2Stream(width=width, height=height, fps=fps)
    return OpenCVCamera(source=source, width=width, height=height, fps=fps)
