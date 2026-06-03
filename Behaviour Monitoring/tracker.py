"""
tracker.py – Object tracking module.

Provides two tracker implementations:
  1. CentroidTracker  – lightweight, no Kalman filter, suitable for low-density
                        scenes (≤ 10 chickens) on a Raspberry Pi 4.
  2. SORTTracker      – Simple Online and Realtime Tracking using Kalman filters
                        and the Hungarian algorithm; more robust for faster motion.

Both trackers share the same interface:

    tracker = CentroidTracker(max_disappeared=30)
    tracks  = tracker.update(detections)   # list[Detection] → dict[id → Track]

Each ``Track`` holds:
  - id           : persistent integer ID
  - bbox         : (x1, y1, x2, y2) in pixels
  - centroid     : (cx, cy)
  - disappeared  : consecutive frames since last matched detection
  - history      : deque of (cx, cy) positions
  - behaviour    : current behaviour label
  - behaviour_confidence : classifier confidence

Usage:
    from tracker import CentroidTracker, SORTTracker
    from model_utils import Detection

    tracker = CentroidTracker(max_disappeared=config.MAX_DISAPPEARED)
    tracks  = tracker.update(detections)
    for tid, track in tracks.items():
        print(tid, track.behaviour, track.centroid)
"""

from __future__ import annotations

import logging
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment  # type: ignore
from scipy.spatial import distance as dist        # type: ignore

from model_utils import Detection
import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Track dataclass
# ---------------------------------------------------------------------------

@dataclass
class Track:
    """Represents a single tracked chicken across frames.

    Attributes
    ----------
    id : int
        Unique integer identifier assigned at creation.
    bbox : tuple[int,int,int,int]
        Latest bounding box (x1, y1, x2, y2).
    centroid : tuple[int,int]
        Latest centroid pixel coordinates.
    disappeared : int
        Consecutive frames during which the chicken was not detected.
    history : deque
        Ring buffer of centroid positions (cx, cy) over time.
    behaviour : str
        Most recent behaviour label.
    behaviour_confidence : float
        Classifier confidence for ``behaviour``.
    behaviour_window : deque
        Raw behaviour labels for majority-vote smoothing.
    frame_count : int
        Total frames this track has been alive.
    """
    id: int
    bbox: tuple = (0, 0, 0, 0)
    centroid: tuple = (0, 0)
    disappeared: int = 0
    history: deque = field(default_factory=lambda: deque(maxlen=config.MAX_TRACK_HISTORY))
    behaviour: str = "other"
    behaviour_confidence: float = 0.0
    behaviour_window: deque = field(
        default_factory=lambda: deque(maxlen=config.BEHAVIOUR_WINDOW_FRAMES)
    )
    frame_count: int = 0

    def distance_moved(self, n_frames: int = 30) -> float:
        """Total Euclidean distance moved over the last ``n_frames`` frames."""
        pts = list(self.history)[-n_frames:]
        if len(pts) < 2:
            return 0.0
        return sum(
            dist.euclidean(pts[i], pts[i + 1])
            for i in range(len(pts) - 1)
        )

    def update_behaviour(self, label: str, confidence: float):
        """Push a new raw label into the smoothing window and recompute majority."""
        self.behaviour_window.append(label)
        self.behaviour_confidence = confidence
        # Majority vote
        from collections import Counter
        counts = Counter(self.behaviour_window)
        self.behaviour = counts.most_common(1)[0][0]


# ---------------------------------------------------------------------------
# Utility: IoU
# ---------------------------------------------------------------------------

def _iou(boxA: tuple, boxB: tuple) -> float:
    """Compute IoU of two (x1, y1, x2, y2) boxes."""
    x1 = max(boxA[0], boxB[0]); y1 = max(boxA[1], boxB[1])
    x2 = min(boxA[2], boxB[2]); y2 = min(boxA[3], boxB[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    aA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    aB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = aA + aB - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# 1. Centroid Tracker (simple, Pi-friendly)
# ---------------------------------------------------------------------------

class CentroidTracker:
    """Assign consistent IDs by matching centroids across frames using
    Euclidean distance.

    Parameters
    ----------
    max_disappeared : int
        Number of consecutive frames a track is held without a match before
        being deleted.
    max_distance : int
        Maximum pixel distance to consider two centroids the same chicken.
    """

    def __init__(
        self,
        max_disappeared: int = config.MAX_DISAPPEARED,
        max_distance: int = 80,
    ):
        self.next_id       = 0
        self.tracks: Dict[int, Track] = OrderedDict()
        self.max_disappeared = max_disappeared
        self.max_distance    = max_distance

    # ------------------------------------------------------------------
    def _register(self, det: Detection):
        track = Track(
            id=self.next_id,
            bbox=det.bbox,
            centroid=det.centroid,
        )
        track.history.append(det.centroid)
        self.tracks[self.next_id] = track
        self.next_id += 1

    def _deregister(self, tid: int):
        del self.tracks[tid]

    # ------------------------------------------------------------------
    def update(self, detections: List[Detection]) -> Dict[int, Track]:
        """Match incoming detections to existing tracks.

        Parameters
        ----------
        detections : list[Detection]
            Detections from the current frame.

        Returns
        -------
        dict[int, Track]
            All currently active tracks (including recently disappeared ones).
        """
        if not detections:
            # Mark all tracks as disappeared
            for tid in list(self.tracks.keys()):
                self.tracks[tid].disappeared += 1
                if self.tracks[tid].disappeared > self.max_disappeared:
                    self._deregister(tid)
            return self.tracks

        det_centroids = np.array([d.centroid for d in detections], dtype=np.float32)

        if not self.tracks:
            for d in detections:
                self._register(d)
            return self.tracks

        track_ids      = list(self.tracks.keys())
        track_centroids = np.array(
            [self.tracks[tid].centroid for tid in track_ids], dtype=np.float32
        )

        # Pairwise distance matrix [n_tracks × n_dets]
        D = dist.cdist(track_centroids, det_centroids)

        # Greedy matching: sort by distance, assign if below threshold
        row_ind, col_ind = linear_sum_assignment(D)

        used_rows = set()
        used_cols = set()

        for r, c in zip(row_ind, col_ind):
            if D[r, c] > self.max_distance:
                continue
            tid = track_ids[r]
            self.tracks[tid].centroid    = detections[c].centroid
            self.tracks[tid].bbox        = detections[c].bbox
            self.tracks[tid].disappeared = 0
            self.tracks[tid].frame_count += 1
            self.tracks[tid].history.append(detections[c].centroid)
            used_rows.add(r)
            used_cols.add(c)

        # Unmatched tracks
        for r in set(range(len(track_ids))) - used_rows:
            tid = track_ids[r]
            self.tracks[tid].disappeared += 1
            if self.tracks[tid].disappeared > self.max_disappeared:
                self._deregister(tid)

        # Unmatched detections → new tracks
        for c in set(range(len(detections))) - used_cols:
            self._register(detections[c])

        return self.tracks


# ---------------------------------------------------------------------------
# 2. SORT Tracker (Kalman filter + Hungarian)
# ---------------------------------------------------------------------------

class KalmanBoxTracker:
    """Kalman filter for a single bounding box.

    State vector: [x, y, s, r, dx, dy, ds]
      x, y = centre coords
      s    = scale (area)
      r    = aspect ratio (constant)
      dx, dy, ds = velocities
    """

    _count = 0

    def __init__(self, bbox: tuple):
        from filterpy.kalman import KalmanFilter  # type: ignore
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([
            [1,0,0,0,1,0,0],
            [0,1,0,0,0,1,0],
            [0,0,1,0,0,0,1],
            [0,0,0,1,0,0,0],
            [0,0,0,0,1,0,0],
            [0,0,0,0,0,1,0],
            [0,0,0,0,0,0,1],
        ], dtype=np.float32)
        self.kf.H = np.array([
            [1,0,0,0,0,0,0],
            [0,1,0,0,0,0,0],
            [0,0,1,0,0,0,0],
            [0,0,0,1,0,0,0],
        ], dtype=np.float32)
        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P         *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = self._bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker._count
        KalmanBoxTracker._count += 1
        self.hit_streak = 0
        self.age        = 0

    @staticmethod
    def _bbox_to_z(bbox):
        """(x1,y1,x2,y2) → (cx, cy, s, r)."""
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w / 2.0
        y = bbox[1] + h / 2.0
        s = w * h
        r = w / float(h) if h > 0 else 1.0
        return np.array([[x], [y], [s], [r]], dtype=np.float32)

    @staticmethod
    def _z_to_bbox(z, score: float = 0.0):
        """State vector → (x1,y1,x2,y2,score)."""
        w = np.sqrt(abs(z[2] * z[3]))
        h = z[2] / w if w > 0 else 1.0
        return (
            int(z[0] - w / 2), int(z[1] - h / 2),
            int(z[0] + w / 2), int(z[1] + h / 2),
        )

    def predict(self):
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.age += 1
        self.kf.predict()
        return self._z_to_bbox(self.kf.x)

    def update(self, bbox: tuple):
        self.time_since_update = 0
        self.hit_streak += 1
        self.kf.update(self._bbox_to_z(bbox))

    @property
    def state_bbox(self):
        return self._z_to_bbox(self.kf.x)


class SORTTracker:
    """SORT: Simple Online and Realtime Tracking.

    Reference: Bewley et al., 2016 (https://arxiv.org/abs/1602.00763)

    Parameters
    ----------
    max_disappeared : int
        Max frames without a match before removing a track.
    min_hits : int
        Minimum number of consecutive detections before a track is considered
        confirmed.
    iou_threshold : float
        Minimum IoU for matching.
    """

    def __init__(
        self,
        max_disappeared: int = config.MAX_DISAPPEARED,
        min_hits: int = 3,
        iou_threshold: float = config.IOU_MATCH_THRESHOLD,
    ):
        self.max_disappeared = max_disappeared
        self.min_hits        = min_hits
        self.iou_threshold   = iou_threshold
        self._kf_trackers: List[KalmanBoxTracker] = []
        self.tracks: Dict[int, Track] = {}
        self._frame_count = 0

    # ------------------------------------------------------------------
    def _associate(
        self,
        dets: List[Detection],
        trks: List[tuple],
    ):
        """Hungarian algorithm matching using IoU cost matrix."""
        if not trks:
            return np.empty((0, 2), dtype=int), list(range(len(dets))), []
        if not dets:
            return np.empty((0, 2), dtype=int), [], list(range(len(trks)))

        iou_matrix = np.zeros((len(dets), len(trks)), dtype=np.float32)
        for d, det in enumerate(dets):
            for t, trk in enumerate(trks):
                iou_matrix[d, t] = _iou(det.bbox, trk)

        det_idx, trk_idx = linear_sum_assignment(-iou_matrix)
        matched, unmatched_d, unmatched_t = [], [], []

        for d, t in zip(det_idx, trk_idx):
            if iou_matrix[d, t] < self.iou_threshold:
                unmatched_d.append(d)
                unmatched_t.append(t)
            else:
                matched.append([d, t])

        for d in range(len(dets)):
            if d not in det_idx:
                unmatched_d.append(d)
        for t in range(len(trks)):
            if t not in trk_idx:
                unmatched_t.append(t)

        return np.array(matched), unmatched_d, unmatched_t

    # ------------------------------------------------------------------
    def update(self, detections: List[Detection]) -> Dict[int, Track]:
        """Update all Kalman filter trackers and return active tracks."""
        self._frame_count += 1

        # Predict next state for each existing KF tracker
        trk_bboxes = [kft.predict() for kft in self._kf_trackers]

        matched, unmatched_dets, unmatched_trks = self._associate(
            detections, trk_bboxes
        )

        # Update matched trackers
        for d, t in matched:
            self._kf_trackers[t].update(detections[d].bbox)

        # Create new trackers for unmatched detections
        for d in unmatched_dets:
            self._kf_trackers.append(KalmanBoxTracker(detections[d].bbox))

        # Remove dead trackers
        alive = []
        for kft in self._kf_trackers:
            if kft.time_since_update <= self.max_disappeared:
                alive.append(kft)
        self._kf_trackers = alive

        # Build Track objects
        new_tracks: Dict[int, Track] = {}
        for kft in self._kf_trackers:
            if kft.hit_streak < self.min_hits and self._frame_count > self.min_hits:
                continue
            bx = kft.state_bbox
            cx, cy = (bx[0] + bx[2]) // 2, (bx[1] + bx[3]) // 2
            tid = kft.id

            if tid in self.tracks:
                t = self.tracks[tid]
                t.bbox        = bx
                t.centroid    = (cx, cy)
                t.disappeared = kft.time_since_update
                t.frame_count += 1
                t.history.append((cx, cy))
            else:
                t = Track(id=tid, bbox=bx, centroid=(cx, cy))
                t.history.append((cx, cy))

            new_tracks[tid] = t

        self.tracks = new_tracks
        return self.tracks


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_tracker(tracker_type: str = config.TRACKER_TYPE):
    """Return a configured tracker based on ``config.TRACKER_TYPE``.

    Parameters
    ----------
    tracker_type : str
        ``"centroid"`` or ``"sort"``.
    """
    if tracker_type == "sort":
        try:
            import filterpy  # noqa: F401
        except ImportError:
            logger.warning(
                "filterpy not installed (required for SORT). "
                "Falling back to centroid tracker. "
                "Install with: pip install filterpy"
            )
            return CentroidTracker()
        return SORTTracker()
    return CentroidTracker()
