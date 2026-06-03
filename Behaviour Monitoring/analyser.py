"""
analyser.py – Behaviour analysis engine.

Consumes the ``tracks`` dictionary produced by the tracker each frame and
maintains per-chicken behaviour statistics.  Checks configurable alert
thresholds and dispatches alerts via the AlertDispatcher.

Responsibilities:
  • Maintain a rolling behaviour history per track
  • Compute time budgets (seconds in each behaviour)
  • Estimate activity level (pixel distance moved per minute)
  • Detect inactivity events (resting > threshold)
  • Detect aggression clusters (overlapping bboxes + aggression label)
  • Detect feeding/drinking drop-offs
  • Export per-frame and per-minute summaries

Usage:
    analyser = BehaviourAnalyser()
    analyser.update(tracks, fps=15.0)
    summary = analyser.get_summary()
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

import config
from logger import AlertDispatcher, BehaviourLogger
from tracker import Track

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-track statistics
# ---------------------------------------------------------------------------

@dataclass
class TrackStats:
    """Accumulated statistics for one tracked chicken.

    Parameters
    ----------
    track_id : int
    """
    track_id:    int
    # Time-budget counters (seconds)
    time_in_behaviour: Dict[str, float] = field(
        default_factory=lambda: defaultdict(float)
    )
    # Activity
    distance_last_minute: float = 0.0
    distance_history:     List[Tuple[float, float]] = field(default_factory=list)
    # Aggression
    aggression_events:    List[float] = field(default_factory=list)  # timestamps
    # Inactivity
    resting_since: float | None = None   # timestamp when resting started
    inactivity_alerted:  bool   = False
    # Last known behaviour
    last_behaviour: str   = "other"
    last_behaviour_ts: float = 0.0


# ---------------------------------------------------------------------------
# Overlap / proximity check
# ---------------------------------------------------------------------------

def _boxes_overlap(b1: Tuple, b2: Tuple) -> bool:
    """Return True if two bounding boxes (x1,y1,x2,y2) overlap."""
    return not (b1[2] <= b2[0] or b2[2] <= b1[0] or
                b1[3] <= b2[1] or b2[3] <= b1[1])


# ---------------------------------------------------------------------------
# Main Analyser
# ---------------------------------------------------------------------------

class BehaviourAnalyser:
    """Processes tracker output frame-by-frame and maintains statistics.

    Parameters
    ----------
    fps : float
        Expected camera frame rate (used for time-budget conversion).
        Can be updated dynamically via the ``fps`` argument to ``update()``.
    inactivity_threshold_min : int
        Alert if a chicken rests continuously for this many minutes.
    aggression_rate_threshold : int
        Alert if aggression events exceed this count per minute.
    """

    def __init__(
        self,
        fps: float = 15.0,
        inactivity_threshold_min: int = config.INACTIVITY_THRESHOLD_MIN,
        aggression_rate_threshold: int = config.AGGRESSION_RATE_PER_MIN,
    ):
        self.fps                       = fps
        self.inactivity_threshold_sec  = inactivity_threshold_min * 60
        self.aggression_rate_threshold = aggression_rate_threshold

        self._stats: Dict[int, TrackStats] = {}
        self._alerts = AlertDispatcher()
        self._beh_logger = BehaviourLogger()

        # Minute-bucket accumulators for log writing
        self._last_log_ts: float = time.time()

        # Baseline feeding fraction (estimated over first 5 minutes)
        self._feed_baseline: float | None = None
        self._feed_baseline_samples: List[float] = []
        self._feed_alerted = False

    # ------------------------------------------------------------------

    def update(
        self,
        tracks: Dict[int, Track],
        fps: float | None = None,
    ) -> None:
        """Process one frame's worth of tracking data.

        Parameters
        ----------
        tracks : dict[int, Track]
            Output of ``tracker.update()`` for the current frame.
        fps : float | None
            If provided, overrides the stored FPS estimate.
        """
        if fps:
            self.fps = fps

        now = time.time()
        sec_per_frame = 1.0 / max(self.fps, 1.0)

        # ── Initialise stats for new track IDs ──────────────────────────
        for tid in tracks:
            if tid not in self._stats:
                self._stats[tid] = TrackStats(track_id=tid)

        # ── Update each track ──────────────────────────────────────────
        for tid, track in tracks.items():
            if track.disappeared > 0:
                continue  # Skip tracks that are currently unmatched

            stats = self._stats[tid]
            beh   = track.behaviour

            # Time budget
            stats.time_in_behaviour[beh] += sec_per_frame

            # Activity: record distance
            history_pts = list(track.history)
            if len(history_pts) >= 2:
                d = float(np.linalg.norm(
                    np.array(history_pts[-1]) - np.array(history_pts[-2])
                ))
                stats.distance_history.append((now, d))

            # Trim old distance records (keep last 60 seconds)
            cutoff = now - 60.0
            stats.distance_history = [
                (t, v) for t, v in stats.distance_history if t >= cutoff
            ]
            stats.distance_last_minute = sum(v for _, v in stats.distance_history)

            # ── Inactivity detection ─────────────────────────────────────
            if beh == "resting":
                if stats.resting_since is None:
                    stats.resting_since = now
                elif (not stats.inactivity_alerted and
                      now - stats.resting_since >= self.inactivity_threshold_sec):
                    msg = (f"INACTIVITY ALERT: Chicken #{tid} has been resting "
                           f"for {int((now - stats.resting_since) // 60)} minutes.")
                    self._alerts.send(msg)
                    stats.inactivity_alerted = True
                    logger.warning(msg)
            else:
                stats.resting_since     = None
                stats.inactivity_alerted = False

            stats.last_behaviour    = beh
            stats.last_behaviour_ts = now

        # ── Aggression cross-track check ─────────────────────────────
        track_list = [
            (tid, trk) for tid, trk in tracks.items() if trk.disappeared == 0
        ]
        for i in range(len(track_list)):
            for j in range(i + 1, len(track_list)):
                tid_a, trk_a = track_list[i]
                tid_b, trk_b = track_list[j]
                if (
                    (trk_a.behaviour == "aggression" or trk_b.behaviour == "aggression")
                    and _boxes_overlap(trk_a.bbox, trk_b.bbox)
                ):
                    ts = now
                    self._stats[tid_a].aggression_events.append(ts)
                    self._stats[tid_b].aggression_events.append(ts)

        # Check aggression rate
        for tid, stats in self._stats.items():
            cutoff = now - 60.0
            stats.aggression_events = [
                t for t in stats.aggression_events if t >= cutoff
            ]
            rate = len(stats.aggression_events)
            if rate >= self.aggression_rate_threshold:
                msg = (f"AGGRESSION ALERT: Chicken #{tid} involved in "
                       f"{rate} aggressive interactions in the last minute.")
                self._alerts.send(msg)

        # ── Feeding drop-off detection ────────────────────────────────
        if tracks:
            total   = len(tracks)
            feeding = sum(
                1 for trk in tracks.values()
                if trk.behaviour == "feeding" and trk.disappeared == 0
            )
            feed_frac = feeding / total if total else 0.0

            if self._feed_baseline is None:
                # Collect samples for first 5 minutes
                self._feed_baseline_samples.append(feed_frac)
                if len(self._feed_baseline_samples) >= int(5 * 60 * self.fps):
                    self._feed_baseline = float(
                        np.mean(self._feed_baseline_samples)
                    )
                    logger.info("Feed baseline established: %.2f", self._feed_baseline)
            else:
                if (not self._feed_alerted and self._feed_baseline > 0.05 and
                        feed_frac < self._feed_baseline * config.FEED_DROP_THRESHOLD):
                    msg = (
                        f"FEEDING DROP ALERT: Feeding fraction ({feed_frac:.0%}) "
                        f"is well below baseline ({self._feed_baseline:.0%})."
                    )
                    self._alerts.send(msg)
                    self._feed_alerted = True

                # Reset alert when feeding recovers
                if feed_frac >= self._feed_baseline * 0.8:
                    self._feed_alerted = False

        # ── Periodic CSV log ──────────────────────────────────────────
        if now - self._last_log_ts >= config.LOG_INTERVAL_SEC:
            self._write_log(tracks, now)
            self._last_log_ts = now

    # ------------------------------------------------------------------

    def _write_log(self, tracks: Dict[int, Track], ts: float):
        """Write a behaviour summary row per track to CSV."""
        summary = {}
        for tid, track in tracks.items():
            if track.disappeared > 0:
                continue
            stats = self._stats.get(tid)
            summary[tid] = {
                "behaviour":        track.behaviour,
                "confidence":       track.behaviour_confidence,
                "cx":               track.centroid[0],
                "cy":               track.centroid[1],
                "elapsed_sec":      stats.time_in_behaviour.get(track.behaviour, 0)
                                    if stats else 0,
            }
        if summary:
            self._beh_logger.write_summary(summary)

    # ------------------------------------------------------------------

    def get_summary(self) -> Dict[int, Dict]:
        """Return current behaviour statistics for all tracks.

        Returns
        -------
        dict
            ``{ track_id: { 'behaviour': str, 'time_budget': dict,
                            'distance_last_minute': float,
                            'aggression_count': int } }``
        """
        result = {}
        for tid, stats in self._stats.items():
            result[tid] = {
                "behaviour":             stats.last_behaviour,
                "time_budget":           dict(stats.time_in_behaviour),
                "distance_last_minute":  round(stats.distance_last_minute, 1),
                "aggression_last_minute": len(stats.aggression_events),
                "resting_since":         stats.resting_since,
            }
        return result

    def get_flock_summary(self) -> Dict:
        """Return aggregate statistics across all tracked chickens.

        Useful for dashboards and MQTT status messages.
        """
        behaviours = defaultdict(int)
        total_distance = 0.0
        total_aggression = 0
        n = len(self._stats)

        for stats in self._stats.values():
            behaviours[stats.last_behaviour] += 1
            total_distance  += stats.distance_last_minute
            total_aggression += len(stats.aggression_events)

        return {
            "n_tracked":          n,
            "behaviour_counts":   dict(behaviours),
            "avg_distance_px_pm": round(total_distance / max(n, 1), 1),
            "total_aggression":   total_aggression,
            "feed_baseline":      self._feed_baseline,
        }
