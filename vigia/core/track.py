"""ByteTrack (from supervision 0.29.1, MIT) and the conversion to plain Tracks for the logic modules."""
from __future__ import annotations

import numpy as np
import supervision as sv

from .types import Track


class Tracker:
    def __init__(self, fps: float, lost_seconds: float = 1.0, activation: float = 0.25):
        # supervision scales lost_track_buffer by frame_rate / 30 internally, so 30 frames == 1 s at any fps
        self._bt = sv.ByteTrack(track_activation_threshold=activation, lost_track_buffer=int(30 * lost_seconds),
                                frame_rate=max(1, round(fps)))

    def update(self, dets: sv.Detections) -> sv.Detections:
        return self._bt.update_with_detections(dets)


def to_tracks(dets: sv.Detections, keypoints: np.ndarray | None = None) -> list[Track]:
    """sv.Detections -> list[Track]; untracked detections get id -1. `keypoints` is (N, 17, 3) or None."""
    ids = dets.tracker_id if dets.tracker_id is not None else np.full(len(dets), -1)
    names = dets.data.get("class_name", np.array(["object"] * len(dets), dtype=object))
    conf = dets.confidence if dets.confidence is not None else np.ones(len(dets))
    return [Track(int(ids[i]), tuple(float(v) for v in dets.xyxy[i]), str(names[i]), float(conf[i]),
                  None if keypoints is None else keypoints[i]) for i in range(len(dets))]
