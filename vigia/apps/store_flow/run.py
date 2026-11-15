"""vigia store --source samples/store_flow/checkouts.mp4 --out docs/gifs/store.gif"""
from __future__ import annotations

import argparse

import numpy as np
import supervision as sv

from vigia.core import draw
from vigia.core.detect import Detector
from vigia.core.scene import Scene
from vigia.core.track import Tracker, to_tracks
from vigia.core.types import Event, Track
from vigia.core.video import SourceError

from .logic import KINDS, StoreLogic

STATE_STATUS = {"TRACKING": "info", "PRODUCT": "info", "PAID": "ok", "REVIEW": "warn", "STAFF": "muted",
                "LEFT": "muted"}
ZONE_STATUS = {"checkout": "ok", "staff": "muted", "exit": "info", "shelf": "info"}


def nested_in(xyxy: np.ndarray, contain: float = 0.85, top: float = 0.1) -> np.ndarray:
    """For each box, the index of the bigger box it is a cut-short copy of, else -1. D-FINE sometimes returns a whole
    person and also the part above a counter or a cart. Only boxes nested in a bigger one *and* sharing its top
    edge count, so a child in front of an adult is kept."""
    out = np.full(len(xyxy), -1)
    area = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    for i, a in enumerate(xyxy):
        for j, b in enumerate(xyxy):
            if area[j] <= area[i] or (out[i] >= 0 and area[j] <= area[out[i]]):
                continue
            inter = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
            if inter >= contain * area[i] and abs(a[1] - b[1]) <= top * (b[3] - b[1]):
                out[i] = j
    return out


def merge_nested(dets: sv.Detections) -> sv.Detections:
    """Keep the whole-body box with the pair's best score. The whole body often scores ~0.45 and the cut copy ~0.85;
    left low, ByteTrack (which weighs overlap by score) starts a new id each time one of the two drops out."""
    into = nested_in(dets.xyxy)
    for i, j in enumerate(into):
        if j >= 0:
            dets.confidence[j] = max(dets.confidence[j], dets.confidence[i])
    return dets[into < 0]


class StoreApp:
    name = "store"
    blur_faces = True

    @staticmethod
    def add_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--dwell", type=float, default=4.0, help="seconds in a shelf zone that count as picking a product")

    def setup(self, scene: Scene | None, fps: float, size: tuple[int, int], args: argparse.Namespace) -> None:
        if scene is None or not any(scene.zones_like(k) for k in KINDS):
            raise SourceError("store needs zones named checkout*, exit*, staff*, shelf-*: "
                              "vigia zones <video> --out scenes/store.json")
        self.det = Detector(args.model, labels={"person"}, device=args.device)
        self.tracker = Tracker(fps)
        self.logic = StoreLogic(scene, args.dwell)
        self.zones = scene.zones
        self.tracks: list[Track] = []

    def process(self, t: float, frame: np.ndarray) -> list[Event]:
        dets = merge_nested(self.det(frame))
        self.tracks = [tr for tr in to_tracks(self.tracker.update(dets)) if tr.id >= 0]
        return self.logic.update(t, self.tracks)

    def draw(self, frame: np.ndarray) -> np.ndarray:
        for name, poly in self.zones.items():
            kind = next((k for k in KINDS if name.startswith(k)), None)
            draw.polygon(frame, poly, ZONE_STATUS.get(kind, "muted"), name.upper(), alpha=0.15)
        in_view = dict.fromkeys(STATE_STATUS, 0)
        for tr in self.tracks:
            p = self.logic.paths.get(tr.id)
            state = p.state if p else "TRACKING"
            in_view[state] += 1
            draw.box(frame, tr.box, f"{tr.id} {state}", STATE_STATUS[state])
        draw.panel(frame, "People in view", [(s.title(), str(n), STATE_STATUS[s]) for s, n in in_view.items()],
                   corner="br")
        return frame

    def summary(self) -> dict:
        lg = self.logic
        return {"model": self.det.name,
                "entered_state": lg.counts,  # per tracker id: an id switch makes the same person count again
                "zones": {z: {"visitors": lg.zone_visitors.get(z, 0), "person_seconds": round(s, 1),
                              "mean_seconds": round(s / max(1, lg.zone_visitors.get(z, 0)), 1)}
                          for z, s in lg.zone_seconds.items()}}


APP = StoreApp
