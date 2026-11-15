"""vigia count --source samples/people_counter/overhead-walkers.mp4 --out docs/gifs/count.gif"""
from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from vigia.core import draw
from vigia.core.detect import Detector
from vigia.core.runner import ROOT
from vigia.core.scene import Scene
from vigia.core.track import Tracker, to_tracks
from vigia.core.types import Event, Track
from vigia.core.video import SourceError

from .logic import CounterLogic, slot

TRAIL = 1.5   # seconds of feet positions drawn behind each person
FLASH = 1.0   # seconds a counted crossing stays highlighted


def _pts(points) -> np.ndarray:
    return np.array([(round(x), round(y)) for x, y in points], np.int32)


def _ipt(p) -> tuple[int, int]:
    return round(float(p[0])), round(float(p[1]))


def _badge(frame: np.ndarray, org: tuple[int, int], text: str, status: str) -> int:
    """A big filled label (draw.chip with more padding), bottom-left corner at `org`; returns its right edge."""
    k = max(0.5, frame.shape[0] / 720)
    s, pad = 0.8 * k, round(12 * k)
    th = max(1, round(s * 1.6))
    (tw, tht), base = cv2.getTextSize(text, draw.FONT, s, th)
    x, y = org
    y = max(y, tht + base + 2 * pad)
    cv2.rectangle(frame, (x, y - tht - base - 2 * pad), (x + tw + 2 * pad, y), draw.PALETTE[status], -1)
    cv2.putText(frame, text, (x + pad, y - base - pad), draw.FONT, s, draw.WHITE, th, cv2.LINE_AA)
    return x + tw + 2 * pad


class CountApp:
    name = "count"
    blur_faces = True

    @staticmethod
    def add_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--line", default="door", help="name of the counting line in the scene")
        p.add_argument("--flip", action="store_true", help="count the other direction as IN")
        p.add_argument("--min-travel", type=float, default=20.0,
                       help="pixels the feet must travel across the line (half on each side) to count")
        p.add_argument("--snapshots", default=None, nargs="?", const=str(ROOT / "out" / "count-crossings"),
                       help="save a crop (head pixelated) of every counted crossing, to check the counts by eye; "
                            "default folder out/count-crossings")

    def setup(self, scene: Scene | None, fps: float, size: tuple[int, int], args: argparse.Namespace) -> None:
        if scene is None or args.line not in scene.lines:
            raise SourceError(f"count needs a line named {args.line!r}: vigia zones <video> --out scenes/count.json")
        self.det = Detector(args.model, labels={"person"}, device=args.device)
        self.tracker = Tracker(fps)
        self.logic = CounterLogic(scene, args.line, -1 if args.flip else 1, args.min_travel)
        self.tracks: list[Track] = []
        self.trails: dict[int, deque] = {}
        self.crossed: dict[int, tuple[float, str]] = {}  # track id -> (t, "IN"/"OUT") of its last counted crossing
        self.snap_dir = Path(args.snapshots) if args.snapshots else None
        self.blur = None
        if self.snap_dir:
            self.snap_dir.mkdir(parents=True, exist_ok=True)
            if not args.no_blur:
                from vigia.core.privacy import FaceBlur
                self.blur = FaceBlur()
        self.t = 0.0

    def process(self, t: float, frame: np.ndarray) -> list[Event]:
        self.t = t
        dets = self.det(frame)
        tracked = to_tracks(self.tracker.update(dets))
        # Detections the tracker has not confirmed yet stay in with id -1 so the runner pixelates their heads too;
        # the counter and the drawing ignore them.
        self.tracks = tracked + to_tracks(dets)
        for tr in tracked:
            self.trails.setdefault(tr.id, deque()).append((t, tr.bottom_center))
        for tid, trail in list(self.trails.items()):
            while trail and t - trail[0][0] > TRAIL:
                trail.popleft()
            if not trail:
                del self.trails[tid]
        events = self.logic.update(t, self.tracks)
        for ev in events:
            self.crossed[ev.track_id] = (t, "IN" if ev.kind == "entered" else "OUT")
            if self.snap_dir:
                ev.data["snapshot"] = self.snapshot(frame, ev)
        return events

    def snapshot(self, frame: np.ndarray, ev: Event) -> str:
        """Crop around the person at the moment they are counted, with the line and their trail, head pixelated."""
        tr = next(tr for tr in self.tracks if tr.id == ev.track_id)
        img = frame.copy()
        if self.blur:
            self.blur(img, [p.box for p in self.tracks])
        status = "ok" if ev.kind == "entered" else "warn"
        draw.line(img, self.logic.a, self.logic.b, "info")
        cv2.polylines(img, [_pts(p for _, p in self.trails[tr.id])], False, draw.PALETTE[status], 3, cv2.LINE_AA)
        draw.box(img, tr.box, status=status)
        H, W = img.shape[:2]
        x1, y1, x2, y2 = tr.box
        m = 0.6 * tr.h
        crop = img[max(0, int(y1 - m)):min(H, int(y2 + m)), max(0, int(x1 - m)):min(W, int(x2 + m))]
        path = self.snap_dir / f"{ev.t:07.2f}s-{'in' if ev.kind == 'entered' else 'out'}-id{tr.id}.jpg"
        cv2.imwrite(str(path), crop)
        return str(path)

    def draw(self, frame: np.ndarray) -> np.ndarray:
        lg = self.logic
        for tr in self.tracks:
            if tr.id < 0:
                continue
            hit = self.crossed.get(tr.id)
            flash = hit is not None and self.t - hit[0] <= FLASH
            status = ("ok" if hit[1] == "IN" else "warn") if flash else "info"
            if len(self.trails.get(tr.id, ())) > 1:
                cv2.polylines(frame, [_pts(p for _, p in self.trails[tr.id])], False, draw.PALETTE[status], 3,
                              cv2.LINE_AA)
            draw.box(frame, tr.box, f"#{tr.id} {hit[1]}" if flash else f"#{tr.id}", status)
        self._line(frame)
        n_in, n_out = lg.per_15min.get(slot(self.t), (0, 0))
        draw.panel(frame, "People counter", [
            ("INSIDE NOW", str(lg.inside_now), "info"),
            ("Total entries (IN)", str(lg.entered), "ok"),
            ("Total exits (OUT)", str(lg.exited), "warn"),
            (f"Since {slot(self.t)}", f"+{n_in} in, -{n_out} out", "muted")])
        return frame

    def _line(self, frame: np.ndarray) -> None:
        """The line with a green stripe on its IN side and an orange one on its OUT side, an arrow towards IN and
        the three counts as coloured badges at its start."""
        H = frame.shape[0]
        a, b = np.array(self.logic.a), np.array(self.logic.b)
        n = np.array([a[1] - b[1], b[0] - a[0]]) / self.logic.length * self.logic.in_side  # unit normal towards IN
        w = max(4, round(H / 135))
        for side, status in ((1, "ok"), (-1, "warn")):
            off = n * side * (w / 2 + 3)
            cv2.line(frame, _ipt(a + off), _ipt(b + off), draw.PALETTE[status], w, cv2.LINE_AA)
        draw.line(frame, a, b, "info")
        mid, r = a + 0.85 * (b - a), 0.06 * H  # arrow near the far end, away from the badges
        cv2.arrowedLine(frame, _ipt(mid - n * r), _ipt(mid + n * r), draw.WHITE, max(3, round(H / 240)), cv2.LINE_AA,
                        tipLength=0.35)
        draw.chip(frame, mid + n * r + (0.01 * H, 0), "IN", "ok")
        draw.chip(frame, mid - n * r + (0.01 * H, 0.02 * H), "OUT", "warn")
        x, y = _ipt(a - n * (w + 6))
        lg = self.logic
        for text, status in ((f"IN {lg.entered}", "ok"), (f"OUT {lg.exited}", "warn"),
                             (f"INSIDE NOW {lg.inside_now}", "info")):
            x = _badge(frame, (x, y), text, status) + round(H / 135)

    def summary(self) -> dict:
        lg = self.logic
        return {"model": self.det.name, "entered": lg.entered, "exited": lg.exited, "inside_now": lg.inside_now,
                "per_15min": lg.per_15min}


APP = CountApp
