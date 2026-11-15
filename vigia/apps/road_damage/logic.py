"""Road damage found on video, each one counted once, with a report a city can act on: when it was first seen,
what it is, how sure the model was and a picture of it. Decides on tracked boxes only; run.py saves the pictures."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from vigia.core.types import Box, Event, Track

FORGET = 5.0  # seconds after which a half-seen (not yet counted) track is dropped


@dataclass
class DamageRecord:
    first_t: float
    label: str        # the class of its most confident view
    max_score: float  # highest confidence while counted
    crop_path: str
    box: Box          # where it was seen with max_score: the crop
    last_t: float


def iou(a: Box, b: Box) -> float:
    inter = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def mostly_inside(box: Box, others: list[Box], share: float = 0.5) -> bool:
    """True if at least `share` of `box` lies inside one of `others` (a crack 'seen' on a car's side or wheel)."""
    area = (box[2] - box[0]) * (box[3] - box[1])
    for o in others:
        inter = max(0.0, min(box[2], o[2]) - max(box[0], o[0])) * max(0.0, min(box[3], o[3]) - max(box[1], o[1]))
        if area > 0 and inter / area >= share:
            return True
    return False


def mmss(s: float) -> str:
    return f"{int(s // 60)}:{int(s % 60):02d}"


class DamageLogic:
    def __init__(self, min_frames: int = 3, crop_dir: str = "out/road-crops", rematch_seconds: float = 5.0,
                 rematch_iou: float = 0.3):
        self.min_frames, self.crop_dir = min_frames, crop_dir.rstrip("/")
        self.rematch_seconds, self.rematch_iou = rematch_seconds, rematch_iou
        self.report: list[DamageRecord] = []
        self.improved: list[int] = []  # report rows whose best view changed this frame: save their crop now
        self._pending: dict[int, tuple[int, float, float]] = {}  # track id -> (frames seen, first t, last t)
        self.row_of: dict[int, int] = {}  # track id -> report row

    def _same_spot(self, t: float, box: Box) -> int | None:
        """A record seen lately at this spot: the same damage under a new track id (a car hid it for a while).
        Only right for a camera that stays still; rematch_seconds=0 turns it off for a moving one."""
        for i, r in enumerate(self.report):
            if t - r.last_t <= self.rematch_seconds and iou(box, r.box) >= self.rematch_iou:
                return i
        return None

    def update(self, t: float, tracks: list[Track]) -> list[Event]:
        self.improved = []
        events = []
        for tr in tracks:
            if tr.id < 0:
                continue
            row = self.row_of.get(tr.id)
            if row is None:
                n, first, _ = self._pending.get(tr.id, (0, t, t))
                if n + 1 < self.min_frames:
                    self._pending[tr.id] = (n + 1, first, t)
                    continue
                self._pending.pop(tr.id, None)
                row = self._same_spot(t, tr.box)
                if row is None:
                    row = len(self.report)
                    self.report.append(DamageRecord(first, tr.label, tr.score,
                                                    f"{self.crop_dir}/damage-{row + 1:03d}.jpg", tr.box, t))
                    self.improved.append(row)
                    events.append(Event(t, "damage_found", tr.id,
                                        {"id": row + 1, "label": tr.label, "score": round(tr.score, 2)}))
                self.row_of[tr.id] = row
            rec = self.report[row]
            rec.last_t = t
            if tr.score > rec.max_score:
                rec.max_score, rec.label, rec.box = tr.score, tr.label, tr.box
                if row not in self.improved:
                    self.improved.append(row)
        self._pending = {k: v for k, v in self._pending.items() if t - v[2] <= FORGET}
        return events

    def write_csv(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["id", "first_seen", "first_t", "label", "max_score", "crop_path"])
            for i, r in enumerate(self.report, start=1):
                w.writerow([i, mmss(r.first_t), f"{r.first_t:.1f}", r.label, f"{r.max_score:.2f}", r.crop_path])
