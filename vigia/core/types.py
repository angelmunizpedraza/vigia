"""The two values every demo passes around: what is tracked in a frame, and what happened."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

Pt = tuple[float, float]
Box = tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels


@dataclass(frozen=True)
class Track:
    id: int
    box: Box
    label: str
    score: float
    keypoints: np.ndarray | None = None  # (17, 3) COCO order: x, y, confidence

    @property
    def w(self) -> float:
        return self.box[2] - self.box[0]

    @property
    def h(self) -> float:
        return self.box[3] - self.box[1]

    @property
    def center(self) -> Pt:
        return ((self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2)

    @property
    def bottom_center(self) -> Pt:
        return ((self.box[0] + self.box[2]) / 2, self.box[3])


@dataclass
class Event:
    t: float
    kind: str
    track_id: int | None = None
    data: dict = field(default_factory=dict)
