"""How many people went in, how many came out, and how many are inside now, from feet crossing a line.

A person's feet (bottom centre of the box) must move from one side of the line to the other, at least
`min_travel / 2` px clear of it on each side, so someone hesitating on the line is counted once. Only tracks with
an identity are used; each one counts at most once per direction until it crosses back.
"""
from __future__ import annotations

import math

from vigia.core.scene import Scene, side_of_line
from vigia.core.types import Event, Track

FORGET_AFTER = 5.0  # seconds without seeing a track before its side is dropped (the tracker gives up after 1 s)


def slot(t: float) -> str:
    """Start of the 15-minute slot holding t, as HH:MM of elapsed time."""
    b = int(t // 900) * 900
    return f"{b // 3600:02d}:{b % 3600 // 60:02d}"


class CounterLogic:
    def __init__(self, scene: Scene, line: str = "door", in_side: int = 1, min_travel: float = 20.0):
        self.a, self.b = scene.lines[line]
        self.in_side, self.min_travel = in_side, min_travel
        self.length = math.dist(self.a, self.b)
        self.entered = self.exited = 0
        self.per_15min: dict[str, tuple[int, int]] = {}
        self.sides: dict[int, tuple[int, float]] = {}  # track id -> (last confident side +1/-1, last seen t)

    @property
    def inside_now(self) -> int:
        return max(0, self.entered - self.exited)

    def side(self, p) -> int:
        """+1 or -1 when the point is clear of the line by min_travel / 2, else 0 (on the line, undecided)."""
        d = side_of_line(p, self.a, self.b) / self.length
        return 0 if abs(d) < self.min_travel / 2 else (1 if d > 0 else -1)

    def within_door(self, p) -> bool:
        """The feet are in front of the segment, not beside it (matters for a short door line)."""
        (ax, ay), (bx, by) = self.a, self.b
        u = ((p[0] - ax) * (bx - ax) + (p[1] - ay) * (by - ay)) / self.length ** 2
        return 0.0 <= u <= 1.0

    def update(self, t: float, tracks: list[Track]) -> list[Event]:
        events = []
        for tr in tracks:
            if tr.label != "person" or tr.id < 0:
                continue
            p = tr.bottom_center
            s = self.side(p)
            prev = self.sides.get(tr.id)
            if s == 0:
                if prev:
                    self.sides[tr.id] = (prev[0], t)
                continue
            self.sides[tr.id] = (s, t)
            if prev is None or prev[0] == s or not self.within_door(p):
                continue
            went_in = s == self.in_side
            if went_in:
                self.entered += 1
            else:
                self.exited += 1
            n_in, n_out = self.per_15min.get(slot(t), (0, 0))
            self.per_15min[slot(t)] = (n_in + went_in, n_out + (not went_in))
            events.append(Event(t, "entered" if went_in else "exited", tr.id,
                                {"inside_now": self.inside_now, "entered": self.entered, "exited": self.exited,
                                 "feet": [round(p[0]), round(p[1])]}))
        for tid, (_, seen) in list(self.sides.items()):
            if t - seen > FORGET_AFTER:
                del self.sides[tid]
        return events
