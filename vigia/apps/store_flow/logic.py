"""Shop paths, one per tracked person: browsing, a product picked at a shelf, a checkout, leaving.

A person who picked a product (stayed `dwell` s at a shelf) and reaches the exit without passing a checkout since is
marked REVIEW: a moment worth a human look, never a verdict. People are told apart only by the tracker's id: no face
recognition, no identity. Staff are recognised only by standing in a staff zone, and are then left out.

Zones by name prefix: `shelf-*`, `checkout*`, `exit*`, `staff*` (anything else, like `entrance`, is only drawn).
A person is in a zone when their feet (bottom centre of the box) are inside it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from vigia.core.scene import Scene, point_in_polygon
from vigia.core.types import Event, Track

KINDS = ("staff", "checkout", "exit", "shelf")  # priority when zones overlap
STATES = ("PRODUCT", "PAID", "REVIEW", "STAFF", "LEFT")  # counted on entry; TRACKING is the start
FORGET_AFTER = 30.0  # s without seeing a track before its path is dropped (the tracker gives up after 1 s)


@dataclass
class Path:
    state: str = "TRACKING"
    last_t: float = 0.0
    zone: str | None = None
    seen_inside: bool = False      # seen outside the exit zone, so reaching it means leaving, not coming in
    shelf: str | None = None       # current shelf stretch: zone, start, last time seen in it
    shelf_start: float = 0.0
    shelf_last: float = 0.0
    visited: set[str] = field(default_factory=set)


class StoreLogic:
    def __init__(self, scene: Scene, dwell: float = 4.0, gap_tolerance: float = 1.0):
        self.zones = {kind: scene.zones_like(kind) for kind in KINDS}
        self.dwell, self.gap_tolerance = dwell, gap_tolerance
        self.height = scene.height
        self.paths: dict[int, Path] = {}
        self.counts = dict.fromkeys(STATES, 0)
        self.zone_seconds: dict[str, float] = {}   # person-seconds in each zone, staff left out
        self.zone_visitors: dict[str, int] = {}    # distinct tracks that stood in each zone, staff left out

    def zone_of(self, track: Track) -> tuple[str | None, str | None]:
        x, y = track.bottom_center
        feet = (x, min(y, self.height - 1))  # a box cut by the bottom edge still counts as inside the frame
        for kind in KINDS:
            for name, poly in self.zones[kind].items():
                if point_in_polygon(feet, poly):
                    return kind, name
        return None, None

    def _to(self, p: Path, state: str, t: float, tid: int, kind: str, **data) -> Event:
        p.state = state
        self.counts[state] += 1
        return Event(t, kind, tid, data)

    def update(self, t: float, tracks: list[Track]) -> list[Event]:
        events = []
        for tr in tracks:
            if tr.label != "person" or tr.id < 0:
                continue
            p = self.paths.get(tr.id)
            if p is None:
                p = self.paths[tr.id] = Path(last_t=t)
            dt = t - p.last_t if t - p.last_t <= self.gap_tolerance else 0.0
            p.last_t = t
            kind, name = self.zone_of(tr)
            if kind == "staff" and p.state != "STAFF":
                events.append(self._to(p, "STAFF", t, tr.id, "staff", zone=name))
            if p.state == "STAFF":
                continue
            if name:
                self.zone_seconds[name] = self.zone_seconds.get(name, 0.0) + dt
                if name not in p.visited:
                    p.visited.add(name)
                    self.zone_visitors[name] = self.zone_visitors.get(name, 0) + 1
            p.zone = name
            if p.state in ("LEFT", "REVIEW"):
                continue
            if kind == "exit":
                if p.seen_inside and p.state == "PRODUCT":
                    events.append(self._to(p, "REVIEW", t, tr.id, "exit_without_checkout", zone=name))
                elif p.seen_inside:
                    events.append(self._to(p, "LEFT", t, tr.id, "left", zone=name))
                continue
            p.seen_inside = True
            if kind == "checkout" and p.state in ("TRACKING", "PRODUCT"):
                events.append(self._to(p, "PAID", t, tr.id, "paid", zone=name))
            elif kind == "shelf" and p.state in ("TRACKING", "PAID"):  # a product picked after paying counts too
                if p.shelf != name or t - p.shelf_last > self.gap_tolerance:
                    p.shelf, p.shelf_start = name, t
                p.shelf_last = t
                if t - p.shelf_start >= self.dwell - 1e-9:
                    events.append(self._to(p, "PRODUCT", t, tr.id, "product", zone=name))
        for tid in [k for k, p in self.paths.items() if t - p.last_t > FORGET_AFTER]:
            del self.paths[tid]
        return events
