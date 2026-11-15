"""People per square metre on a grid laid on the ground, and an alert when a cell stays packed.

The ground is measured in one of three ways: a homography from 4 image points with their positions in metres
(any camera angle), a known px-per-metre (camera looking straight down), or, failing both, a scale estimated from
the median person box width (straight down only, flagged as an estimate).
"""
from __future__ import annotations

import math

import numpy as np

from vigia.core.scene import Scene, box_fraction_in_polygon, point_in_polygon
from vigia.core.types import Event, Pt, Track

PERSON_M = 0.45  # adult shoulder width: what a person box spans when seen from straight above


def homography(image_pts: list[Pt], ground_pts: list[Pt]) -> np.ndarray:
    """3x3 matrix mapping 4 image points (px) onto their ground positions (m): the 8 DLT equations, h33 = 1."""
    a, b = [], []
    for (x, y), (u, v) in zip(image_pts, ground_pts, strict=True):
        a += [[x, y, 1, 0, 0, 0, -u * x, -u * y], [0, 0, 0, x, y, 1, -v * x, -v * y]]
        b += [u, v]
    return np.append(np.linalg.solve(np.array(a, float), np.array(b, float)), 1.0).reshape(3, 3)


def homography_from_scene(scene: Scene | None) -> np.ndarray | None:
    """Zone `calibration` (4 image points) + params `calibration_m` (the same 4 points in metres), if present."""
    if scene is None or "calibration" not in scene.zones or "calibration_m" not in scene.params:
        return None
    return homography(scene.zones["calibration"], [tuple(p) for p in scene.params["calibration_m"]])


def apply_homography(h: np.ndarray, pts) -> np.ndarray:
    p = np.c_[np.asarray(pts, float).reshape(-1, 2), np.ones(len(pts))] @ h.T
    return p[:, :2] / p[:, 2:3]


class DensityLogic:
    def __init__(self, scene: Scene, cell_m: float = 2.0, px_per_m: float | None = None,
                 homography: np.ndarray | None = None, amber: float = 2.0, red: float = 4.0, hold: float = 3.0,
                 person_m: float = PERSON_M, min_people: int = 10):
        w, h = scene.width, scene.height
        # where people can stand; under a homography never the whole image, whose top may reach the horizon
        self.region = (scene.zones.get("area") or (scene.zones.get("calibration") if homography is not None else None)
                       or [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)])
        self.cell_m, self.amber, self.red, self.hold = cell_m, amber, red, hold
        self.person_m, self.min_people = person_m, min_people
        self.H = homography
        self.estimated = homography is None and px_per_m is None
        self.px_per_m = None if homography is not None else px_per_m
        self.grid = np.zeros((0, 0))                  # people/m2 per (row, col); NaN = cell mostly outside
        self.level: dict[tuple[int, int], str] = {}   # "ok" | "warn" | "alert" per evaluated cell
        self.people = 0
        self.peak = 0.0
        self._to_ground = np.eye(3)
        self._origin = (0.0, 0.0)
        self._red_since: dict[tuple[int, int], float] = {}
        self._alerted: set[tuple[int, int]] = set()

    def level_of(self, d: float) -> str:
        return "alert" if d >= self.red else "warn" if d >= self.amber else "ok"

    @property
    def peak_level(self) -> str:
        return self.level_of(self.peak)

    def cell_polygon(self, row: int, col: int) -> list[Pt]:
        """The cell's 4 corners in image pixels (a perspective quad under a homography)."""
        c, (x0, y0) = self.cell_m, self._origin
        corners = [(x0 + col * c, y0 + row * c), (x0 + (col + 1) * c, y0 + row * c),
                   (x0 + (col + 1) * c, y0 + (row + 1) * c), (x0 + col * c, y0 + (row + 1) * c)]
        return [tuple(p) for p in apply_homography(np.linalg.inv(self._to_ground), corners)]

    def update(self, t: float, tracks: list[Track]) -> list[Event]:
        people = [tr for tr in tracks if tr.label == "person"]
        if self.estimated and len(people) >= self.min_people:  # per frame, so a drone climbing or zooming is followed
            self.px_per_m = float(np.median([p.w for p in people])) / self.person_m
        # under a homography people stand on their feet; seen from straight above, the box centre is the person
        pts = [p.bottom_center if self.H is not None else p.center for p in people]
        pts = [p for p in pts if point_in_polygon(p, self.region)]
        self.people = len(pts)
        if self.H is None and self.px_per_m is None:
            self.grid, self.level, self.peak = np.zeros((0, 0)), {}, 0.0
            return self._alerts(t)
        self._to_ground = self.H if self.H is not None else np.diag([1 / self.px_per_m, 1 / self.px_per_m, 1.0])
        region = apply_homography(self._to_ground, self.region)
        (x0, y0), (x1, y1) = region.min(0), region.max(0)
        self._origin, c = (float(x0), float(y0)), self.cell_m
        rows, cols = max(1, math.ceil((y1 - y0) / c - 1e-9)), max(1, math.ceil((x1 - x0) / c - 1e-9))
        poly = [tuple(p) for p in region]
        area = np.full((rows, cols), np.nan)
        for r in range(rows):
            for k in range(cols):
                frac = box_fraction_in_polygon((x0 + k * c, y0 + r * c, x0 + (k + 1) * c, y0 + (r + 1) * c), poly)
                if frac >= 0.5:  # thin slivers at the edge would turn one person into a huge density
                    area[r, k] = c * c * frac
        counts = np.zeros((rows, cols))
        if pts:
            idx = np.floor((apply_homography(self._to_ground, pts) - (x0, y0)) / c).astype(int)
            for k, r in idx:
                if 0 <= r < rows and 0 <= k < cols:
                    counts[r, k] += 1
        self.grid = counts / area
        self.level = {(r, k): self.level_of(float(self.grid[r, k]))
                      for r in range(rows) for k in range(cols) if not np.isnan(self.grid[r, k])}
        self.peak = float(np.nanmax(self.grid)) if self.level else 0.0
        return self._alerts(t)

    def _alerts(self, t: float) -> list[Event]:
        """`density_alert` once per cell when it stays >= red for `hold` s; timed by t, so frames may be sparse."""
        red = {cell for cell, lv in self.level.items() if lv == "alert"}
        for cell in set(self._red_since) - red:
            del self._red_since[cell]
            self._alerted.discard(cell)
        events = []
        for cell in sorted(red):
            since = self._red_since.setdefault(cell, t)
            if t - since >= self.hold - 1e-9 and cell not in self._alerted:
                self._alerted.add(cell)
                events.append(Event(t, "density_alert", data={
                    "cell": list(cell), "density": round(float(self.grid[cell]), 2), "since": since,
                    "estimated_scale": self.estimated}))
        return events
