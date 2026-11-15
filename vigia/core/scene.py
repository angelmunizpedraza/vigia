"""Zones, lines and parameters of a camera view, stored in 0-1 coordinates so they survive a resize."""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path

from .types import Box, Pt


@dataclass
class Scene:
    width: int
    height: int
    lines: dict[str, tuple[Pt, Pt]] = field(default_factory=dict)
    zones: dict[str, list[Pt]] = field(default_factory=dict)
    params: dict = field(default_factory=dict)

    def zones_like(self, prefix: str) -> dict[str, list[Pt]]:
        return {k: v for k, v in self.zones.items() if k.startswith(prefix)}


def load_scene(path: Path, width: int, height: int) -> Scene:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    aspect = d.get("aspect")
    if aspect and abs(aspect - width / height) / aspect > 0.05:
        warnings.warn(f"scene {path} was drawn for aspect {aspect:.3f}, frame is {width / height:.3f}", UserWarning)

    def px(p) -> Pt:
        return (float(p[0]) * width, float(p[1]) * height)

    return Scene(width, height,
                 {k: (px(a), px(b)) for k, (a, b) in d.get("lines", {}).items()},
                 {k: [px(p) for p in poly] for k, poly in d.get("zones", {}).items()},
                 d.get("params", {}))


def save_scene(path: Path, scene: Scene) -> None:
    w, h = scene.width, scene.height

    def n(p: Pt) -> list[float]:
        return [round(p[0] / w, 4), round(p[1] / h, 4)]

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({
        "aspect": round(w / h, 4),
        "lines": {k: [n(a), n(b)] for k, (a, b) in scene.lines.items()},
        "zones": {k: [n(p) for p in poly] for k, poly in scene.zones.items()},
        "params": scene.params,
    }, indent=2), encoding="utf-8")


def point_in_polygon(p: Pt, poly: list[Pt]) -> bool:
    x, y = p
    inside = False
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def side_of_line(p: Pt, a: Pt, b: Pt) -> float:
    """Cross product sign: >0 on one side of a→b, <0 on the other, 0 on the line."""
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def _area(poly: list[Pt]) -> float:
    return abs(sum(poly[i][0] * poly[i - 1][1] - poly[i - 1][0] * poly[i][1] for i in range(len(poly)))) / 2


def _clip(poly: list[Pt], box: Box) -> list[Pt]:
    """Sutherland-Hodgman: the part of `poly` (may be concave) inside the axis-aligned `box`."""
    x1, y1, x2, y2 = box
    edges = [(lambda p: p[0] >= x1, lambda a, b: _cut_x(a, b, x1)),
             (lambda p: p[0] <= x2, lambda a, b: _cut_x(a, b, x2)),
             (lambda p: p[1] >= y1, lambda a, b: _cut_y(a, b, y1)),
             (lambda p: p[1] <= y2, lambda a, b: _cut_y(a, b, y2))]
    out = list(poly)
    for inside, cut in edges:
        src, out = out, []
        for i, cur in enumerate(src):
            prev = src[i - 1]
            if inside(cur):
                if not inside(prev):
                    out.append(cut(prev, cur))
                out.append(cur)
            elif inside(prev):
                out.append(cut(prev, cur))
        if not out:
            break
    return out


def _cut_x(a: Pt, b: Pt, x: float) -> Pt:
    return (x, a[1] + (b[1] - a[1]) * (x - a[0]) / (b[0] - a[0]))


def _cut_y(a: Pt, b: Pt, y: float) -> Pt:
    return (a[0] + (b[0] - a[0]) * (y - a[1]) / (b[1] - a[1]), y)


def box_fraction_in_polygon(box: Box, poly: list[Pt]) -> float:
    """Share of the box's area that lies inside the polygon (0-1)."""
    area = (box[2] - box[0]) * (box[3] - box[1])
    if area <= 0:
        return 0.0
    clipped = _clip(poly, box)
    return min(1.0, _area(clipped) / area) if len(clipped) >= 3 else 0.0
