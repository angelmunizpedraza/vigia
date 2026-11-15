"""Draw zones and lines on a video's first frame and save them as a scene JSON.

vigia zones samples/tables/cafe.mp4 --out scenes/tables.json
  left click  add a point to the current shape
  z           start a new zone (name asked in the terminal)
  l           start a new line (2 points)
  u           undo the last point
  s           save and quit        q  quit without saving
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from vigia.core import draw
from vigia.core.scene import Scene, load_scene, save_scene
from vigia.core.video import SourceError, frames


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="vigia zones")
    p.add_argument("video")
    p.add_argument("--out", required=True)
    p.add_argument("--at", type=float, default=0.0, help="use the frame at this second")
    a = p.parse_args(argv)
    frame = None
    try:
        for t, f in frames(a.video):
            frame = f
            if t >= a.at:
                break
    except SourceError as e:
        print(f"error: {e}")
        return 2
    if frame is None:
        print(f"error: cannot read {a.video}")
        return 2
    h, w = frame.shape[:2]
    scene = load_scene(Path(a.out), w, h) if Path(a.out).exists() else Scene(w, h)
    r = min(1.0, 1280 / w, 800 / h)
    current: dict = {"kind": None, "name": None, "pts": []}

    def commit() -> None:
        if current["kind"] == "zone" and len(current["pts"]) >= 3:
            scene.zones[current["name"]] = list(current["pts"])
        elif current["kind"] == "line" and len(current["pts"]) == 2:
            scene.lines[current["name"]] = tuple(current["pts"])
        current.update(kind=None, name=None, pts=[])

    def on_mouse(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN and current["kind"]:
            current["pts"].append((x / r, y / r))
            if current["kind"] == "line" and len(current["pts"]) == 2:
                commit()

    win = "vigia zones  (z zone, l line, u undo, s save, q quit)"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        vis = frame.copy()
        for name, poly in scene.zones.items():
            draw.polygon(vis, poly, "info", name)
        for name, (pa, pb) in scene.lines.items():
            draw.line(vis, pa, pb, "warn", name)
        for pt in current["pts"]:
            cv2.circle(vis, (int(pt[0]), int(pt[1])), 6, draw.PALETTE["alert"], -1)
        cv2.imshow(win, cv2.resize(vis, (round(w * r), round(h * r))))
        k = cv2.waitKey(30) & 0xFF
        if k in (ord("z"), ord("l")):
            commit()
            current.update(kind="zone" if k == ord("z") else "line", name=input("name: ").strip() or "zone", pts=[])
        elif k == ord("u") and current["pts"]:
            current["pts"].pop()
        elif k == ord("s"):
            commit()
            save_scene(Path(a.out), scene)
            print(f"saved {a.out}: zones {list(scene.zones)}, lines {list(scene.lines)}")
            break
        elif k == ord("q"):
            break
    cv2.destroyAllWindows()
    return 0
