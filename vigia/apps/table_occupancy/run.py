"""vigia tables --source samples/table_occupancy/cafe-from-above.mp4 --out docs/gifs/tables.gif"""
from __future__ import annotations

import argparse

import numpy as np

from vigia.core import draw
from vigia.core.detect import Detector
from vigia.core.scene import Scene
from vigia.core.track import to_tracks
from vigia.core.types import Event, Track
from vigia.core.video import SourceError

from .logic import TableLogic


def mmss(s: float) -> str:
    return f"{int(s // 60)}:{int(s % 60):02d}"


class TableApp:
    name = "tables"
    blur_faces = True

    @staticmethod
    def add_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--occupy-after", type=float, default=3.0, help="seconds someone must stay to occupy a table")
        p.add_argument("--free-after", type=float, default=10.0, help="seconds empty before a table is free again")

    def setup(self, scene: Scene | None, fps: float, size: tuple[int, int], args: argparse.Namespace) -> None:
        if scene is None or not scene.zones_like("table"):
            raise SourceError("tables needs zones named table-*: vigia zones <video> --out scenes/tables.json")
        self.det = Detector(args.model, labels={"person"}, device=args.device)
        self.logic = TableLogic(scene, args.occupy_after, args.free_after)
        self.tracks: list[Track] = []
        self.t = 0.0

    def process(self, t: float, frame: np.ndarray) -> list[Event]:
        self.t = t
        self.tracks = to_tracks(self.det(frame))
        return self.logic.update(t, self.tracks)

    def draw(self, frame: np.ndarray) -> np.ndarray:
        for tr in self.tracks:
            draw.box(frame, tr.box, status="muted")
        free = []
        for name, poly in self.logic.tables.items():
            st = self.logic.status[name]
            occ = st.state == "OCCUPIED"
            short = name.replace("table-", "T")
            since = mmss(max(0.0, self.t - st.since)) if (occ or st.since) else ""
            draw.polygon(frame, poly, "alert" if occ else "ok", f"{short} {st.state} {since}".strip())
            if not occ:
                free.append(short)
        draw.panel(frame, "Table occupancy", [
            ("Free", f"{len(free)} of {len(self.logic.tables)}", "ok"),
            ("Free tables", ", ".join(free) or "none", "ok" if free else "alert"),
            ("Occupied time", f"{self.logic.occupancy_ratio():.0%}", "info")], corner="bl")
        return frame

    def summary(self) -> dict:
        return {"model": self.det.name, "occupancy": round(self.logic.occupancy_ratio(), 3),
                "tables": {k: {"state": s.state, "occupied_seconds": round(s.occupied_seconds, 1)}
                           for k, s in self.logic.status.items()}}


APP = TableApp
