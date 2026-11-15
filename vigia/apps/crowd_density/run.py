"""vigia crowd --source samples/crowd_density/street-festival.mp4 --every 30 --seconds 24 --out docs/gifs/crowd.gif"""
from __future__ import annotations

import argparse

import cv2
import numpy as np

from vigia.core import draw
from vigia.core.detect import Detector
from vigia.core.scene import Scene
from vigia.core.track import to_tracks
from vigia.core.types import Event, Track
from vigia.core.video import SourceError

from .logic import PERSON_M, DensityLogic, homography_from_scene


class CrowdApp:
    name = "crowd"
    blur_faces = True

    @staticmethod
    def add_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--tile", type=int, default=320,
                       help="tile side in px; each tile is resized to 640, so smaller tiles zoom in on tiny people")
        p.add_argument("--conf", type=float, default=0.3, help="detection confidence (people seen from above score low)")
        p.add_argument("--cell-m", type=float, help="grid cell side in metres (default: scene params, else 2)")
        p.add_argument("--px-per-m", type=float, help="known scale of a camera looking straight down (no estimate)")
        p.add_argument("--person-m", type=float,
                       help="metres one person box spans, for the estimated scale (default: scene params, else 0.45)")
        p.add_argument("--amber", type=float, default=2.0, help="people/m2 where movement gets restricted")
        p.add_argument("--red", type=float, default=4.0, help="people/m2 where crush risk starts to rise")
        p.add_argument("--hold", type=float, default=3.0, help="seconds a cell must stay at red before an alert")

    def setup(self, scene: Scene | None, fps: float, size: tuple[int, int], args: argparse.Namespace) -> None:
        scene = scene or Scene(*size)
        prm = scene.params
        try:
            H = homography_from_scene(scene)
        except (ValueError, TypeError, np.linalg.LinAlgError) as e:
            raise SourceError(f"bad calibration in the scene (4 image points, 4 points in metres, no 3 in line): {e}")
        cell_m = args.cell_m or prm.get("cell_m", 2.0)
        px_per_m = args.px_per_m or prm.get("px_per_m")
        person_m = args.person_m or prm.get("person_m", PERSON_M)
        if min(cell_m, person_m, px_per_m or 1) <= 0 or args.tile < 32:
            raise SourceError("cell_m, person_m and px_per_m must be positive, tile at least 32 px")
        self.det = Detector(args.model, labels={"person"}, conf=args.conf, device=args.device)
        self.tile = args.tile
        self.logic = DensityLogic(scene, cell_m=cell_m, px_per_m=px_per_m, homography=H, amber=args.amber,
                                  red=args.red, hold=args.hold, person_m=person_m)
        self.tracks: list[Track] = []
        self.peak_max, self.alerts = 0.0, 0

    def process(self, t: float, frame: np.ndarray) -> list[Event]:
        self.tracks = to_tracks(self.det.tiled(frame, tile=self.tile, overlap=0.25))
        events = self.logic.update(t, self.tracks)
        self.peak_max = max(self.peak_max, self.logic.peak)
        self.alerts += len(events)
        return events

    def level_text(self, lv: str) -> str:
        lg = self.logic
        return {"ok": f"OK (under {lg.amber:g}/m2)", "warn": f"DENSE ({lg.amber:g}-{lg.red:g}/m2)",
                "alert": f"ALERT ({lg.red:g}+/m2)"}[lv]

    def draw(self, frame: np.ndarray) -> np.ndarray:
        lg = self.logic
        s = max(1, round(frame.shape[0] / 720))
        cells = {cell: np.round(lg.cell_polygon(*cell)).astype(np.int32) for cell in lg.level}
        # translucent cells: the colour says the level, the opacity grows from 0 (empty) to 0.6 (red)
        alpha = np.zeros(frame.shape[:2], np.float32)
        colour = np.zeros_like(frame)
        for cell, poly in cells.items():
            d = float(lg.grid[cell])
            if d > 0:
                cv2.fillPoly(alpha, [poly], 0.6 * min(1.0, d / lg.red) ** 0.5)
                cv2.fillPoly(colour, [poly], draw.PALETTE[lg.level[cell]])
        cv2.polylines(alpha, list(cells.values()), True, 0.35, 2 * s)
        cv2.polylines(colour, list(cells.values()), True, draw.PALETTE["muted"], 2 * s)
        a = alpha[..., None]
        frame[:] = (frame * (1 - a) + colour * a).astype(np.uint8)
        r = max(3, round(0.2 * float(np.median([tr.w for tr in self.tracks])))) if self.tracks else 3
        for tr in self.tracks:
            cv2.circle(frame, tuple(round(v) for v in tr.center), r, draw.WHITE, -1, cv2.LINE_AA)
        peak = max(cells, key=lambda c: lg.grid[c], default=None) if lg.peak > 0 else None
        for cell, poly in cells.items():
            d = float(lg.grid[cell])
            if cell == peak or d >= lg.amber / 2:
                text = f"PEAK {d:.1f}/m2" if cell == peak else f"{d:.1f}"
                draw.chip(frame, (poly[:, 0].min() + 2 * s, poly[:, 1].max() - 2 * s), text, lg.level[cell])
        est = " EST." if lg.estimated else ""
        scale = ("calibrated" if lg.H is not None else
                 f"EST. {lg.px_per_m:.0f} px/m" if lg.estimated and lg.px_per_m else
                 "EST. waiting for people" if lg.estimated else f"{lg.px_per_m:g} px/m")
        draw.panel(frame, f"Crowd density{est}", [
            ("People counted", str(lg.people), "info"),
            ("Peak cell", f"{lg.peak:.1f}/m2{est}", lg.peak_level),
            ("Level", self.level_text(lg.peak_level), lg.peak_level),
            ("Scale", f"{scale}, {lg.cell_m:g} m cells", "warn" if lg.estimated else "ok")], corner="br")
        return frame

    def summary(self) -> dict:
        lg = self.logic
        return {"model": self.det.name, "tile": self.tile, "estimated_scale": lg.estimated,
                "px_per_m": None if lg.px_per_m is None else round(lg.px_per_m, 1),
                "people_last_frame": lg.people, "peak_density_max": round(self.peak_max, 2),
                "density_alerts": self.alerts}


APP = CrowdApp
