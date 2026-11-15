"""vigia road --source samples/road_damage/pothole-kumasi.mp4 --scene scenes/road-kumasi.json --out docs/gifs/road.gif"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import supervision as sv

from vigia.core import draw
from vigia.core.detect import Detector
from vigia.core.runner import ROOT
from vigia.core.scene import Scene, point_in_polygon
from vigia.core.track import Tracker, to_tracks
from vigia.core.types import Event, Track

from .detector import RoadDetector
from .logic import DamageLogic, mostly_inside

STATUS = {"pothole": "alert", "alligator crack": "warn"}  # line cracks: "info"
VEHICLES = {"car", "bus", "truck", "motorbike", "bicycle"}


class RoadApp:
    name = "road"
    blur_faces = True

    @staticmethod
    def add_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--conf", type=float, default=0.3, help="confidence a damage needs to start being tracked")
        p.add_argument("--min-frames", type=int, default=3, help="frames a damage must be tracked before it counts")
        p.add_argument("--tile", type=int, help="cut the road area into squares of N px (zoom; default: scene's "
                                                "params.tile, else the whole frame as one image)")
        p.add_argument("--rematch", type=float, default=5.0,
                       help="seconds a counted damage keeps its spot, so a new track there is not counted again "
                            "(fixed camera; use 0 on a moving vehicle)")
        p.add_argument("--report", default=str(ROOT / "out" / "road-report.csv"), help="CSV report path")

    def setup(self, scene: Scene | None, fps: float, size: tuple[int, int], args: argparse.Namespace) -> None:
        W, H = size
        self.zone = scene.zones.get("road") if scene else None
        if self.zone:
            xs, ys = [p[0] for p in self.zone], [p[1] for p in self.zone]
            self.roi = (max(0, int(min(xs))), max(0, int(min(ys))), min(W, int(max(xs))), min(H, int(max(ys))))
        else:
            self.roi = None
        self.tile = args.tile if args.tile is not None else int(scene.params.get("tile", 0)) if scene else 0
        # Weak boxes (>= 0.15) only keep an existing track alive (ByteTrack's second pass); a new track needs
        # --conf: supervision starts tracks at activation + 0.1.
        self.det = RoadDetector(0.15, args.device)
        self.tracker = Tracker(fps, lost_seconds=2.0, activation=max(0.15, args.conf - 0.1))
        # people: their heads are pixelated in saved output; vehicles: damage seen on a car is not road damage
        self.objects = Detector(args.model, labels={"person"} | VEHICLES, device=args.device)
        self.report = Path(args.report).resolve()
        crops = self.report.with_name(self.report.stem.removesuffix("-report") + "-crops")  # out/road-crops
        crops.mkdir(parents=True, exist_ok=True)
        for old in crops.glob("damage-*.jpg"):  # crops from a previous run would not match the new report
            old.unlink()
        shown = crops.relative_to(ROOT).as_posix() if crops.is_relative_to(ROOT) else crops.as_posix()
        self.logic = DamageLogic(args.min_frames, shown, args.rematch)
        self.damage: list[Track] = []
        self.tracks: list[Track] = []
        self.blur = None
        if not args.no_blur:  # crops are shared with the report: pixelate faces and plates before cutting them
            from vigia.core.privacy import FaceBlur
            self.blur = FaceBlur()

    def process(self, t: float, frame: np.ndarray) -> list[Event]:
        objects = to_tracks(self.objects(frame))
        vehicles = [o.box for o in objects if o.label in VEHICLES]
        d = self.det(frame, self.roi, self.tile)
        if len(d):
            keep = [not mostly_inside(tuple(b), vehicles) for b in d.xyxy]
            if self.zone:
                centres = d.get_anchors_coordinates(sv.Position.CENTER)
                keep = [k and point_in_polygon(tuple(c), self.zone) for k, c in zip(keep, centres)]
            d = d[np.array(keep, bool)]
        self.damage = to_tracks(self.tracker.update(d))
        self.tracks = self.damage + [o for o in objects if o.label == "person"]
        events = self.logic.update(t, self.damage)
        for row in self.logic.improved:
            self._save_crop(frame, self.logic.report[row])
        return events

    def _save_crop(self, frame: np.ndarray, rec) -> None:
        if self.blur:
            from vigia.core.runner import people_to_blur, vehicles_to_blur
            frame = self.blur(frame.copy(), people_to_blur(self), vehicles_to_blur(self))
        H, W = frame.shape[:2]
        x1, y1, x2, y2 = rec.box
        pad = max(24.0, 0.25 * max(x2 - x1, y2 - y1))
        crop = frame[max(0, int(y1 - pad)):min(H, int(y2 + pad)), max(0, int(x1 - pad)):min(W, int(x2 + pad))]
        cv2.imwrite(str(ROOT / rec.crop_path), crop)

    def draw(self, frame: np.ndarray) -> np.ndarray:
        if self.zone:
            draw.polygon(frame, self.zone, "muted", "ROAD AREA", alpha=0.08)
        for tr in self.damage:
            row = self.logic.row_of.get(tr.id)
            if row is None:
                draw.box(frame, tr.box, f"{tr.label}?", "muted")  # not counted yet: needs more frames
            else:  # the class and score of its best view, as in the panel and the report
                rec = self.logic.report[row]
                draw.box(frame, tr.box, f"#{row + 1} {rec.label} {rec.max_score:.0%}", STATUS.get(rec.label, "info"))
        by = self.by_class()
        draw.panel(frame, "Road damage", [
            ("Counted", str(len(self.logic.report)), "info"),
            ("Potholes", str(by.get("pothole", 0)), "alert"),
            ("Alligator cracks", str(by.get("alligator crack", 0)), "warn"),
            ("Line cracks", str(by.get("longitudinal crack", 0) + by.get("transverse crack", 0)), "info"),
            ("Report", self.report.name, "muted")], corner="tl")
        return frame

    def by_class(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in self.logic.report:
            out[r.label] = out.get(r.label, 0) + 1
        return out

    def summary(self) -> dict:
        self.logic.write_csv(self.report)  # the runner calls this once at the end, also after Ctrl+C
        return {"model": self.det.name, "damage_counted": len(self.logic.report), "by_class": self.by_class(),
                "report": str(self.report)}


APP = RoadApp
