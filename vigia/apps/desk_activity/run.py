"""vigia desk --source samples/desk_activity/office.mp4 --out docs/gifs/desk.gif --every 3"""
from __future__ import annotations

import argparse

import numpy as np

from vigia.core import draw
from vigia.core.detect import Detector
from vigia.core.scene import Scene
from vigia.core.track import to_tracks
from vigia.core.types import Event, Track
from vigia.core.video import SourceError

from .logic import DeskLogic

STATUS = {"WORKING": "ok", "AWAY": "muted", "PHONE": "warn"}


def mmss(s: float) -> str:
    return f"{int(s // 60)}:{int(s % 60):02d}"


class DeskApp:
    name = "desk"
    blur_faces = True

    @staticmethod
    def add_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--away-after", type=float, default=10.0, help="seconds empty before a desk counts as away")
        p.add_argument("--phone-after", type=float, default=3.0, help="seconds a phone must be seen in hand")

    def setup(self, scene: Scene | None, fps: float, size: tuple[int, int], args: argparse.Namespace) -> None:
        if scene is None or not scene.zones_like("desk"):
            raise SourceError("desk needs zones named desk-*: vigia zones <video> --out scenes/desk.json")
        self.det = Detector(args.model, labels={"person", "cell phone"}, device=args.device)
        self.logic = DeskLogic(scene, args.away_after, args.phone_after)
        self.tracks: list[Track] = []
        self.t = 0.0

    def process(self, t: float, frame: np.ndarray) -> list[Event]:
        self.t = t
        self.tracks = to_tracks(self.det(frame))
        return self.logic.update(t, self.tracks)

    def draw(self, frame: np.ndarray) -> np.ndarray:
        lg = self.logic
        seated = [p for p in self.tracks if p.label == "person" and any(lg.seated(p, z) for z in lg.desks.values())]
        for tr in self.tracks:
            if tr.label == "person":
                draw.box(frame, tr.box, status="muted")
            elif any(lg.in_hands(tr, p) for p in seated):  # phone-like objects elsewhere (sockets...) are not drawn
                draw.box(frame, tr.box, "phone", "warn")
        rows = []
        for name, poly in self.logic.desks.items():
            st = self.logic.status[name]
            short = name.removeprefix("desk-").upper()
            draw.polygon(frame, poly, STATUS[st.state], f"{short} {st.state} {mmss(max(0.0, self.t - st.since))}")
            s = st.seconds
            rows.append((short, f"{st.state:<7}  work {mmss(s['WORKING'])}  away {mmss(s['AWAY'])}  "
                                f"phone {mmss(s['PHONE'])}", STATUS[st.state]))
        used = sum(st.state != "AWAY" for st in self.logic.status.values())
        draw.panel(frame, "Desk use", [("In use now", f"{used} of {len(rows)}", "info")] + rows, corner="tr")
        return frame

    def summary(self) -> dict:
        return {"model": self.det.name,
                "desks": {k: {"state": s.state, "seconds": {n: round(v, 1) for n, v in s.seconds.items()}}
                          for k, s in self.logic.status.items()}}


APP = DeskApp
