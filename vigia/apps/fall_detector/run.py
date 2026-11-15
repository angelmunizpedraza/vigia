"""vigia fall --source "samples/fall_detector/.../Subject 1/Fall/05.mp4" --out docs/gifs/fall.gif"""
from __future__ import annotations

import argparse

import numpy as np

from vigia.core import draw
from vigia.core.detect import Detector
from vigia.core.pose import L_HIP, L_SHOULDER, R_HIP, R_SHOULDER, Pose, midpoint
from vigia.core.scene import Scene
from vigia.core.track import Tracker, to_tracks
from vigia.core.types import Event, Track

from .logic import FallLogic

STATUS = {"STANDING": "ok", "WALKING": "ok", "SITTING": "info", "LYING": "info", "FALLING": "warn", "FALL": "alert"}


def mmss(s: float) -> str:
    return f"{int(s // 60)}:{int(s % 60):02d}"


class FallApp:
    name = "fall"
    blur_faces = True

    @staticmethod
    def add_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--descent-speed", type=float, default=0.4, help="torso drop speed (body heights/s) that starts FALLING")
        p.add_argument("--lying-angle", type=float, default=60.0, help="torso angle from vertical (deg) that counts as down")
        p.add_argument("--confirm-after", type=float, default=1.5, help="seconds down after a fast descent before FALL")
        p.add_argument("--recover-window", type=float, default=1.5, help="seconds after a fast descent to end up down")

    def setup(self, scene: Scene | None, fps: float, size: tuple[int, int], args: argparse.Namespace) -> None:
        if getattr(self, "det", None) is None:  # scripts/eval_fall.py calls setup once per clip: load models once
            self.det = Detector(args.model, labels={"person"}, device=args.device)
            self.pose = Pose()
        self.tracker = Tracker(fps)
        self.logic = FallLogic(args.descent_speed, args.lying_angle, args.confirm_after, args.recover_window)
        self.tracks: list[Track] = []
        self.last_fall: float | None = None

    def process(self, t: float, frame: np.ndarray) -> list[Event]:
        dets = self.tracker.update(self.det(frame))
        self.tracks = to_tracks(dets, self.pose(frame, dets.xyxy))
        events = self.logic.update(t, self.tracks)
        if events:
            self.last_fall = t
        return events

    def draw(self, frame: np.ndarray) -> np.ndarray:
        for tr in self.tracks:
            st = self.logic.status.get(tr.id)
            if st is None:
                continue
            status = STATUS[st.state]
            if st.source == "pose":
                s = midpoint(tr.keypoints, L_SHOULDER, R_SHOULDER)
                h = midpoint(tr.keypoints, L_HIP, R_HIP)
                draw.line(frame, s, h, status)
            label = "FALL DETECTED" if st.state == "FALL" else st.state
            draw.box(frame, tr.box, label, status)
        states = self.logic.counts()
        alarm = states.get("FALL", 0)
        draw.panel(frame, "Fall detector", [
            ("People", str(len(self.logic.status)), "info"),
            ("Now", "FALL DETECTED" if alarm else ("FALLING" if states.get("FALLING") else "no fall"),
             "alert" if alarm else ("warn" if states.get("FALLING") else "ok")),
            ("Falls", f"{self.logic.falls}" + (f", last at {mmss(self.last_fall)}" if self.last_fall is not None else ""),
             "alert" if self.logic.falls else "muted")], corner="tr")
        return frame

    def summary(self) -> dict:
        return {"model": self.det.name, "falls": self.logic.falls,
                "people": {k: s.state for k, s in self.logic.status.items()}}


APP = FallApp
