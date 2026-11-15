"""vigia exam --source samples/exam_monitor/exam-room.mp4 --every 4 --out docs/gifs/exam.gif"""
from __future__ import annotations

import argparse

import numpy as np
import supervision as sv

from vigia.core import draw
from vigia.core.detect import Detector
from vigia.core.pose import Pose
from vigia.core.scene import Scene
from vigia.core.track import Tracker, to_tracks
from vigia.core.types import Event, Track

from .logic import STATES, ExamLogic

STATUS = {"NORMAL": "ok", "REVIEW": "warn", "FLAGGED": "alert"}


def mmss(s: float) -> str:
    return f"{int(s // 60)}:{int(s % 60):02d}"


class ExamApp:
    name = "exam"
    blur_faces = True

    @staticmethod
    def add_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--person-conf", type=float, default=0.4, help="detector confidence for people")
        p.add_argument("--phone-conf", type=float, default=0.4, help="detector confidence for phones (small objects)")
        p.add_argument("--side-look-deg", type=float, default=35.0, help="head turn that counts as a side look")
        p.add_argument("--review", type=float, default=3.0, help="seconds of evidence before REVIEW")
        p.add_argument("--flagged", type=float, default=8.0, help="seconds of evidence before FLAGGED")

    def setup(self, scene: Scene | None, fps: float, size: tuple[int, int], args: argparse.Namespace) -> None:
        self.person_conf, self.phone_conf = args.person_conf, args.phone_conf
        self.det = Detector(args.model, labels={"person", "cell phone"}, conf=min(self.person_conf, self.phone_conf),
                            device=args.device)
        self.pose = Pose()
        self.tracker = Tracker(fps)
        self.logic = ExamLogic(side_look_deg=args.side_look_deg, review=args.review, flagged=args.flagged)
        self.tracks: list[Track] = []
        self.phones: list[Track] = []

    def process(self, t: float, frame: np.ndarray) -> list[Event]:
        d = self.det(frame)
        names = d.data["class_name"]
        people = d[(names == "person") & (d.confidence >= self.person_conf)]
        # D-FINE has no NMS step: drop a person box lying mostly inside a more confident one (upper-body duplicates)
        people = self.tracker.update(people.with_nms(0.8, overlap_metric=sv.OverlapMetric.IOS))
        self.tracks = to_tracks(people, self.pose(frame, people.xyxy))
        self.phones = to_tracks(d[(names == "cell phone") & (d.confidence >= self.phone_conf)])
        return self.logic.update(t, self.tracks + self.phones)

    def draw(self, frame: np.ndarray) -> np.ndarray:
        for ph in self.phones:
            draw.box(frame, ph.box, status="warn")
        shown = []
        for tr in self.tracks:
            st = self.logic.students.get(tr.id)
            if st is not None:
                draw.box(frame, tr.box, status=STATUS[st.state])
                shown.append((STATES.index(st.state), bool(st.signals), tr, st))
        # labels after every box, the most important last, so no box line or other label covers a student number
        for _, _, tr, st in sorted(shown, key=lambda s: s[:2]):
            signals = f" {' + '.join(st.signals)}" if st.signals else ""
            draw.chip(frame, tr.box[:2], f"#{tr.id} {st.state}{signals}", STATUS[st.state])
        counts = self.logic.counts()
        draw.panel(frame, "Students", [(s.title(), str(counts[s]), STATUS[s]) for s in STATES], corner="tl")
        rows = [(mmss(e.data["from"]), f"#{e.track_id} {e.kind.replace('_started', '')}: "
                 f"{', '.join(e.data['signals']).lower()}", "warn" if e.kind == "review_started" else "alert")
                for e in self.logic.reviews[-4:]]
        draw.panel(frame, "For human review", rows or [("nothing yet", "", "muted")], corner="tr")
        return frame

    def summary(self) -> dict:
        return {"model": self.det.name,
                "reviews": [{"t": round(e.t, 2), "from": round(e.data["from"], 2), "student": e.track_id,
                             "kind": e.kind, "signals": e.data["signals"]} for e in self.logic.reviews],
                "students": {k: {"state": s.state, "score": round(s.score, 1)} for k, s in self.logic.students.items()}}


APP = ExamApp
