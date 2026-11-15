"""vigia drowsy --source samples/drowsiness/man-yawning.mp4 --out docs/gifs/drowsy.gif"""
from __future__ import annotations

import argparse

import cv2
import numpy as np

from vigia.core import draw
from vigia.core.models import ensure
from vigia.core.scene import Scene
from vigia.core.types import Event, Track
from vigia.core.video import SourceError

from .board import SerialBoard, VirtualBoard
from .logic import LEFT_EYE, RIGHT_EYE, DrowsyLogic, eye_aspect_ratio

# The only blendshapes read. Nothing else: no expression or emotion is inferred (EU AI Act art. 5(1)(f)).
BLENDSHAPES = ("eyeBlinkLeft", "eyeBlinkRight", "jawOpen")
STATUS = {"CALIBRATING": "info", "EYES OPEN": "ok", "EYES CLOSED": "warn", "ALARM": "alert", "NO FACE": "muted"}


class DrowsyApp:
    name = "drowsy"
    blur_faces = False  # the eyes are the subject; the sample is a stock actor (Pexels licence)

    @staticmethod
    def add_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--closed-alarm", type=float, default=1.5, help="seconds with eyes closed before the alarm")
        p.add_argument("--calib-seconds", type=float, default=5.0, help="seconds to learn the open-eye EAR")
        p.add_argument("--blink-threshold", type=float, default=0.5,
                       help="eyes count as closed when the mean eyeBlink score reaches this (0-1)")
        p.add_argument("--yawn-open", type=float, default=0.25, help="jawOpen score that counts as a wide-open jaw")
        p.add_argument("--yawn-seconds", type=float, default=0.15, help="seconds the jaw must stay open for a yawn")
        p.add_argument("--signal", default="blink", choices=["blink", "ear"],
                       help="blink: eyeBlink blendshapes (EAR only if they are missing); ear: calibrated EAR only")
        p.add_argument("--board", default="virtual", help="virtual (drawn on screen), none, or serial:<port>, "
                                                          "e.g. serial:COM3 or serial:/dev/ttyACM0")

    def setup(self, scene: Scene | None, fps: float, size: tuple[int, int], args: argparse.Namespace) -> None:
        try:
            import mediapipe as mp
        except ImportError as e:
            raise SourceError("drowsy needs MediaPipe: uv sync --extra face") from e
        self.board = self._board(args.board)
        vision = mp.tasks.vision
        opts = vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(ensure("face-landmarker"))),
            running_mode=vision.RunningMode.VIDEO, num_faces=1, output_face_blendshapes=True)
        self.mp = mp
        self.landmarker = vision.FaceLandmarker.create_from_options(opts)
        self.logic = DrowsyLogic(args.closed_alarm, args.calib_seconds, blink_threshold=args.blink_threshold,
                                 yawn_open=args.yawn_open, yawn_seconds=args.yawn_seconds)
        self.use_blink = args.signal == "blink"
        self.tracks: list[Track] = []  # no person boxes: this demo never pixelates
        self.t, self._ts = 0.0, -1
        self.pts: np.ndarray | None = None
        self.blink: float | None = None
        self.ear: float | None = None
        self.alarms = 0
        self.face_frames = self.frames = 0

    @staticmethod
    def _board(spec: str):
        if spec == "none":
            return None
        if spec == "virtual":
            return VirtualBoard()
        if not spec.startswith("serial:") or len(spec) <= len("serial:"):
            raise SourceError(f"--board must be virtual, none or serial:<port>, not {spec!r}")
        port = spec[len("serial:"):]
        try:
            return SerialBoard(port)
        except ImportError as e:
            raise SourceError("--board serial needs pyserial: uv sync --extra arduino") from e
        except (OSError, ValueError) as e:  # serial.SerialException is an OSError
            raise SourceError(f"cannot open serial port {port}: {e}") from e

    def process(self, t: float, frame: np.ndarray) -> list[Event]:
        self.t = t
        self.frames += 1
        self._ts = max(int(t * 1000), self._ts + 1)  # VIDEO mode needs strictly increasing milliseconds
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.landmarker.detect_for_video(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb), self._ts)
        bl = br = jaw = None
        self.pts = self.ear = self.blink = None
        if res.face_landmarks:
            self.face_frames += 1
            h, w = frame.shape[:2]
            self.pts = np.array([(p.x * w, p.y * h) for p in res.face_landmarks[0]])
            self.ear = (eye_aspect_ratio(self.pts[list(LEFT_EYE)]) + eye_aspect_ratio(self.pts[list(RIGHT_EYE)])) / 2
            scores = {c.category_name: c.score for c in (res.face_blendshapes or [[]])[0]
                      if c.category_name in BLENDSHAPES}
            jaw = scores.get("jawOpen")
            if self.use_blink:
                bl, br = scores.get("eyeBlinkLeft"), scores.get("eyeBlinkRight")
                if bl is not None and br is not None:
                    self.blink = (bl + br) / 2
        events = self.logic.update(t, bl, br, self.ear, jaw)
        self.alarms += sum(e.kind == "eyes_closed_alarm" for e in events)
        if self.board:
            self.board.update(self.logic.state == "ALARM")
        return events

    def _label(self) -> str:
        lg, closed = self.logic, self.logic.closed_for(self.t)
        if lg.state == "ALARM":
            return f"ALARM: EYES CLOSED {closed:.1f} s"
        if lg.state == "EYES CLOSED":
            return f"EYES CLOSED {closed:.1f} s"
        return lg.state

    def draw(self, frame: np.ndarray) -> np.ndarray:
        lg = self.logic
        status = STATUS[lg.state]
        if self.pts is not None:
            s = max(0.5, frame.shape[0] / 720)
            for eye in (LEFT_EYE, RIGHT_EYE):
                cv2.polylines(frame, [self.pts[list(eye)].round().astype(np.int32)], True, draw.PALETTE[status],
                              max(2, round(1.5 * s)), cv2.LINE_AA)
            (x1, y1), (x2, y2) = self.pts.min(axis=0), self.pts.max(axis=0)
            m = 0.08 * (x2 - x1)
            draw.box(frame, (x1 - m, y1 - m, x2 + m, y2 + m), self._label(), status)
            if lg.yawn_seen is not None and self.t - lg.yawn_seen <= 1.0:
                draw.chip(frame, (x1 - m, y2 + m + 40 * s), f"YAWN ({lg.yawns})", "warn")
        if self.blink is not None:
            eyes = (f"closure {self.blink:.2f} (closed at {lg.blink_threshold:.2f})",
                    "warn" if self.blink >= lg.blink_threshold else "ok")
        elif self.ear is not None:
            eyes = (f"EAR {self.ear:.2f} (closed below {lg.ear_threshold:.2f})",
                    "warn" if self.ear < lg.ear_threshold else "ok")
        else:
            eyes = ("not seen", "muted")
        draw.panel(frame, "Drowsiness", [
            ("State", self._label(), status),
            ("Eyes", *eyes),
            (f"PERCLOS {lg.perclos_window:.0f} s", f"{lg.perclos:.0%} of the time closed", "info"),
            ("Yawns", str(lg.yawns), "warn" if lg.yawns else "info")], corner="tl")
        if self.board:
            self.board.draw(frame)
        return frame

    def summary(self) -> dict:
        lg = self.logic
        out = {"frames": self.frames, "frames_with_face": self.face_frames, "alarms": self.alarms,
               "yawns": lg.yawns, "perclos": round(lg.perclos, 3), "state": lg.state,
               "ear_threshold": round(lg.ear_threshold, 3), "calibrated": lg.calibrated,
               "signal": "blink" if self.use_blink else "ear"}
        if self.board:
            out["board"] = self.board.title + (f" LOST ({self.board.lost_reason})" if self.board.error else "")
        return out


APP = DrowsyApp
