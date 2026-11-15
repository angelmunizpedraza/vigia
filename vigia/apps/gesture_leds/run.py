"""vigia hands --source samples/gesture_leds/hand-signs.mp4 --out docs/gifs/hands.gif"""
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
from .logic import FINGERS, GestureLogic, fingers_up


class HandsApp:
    name = "hands"
    blur_faces = True  # no faces expected in a hand close-up; kept on in case one comes into view

    @staticmethod
    def add_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--stable-frames", type=int, default=3,
                       help="frames a new finger reading must hold before the LEDs change")
        p.add_argument("--hands", type=int, default=1, choices=[1, 2],
                       help="hands to find; with 2, the larger (nearer) hand drives the LEDs")
        p.add_argument("--board", default="virtual", help="virtual (drawn on screen), none, or serial:<port>, "
                                                          "e.g. serial:COM3 or serial:/dev/ttyACM0")
        p.add_argument("--no-mirror", action="store_true",
                       help="show a webcam as the camera sees it (by default it is mirrored like a selfie view)")

    def setup(self, scene: Scene | None, fps: float, size: tuple[int, int], args: argparse.Namespace) -> None:
        try:
            import mediapipe as mp
        except ImportError as e:
            raise SourceError("hands needs MediaPipe: uv sync --extra face") from e
        self.board = self._board(args.board)
        vision = mp.tasks.vision
        opts = vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(ensure("hand-landmarker"))),
            running_mode=vision.RunningMode.VIDEO, num_hands=args.hands)
        self.mp = mp
        self.landmarker = vision.HandLandmarker.create_from_options(opts)
        self.bones = [(c.start, c.end) for c in vision.HandLandmarksConnections.HAND_CONNECTIONS]
        self.logic = GestureLogic(args.stable_frames)
        # MediaPipe always sees the raw camera frame, so its Left/Right is the person's real hand (checked on the
        # samples: flipping a frame flips the label). Only the drawing is mirrored, and the LEDs follow landmark
        # numbers, never screen sides, so mirroring cannot swap the thumb and pinky LEDs.
        self.mirror = str(args.source).isdigit() and not args.no_mirror
        self.tracks: list[Track] = []  # no person boxes: only faces YuNet finds are pixelated
        self.hands: list[tuple[np.ndarray, str, tuple[int, ...]]] = []  # (21x3 pixels, handedness, fingers)
        self._ts = -1
        self.frames = self.hand_frames = self.changes = 0

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
        self.frames += 1
        self._ts = max(int(t * 1000), self._ts + 1)  # VIDEO mode needs strictly increasing milliseconds
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.landmarker.detect_for_video(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb), self._ts)
        h, w = frame.shape[:2]
        self.hands = []
        for lms, cats in zip(res.hand_landmarks, res.handedness):
            p = np.array([(q.x * w, q.y * h, q.z * w) for q in lms])  # same unit on x and y, or angles get squashed
            self.hands.append((p, cats[0].category_name, fingers_up(p, cats[0].category_name)))
        # the larger hand on screen (nearer the camera) drives the LEDs
        self.hands.sort(key=lambda hd: -np.prod(np.ptp(hd[0][:, :2], axis=0)))
        self.hand_frames += bool(self.hands)
        events = self.logic.update(t, self.hands[0][2] if self.hands else None)
        for e in events:
            e.data["hand"] = self.hands[0][1] if self.hands else None
        self.changes += len(events)
        if self.board:
            self.board.update(self.logic.mask)
        return events

    def draw(self, frame: np.ndarray) -> np.ndarray:
        W = frame.shape[1]
        if self.mirror:
            frame = cv2.flip(frame, 1)
        s = max(0.5, frame.shape[0] / 720)
        for k, (p, side, fingers) in enumerate(self.hands):
            pts = p[:, :2].copy()
            if self.mirror:
                pts[:, 0] = W - 1 - pts[:, 0]
            ipts = pts.round().astype(int).tolist()  # cv2 wants plain ints
            for a, b in self.bones:
                f = (b - 1) // 4  # landmarks 1-4 are the thumb (0), 5-8 the index (1) ... 17-20 the pinky (4)
                status = "ok" if a > 0 and (a - 1) // 4 == f and fingers[f] else "muted"  # palm bones stay muted
                cv2.line(frame, tuple(ipts[a]), tuple(ipts[b]), draw.PALETTE[status],
                         max(2, round((3 if status == "ok" else 2) * s)), cv2.LINE_AA)
            for xy in ipts:
                cv2.circle(frame, tuple(xy), max(3, round(3 * s)), draw.WHITE, -1, cv2.LINE_AA)
                cv2.circle(frame, tuple(xy), max(3, round(3 * s)), draw.DARK, 1, cv2.LINE_AA)
            label = f"{side.upper()} HAND" + (" (LEDS)" if k == 0 and len(self.hands) > 1 else "")
            draw.chip(frame, (pts[:, 0].min(), pts[:, 1].min() - 8 * s), label, "info")
        bottom = self.board.height(frame) if self.board else 0
        up = self.logic.mask.count("1")
        rows = [("Hand", self.hands[0][1] if self.hands else "none", "info" if self.hands else "muted"),
                ("Fingers up", str(up), "ok" if up else "muted"),
                ("Serial line", f"LEDS {self.logic.mask}", "info")]
        if self.board:
            rows.append(("Board", f"{self.board.title} {self.board.error}".strip(),
                         "alert" if self.board.error else "info"))
        draw.panel(frame[:frame.shape[0] - bottom], "Gesture LEDs", rows, corner="bl")  # a view: drawn in place
        if self.board:
            self.board.draw(frame)
        return frame

    def summary(self) -> dict:
        out = {"frames": self.frames, "frames_with_hand": self.hand_frames, "led_changes": self.changes,
               "mask": self.logic.mask, "fingers_up": [f for f, b in zip(FINGERS, self.logic.mask) if b == "1"]}
        if self.board:
            out["board"] = self.board.title + (f" LOST ({self.board.lost_reason})" if self.board.error else "")
        return out


APP = HandsApp
