"""Which fingers are raised, from the joint angles of MediaPipe's 21 hand landmarks, debounced into a 5-bit mask
(thumb..pinky) for five LEDs.

Only angles and distances between the hand's own points are used, never "tip x left of joint x". So a hand turned
sideways or upside down, and a left hand (a mirror image of a right one), give the same answer.
"""
from __future__ import annotations

import numpy as np

from vigia.core.types import Event

FINGERS = ("thumb", "index", "middle", "ring", "pinky")
STRAIGHT_DEG = 155.0   # PIP angle from which a finger counts as straight (sample: raised >= 164, bent <= 147)
MIN_REACH = 0.8        # tip beyond the PIP, away from the wrist, in proximal-phalanx lengths (raised >= 1.0)
THUMB_IP_DEG = 140.0   # thumb IP angle from which the thumb counts as straight
THUMB_CLEAR = 0.4      # thumb tip to the nearest joint of the other fingers, in palm lengths (up >= 0.51, down <= 0.29)


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle at b, in degrees: 180 for a straight joint."""
    u, v = a - b, c - b
    cos = u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def fingers_up(landmarks: np.ndarray, handedness: str) -> tuple[int, int, int, int, int]:
    """1 per raised finger, thumb..pinky. `landmarks` is (21, 2) or (21, 3) in MediaPipe order, with x and y in the
    same unit (pixels, not 0-1 coordinates of a non-square frame, which would squash the angles).

    `handedness` ("Left"/"Right") is part of the interface but not needed: every test here is an angle or a
    distance, which a mirror image keeps, so both hands use the same rule.

    A finger (index..pinky) is up when its middle joint (PIP) is straight and its tip reaches well past that joint,
    away from the wrist. The second test catches a finger curled towards the camera, which looks straight in 2D but
    short. The thumb is up when its IP joint is straight and its tip is clear of the other fingers: a thumb folded
    across the palm or touching a fingertip is down, wherever it points.
    """
    p = np.asarray(landmarks, dtype=float)
    n = np.linalg.norm
    palm = n(p[9] - p[0]) + 1e-9
    clear = min(n(p[4] - p[j]) for j in range(5, 21)) / palm
    up = [int(_angle(p[2], p[3], p[4]) >= THUMB_IP_DEG and clear >= THUMB_CLEAR)]
    for mcp in (5, 9, 13, 17):
        pip, dip, tip = mcp + 1, mcp + 2, mcp + 3
        reach = (n(p[tip] - p[0]) - n(p[pip] - p[0])) / (n(p[pip] - p[mcp]) + 1e-9)
        up.append(int(_angle(p[mcp], p[pip], p[dip]) >= STRAIGHT_DEG and reach >= MIN_REACH))
    return tuple(up)


class GestureLogic:
    """The LED mask changes only after the same raw mask is seen `stable_frames` frames in a row. No hand counts as
    all fingers down, so the LEDs go off when the hand leaves."""

    def __init__(self, stable_frames: int = 3):
        self.stable_frames = max(1, stable_frames)
        self.mask = "00000"   # what the LEDs show
        self.raw = "00000"    # this frame's reading
        self.hand = False
        self._run = 0

    def update(self, t: float, fingers: tuple[int, ...] | None) -> list[Event]:
        self.hand = fingers is not None
        raw = "".join(str(int(f)) for f in fingers) if fingers is not None else "00000"
        self._run = self._run + 1 if raw == self.raw else 1
        self.raw = raw
        if self._run >= self.stable_frames and raw != self.mask:
            self.mask = raw
            return [Event(t, "leds", data={"mask": raw, "fingers": [f for f, b in zip(FINGERS, raw) if b == "1"]})]
        return []
