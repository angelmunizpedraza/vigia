"""Pixelate faces before a frame is saved (YuNet from OpenCV Zoo, MIT). Live preview is left as is.

Faces are searched at 960 px on the long side (20 ms on a Core Ultra 7 258V, 13 of the 14 faces found at 1280 px):
faces smaller than that are a few pixels wide in a 720 px GIF and not recognisable.
YuNet misses faces seen in profile or from above, so the head area of every detected person is pixelated too, and
so is the lower part of every detected vehicle, where number plates are. When pose keypoints exist, the head they
place is pixelated as well (someone bent over or lying has the head anywhere in the box). Regions stay hidden for a
moment (`hold`, seconds) so a frame where the detector misses someone does not show their face.
"""
from __future__ import annotations

import cv2
import numpy as np

from .models import ensure


class FaceBlur:
    def __init__(self, max_side: int = 960, score: float = 0.5, grow: float = 0.25, hold: float = 0.4):
        self._det = cv2.FaceDetectorYN.create(str(ensure("yunet")), "", (320, 320), score, 0.3, 5000)
        self.max_side, self.grow, self.hold = max_side, grow, hold
        self._recent: list[tuple[float, list]] = []  # (t, regions) of the last `hold` seconds

    def faces(self, frame: np.ndarray) -> np.ndarray:
        """Face boxes (x, y, w, h) in frame pixels. Detects on a downscaled copy for large frames."""
        h, w = frame.shape[:2]
        r = min(1.0, self.max_side / max(h, w))
        img = cv2.resize(frame, (round(w * r), round(h * r))) if r < 1 else frame
        self._det.setInputSize((img.shape[1], img.shape[0]))
        _, found = self._det.detect(img)
        return np.zeros((0, 4)) if found is None else found[:, :4] / r

    def __call__(self, frame: np.ndarray, people: list = (), vehicles: list = (), heads: list = (),
                 t: float | None = None) -> np.ndarray:
        """Pixelate YuNet faces, the head area of each person box, the plate area of each vehicle box and each head box
        (x1, y1, x2, y2), plus what was pixelated in the last `hold` seconds when `t` is given."""
        h, w = frame.shape[:2]
        g = self.grow
        boxes = [(x - g * fw, y - g * fh, x + (1 + g) * fw, y + (1 + g) * fh) for x, y, fw, fh in self.faces(frame)]
        boxes += [hb for b in people for hb in head_boxes(b)]
        boxes += [plate_box(b) for b in vehicles]
        boxes += list(heads)
        if t is not None:
            self._recent = [(rt, r) for rt, r in self._recent if t - rt <= self.hold] + [(t, boxes)]
            boxes = [b for _, r in self._recent for b in r]
        for bx1, by1, bx2, by2 in boxes:
            x1, y1, x2, y2 = max(0, int(bx1)), max(0, int(by1)), min(w, int(bx2)), min(h, int(by2))
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            roi = frame[y1:y2, x1:x2]
            k = max(2, min(x2 - x1, y2 - y1) // 6)
            small = cv2.resize(roi, (max(1, (x2 - x1) // k), max(1, (y2 - y1) // k)), interpolation=cv2.INTER_AREA)
            frame[y1:y2, x1:x2] = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
        return frame


def head_boxes(box) -> list[tuple[float, float, float, float]]:
    """Where a head can be in a person box. Upright: the top band at full width (the head is off-centre when the box
    takes in a bag or an arm). Lying (box clearly wider than tall): both ends."""
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    if bw > 1.2 * bh:
        side = min(bh, bw * 0.25)  # the head of a lying person is within the outer quarter of the body
        return [(x1, y1, x1 + side, y2), (x2 - side, y1, x2, y2)]
    side = min(bh * 0.3, bw * 0.8)
    return [(x1, y1 - side * 0.1, x2, y1 + side * 1.1)]


def plate_box(box) -> tuple[float, float, float, float]:
    """Number plates sit low on a vehicle, at the front or the back: the bottom 45 % of its box at full width."""
    x1, y1, x2, y2 = box
    return (x1, y2 - 0.45 * (y2 - y1), x2, y2)


def keypoint_head(kp: np.ndarray, box) -> tuple[float, float, float, float] | None:
    """A square around the face keypoints (nose, eyes, ears) that are at least a little confident, sized from the
    body box so it covers the whole head; None when no face point is visible."""
    pts = kp[:5][kp[:5, 2] >= 0.2, :2]
    if len(pts) == 0:
        return None
    cx, cy = pts.mean(0)
    r = max(0.12 * max(box[2] - box[0], box[3] - box[1]), 0.75 * float(np.ptp(pts, axis=0).max()))
    return (float(cx - r), float(cy - r), float(cx + r), float(cy + r))
