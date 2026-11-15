"""One visual style for every demo. Colour is never the only signal: each state also prints its label.

Okabe-Ito palette (colour-blind safe), in BGR.
"""
from __future__ import annotations

import cv2
import numpy as np

PALETTE = {
    "ok": (115, 158, 0),      # bluish green
    "warn": (0, 159, 230),    # orange
    "alert": (0, 94, 213),    # vermillion
    "info": (233, 180, 86),   # sky blue
    "muted": (160, 160, 160),
}
FONT = cv2.FONT_HERSHEY_DUPLEX
WHITE, DARK = (255, 255, 255), (24, 24, 24)


def _s(frame: np.ndarray) -> float:
    return max(0.5, frame.shape[0] / 720)


def _pt(p) -> tuple[int, int]:
    return int(round(p[0])), int(round(p[1]))


def chip(frame: np.ndarray, org, text: str, status: str = "info", scale: float | None = None) -> None:
    """A filled label with white text, anchored at its bottom-left corner `org`."""
    s = scale or 0.55 * _s(frame)
    th = max(1, round(s * 1.6))
    (tw, tht), base = cv2.getTextSize(text, FONT, s, th)
    x, y = _pt(org)
    y = max(tht + 2 * th + 2, y)
    cv2.rectangle(frame, (x, y - tht - base - 4), (x + tw + 8, y + 2), PALETTE[status], -1)
    cv2.putText(frame, text, (x + 4, y - base + 1), FONT, s, WHITE, th, cv2.LINE_AA)


def box(frame: np.ndarray, xyxy, label: str = "", status: str = "info") -> None:
    x1, y1, x2, y2 = (int(round(v)) for v in xyxy)
    cv2.rectangle(frame, (x1, y1), (x2, y2), PALETTE[status], max(2, round(2 * _s(frame))), cv2.LINE_AA)
    if label:
        chip(frame, (x1, y1), label, status)


def polygon(frame: np.ndarray, pts, status: str = "info", label: str = "", alpha: float = 0.22) -> None:
    poly = np.array([_pt(p) for p in pts], np.int32)
    overlay = frame.copy()
    cv2.fillPoly(overlay, [poly], PALETTE[status])
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, dst=frame)
    cv2.polylines(frame, [poly], True, PALETTE[status], max(2, round(2 * _s(frame))), cv2.LINE_AA)
    if label:
        x, y = poly[:, 0].min(), poly[:, 1].min()
        chip(frame, (x, y), label, status)


def line(frame: np.ndarray, a, b, status: str = "info", label: str = "") -> None:
    cv2.line(frame, _pt(a), _pt(b), PALETTE[status], max(3, round(3 * _s(frame))), cv2.LINE_AA)
    if label:
        chip(frame, _pt(a), label, status)


def panel(frame: np.ndarray, title: str, rows: list[tuple[str, str, str]], corner: str = "tl") -> None:
    """Translucent dashboard: title, then rows of (label, value, status) with a coloured dot."""
    s = 0.6 * _s(frame)
    th = max(1, round(s * 1.6))
    lh = int(34 * s / 0.6)
    pad = int(14 * s / 0.6)
    texts = [cv2.getTextSize(f"{lab}  {val}", FONT, s, th)[0][0] for lab, val, _ in rows]
    width = max([cv2.getTextSize(title, FONT, s * 1.1, th)[0][0]] + texts) + 3 * pad + lh // 2
    height = pad * 2 + lh * (len(rows) + 1)
    H, W = frame.shape[:2]
    x0 = pad if corner.endswith("l") else W - width - pad
    y0 = pad if corner.startswith("t") else H - height - pad
    roi = frame[y0:y0 + height, x0:x0 + width]
    roi[:] = (roi * 0.35 + np.array(DARK) * 0.65).astype(np.uint8)
    cv2.putText(frame, title, (x0 + pad, y0 + pad + int(lh * 0.7)), FONT, s * 1.1, WHITE, th, cv2.LINE_AA)
    for i, (lab, val, status) in enumerate(rows, start=1):
        y = y0 + pad + lh * i + int(lh * 0.7)
        cv2.circle(frame, (x0 + pad + lh // 4, y - lh // 4), max(4, lh // 5), PALETTE[status], -1, cv2.LINE_AA)
        cv2.putText(frame, f"{lab}  {val}", (x0 + pad + lh // 2 + pad // 2, y), FONT, s, WHITE, th, cv2.LINE_AA)
