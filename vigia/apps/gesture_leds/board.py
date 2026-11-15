"""Five LEDs, thumb..pinky: drawn on the frame (`VirtualBoard`, the default, no hardware) or on a real Arduino over
USB serial (`SerialBoard`, sketch in firmware/gestures/).

Serial protocol, 9600 baud, one ASCII line: `LEDS 10110` (thumb..pinky, 1 = on), sent as soon as the mask changes
and again every 0.5 s as a heartbeat. The repeat matters: opening the port resets an Arduino Uno, and lines sent
while it boots are lost, so the board catches up within half a second. If the board hears nothing for 2 s it turns
every LED off. The original write-up drove the board with PyFirmata2 and StandardFirmata; this one-line protocol
replaces it so the link can be tested without a board (pyserial's loop://).
"""
from __future__ import annotations

import math
import time

import cv2
import numpy as np

from vigia.core import draw

from .logic import FINGERS

PINS = (2, 3, 4, 5, 6)  # thumb..pinky, as in firmware/gestures/gestures.ino


class VirtualBoard:
    """A row of five LEDs across the bottom of the frame, each with its finger, ON/off and Arduino pin."""

    title = "virtual"

    def __init__(self) -> None:
        self.mask = "00000"
        self.error = ""

    def update(self, mask: str, now: float | None = None) -> None:
        self.mask = mask

    @staticmethod
    def _layout(frame: np.ndarray) -> tuple[float, float, int, int, int, int]:
        """(scale, font scale, thickness, LED radius, text line height, strip height) for this frame size."""
        H, W = frame.shape[:2]
        s = max(0.5, H / 720)
        slot = W / 5
        fs = min(0.45 * s, 0.8 * slot / cv2.getTextSize("middle", draw.FONT, 1, 1)[0][0])
        r = int(min(14 * s, slot / 5))
        line = int(30 * fs / 0.5)
        return s, fs, max(1, round(fs * 1.6)), r, line, int(10 * s) + 2 * r + 3 * line + int(6 * s)

    def height(self, frame: np.ndarray) -> int:
        return self._layout(frame)[-1]

    def draw(self, frame: np.ndarray) -> None:
        H, W = frame.shape[:2]
        s, fs, th, r, line, h = self._layout(frame)
        roi = frame[H - h:]
        roi[:] = (roi * 0.35 + np.array(draw.DARK) * 0.65).astype(np.uint8)
        slot = W / 5
        for i, (name, pin, bit) in enumerate(zip(FINGERS, PINS, self.mask)):
            on = bit == "1"
            cx, cy = int(slot * (i + 0.5)), H - h + int(10 * s) + r
            cv2.circle(frame, (cx, cy), r, draw.PALETTE["ok"] if on else (50, 50, 50), -1, cv2.LINE_AA)
            if on:  # a highlight so a lit LED reads as lit, not just as a coloured dot
                cv2.circle(frame, (cx - r // 3, cy - r // 3), max(2, r // 4), draw.WHITE, -1, cv2.LINE_AA)
            # an unlit green LED still looks green; this ring also puts the green in a GIF's first-frame palette
            cv2.circle(frame, (cx, cy), r, (200, 200, 200) if on else draw.PALETTE["ok"], max(2, r // 6), cv2.LINE_AA)
            for k, (text, colour) in enumerate([(name, draw.WHITE), ("ON" if on else "off", draw.WHITE),
                                                (f"pin {pin}", draw.PALETTE["muted"])]):
                tw = cv2.getTextSize(text, draw.FONT, fs, th)[0][0]
                cv2.putText(frame, text, (cx - tw // 2, cy + r + line * (k + 1) - line // 4), draw.FONT, fs,
                            colour, th, cv2.LINE_AA)

    def close(self) -> None:
        pass


class SerialBoard(VirtualBoard):
    """Sends the mask to an Arduino and also mirrors it on screen. `port` is COM3, /dev/ttyACM0 or a pyserial URL
    such as loop:// (tests)."""

    heartbeat = 0.5

    def __init__(self, port: str, baud: int = 9600):
        super().__init__()
        import serial  # optional extra: uv sync --extra arduino

        self.title = port
        self._serial_error = serial.SerialException
        self.ser = serial.serial_for_url(port, baudrate=baud, timeout=0, write_timeout=0.5)
        self._last = -math.inf
        self._sent: str | None = None
        self.lost_reason = ""

    def update(self, mask: str, now: float | None = None) -> None:
        super().update(mask)
        if self.error:
            return
        now = time.monotonic() if now is None else now  # wall clock: the board's 2 s timeout is wall clock too
        if mask == self._sent and now - self._last < self.heartbeat:
            return
        try:
            self.ser.write(f"LEDS {mask}\n".encode("ascii"))
            self._sent, self._last = mask, now
        except self._serial_error as e:  # unplugged mid-run: keep going, say it on screen and in the summary
            self.error = "LOST"
            self.lost_reason = str(e)

    def close(self) -> None:
        self.ser.close()
