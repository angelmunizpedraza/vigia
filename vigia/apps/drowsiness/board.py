"""The alarm board: drawn on the frame (`VirtualBoard`, the default, no hardware) or a real Arduino over USB serial
(`SerialBoard`, sketch in firmware/drowsiness/).

Serial protocol, 9600 baud, one ASCII line per message: `ALARM` or `OK` as soon as the state changes, and every
0.5 s `HB` followed by the current state again. The repeat matters: opening the port resets an Arduino Uno, and lines
sent while it boots are lost, so the board catches up within half a second. If the board hears nothing for 2 s it
blinks both LEDs: the computer has stopped watching.
"""
from __future__ import annotations

import math
import time

import cv2
import numpy as np

from vigia.core import draw


class VirtualBoard:
    """Green LED on while all is well, red LED and buzzer on during an alarm, as the real sketch does."""

    title = "Board: virtual"

    def __init__(self) -> None:
        self.alarm = False
        self.error = ""

    def update(self, alarm: bool, now: float | None = None) -> None:
        self.alarm = alarm

    def draw(self, frame: np.ndarray) -> None:
        s = max(0.5, frame.shape[0] / 720)
        fs, th = 0.55 * s, max(1, round(0.55 * s * 1.6))
        r, pad, lh = int(12 * s), int(12 * s), int(36 * s)
        rows = [("Green LED (pin 7)", not self.alarm, "ok"), ("Red LED (pin 8)", self.alarm, "alert"),
                ("Buzzer (pin 9)", self.alarm, "alert")]
        texts = [f"{name}  {'ON' if on else 'off'}" for name, on, _ in rows]
        title = f"{self.title}  {self.error}".strip()
        tw = max(cv2.getTextSize(x, draw.FONT, fs, th)[0][0] for x in texts + [title])
        icon_w = 2 * r + int(24 * s)  # room for the buzzer's sound waves
        W, H = tw + icon_w + 3 * pad, lh * (len(rows) + 1) + 2 * pad
        x0, y0 = frame.shape[1] - W - pad, pad
        roi = frame[y0:y0 + H, x0:x0 + W]
        roi[:] = (roi * 0.35 + np.array(draw.DARK) * 0.65).astype(np.uint8)
        cv2.putText(frame, title, (x0 + pad, y0 + pad + int(lh * 0.65)), draw.FONT, fs,
                    draw.PALETTE["alert"] if self.error else draw.WHITE, th, cv2.LINE_AA)
        off = (70, 70, 70)
        for i, ((name, on, status), text) in enumerate(zip(rows, texts), start=1):
            cy = y0 + pad + lh * i + lh // 2
            cx = x0 + pad + r
            colour = draw.PALETTE[status] if on else off
            if name.startswith("Buzzer"):
                cv2.circle(frame, (cx, cy), r, (25, 25, 25), -1, cv2.LINE_AA)
                cv2.circle(frame, (cx, cy), r, colour, max(2, r // 5), cv2.LINE_AA)
                cv2.circle(frame, (cx, cy), max(2, r // 4), colour, -1, cv2.LINE_AA)
                if on:
                    for k in (1.4, 1.8):
                        cv2.ellipse(frame, (cx, cy), (int(r * k), int(r * k)), 0, -40, 40, colour,
                                    max(2, r // 6), cv2.LINE_AA)
            else:
                cv2.circle(frame, (cx, cy), r, colour, -1, cv2.LINE_AA)
                if on:  # a highlight so a lit LED reads as lit, not just as a coloured dot
                    cv2.circle(frame, (cx - r // 3, cy - r // 3), max(2, r // 4), (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(frame, (cx, cy), r, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(frame, text, (x0 + pad + icon_w + pad // 2, cy + int(lh * 0.2)), draw.FONT, fs,
                        draw.WHITE, th, cv2.LINE_AA)

    def close(self) -> None:
        pass


class SerialBoard(VirtualBoard):
    """Sends the state to an Arduino and also mirrors it on screen. `port` is COM3, /dev/ttyACM0 or a pyserial
    URL such as loop:// (tests)."""

    heartbeat = 0.5

    def __init__(self, port: str, baud: int = 9600):
        super().__init__()
        import serial  # optional extra: uv sync --extra arduino

        self.title = f"Board: {port}"
        self._serial_error = serial.SerialException
        self.ser = serial.serial_for_url(port, baudrate=baud, timeout=0, write_timeout=0.5)
        self._beat = -math.inf
        self._sent: bytes | None = None
        self.lost_reason = ""

    def update(self, alarm: bool, now: float | None = None) -> None:
        super().update(alarm)
        if self.error:
            return
        now = time.monotonic() if now is None else now  # wall clock: the board's 2 s timeout is wall clock too
        msg = b"ALARM\n" if alarm else b"OK\n"
        try:
            beat = now - self._beat >= self.heartbeat
            if beat:
                self.ser.write(b"HB\n")
                self._beat = now
            if beat or msg != self._sent:
                self.ser.write(msg)
                self._sent = msg
        except self._serial_error as e:  # unplugged mid-run: keep watching, say it on screen and in the summary
            self.error = "LOST"
            self.lost_reason = str(e)

    def close(self) -> None:
        self.ser.close()
