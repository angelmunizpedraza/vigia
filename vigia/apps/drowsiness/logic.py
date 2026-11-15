"""Eyes-closed alarm from per-frame face signals. Only physical states (eye closure, jaw opening): no emotion
inference, which the EU AI Act bans at work and in education (art. 5(1)(f); eye closure and yawning are physical
states, recital 18)."""
from __future__ import annotations

from collections import deque

import numpy as np

from vigia.core.types import Event

# MediaPipe face-mesh points for the eye aspect ratio, p1..p6 around each eye (corner, upper lid x2, corner, lower lid x2).
LEFT_EYE = (33, 160, 158, 133, 153, 144)
RIGHT_EYE = (362, 385, 387, 263, 373, 380)


def eye_aspect_ratio(pts: np.ndarray) -> float:
    """EAR of one eye from its 6 points (Soukupova and Cech, 2016): lid gap over eye width, ~0 when closed."""
    p = np.asarray(pts, float)
    return float((np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])) / (2 * np.linalg.norm(p[0] - p[3]) + 1e-9))


class DrowsyLogic:
    def __init__(self, closed_alarm: float = 1.5, calib_seconds: float = 5.0, perclos_window: float = 60.0,
                 blink_threshold: float = 0.5, yawn_open: float = 0.25, yawn_seconds: float = 0.15,
                 reopen_seconds: float = 0.2, default_ear_threshold: float = 0.2, yawn_merge: float = 6.0):
        self.closed_alarm, self.calib_seconds, self.perclos_window = closed_alarm, calib_seconds, perclos_window
        self.blink_threshold, self.reopen_seconds = blink_threshold, reopen_seconds
        self.yawn_open, self.yawn_seconds = yawn_open, yawn_seconds
        self.yawn_merge = yawn_merge  # a yawn lasts ~6 s; a hand over the mouth splits it into two sightings
        self.ear_threshold = default_ear_threshold
        self.calibrated = False
        self.state = "NO FACE"
        self.perclos = 0.0
        self.yawns = 0
        self.yawn_seen: float | None = None  # last time an open jaw of the current or last yawn was seen
        self.closed_since: float | None = None
        self._open_since: float | None = None  # eyes read open inside a closure, not yet confirmed
        self._calib_start: float | None = None
        self._calib: list[float] = []
        self._window: deque[tuple[float, float, bool]] = deque()  # (t, dt, closed) for PERCLOS
        self._jaw_since: float | None = None
        self._yawn_start: float | None = None
        self._last_t: float | None = None

    def eyes_closed(self, blink_left, blink_right, ear) -> bool | None:
        """Blendshapes first; EAR against the calibrated threshold when they are missing; None without a face."""
        if blink_left is not None and blink_right is not None:
            return (blink_left + blink_right) / 2 >= self.blink_threshold
        if ear is not None:
            return ear < self.ear_threshold
        return None

    def closed_for(self, t: float) -> float:
        return 0.0 if self.closed_since is None else t - self.closed_since

    def update(self, t: float, blink_left: float | None = None, blink_right: float | None = None,
               ear: float | None = None, jaw_open: float | None = None) -> list[Event]:
        events: list[Event] = []
        dt = 0.0 if self._last_t is None else min(1.0, max(0.0, t - self._last_t))
        self._last_t = t
        self._calibrate(t, ear, events)

        closed = self.eyes_closed(blink_left, blink_right, ear)
        if closed:
            self._open_since = None
            if self.closed_since is None:
                self.closed_since = t
        elif closed is False and self.closed_since is not None:
            self._open_since = t if self._open_since is None else self._open_since
            if t - self._open_since >= self.reopen_seconds - 1e-9:
                if self.state == "ALARM":
                    events.append(Event(t, "alarm_cleared",
                                        data={"closed_for": round(self._open_since - self.closed_since, 2)}))
                self.closed_since = self._open_since = None
        # closed is None (no face): a closure carries on, so a head dropping out of view keeps the alarm

        if self.closed_for(t) >= self.closed_alarm - 1e-9:
            if self.state != "ALARM":
                events.append(Event(t, "eyes_closed_alarm", data={"closed_for": round(self.closed_for(t), 2)}))
            self.state = "ALARM"
        elif self.closed_since is not None:
            self.state = "EYES CLOSED"
        elif closed is None:
            self.state = "NO FACE"
        else:
            self.state = "EYES OPEN" if self.calibrated else "CALIBRATING"

        if closed is not None:
            self._window.append((t, dt, closed))
        while self._window and self._window[0][0] <= t - self.perclos_window:
            self._window.popleft()
        total = sum(d for _, d, _ in self._window)
        self.perclos = sum(d for _, d, c in self._window if c) / total if total > 0 else 0.0

        self._yawn(t, jaw_open, events)
        return events

    def _calibrate(self, t: float, ear: float | None, events: list[Event]) -> None:
        """Learn the open-eye EAR over the first `calib_seconds` with a face. The 90th percentile, not the median,
        so it still works when the eyes are closed for most of the window; if it never sees open eyes it keeps the
        default."""
        if self.calibrated or (ear is None and self._calib_start is None):
            return
        if self._calib_start is None:
            self._calib_start = t
        if ear is not None:
            self._calib.append(ear)
        if t - self._calib_start < self.calib_seconds - 1e-9:
            return
        self.calibrated = True
        open_ear = float(np.percentile(self._calib, 90)) if len(self._calib) >= 5 else 0.0
        default = open_ear < 0.18  # open eyes gave EAR 0.21-0.31 on the sample; below 0.18 it saw no open eyes
        if not default:
            self.ear_threshold = 0.75 * open_ear
        events.append(Event(t, "calibrated", data={"open_ear": round(open_ear, 3), "default": default,
                                                   "ear_threshold": round(self.ear_threshold, 3)}))
        self._calib = []

    def _yawn(self, t: float, jaw_open: float | None, events: list[Event]) -> None:
        if jaw_open is None or jaw_open < self.yawn_open:
            self._jaw_since = None
            return
        if self._jaw_since is None:
            self._jaw_since = t
        if self._yawn_start is not None and self._jaw_since - self._yawn_start < self.yawn_merge:
            self.yawn_seen = t  # the same yawn, seen again
        elif t - self._jaw_since >= self.yawn_seconds - 1e-9:
            self.yawns += 1
            self._yawn_start, self.yawn_seen = self._jaw_since, t
            events.append(Event(t, "yawn", data={"count": self.yawns}))
