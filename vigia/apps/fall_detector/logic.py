"""Did someone fall, or did they sit or lie down on purpose? Per tracked person, from pose keypoints (or the box when
they are missing): the torso angle from vertical, how fast the torso centre drops in body heights per second, and how
far the box is below the person's own standing height.

A fall is a fast descent followed by staying down. Lying down slowly is LYING, never an alarm, and getting back up
soon after a fast descent cancels it.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

from vigia.core.pose import L_HIP, L_SHOULDER, R_HIP, R_SHOULDER, midpoint
from vigia.core.types import Event, Track


@dataclass
class FallStatus:
    state: str = "STANDING"        # STANDING | WALKING | SITTING | FALLING | LYING | FALL
    since: float = 0.0             # when the current state began
    angle: float = 0.0             # torso angle from vertical in degrees (box angle when source == "box")
    speed: float = 0.0             # downward speed of the torso centre, body heights per second
    height: float = 1.0            # box height / own standing height
    source: str = "box"            # "pose" when angle and centre come from shoulders and hips
    baseline: float = 0.0          # standing box height in px
    last_seen: float = 0.0
    descent_at: float | None = None  # start of the fast descent being checked
    peak_speed: float = 0.0
    down_since: float | None = None
    up_since: float | None = None
    center: tuple[float, float] = (0.0, 0.0)          # box centre, to hand the status over after an id switch
    history: deque = field(default_factory=deque)    # (t, torso centre or None, box centre)
    standing: deque = field(default_factory=deque)   # (t, box height) while standing


class FallLogic:
    def __init__(self, descent_speed: float = 0.4, lying_angle: float = 60.0, confirm_after: float = 1.5,
                 recover_window: float = 1.5, sit_ratio: float = 0.8, down_ratio: float = 0.45,
                 walk_speed: float = 0.3, speed_window: float = 1.0, up_hold: float = 0.5,
                 forget_after: float = 3.0, min_conf: float = 0.3):
        """descent_speed: body heights/s that start FALLING. lying_angle: degrees from vertical that count as down.
        confirm_after: seconds down after a fast descent before FALL. recover_window: seconds after a fast descent
        to end up down; still upright by then means a stumble or a brisk sit, not a fall. sit_ratio / down_ratio:
        box height vs standing height below which someone is sitting / down. up_hold: seconds upright that cancel."""
        self.descent_speed, self.lying_angle = descent_speed, lying_angle
        self.confirm_after, self.recover_window = confirm_after, recover_window
        self.sit_ratio, self.down_ratio, self.walk_speed = sit_ratio, down_ratio, walk_speed
        self.speed_window, self.up_hold, self.forget_after, self.min_conf = speed_window, up_hold, forget_after, min_conf
        self.status: dict[int, FallStatus] = {}
        self.falls = 0

    def measure(self, tr: Track) -> tuple[float, float, float, str]:
        """(angle, x, y, source): torso angle and centre from shoulders/hips, else the box (angle of its diagonal)."""
        kp = tr.keypoints
        if kp is not None:
            s = midpoint(kp, L_SHOULDER, R_SHOULDER, self.min_conf)
            h = midpoint(kp, L_HIP, R_HIP, self.min_conf)
            # a torso shorter than 15 % of the box's long side is foreshortened or a bad pose: the box says more
            if s and h and math.dist(s, h) >= 0.15 * max(tr.w, tr.h):
                angle = math.degrees(math.atan2(abs(s[0] - h[0]), h[1] - s[1]))  # 0 upright, 90 flat, 180 head-down
                return angle, (s[0] + h[0]) / 2, (s[1] + h[1]) / 2, "pose"
        return math.degrees(math.atan2(tr.w, tr.h)), tr.center[0], tr.center[1], "box"

    def _speeds(self, st: FallStatus, t: float, torso, box) -> tuple[float, float]:
        """Fastest downward speed of the torso centre over any span of 0.15 s to `speed_window` s (spans across
        missed detections count, so a fall the detector half-missed still shows), and the sideways speed, both in
        body heights per second. The box centre stands in when either end of a span has no torso."""
        while st.history and t - st.history[0][0] > self.speed_window + 1e-6:
            st.history.popleft()
        down, side = 0.0, 0.0
        for ht, htorso, hbox in st.history:
            dt = t - ht
            if dt >= 0.15:
                a, b = (htorso, torso) if htorso and torso else (hbox, box)
                down = max(down, (b[1] - a[1]) / dt)
                side = max(side, abs(b[0] - a[0]) / dt)
        st.history.append((t, torso, box))
        return down / st.baseline, side / st.baseline

    def _posture(self, st: FallStatus, down: bool, sideways: float) -> str:
        if down:
            return "LYING"
        if st.height < self.sit_ratio:
            return "SITTING"
        return "WALKING" if sideways >= self.walk_speed else "STANDING"

    def _set(self, st: FallStatus, state: str, t: float) -> None:
        if state != st.state:
            st.state, st.since = state, t

    def update(self, t: float, tracks: list[Track]) -> list[Event]:
        events = []
        for tr in tracks:
            if tr.label != "person":
                continue
            st = self.status.get(tr.id) or self._handover(tr, t)
            if st is None:
                st = self.status[tr.id] = FallStatus(since=t, baseline=max(tr.h, tr.w, 1.0))
            st.last_seen, st.center = t, tr.center
            st.angle, x, y, st.source = self.measure(tr)
            st.height = tr.h / st.baseline
            st.speed, sideways = self._speeds(st, t, (x, y) if st.source == "pose" else None, tr.center)
            # bent over with the legs straight (box still near standing height) is not down
            down = (st.angle >= self.lying_angle and st.height < self.sit_ratio) or st.height <= self.down_ratio
            posture = self._posture(st, down, sideways)

            if st.state == "FALLING":
                st.peak_speed = max(st.peak_speed, st.speed)
                if down:
                    st.down_since = st.down_since if st.down_since is not None else t
                    st.up_since = None
                elif st.down_since is not None:
                    st.up_since = st.up_since if st.up_since is not None else t
                if down and t - st.down_since >= self.confirm_after:
                    self._set(st, "FALL", t)
                    self.falls += 1
                    events.append(Event(t, "fall_detected", tr.id, {
                        "descent_at": round(st.descent_at, 2), "peak_speed": round(st.peak_speed, 2),
                        "torso_angle": round(st.angle), "source": st.source}))
                elif st.down_since is None and t - st.descent_at > self.recover_window:
                    self._set(st, posture, t)       # never went down: a stumble, a bend or a brisk sit
                elif st.up_since is not None and t - st.up_since >= self.up_hold:
                    self._set(st, posture, t)       # got back up before the alarm
            elif st.state == "FALL":
                st.up_since = None if down else (st.up_since if st.up_since is not None else t)
                if st.up_since is not None and t - st.up_since >= self.up_hold:
                    self._set(st, posture, t)
            elif st.speed >= self.descent_speed:
                self._set(st, "FALLING", t)
                st.descent_at, st.peak_speed = t, st.speed
                st.down_since, st.up_since = (t if down else None), None
            else:
                self._set(st, posture, t)

            if st.state in ("STANDING", "WALKING") and st.angle < 30:
                st.standing.append((t, tr.h))
                while t - st.standing[0][0] > 3.0:
                    st.standing.popleft()
                st.baseline = max(h for _, h in st.standing)
        for tid in [k for k, s in self.status.items() if t - s.last_seen > self.forget_after]:
            del self.status[tid]
        return events

    def _handover(self, tr: Track, t: float) -> FallStatus | None:
        """A new id next to someone lost moments ago is the same person (the tracker loses people mid-fall):
        it keeps their state, history and standing height."""
        lost = [(k, s) for k, s in self.status.items() if s.last_seen < t
                and math.dist(s.center, tr.center) <= s.baseline]
        if not lost:
            return None
        k, st = max(lost, key=lambda ks: ks[1].last_seen)
        self.status[tr.id] = self.status.pop(k)
        return st

    def counts(self) -> dict[str, int]:
        """How many tracked people are in each state right now."""
        out: dict[str, int] = {}
        for s in self.status.values():
            out[s.state] = out.get(s.state, 0) + 1
        return out
