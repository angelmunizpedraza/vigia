"""Exam moments a human should review: a phone at a student's hand or desk, or a sustained look to the side.

Each signal adds evidence to a per-student score (units ~ seconds of evidence) that drains away when the signal
stops, so a brief glance never reaches REVIEW. It never decides anything: it timestamps moments for a person to
watch. Looking down at the paper is not a signal. No face recognition, no emotions, no identity: ids are tracker
numbers that last as long as the track.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from vigia.core.types import Event, Track

NOSE, L_EAR, R_EAR, L_SHOULDER, R_SHOULDER = 0, 3, 4, 5, 6
L_ELBOW, R_ELBOW, L_WRIST, R_WRIST = 7, 8, 9, 10
MIN_CONF = 0.3
# Nose offset from the shoulder midpoint, in shoulder widths, at a 90-degree head turn: the nose sits ~10 cm in
# front of the neck, the shoulder keypoints are ~35-40 cm apart. Calibrated by eye on the sample clip.
NOSE_REACH = 0.5
# A turned torso also shifts the nose against the shoulders. If both ears are visible and the nose sits in the
# middle half between them, the face is towards the camera and it is not a side look.
EARS_CENTRED = 0.25
MIN_SHOULDER_SHARE = 0.25  # shoulders narrower than this share of the box: profile view or a bad box, no reading
FOREARM_PER_SHOULDER = 0.66  # body scale from a forearm when the shoulders are out of view (close-ups)
FOREARM_CONF = 0.5  # the forearm fallback needs clear keypoints: noise on a bag scores 0.1-0.35 on the sample
# Small phones flicker in and out of detection, and a pen or a calculator can pass for one in a single frame:
# a phone counts once seen twice within PHONE_HOLD seconds, and stays counted for PHONE_HOLD after a sighting.
PHONE_HOLD = 0.5
STATES = ("NORMAL", "REVIEW", "FLAGGED")
HYSTERESIS = 0.5  # a state is left only when the score falls below half its threshold: no flicker


def _ok(kp: np.ndarray, *idx: int) -> bool:
    return all(kp[i, 2] >= MIN_CONF for i in idx)


def shoulder_width(kp: np.ndarray | None) -> float | None:
    if kp is None or not _ok(kp, L_SHOULDER, R_SHOULDER):
        return None
    w = float(np.hypot(*(kp[L_SHOULDER, :2] - kp[R_SHOULDER, :2])))
    return w if w > 1 else None


def body_scale(tr: Track) -> float | None:
    """Shoulder width in pixels, or one estimated from the longer clearly visible forearm. Shoulders a sliver of the
    box wide are noise (keypoints guessed on a bag or an arm close-up) and must not shrink the phone radius."""
    kp = tr.keypoints
    if kp is None:
        return None
    sw = shoulder_width(kp)
    if sw is not None and sw >= MIN_SHOULDER_SHARE * tr.w:
        return sw
    arms = [math.dist(kp[e, :2], kp[w, :2]) for e, w in ((L_ELBOW, L_WRIST), (R_ELBOW, R_WRIST))
            if kp[e, 2] >= FOREARM_CONF and kp[w, 2] >= FOREARM_CONF]
    return max(arms) / FOREARM_PER_SHOULDER if arms and max(arms) > 1 else None


def head_turn(tr: Track) -> float | None:
    """Signed head turn in degrees (+ = towards image right) from the nose's sideways offset to the shoulder
    midpoint. Bowing the head over the paper moves the nose down, not sideways, so it reads ~0."""
    kp = tr.keypoints
    sw = shoulder_width(kp)
    if sw is None or sw < MIN_SHOULDER_SHARE * tr.w or not _ok(kp, NOSE):
        return None
    r = (kp[NOSE, 0] - (kp[L_SHOULDER, 0] + kp[R_SHOULDER, 0]) / 2) / sw
    return math.degrees(math.asin(max(-1.0, min(1.0, r / NOSE_REACH))))


def side_look(tr: Track, min_deg: float) -> bool:
    turn = head_turn(tr)
    if turn is None or abs(turn) < min_deg:
        return False
    kp = tr.keypoints
    if _ok(kp, L_EAR, R_EAR):
        lo, hi = sorted((kp[L_EAR, 0], kp[R_EAR, 0]))
        if abs(kp[NOSE, 0] - (lo + hi) / 2) < EARS_CENTRED * (hi - lo):
            return False
    return True


def assign_phones(phones: list[Track], people: list[Track], radius: float) -> dict[int, int]:
    """Phones per student id: the nearest wrist within `radius` x that student's shoulder width wins; a phone in
    nobody's reach goes to the student whose box contains its centre (smallest box if several)."""
    out: dict[int, int] = {}
    for ph in phones:
        cx, cy = ph.center
        best, best_d = None, math.inf
        for p in people:
            scale = body_scale(p)
            if scale is None:
                continue
            for w in (L_WRIST, R_WRIST):
                if _ok(p.keypoints, w):
                    d = math.hypot(p.keypoints[w, 0] - cx, p.keypoints[w, 1] - cy)
                    if d <= radius * scale and d < best_d:
                        best, best_d = p.id, d
        if best is None:
            inside = [p for p in people if p.box[0] <= cx <= p.box[2] and p.box[1] <= cy <= p.box[3]]
            if inside:
                best = min(inside, key=lambda p: p.w * p.h).id
        if best is not None:
            out[best] = out.get(best, 0) + 1
    return out


@dataclass
class Student:
    state: str = "NORMAL"            # NORMAL | REVIEW | FLAGGED
    score: float = 0.0               # ~ seconds of evidence, drains at `decay` per second
    signals: tuple[str, ...] = ()    # what is seen now: "PHONE", "SIDE LOOK"
    since: float | None = None       # when the current build-up of evidence began
    last_seen: float = 0.0
    phone_t: float | None = None     # last time a phone was assigned to this student
    phone_prev: float | None = None  # the sighting before that


class ExamLogic:
    def __init__(self, phone_radius: float = 0.6, side_look_deg: float = 35.0, rise: float = 1.0,
                 decay: float = 0.15, review: float = 3.0, flagged: float = 8.0, max_dt: float = 1.0):
        self.phone_radius, self.side_look_deg = phone_radius, side_look_deg
        self.rise, self.decay = rise, decay
        self.thresholds = {"REVIEW": review, "FLAGGED": flagged}
        self.max_dt = max_dt  # a stalled live source must not turn one frame into seconds of evidence
        self.students: dict[int, Student] = {}
        self.present: list[int] = []
        self.reviews: list[Event] = []
        self._last_t: float | None = None

    def update(self, t: float, tracks: list[Track]) -> list[Event]:
        dt = 0.0 if self._last_t is None else min(self.max_dt, max(0.0, t - self._last_t))
        self._last_t = t
        # a student is a tracked person with a visible body (shoulders or a forearm): not a coat, bag or a sliver
        people = [tr for tr in tracks if tr.label == "person" and tr.id >= 0
                  and (tr.id in self.students or body_scale(tr) is not None)]
        phones = assign_phones([tr for tr in tracks if tr.label == "cell phone"], people, self.phone_radius)
        self.present = [p.id for p in people]
        seen = {p.id: p for p in people}
        events = []
        for p in people:
            self.students.setdefault(p.id, Student(since=t))
        for sid, st in self.students.items():
            signals = []
            if sid in seen:
                st.last_seen = t
                if phones.get(sid):
                    st.phone_prev, st.phone_t = st.phone_t, t
                confirmed = st.phone_prev is not None and st.phone_t - st.phone_prev <= PHONE_HOLD
                if confirmed and t - st.phone_t <= PHONE_HOLD:
                    signals.append("PHONE")
                if side_look(seen[sid], self.side_look_deg):
                    signals.append("SIDE LOOK")
            prev, st.signals = st.signals, tuple(signals)
            if signals:
                if st.score == 0 and not prev:
                    st.since = t
                st.score += self.rise * dt
            else:
                st.score = max(0.0, st.score - self.decay * dt)
            events += self._transition(t, sid, st)
        return events

    def _transition(self, t: float, sid: int, st: Student) -> list[Event]:
        events = []
        for name in STATES[STATES.index(st.state) + 1:]:
            if st.score >= self.thresholds[name]:
                st.state = name
                kind = "review_started" if name == "REVIEW" else "flagged"
                events.append(Event(t, kind, sid, {"from": st.since, "score": round(st.score, 1),
                                                   "signals": list(st.signals)}))
        while st.state != "NORMAL" and st.score < HYSTERESIS * self.thresholds[st.state]:
            st.state = STATES[STATES.index(st.state) - 1]
        self.reviews += events
        return events

    def counts(self) -> dict[str, int]:
        """Students in view now, per state."""
        c = dict.fromkeys(STATES, 0)
        for sid in self.present:
            c[self.students[sid].state] += 1
        return c
