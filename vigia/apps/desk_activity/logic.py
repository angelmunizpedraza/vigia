"""How is each desk used: someone working at it, away from it, or on the phone? Debounced, and kept per desk zone
(names starting with "desk"), never per person: the question is which desks are used, not who sits there."""
from __future__ import annotations

from dataclasses import dataclass, field

from vigia.core.scene import Scene, box_fraction_in_polygon
from vigia.core.types import Box, Event, Track

STATES = ("WORKING", "AWAY", "PHONE")


@dataclass
class DeskStatus:
    state: str = "AWAY"             # WORKING | AWAY | PHONE
    since: float = 0.0              # when the current state began (dated back to the evidence)
    seconds: dict[str, float] = field(default_factory=lambda: dict.fromkeys(STATES, 0.0))
    first_seen: float | None = None  # current stretch with someone seated at the desk
    last_seen: float | None = None
    phone_first: float | None = None  # current stretch with a phone in the seated person's hands
    phone_last: float | None = None


def _stretch(first: float | None, last: float | None, t: float, gap: float) -> tuple[float, float]:
    """Extend a run of sightings to `t`, or start a new one if it was interrupted for longer than `gap`."""
    return (t if first is None or last is None or t - last > gap else first), t


def _grow(box: Box, f: float) -> Box:
    x1, y1, x2, y2 = box
    dx, dy = (x2 - x1) * f, (y2 - y1) * f
    return x1 - dx, y1 - dy, x2 + dx, y2 + dy


class DeskLogic:
    def __init__(self, scene: Scene, away_after: float = 10.0, phone_after: float = 3.0, min_overlap: float = 0.25,
                 work_after: float = 2.0, gap_tolerance: float = 1.0):
        self.desks = scene.zones_like("desk")
        self.away_after, self.phone_after, self.min_overlap = away_after, phone_after, min_overlap
        self.work_after, self.gap_tolerance = work_after, gap_tolerance
        self.status = {name: DeskStatus() for name in self.desks}
        self._last_t: float | None = None

    def seated(self, track: Track, poly) -> bool:
        """Seated people cover the desk area with the lower half of their box (chair, lap, desk top)."""
        x1, y1, x2, y2 = track.box
        return box_fraction_in_polygon((x1, (y1 + y2) / 2, x2, y2), poly) >= self.min_overlap

    @staticmethod
    def in_hands(phone: Track, person: Track) -> bool:
        """A phone counts when its centre is inside the seated person's box (a little grown): in the hands or at
        the ear. Not required to be inside the desk polygon, which would miss a phone held up to the ear."""
        x, y = phone.center
        x1, y1, x2, y2 = _grow(person.box, 0.1)
        return x1 <= x <= x2 and y1 <= y <= y2

    def update(self, t: float, tracks: list[Track]) -> list[Event]:
        first_frame = self._last_t is None
        dt = 0.0 if first_frame else max(0.0, t - self._last_t)
        self._last_t = t
        people = [tr for tr in tracks if tr.label == "person"]
        phones = [tr for tr in tracks if tr.label == "cell phone"]
        events = []
        for name, poly in self.desks.items():
            st = self.status[name]
            st.seconds[st.state] += dt
            here = [p for p in people if self.seated(p, poly)]
            if here:
                st.first_seen, st.last_seen = _stretch(st.first_seen, st.last_seen, t, self.gap_tolerance)
            if any(self.in_hands(ph, p) for ph in phones for p in here):
                st.phone_first, st.phone_last = _stretch(st.phone_first, st.phone_last, t, self.gap_tolerance)
            new, since = st.state, st.since
            if first_frame:  # nothing to debounce against yet: take the first frame as it is
                st.state, st.since = ("WORKING" if here else "AWAY"), t
                continue
            if st.state == "AWAY":
                if here and t - st.first_seen >= self.work_after:
                    new, since = "WORKING", st.first_seen
            elif st.last_seen is not None and t - st.last_seen >= self.away_after:
                new, since = "AWAY", st.last_seen
            elif st.state == "WORKING" and st.phone_last == t and t - st.phone_first >= self.phone_after:
                new, since = "PHONE", st.phone_first
            elif st.state == "PHONE" and t - st.phone_last >= self.phone_after:
                new, since = "WORKING", st.phone_last
            if new != st.state:
                # the debounce wait was counted in the old state: move it to the new one, where it belongs
                back = t - max(since, st.since)
                st.seconds[st.state] -= back
                st.seconds[new] += back
                st.state, st.since = new, max(since, st.since)
                events.append(Event(t, f"desk_{new.lower()}", data={"desk": name, "since": round(st.since, 1)}))
        return events
