"""Is each table free or occupied? Debounced so a waiter walking past or a customer hidden for a moment does
not flip it. Works on table zones (names starting with "table"), never on who the person is."""
from __future__ import annotations

from dataclasses import dataclass

from vigia.core.scene import Scene, box_fraction_in_polygon
from vigia.core.types import Event, Track


@dataclass
class TableStatus:
    state: str = "FREE"            # FREE | OCCUPIED
    since: float = 0.0             # when the current state began
    occupied_seconds: float = 0.0
    first_seen: float | None = None  # start of the current stretch with someone at the table
    last_seen: float | None = None


class TableLogic:
    def __init__(self, scene: Scene, occupy_after: float = 3.0, free_after: float = 10.0,
                 min_overlap: float = 0.3, gap_tolerance: float = 1.0):
        self.tables = scene.zones_like("table")
        self.occupy_after, self.free_after = occupy_after, free_after
        self.min_overlap, self.gap_tolerance = min_overlap, gap_tolerance
        self.status = {name: TableStatus() for name in self.tables}
        self._start: float | None = None
        self._last_t: float | None = None

    def at_table(self, track: Track, poly) -> bool:
        """Seated people overlap the table area with the lower half of their box (legs, chair, lap)."""
        x1, y1, x2, y2 = track.box
        lower = (x1, (y1 + y2) / 2, x2, y2)
        return box_fraction_in_polygon(lower, poly) >= self.min_overlap

    def update(self, t: float, tracks: list[Track]) -> list[Event]:
        if self._start is None:
            self._start = t
        dt = 0.0 if self._last_t is None else max(0.0, t - self._last_t)
        self._last_t = t
        people = [tr for tr in tracks if tr.label == "person"]
        events = []
        for name, poly in self.tables.items():
            st = self.status[name]
            if st.state == "OCCUPIED":
                st.occupied_seconds += dt
            if any(self.at_table(p, poly) for p in people):
                if st.first_seen is None or (st.last_seen is not None and t - st.last_seen > self.gap_tolerance):
                    st.first_seen = t
                st.last_seen = t
            if st.state == "FREE" and st.first_seen is not None and st.last_seen == t \
                    and t - st.first_seen >= self.occupy_after:
                st.state, st.since = "OCCUPIED", st.first_seen
                events.append(Event(t, "table_occupied", data={"table": name}))
            elif st.state == "OCCUPIED" and st.last_seen is not None and st.last_seen != t \
                    and t - st.last_seen >= self.free_after:
                taken = st.last_seen - st.since
                st.state, st.since, st.first_seen = "FREE", st.last_seen, None
                events.append(Event(t, "table_freed", data={"table": name, "occupied_for": round(taken, 1)}))
        return events

    def occupancy_ratio(self) -> float:
        """Share of table-time occupied since the start (0-1)."""
        if not self.tables or self._start is None or self._last_t is None or self._last_t <= self._start:
            return 0.0
        total = (self._last_t - self._start) * len(self.tables)
        return min(1.0, sum(s.occupied_seconds for s in self.status.values()) / total)
