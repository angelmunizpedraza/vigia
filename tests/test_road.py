import csv

from vigia.apps.road_damage.logic import DamageLogic, mostly_inside
from vigia.core.types import Track

POTHOLE = (100, 100, 200, 160)
CRACK = (500, 300, 700, 340)


def dmg(tid, box=POTHOLE, label="pothole", score=0.8):
    return Track(tid, box, label, score)


def feed(logic, frames, start=0.0, fps=10):
    events = []
    for i, tracks in enumerate(frames):
        events += logic.update(start + i / fps, tracks)
    return events, start + len(frames) / fps


def test_damage_counted_once_per_track():
    lg = DamageLogic(min_frames=3)
    ev, _ = feed(lg, [[dmg(1, score=s), dmg(2, CRACK, "alligator crack", 0.5)] for s in (0.5, 0.6, 0.9, 0.7, 0.8)] * 4)
    assert [e.kind for e in ev] == ["damage_found", "damage_found"]
    assert len(lg.report) == 2
    pot = next(r for r in lg.report if r.label == "pothole")
    assert pot.first_t == 0.0 and pot.max_score == 0.9 and pot.crop_path.endswith("damage-001.jpg")


def test_short_lived_detection_ignored():
    lg = DamageLogic(min_frames=3)
    ev, _ = feed(lg, [[dmg(1)], [dmg(1)], [], [], [dmg(-1)], [dmg(-1)], [dmg(-1)]])
    assert not ev and not lg.report


def test_report_csv_columns(tmp_path):
    lg = DamageLogic(min_frames=1)
    feed(lg, [[dmg(1)], [dmg(2, CRACK, "transverse crack", 0.42)]], start=61.25)
    lg.write_csv(tmp_path / "r.csv")
    rows = list(csv.DictReader(open(tmp_path / "r.csv", newline="", encoding="utf-8")))
    assert list(rows[0]) == ["id", "first_seen", "first_t", "label", "max_score", "crop_path"]
    assert rows[0]["first_seen"] == "1:01" and rows[0]["first_t"] == "61.2" and rows[0]["label"] == "pothole"
    assert rows[1]["max_score"] == "0.42" and rows[1]["crop_path"].endswith("damage-002.jpg")


def test_best_crop_follows_max_score():
    lg = DamageLogic(min_frames=2)
    lg.update(0.0, [dmg(1, score=0.5)])
    lg.update(0.1, [dmg(1, score=0.6)])
    assert lg.improved == [0]  # counted now: save its first crop
    lg.update(0.2, [dmg(1, score=0.55)])
    assert lg.improved == []
    lg.update(0.3, [dmg(1, (102, 101, 203, 162), "alligator crack", 0.95)])
    assert lg.improved == [0] and lg.report[0].box == (102, 101, 203, 162)
    assert lg.report[0].label == "alligator crack"  # the class of its most confident view


def test_new_id_on_the_same_spot_is_not_recounted():
    # a car hides the pothole for 3 s and the tracker gives it a new id when it reappears
    lg = DamageLogic(min_frames=3, rematch_seconds=5.0)
    _, t = feed(lg, [[dmg(1)]] * 10)
    _, t = feed(lg, [[]] * 30, start=t)
    ev, t = feed(lg, [[dmg(7, (104, 98, 205, 163))]] * 10, start=t)
    assert not ev and len(lg.report) == 1
    # a different damage elsewhere still counts, and the same spot long after does too
    ev, t = feed(lg, [[dmg(8, CRACK, "longitudinal crack")]] * 5, start=t)
    _, t = feed(lg, [[]] * 60, start=t)
    ev2, _ = feed(lg, [[dmg(9)]] * 5, start=t)
    assert len(ev) == 1 and len(ev2) == 1 and len(lg.report) == 3


def test_rematch_off_counts_every_track():
    lg = DamageLogic(min_frames=1, rematch_seconds=0.0)
    feed(lg, [[dmg(1)], [], [dmg(2)]])
    assert len(lg.report) == 2


def test_damage_on_a_vehicle_is_dropped():
    car = (80, 40, 400, 220)
    assert mostly_inside(POTHOLE, [CRACK, car])          # wheel or bodywork 'pothole'
    assert not mostly_inside(CRACK, [car])
    assert not mostly_inside((300, 150, 500, 250), [car])  # a damage only partly behind a car stays (35 %)
    assert not mostly_inside(POTHOLE, [])


def test_road_empty_frames():
    lg = DamageLogic()
    feed(lg, [[dmg(1)]] * 2)
    ev, _ = feed(lg, [[]] * 100, start=0.2)
    assert not ev and not lg.report and lg.improved == []
    assert not lg._pending  # half-seen damages are forgotten
