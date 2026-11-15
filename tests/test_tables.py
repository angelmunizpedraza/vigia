from vigia.apps.table_occupancy.logic import TableLogic
from vigia.core.scene import Scene
from vigia.core.types import Track

SCENE = Scene(1000, 1000, zones={"table-1": [(100, 100), (400, 100), (400, 400), (100, 400)],
                                 "table-2": [(600, 600), (900, 600), (900, 900), (600, 900)]})
AT_T1 = Track(1, (150, 150, 250, 380), "person", 0.9)   # lower half inside table-1
PASSING = Track(2, (300, 50, 380, 300), "person", 0.9)  # waiter crossing the edge of table-1


def run(logic, seconds, tracks, start=0.0, fps=10):
    events = []
    n = round(seconds * fps)
    for i in range(n):
        events += logic.update(start + i / fps, tracks)
    return events, start + n / fps


def test_table_occupies_after_delay():
    lg = TableLogic(SCENE, occupy_after=3.0)
    ev, t = run(lg, 2.5, [AT_T1])
    assert lg.status["table-1"].state == "FREE" and not ev
    ev, t = run(lg, 1.0, [AT_T1], start=t)
    assert lg.status["table-1"].state == "OCCUPIED"
    assert [e.kind for e in ev] == ["table_occupied"] and ev[0].data["table"] == "table-1"
    assert lg.status["table-2"].state == "FREE"


def test_waiter_passing_does_not_flip():
    lg = TableLogic(SCENE, occupy_after=3.0)
    ev, t = run(lg, 2.0, [AT_T1])
    run(lg, 5.0, [], start=t)
    assert lg.status["table-1"].state == "FREE"


def test_brief_occlusion_keeps_occupied():
    lg = TableLogic(SCENE, occupy_after=3.0, free_after=10.0)
    _, t = run(lg, 4.0, [AT_T1])
    _, t = run(lg, 5.0, [], start=t)
    assert lg.status["table-1"].state == "OCCUPIED"
    ev, t = run(lg, 6.0, [], start=t)
    assert lg.status["table-1"].state == "FREE" and [e.kind for e in ev] == ["table_freed"]


def test_short_detection_gaps_do_not_reset_the_wait():
    lg = TableLogic(SCENE, occupy_after=3.0)
    t = 0.0
    for _ in range(4):  # 0.8 s seen, 0.2 s missed, repeated
        _, t = run(lg, 0.8, [AT_T1], start=t)
        _, t = run(lg, 0.2, [], start=t)
    assert lg.status["table-1"].state == "OCCUPIED"


def test_table_occupied_survives_id_switch():
    lg = TableLogic(SCENE, occupy_after=3.0)
    _, t = run(lg, 2.0, [AT_T1])
    _, t = run(lg, 2.0, [Track(99, AT_T1.box, "person", 0.9)], start=t)
    assert lg.status["table-1"].state == "OCCUPIED"


def test_occupancy_ratio():
    lg = TableLogic(SCENE, occupy_after=0.0, free_after=0.0)
    _, t = run(lg, 10.0, [AT_T1])
    # table-1 occupied all the time, table-2 never: 50 %
    assert abs(lg.occupancy_ratio() - 0.5) < 0.02
    assert abs(lg.status["table-1"].occupied_seconds - 10.0) < 0.2


def test_tables_empty_frames():
    lg = TableLogic(SCENE)
    run(lg, 10.0, [])
    assert all(s.state == "FREE" for s in lg.status.values())
    assert lg.occupancy_ratio() == 0.0


def test_non_table_zones_ignored():
    sc = Scene(1000, 1000, zones={**SCENE.zones, "door": [(0, 0), (10, 0), (10, 10)]})
    assert set(TableLogic(sc).status) == {"table-1", "table-2"}


def test_freed_event_reports_how_long_it_was_taken():
    lg = TableLogic(SCENE, occupy_after=3.0, free_after=10.0)
    _, t = run(lg, 40.0, [AT_T1])          # someone at table-1 from 0 to ~40 s
    ev, t = run(lg, 11.0, [], start=t)
    [freed] = [e for e in ev if e.kind == "table_freed"]
    assert abs(freed.data["occupied_for"] - 39.9) < 0.2
