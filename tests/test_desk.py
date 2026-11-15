from vigia.apps.desk_activity.logic import DeskLogic
from vigia.core.scene import Scene
from vigia.core.types import Track

SCENE = Scene(1000, 1000, zones={"desk-1": [(100, 500), (400, 500), (400, 900), (100, 900)],
                                 "desk-2": [(600, 500), (900, 500), (900, 900), (600, 900)]})
SEATED = Track(-1, (150, 300, 350, 800), "person", 0.9)       # lower half inside desk-1
PHONE = Track(-1, (230, 450, 270, 500), "cell phone", 0.6)    # in the seated person's hands
PHONE_ON_DESK = Track(-1, (370, 860, 395, 890), "cell phone", 0.6)  # lying on desk-1, away from the person
NEAR_EDGE = Track(-1, (380, 200, 480, 700), "person", 0.9)    # standing just beside desk-1
CROSSING = Track(-1, (200, 300, 300, 800), "person", 0.9)     # walking through desk-1's area


def run(logic, seconds, tracks, start=0.0, fps=10):
    events = []
    n = round(seconds * fps)
    for i in range(n):
        events += logic.update(start + i / fps, tracks)
    return events, start + n / fps


def test_away_after_delay():
    lg = DeskLogic(SCENE, away_after=10.0)
    ev, t = run(lg, 5.0, [SEATED])
    assert lg.status["desk-1"].state == "WORKING" and lg.status["desk-2"].state == "AWAY"
    ev, t = run(lg, 9.0, [], start=t)
    assert lg.status["desk-1"].state == "WORKING" and not ev
    ev, t = run(lg, 2.0, [], start=t)
    assert lg.status["desk-1"].state == "AWAY"
    assert [e.kind for e in ev] == ["desk_away"] and ev[0].data["desk"] == "desk-1"
    assert abs(lg.status["desk-1"].since - 4.9) < 0.15  # empty since the last frame with someone there


def test_phone_needs_persistence():
    lg = DeskLogic(SCENE, phone_after=3.0)
    _, t = run(lg, 3.0, [SEATED])
    ev, t = run(lg, 2.0, [SEATED, PHONE], start=t)
    assert lg.status["desk-1"].state == "WORKING" and not ev
    _, t = run(lg, 2.0, [SEATED], start=t)
    ev, t = run(lg, 3.5, [SEATED, PHONE], start=t)
    assert lg.status["desk-1"].state == "PHONE" and [e.kind for e in ev] == ["desk_phone"]


def test_phone_lying_on_the_desk_is_not_phone():
    lg = DeskLogic(SCENE, phone_after=3.0)
    run(lg, 10.0, [SEATED, PHONE_ON_DESK])
    assert lg.status["desk-1"].state == "WORKING"


def test_phone_without_anyone_seated_is_not_phone():
    lg = DeskLogic(SCENE, phone_after=3.0)
    run(lg, 10.0, [PHONE])
    assert lg.status["desk-1"].state == "AWAY"


def test_returns_to_working():
    lg = DeskLogic(SCENE, away_after=10.0, phone_after=3.0, work_after=2.0)
    _, t = run(lg, 2.0, [SEATED])
    _, t = run(lg, 12.0, [], start=t)
    assert lg.status["desk-1"].state == "AWAY"
    ev, t = run(lg, 1.5, [SEATED], start=t)
    assert lg.status["desk-1"].state == "AWAY" and not ev
    ev, t = run(lg, 1.0, [SEATED], start=t)
    assert lg.status["desk-1"].state == "WORKING" and [e.kind for e in ev] == ["desk_working"]
    _, t = run(lg, 4.0, [SEATED, PHONE], start=t)
    assert lg.status["desk-1"].state == "PHONE"
    ev, t = run(lg, 3.5, [SEATED], start=t)  # phone put away
    assert lg.status["desk-1"].state == "WORKING" and [e.kind for e in ev] == ["desk_working"]


def test_seconds_per_state():
    lg = DeskLogic(SCENE, away_after=10.0, phone_after=3.0, work_after=2.0)
    _, t = run(lg, 20.0, [SEATED])           # working 0-20
    _, t = run(lg, 15.0, [], start=t)        # empty 20-35 (AWAY from 30, dated back to 20)
    _, t = run(lg, 10.0, [SEATED, PHONE], start=t)  # back with a phone 35-45 (PHONE dated back to 35)
    s = lg.status["desk-1"].seconds
    assert lg.status["desk-1"].state == "PHONE"
    assert abs(s["WORKING"] - 20.0) < 0.3 and abs(s["AWAY"] - 15.0) < 0.3 and abs(s["PHONE"] - 10.0) < 0.3
    assert abs(sum(s.values()) - 44.9) < 0.01  # every second is in exactly one state
    assert abs(lg.status["desk-2"].seconds["AWAY"] - 44.9) < 0.01


def test_walking_past_is_not_working():
    lg = DeskLogic(SCENE, work_after=2.0)
    run(lg, 5.0, [NEAR_EDGE])  # lower half barely overlaps the desk
    assert lg.status["desk-1"].state == "AWAY"
    lg = DeskLogic(SCENE, work_after=2.0)
    _, t = run(lg, 0.5, [])
    _, t = run(lg, 1.5, [CROSSING], start=t)  # crosses the desk area in 1.5 s
    run(lg, 5.0, [], start=t)
    assert lg.status["desk-1"].state == "AWAY"


def test_short_detection_gaps_do_not_reset_the_wait():
    lg = DeskLogic(SCENE, work_after=2.0)
    _, t = run(lg, 0.5, [])
    for _ in range(3):  # 0.8 s seen, 0.2 s missed, repeated
        _, t = run(lg, 0.8, [SEATED], start=t)
        _, t = run(lg, 0.2, [], start=t)
    run(lg, 0.2, [SEATED], start=t)
    assert lg.status["desk-1"].state == "WORKING"


def test_desk_empty_frames():
    lg = DeskLogic(SCENE)
    ev, _ = run(lg, 10.0, [])
    assert not ev
    assert all(s.state == "AWAY" for s in lg.status.values())
    assert all(abs(s.seconds["AWAY"] - 9.9) < 0.01 and s.seconds["WORKING"] == 0 for s in lg.status.values())


def test_non_desk_zones_ignored():
    sc = Scene(1000, 1000, zones={**SCENE.zones, "door": [(0, 0), (10, 0), (10, 10)]})
    assert set(DeskLogic(sc).status) == {"desk-1", "desk-2"}
