import math

import numpy as np

from vigia.apps.fall_detector.logic import FallLogic

H = 400.0                 # standing body height in px
FEET = (500.0, 900.0)     # where the feet touch the floor
FPS = 10


def body(angle_deg: float = 0.0, x: float = FEET[0], hip_drop: float = 0.0, keypoints: bool = True,
         tid: int = 1) -> dict:
    """A person as a rigid stick pivoting on the feet: angle 0 = standing, 90 = lying on the floor.
    `hip_drop` (0-1 of H) lowers the upper body with the torso upright, as when sitting on a chair."""
    a = math.radians(angle_deg)
    ux, uy = math.sin(a), -math.cos(a)          # unit vector from the feet towards the head
    fx, fy = x, FEET[1]

    def along(d):                               # point at d * H from the feet along the body
        return fx + ux * d * H, fy + uy * d * H + (0 if d < 0.25 else hip_drop * H)

    hip, shoulder, head = along(0.5), along(0.8), along(1.0)
    th = 0.1 * H                                # half thickness of the body
    xs = [fx, head[0]]
    ys = [fy, head[1]]
    box = (min(xs) - th * abs(uy) - 0.05 * H, min(ys) - th * abs(ux),
           max(xs) + th * abs(uy) + 0.05 * H, max(ys) + th * abs(ux))
    kp = None
    if keypoints:
        kp = np.zeros((17, 3), np.float32)
        for i, p in ((5, shoulder), (6, shoulder), (11, hip), (12, hip)):
            kp[i] = (p[0], p[1], 0.9)
    return {"id": tid, "box": box, "kp": kp}


def tracks(*bodies):
    from vigia.core.types import Track
    return [Track(b["id"], b["box"], "person", 0.9, b["kp"]) for b in bodies]


def run(logic, seconds, pose_at, start=0.0):
    """pose_at(u) -> body dict for u in [0, 1) over `seconds`; returns (events, next t)."""
    events = []
    n = round(seconds * FPS)
    for i in range(n):
        events += logic.update(start + i / FPS, tracks(pose_at(i / n)))
    return events, start + n / FPS


def stand(**kw):
    return lambda u: body(0, **kw)


def rotate(a0, a1, **kw):
    return lambda u: body(a0 + (a1 - a0) * u, **kw)


def test_fast_fall_confirms():
    lg = FallLogic(confirm_after=2.0)
    ev, t = run(lg, 2.0, stand())
    assert lg.status[1].state == "STANDING" and not ev
    ev, t = run(lg, 0.5, rotate(0, 90), start=t)
    ev2, t = run(lg, 0.5, rotate(90, 90), start=t)
    assert lg.status[1].state == "FALLING" and not ev + ev2
    ev, t = run(lg, 2.0, rotate(90, 90), start=t)
    assert lg.status[1].state == "FALL"
    assert [e.kind for e in ev] == ["fall_detected"] and ev[0].track_id == 1
    ev, t = run(lg, 3.0, rotate(90, 90), start=t)
    assert not ev  # one event per fall


def test_slow_lie_down_no_alarm():
    lg = FallLogic()
    _, t = run(lg, 2.0, stand())
    ev, t = run(lg, 4.0, rotate(0, 90), start=t)
    ev2, t = run(lg, 5.0, rotate(90, 90), start=t)
    assert lg.status[1].state == "LYING" and not ev + ev2


def test_recovers_cancels():
    lg = FallLogic(confirm_after=2.0)
    _, t = run(lg, 2.0, stand())
    ev, t = run(lg, 0.5, rotate(0, 90), start=t)
    ev2, t = run(lg, 1.0, rotate(90, 90), start=t)
    ev3, t = run(lg, 0.8, rotate(90, 0), start=t)
    ev4, t = run(lg, 2.0, stand(), start=t)
    assert lg.status[1].state == "STANDING" and not ev + ev2 + ev3 + ev4


def test_sitting_not_fall():
    lg = FallLogic()
    _, t = run(lg, 2.0, stand())
    ev, t = run(lg, 0.6, lambda u: body(0, hip_drop=0.3 * u), start=t)  # a brisk sit on a chair
    ev2, t = run(lg, 5.0, lambda u: body(0, hip_drop=0.3), start=t)
    assert lg.status[1].state == "SITTING" and not ev + ev2


def test_box_fallback_without_keypoints():
    lg = FallLogic(confirm_after=2.0)
    _, t = run(lg, 2.0, stand(keypoints=False))
    ev, t = run(lg, 0.5, rotate(0, 90, keypoints=False), start=t)
    ev2, t = run(lg, 3.0, rotate(90, 90, keypoints=False), start=t)
    assert lg.status[1].state == "FALL" and lg.status[1].source == "box"
    assert [e.kind for e in ev + ev2] == ["fall_detected"]


def test_low_confidence_keypoints_use_the_box():
    lg = FallLogic()
    b = body(0)
    b["kp"][:, 2] = 0.1
    lg.update(0.0, tracks(b))
    assert lg.status[1].source == "box"


def test_walking_state():
    lg = FallLogic()
    run(lg, 2.0, lambda u: body(0, x=200 + 600 * u))  # 1.5 body heights in 2 s
    assert lg.status[1].state == "WALKING"


def test_fall_empty_frames():
    lg = FallLogic()
    _, t = run(lg, 1.0, stand())
    for i in range(100):
        assert lg.update(t + i / FPS, []) == []
    assert lg.status == {}  # people not seen for a while are forgotten


def test_fall_hidden_by_missed_detections_still_confirms():
    lg = FallLogic(confirm_after=1.0)
    _, t = run(lg, 2.0, stand())
    for i in range(5):                      # the detector loses the person for the whole descent
        lg.update(t + i / FPS, [])
    t += 0.5
    ev, t = run(lg, 2.0, rotate(90, 90), start=t)
    assert [e.kind for e in ev] == ["fall_detected"]


def test_id_switch_mid_fall_keeps_state():
    lg = FallLogic(confirm_after=1.0)
    _, t = run(lg, 2.0, stand())
    _, t = run(lg, 0.3, rotate(0, 60), start=t)
    ev, t = run(lg, 2.0, rotate(90, 90, tid=7), start=t)  # the tracker gives the person a new id on the floor
    assert [(e.kind, e.track_id) for e in ev] == [("fall_detected", 7)]
    assert set(lg.status) == {7}


def test_eval_script_parses_both_csv_layouts_and_counts(tmp_path):
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("eval_fall", Path(__file__).parents[1] / "scripts" / "eval_fall.py")
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    old = tmp_path / "a.csv"
    old.write_text("File Name, Length,Description\n01.mp4,08 sec, Sitting on the bed to sleeping, right side.\n")
    new = tmp_path / "b.csv"
    new.write_text("File Name,Length (seconds),Time of Recording,Attire,Description\n"
                   "07.mp4,13,Day,Full Shirt & Pant,Exercising (Moving both hands up and down).\n")
    assert ev.descriptions(old) == {"01.mp4": "sitting on the bed to sleeping, right side."}
    assert ev.kind_of(ev.descriptions(new)["07.mp4"]).startswith("exercise")
    rows = [(3, "Fall", Path("1.mp4"), "", True), (3, "Fall", Path("2.mp4"), "", False),
            (3, "ADL", Path("1.mp4"), "walking in the room.", True), (3, "ADL", Path("2.mp4"), "sitting", False)]
    table = ev.report(rows, "t")
    assert "| 1/2 (50%) | 1/2 (50%) | 50% |" in table
    assert "| walking, standing, other | 1/1 (100%) |" in table


def test_two_people_independent():
    lg = FallLogic(confirm_after=1.0)
    for i in range(20):
        lg.update(i / FPS, tracks(body(0, tid=1), body(0, x=1200, tid=2)))
    t, events = 2.0, []
    for i in range(30):
        a = min(90.0, 90 * i / 5)
        events += lg.update(t + i / FPS, tracks(body(a, tid=1), body(0, x=1200, tid=2)))
    assert [e.track_id for e in events] == [1]
    assert lg.status[2].state == "STANDING"
