import pytest

from vigia.apps.drowsiness.logic import DrowsyLogic

OPEN = dict(blink_left=0.15, blink_right=0.2, ear=0.28, jaw_open=0.0)
CLOSED = dict(blink_left=0.7, blink_right=0.6, ear=0.08, jaw_open=0.0)
NO_FACE = dict(blink_left=None, blink_right=None, ear=None, jaw_open=None)


def run(logic, seconds, signals, start=0.0, fps=10):
    events = []
    n = round(seconds * fps)
    for i in range(n):
        events += logic.update(start + i / fps, **signals)
    return events, start + n / fps


def kinds(events):
    return [e.kind for e in events]


def test_alarm_after_closed_duration():
    lg = DrowsyLogic(closed_alarm=1.5, calib_seconds=2.0)
    _, t = run(lg, 3.0, OPEN)
    assert lg.state == "EYES OPEN"
    ev, t = run(lg, 1.4, CLOSED, start=t)
    assert lg.state == "EYES CLOSED" and "eyes_closed_alarm" not in kinds(ev)
    ev, t = run(lg, 0.3, CLOSED, start=t)
    assert lg.state == "ALARM" and kinds(ev) == ["eyes_closed_alarm"]
    ev, t = run(lg, 1.0, OPEN, start=t)
    assert lg.state == "EYES OPEN" and kinds(ev) == ["alarm_cleared"]
    assert abs(ev[0].data["closed_for"] - 1.7) < 0.15


def test_blink_is_not_alarm():
    lg = DrowsyLogic(closed_alarm=1.5, calib_seconds=1.0)
    t, events = 0.0, []
    for _ in range(10):  # 0.3 s blink every 2 s
        ev, t = run(lg, 1.7, OPEN, start=t)
        events += ev
        ev, t = run(lg, 0.3, CLOSED, start=t)
        events += ev
    events += run(lg, 1.0, OPEN, start=t)[0]
    assert "eyes_closed_alarm" not in kinds(events) and lg.state == "EYES OPEN"


def test_brief_open_reading_does_not_reset_the_timer():
    lg = DrowsyLogic(closed_alarm=1.5, calib_seconds=0.0)
    _, t = run(lg, 1.0, CLOSED)
    _, t = run(lg, 0.1, OPEN, start=t)  # one noisy frame
    ev, t = run(lg, 0.5, CLOSED, start=t)
    assert lg.state == "ALARM" and kinds(ev) == ["eyes_closed_alarm"]


def test_calibration_sets_ear_threshold():
    ear_only = dict(blink_left=None, blink_right=None, jaw_open=None)
    lg = DrowsyLogic(calib_seconds=5.0)
    ev, t = run(lg, 1.0, dict(ear_only, ear=0.30))
    assert lg.state == "CALIBRATING" and lg.ear_threshold == pytest.approx(0.2)
    ev, t = run(lg, 4.1, dict(ear_only, ear=0.30), start=t)
    assert kinds(ev) == ["calibrated"] and lg.calibrated
    assert lg.ear_threshold == pytest.approx(0.75 * 0.30)
    # with no blendshapes, EAR decides: 0.21 is closed now (it was open under the 0.2 default)
    run(lg, 0.5, dict(ear_only, ear=0.21), start=t)
    assert lg.state == "EYES CLOSED"


def test_calibration_survives_closed_eyes():
    # eyes closed 80 % of the calibration window: the 90th percentile still finds the open EAR
    lg = DrowsyLogic(calib_seconds=5.0)
    _, t = run(lg, 1.0, dict(OPEN, ear=0.30))
    run(lg, 4.5, dict(CLOSED, ear=0.08), start=t)
    assert lg.calibrated and lg.ear_threshold == pytest.approx(0.75 * 0.30)
    # never open: keep the default instead of learning a closed-eye threshold
    lg = DrowsyLogic(calib_seconds=5.0)
    ev, _ = run(lg, 6.0, dict(CLOSED, ear=0.08))
    calib = [e for e in ev if e.kind == "calibrated"]
    assert lg.ear_threshold == pytest.approx(0.2) and calib[0].data["default"] is True


def test_calibration_waits_for_a_face():
    lg = DrowsyLogic(calib_seconds=2.0)
    _, t = run(lg, 5.0, NO_FACE)
    assert not lg.calibrated and lg.state == "NO FACE"
    run(lg, 1.0, OPEN, start=t)
    assert lg.state == "CALIBRATING"


def test_alarm_does_not_wait_for_calibration():
    lg = DrowsyLogic(closed_alarm=1.5, calib_seconds=5.0)
    _, t = run(lg, 0.5, OPEN)
    ev, _ = run(lg, 2.0, CLOSED, start=t)
    assert lg.state == "ALARM" and not lg.calibrated


def test_face_lost_while_closed_keeps_the_alarm():
    lg = DrowsyLogic(closed_alarm=1.5, calib_seconds=0.0)
    _, t = run(lg, 1.0, CLOSED)
    run(lg, 2.0, NO_FACE, start=t)  # head dropped out of view
    assert lg.state == "ALARM"


def test_perclos_window():
    lg = DrowsyLogic(perclos_window=10.0, calib_seconds=0.0)
    _, t = run(lg, 10.0, OPEN)
    assert lg.perclos == 0.0
    _, t = run(lg, 5.0, CLOSED, start=t)
    assert lg.perclos == pytest.approx(0.5, abs=0.03)
    _, t = run(lg, 10.0, OPEN, start=t)
    assert lg.perclos == pytest.approx(0.0, abs=0.03)
    run(lg, 5.0, NO_FACE, start=t)  # no face: no evidence either way
    assert lg.perclos == pytest.approx(0.0, abs=0.03)


def test_yawn_needs_duration_and_one_yawn_is_counted_once():
    lg = DrowsyLogic(yawn_open=0.25, yawn_seconds=0.15, calib_seconds=0.0)
    wide = dict(OPEN, jaw_open=0.35)
    _, t = run(lg, 1.0, OPEN)
    _, t = run(lg, 0.1, wide, start=t)  # too short (a syllable)
    _, t = run(lg, 1.0, OPEN, start=t)
    assert lg.yawns == 0
    ev, t = run(lg, 0.3, wide, start=t)
    assert lg.yawns == 1 and kinds(ev) == ["yawn"]
    _, t = run(lg, 3.0, OPEN, start=t)  # the hand covers the mouth: jawOpen reads 0
    ev, t = run(lg, 0.3, wide, start=t)  # same yawn seen again as the hand moves away
    assert lg.yawns == 1 and not ev
    _, t = run(lg, 8.0, OPEN, start=t)
    run(lg, 0.3, wide, start=t)
    assert lg.yawns == 2


def test_drowsy_empty_frames():
    lg = DrowsyLogic()
    ev, _ = run(lg, 10.0, NO_FACE)
    assert ev == [] and lg.state == "NO FACE"
    assert lg.perclos == 0.0 and lg.yawns == 0 and lg.closed_for(10.0) == 0.0


def test_serial_protocol_over_loop():
    pytest.importorskip("serial")
    from vigia.apps.drowsiness.board import SerialBoard

    board = SerialBoard("loop://")
    board.update(False, now=0.0)   # first call: heartbeat + state
    board.update(False, now=0.2)   # nothing new
    board.update(True, now=0.3)    # change: sent at once
    board.update(True, now=0.6)    # heartbeat repeats the state
    board.update(False, now=0.7)
    board.ser.timeout = 0.2
    lines = board.ser.read(200).decode("ascii").split("\n")
    board.close()
    assert lines == ["HB", "OK", "ALARM", "HB", "ALARM", "OK", ""]
