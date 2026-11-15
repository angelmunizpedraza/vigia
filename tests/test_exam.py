import numpy as np

from vigia.apps.exam_monitor.logic import ExamLogic, assign_phones, head_turn
from vigia.core.types import Track

SW = 100.0  # shoulder width of every synthetic student, px


def student(id, x, turn=0.0, nose_dy=-45.0, wrists=(), y=300.0):
    """Seated student facing the camera: shoulders at x +- 50, nose `turn` x shoulder width to the side."""
    kp = np.zeros((17, 3), np.float32)
    kp[5] = (x - SW / 2, y, 0.9)
    kp[6] = (x + SW / 2, y, 0.9)
    kp[0] = (x + turn * SW, y + nose_dy, 0.9)
    for i, (wx, wy) in zip((9, 10), wrists):
        kp[i] = (wx, wy, 0.9)
    return Track(id, (x - 80, y - 110, x + 80, y + 200), "person", 0.9, kp)


def phone(x, y):
    return Track(-1, (x - 10, y - 15, x + 10, y + 15), "cell phone", 0.5)


def run(logic, seconds, tracks, start=0.0, fps=10):
    events = []
    n = round(seconds * fps)
    for i in range(n):
        events += logic.update(start + i / fps, tracks)
    return events, start + n / fps


def test_head_turn_sign_and_missing_points():
    assert head_turn(student(1, 500)) == 0
    assert head_turn(student(1, 500, turn=0.3)) > 35 > 0 > head_turn(student(1, 500, turn=-0.3))
    s = student(1, 500, turn=0.3)
    s.keypoints[0, 2] = 0.1  # nose not visible
    assert head_turn(s) is None
    narrow = student(2, 500, turn=0.3)
    narrow.keypoints[5:7, 0] = (495, 505)  # shoulders 10 px apart in a 160 px box: profile view or a bad box
    assert head_turn(narrow) is None


def test_turned_torso_with_face_between_the_ears_is_normal():
    # torso turned away: the nose is off the shoulder midpoint, but centred between both visible ears
    s = student(1, 400, turn=0.35)
    s.keypoints[3] = (s.keypoints[0, 0] + 30, 250, 0.8)
    s.keypoints[4] = (s.keypoints[0, 0] - 30, 250, 0.8)
    lg = ExamLogic()
    run(lg, 10.0, [s])
    assert lg.students[1].state == "NORMAL" and lg.students[1].signals == ()
    s.keypoints[4] = (s.keypoints[0, 0] - 2, 250, 0.8)  # one ear behind the nose: the head is turned
    run(lg, 0.2, [s], start=10.0)
    assert lg.students[1].signals == ("SIDE LOOK",)


def test_phone_goes_to_nearest_wrist_not_nearest_box():
    # B's hand reaches into A's box; the phone sits in A's box but in B's hand
    a = student(1, 400, wrists=[(380, 420), (420, 420)])
    b = student(2, 560, wrists=[(450, 400), (600, 420)])
    assert assign_phones([phone(455, 405)], [a, b], radius=0.6) == {2: 1}
    lg = ExamLogic()
    run(lg, 1.0, [a, b, phone(455, 405)])
    assert lg.students[2].signals == ("PHONE",) and lg.students[1].signals == ()


def test_phone_out_of_reach_falls_back_to_box():
    a = student(1, 400, wrists=[(300, 450), (330, 450)])
    assert assign_phones([phone(420, 330)], [a], radius=0.6) == {1: 1}
    assert assign_phones([phone(900, 330)], [a], radius=0.6) == {}


def test_phone_wrist_without_shoulders_uses_forearm():
    # close-up: only an arm is in view; the forearm gives the body scale for the wrist radius
    kp = np.zeros((17, 3), np.float32)
    kp[8] = (200, 900, 0.7)
    kp[10] = (500, 600, 0.7)  # forearm ~424 px -> scale ~643 px -> radius ~386 px
    arm = Track(3, (0, 550, 600, 1000), "person", 0.8, kp)
    assert assign_phones([phone(700, 450)], [arm], radius=0.6) == {3: 1}  # 250 px from the wrist, outside the box
    assert assign_phones([phone(950, 300)], [arm], radius=0.6) == {}
    # keypoints guessed off-frame: "shoulders" 20 px apart in a 600 px box must not shrink the radius to 12 px
    kp[5], kp[6] = (290, 990, 0.35), (310, 990, 0.35)
    assert assign_phones([phone(700, 450)], [arm], radius=0.6) == {3: 1}


def test_bag_with_noise_keypoints_is_not_a_student():
    # pose run on a bag at the frame edge: tiny "shoulders" and a faint "forearm"
    kp = np.zeros((17, 3), np.float32)
    kp[5], kp[6] = (50, 900, 0.33), (52, 902, 0.31)
    kp[7], kp[9] = (60, 950, 0.32), (90, 1000, 0.31)
    lg = ExamLogic()
    run(lg, 2.0, [Track(10, (0, 860, 140, 1070), "person", 0.5, kp), student(2, 400)])
    assert set(lg.students) == {2}


def test_flickering_phone_counts_as_continuous():
    lg = ExamLogic()
    s = student(1, 400, wrists=[(380, 420), (420, 420)])
    t = 0.0
    for i in range(40):  # phone detected in every other frame
        lg.update(t, [s, phone(422, 418)] if i % 2 == 0 else [s])
        assert lg.students[1].signals == (("PHONE",) if i >= 2 else ())  # confirmed by the second sighting
        t += 0.1
    assert lg.students[1].state == "REVIEW"


def test_single_frame_phone_is_ignored():
    lg = ExamLogic()
    s = student(1, 400, wrists=[(380, 420), (420, 420)])
    run(lg, 1.0, [s])
    run(lg, 0.1, [s, phone(422, 418)], start=1.0)  # a pen taken for a phone in one frame
    run(lg, 1.0, [s], start=1.1)
    assert lg.students[1].signals == () and lg.students[1].score == 0


def test_phone_in_hand_reviews():
    lg = ExamLogic(review=3.0)
    s = student(1, 400, wrists=[(380, 420), (420, 420)])
    ev, _ = run(lg, 3.5, [s, phone(422, 418)])
    assert lg.students[1].state == "REVIEW"
    assert [e.kind for e in ev] == ["review_started"] and ev[0].data["signals"] == ["PHONE"]


def test_brief_glance_stays_normal():
    lg = ExamLogic()
    _, t = run(lg, 1.0, [student(1, 400, turn=0.35)])
    ev, t = run(lg, 10.0, [student(1, 400)], start=t)
    assert lg.students[1].state == "NORMAL" and not ev


def test_sustained_side_look_reviews():
    lg = ExamLogic(review=3.0, flagged=8.0)
    ev, t = run(lg, 2.5, [student(1, 400, turn=-0.35), student(2, 700)])
    assert lg.students[1].state == "NORMAL" and lg.students[1].signals == ("SIDE LOOK",)
    ev, t = run(lg, 1.0, [student(1, 400, turn=-0.35), student(2, 700)], start=t)
    assert lg.students[1].state == "REVIEW" and lg.students[2].state == "NORMAL"
    assert [(e.kind, e.track_id) for e in ev] == [("review_started", 1)]
    assert ev[0].data["from"] == 0.0  # where a reviewer should start watching
    ev, t = run(lg, 5.5, [student(1, 400, turn=-0.35)], start=t)
    assert lg.students[1].state == "FLAGGED" and [e.kind for e in ev] == ["flagged"]
    assert [e.kind for e in lg.reviews] == ["review_started", "flagged"]


def test_score_decays():
    lg = ExamLogic(rise=1.0, decay=0.15, review=3.0)
    _, t = run(lg, 4.0, [student(1, 400, turn=0.35)])
    top = lg.students[1].score
    assert lg.students[1].state == "REVIEW"
    _, t = run(lg, 5.0, [student(1, 400)], start=t)
    assert abs(lg.students[1].score - (top - 0.75)) < 0.05
    assert lg.students[1].state == "REVIEW"  # hysteresis: no flicker right after the look ends
    _, t = run(lg, 30.0, [student(1, 400)], start=t)
    assert lg.students[1].score == 0 and lg.students[1].state == "NORMAL"


def test_looking_down_is_normal():
    lg = ExamLogic()
    # head bowed over the paper: nose drops to (or below) the shoulder line, still centred
    ev, t = run(lg, 10.0, [student(1, 400, nose_dy=5.0), student(2, 700, turn=0.08, nose_dy=-10.0)])
    assert not ev and all(s.state == "NORMAL" and s.score == 0 for s in lg.students.values())


def test_person_without_visible_body_is_not_a_student():
    # a coat on a chair or a sliver at the frame edge: detected as a person, but no shoulders or forearm
    lg = ExamLogic()
    ev, _ = run(lg, 5.0, [Track(1, (0, 0, 100, 300), "person", 0.9), student(2, 400)])
    assert not ev and set(lg.students) == {2} and lg.counts() == {"NORMAL": 1, "REVIEW": 0, "FLAGGED": 0}


def test_exam_empty_frames():
    lg = ExamLogic()
    _, t = run(lg, 4.0, [student(1, 400, turn=0.35)])
    assert lg.students[1].state == "REVIEW"
    ev, t = run(lg, 100 / 10, [], start=t)  # 100 frames with nobody detected
    assert not ev and lg.counts() == {"NORMAL": 0, "REVIEW": 0, "FLAGGED": 0}
    assert lg.students[1].score < 3.0 and lg.students[1].signals == ()
    run(ExamLogic(), 10.0, [])
