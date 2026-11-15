from vigia.apps.people_counter.logic import CounterLogic
from vigia.core.scene import Scene
from vigia.core.types import Track

# Horizontal line at y=500 drawn left to right: with in_side=1 the "in" side is below it (walking down the frame).
SCENE = Scene(1000, 1000, lines={"door": ((0, 500), (1000, 500))})


def person(tid, y, x=500):
    """A person whose feet (bottom centre of the box) are at (x, y)."""
    return Track(tid, (x - 20, y - 100, x + 20, y), "person", 0.9)


def walk(logic, tid, ys, start=0.0, fps=10, x=500):
    events = []
    for i, y in enumerate(ys):
        events += logic.update(start + i / fps, [person(tid, y, x)])
    return events, start + len(ys) / fps


def test_counts_in_and_out():
    lg = CounterLogic(SCENE)
    ev, t = walk(lg, 1, range(300, 720, 20))
    assert [e.kind for e in ev] == ["entered"] and ev[0].track_id == 1
    ev, t = walk(lg, 2, range(700, 280, -20), start=t)
    assert [e.kind for e in ev] == ["exited"] and ev[0].track_id == 2
    assert (lg.entered, lg.exited, lg.inside_now) == (1, 1, 0)


def test_in_side_flips_direction():
    lg = CounterLogic(SCENE, in_side=-1)
    walk(lg, 1, range(300, 720, 20))
    assert (lg.entered, lg.exited) == (0, 1)


def test_jitter_on_line_counts_once():
    lg = CounterLogic(SCENE, min_travel=12.0)
    ev, t = walk(lg, 1, range(300, 500, 20))
    ev2, t = walk(lg, 1, [495, 505, 497, 503] * 10, start=t)  # hesitates on the line for 4 s
    ev3, t = walk(lg, 1, range(510, 700, 20), start=t)
    assert [e.kind for e in ev + ev2 + ev3] == ["entered"]
    assert (lg.entered, lg.exited) == (1, 0)


def test_one_count_per_direction_until_it_recrosses():
    lg = CounterLogic(SCENE)
    walk(lg, 1, [300, 400, 600, 700])          # in
    walk(lg, 1, [700, 600, 650, 700], start=1)  # stays on the in side: nothing more
    assert (lg.entered, lg.exited) == (1, 0)
    walk(lg, 1, [600, 400, 300], start=2)      # walks back out
    assert (lg.entered, lg.exited, lg.inside_now) == (1, 1, 0)


def test_counter_id_switch_counts_once_more():
    lg = CounterLogic(SCENE)
    _, t = walk(lg, 1, range(300, 620, 20))
    # Same person, new id after an occlusion, already past the line: not counted again.
    _, t = walk(lg, 2, range(620, 800, 20), start=t)
    assert lg.entered == 1
    # Worst case: the new id first appears back on the old side (box jumped), then crosses: one more, never more.
    lg2 = CounterLogic(SCENE)
    _, t = walk(lg2, 1, range(300, 620, 20))
    _, t = walk(lg2, 3, [450, 440] + list(range(520, 800, 20)), start=t)
    assert lg2.entered == 2 and lg2.exited == 0


def test_inside_never_negative():
    lg = CounterLogic(SCENE)
    walk(lg, 1, range(700, 280, -20))
    walk(lg, 2, range(700, 280, -20), start=5)
    assert (lg.exited, lg.inside_now) == (2, 0)
    walk(lg, 3, range(300, 720, 20), start=10)
    assert (lg.entered, lg.inside_now) == (1, 0)


def test_per_15min_buckets():
    lg = CounterLogic(SCENE)
    walk(lg, 1, [300, 700], start=10)    # in at 0:00-0:15
    walk(lg, 2, [700, 300], start=910)   # out at 0:15-0:30
    walk(lg, 3, [300, 700], start=920)   # in at 0:15-0:30
    assert lg.per_15min == {"00:00": (1, 0), "00:15": (1, 1)}


def test_crossing_beside_a_short_door_is_ignored():
    door = Scene(1000, 1000, lines={"door": ((400, 500), (600, 500))})
    lg = CounterLogic(door)
    walk(lg, 1, range(300, 720, 20), x=800)  # walks past outside the door's width
    walk(lg, 2, range(300, 720, 20), x=500)  # walks through it
    assert (lg.entered, lg.exited) == (1, 0)


def test_count_empty_frames():
    lg = CounterLogic(SCENE)
    walk(lg, 1, [300, 320])
    events = []
    for i in range(100):
        events += lg.update(1 + i / 10, [])
    assert not events and (lg.entered, lg.exited, lg.inside_now) == (0, 0, 0)
    assert not lg.sides  # people gone for long are forgotten


def test_untracked_detections_are_ignored():
    lg = CounterLogic(SCENE)
    walk(lg, -1, range(300, 720, 20))
    assert lg.entered == 0
