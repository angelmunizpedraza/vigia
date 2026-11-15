import numpy as np

from vigia.apps.store_flow.logic import StoreLogic
from vigia.core.scene import Scene
from vigia.core.types import Track

# Feet points decide the zone. Open floor is anywhere outside the polygons, e.g. (200, 500).
SCENE = Scene(1000, 1000, zones={
    "shelf-a": [(0, 0), (300, 0), (300, 300), (0, 300)],
    "checkout-1": [(400, 0), (600, 0), (600, 300), (400, 300)],
    "exit": [(800, 0), (1000, 0), (1000, 300), (800, 300)],
    "staff": [(400, 600), (600, 600), (600, 900), (400, 900)],
})
FLOOR, SHELF, CHECKOUT, EXIT, STAFF = (200, 500), (150, 200), (500, 200), (900, 200), (500, 800)


def person(tid, at):
    """A person whose feet (bottom centre of the box) are at `at`."""
    x, y = at
    return Track(tid, (x - 20, y - 100, x + 20, y), "person", 0.9)


def stay(logic, tid, at, seconds, start=0.0, fps=10):
    events = []
    n = round(seconds * fps)
    for i in range(n):
        events += logic.update(start + i / fps, [] if at is None else [person(tid, at)])
    return events, start + n / fps


def walk(logic, tid, steps, start=0.0):
    """steps: [(place or None, seconds), ...] -> all events, end time."""
    events, t = [], start
    for at, seconds in steps:
        ev, t = stay(logic, tid, at, seconds, start=t)
        events += ev
    return events, t


def test_normal_purchase_path():
    lg = StoreLogic(SCENE, dwell=4.0)
    ev, _ = walk(lg, 1, [(FLOOR, 1), (SHELF, 5), (FLOOR, 1), (CHECKOUT, 2), (FLOOR, 1), (EXIT, 1)])
    assert [e.kind for e in ev] == ["product", "paid", "left"]
    assert all(e.track_id == 1 for e in ev) and ev[0].data["zone"] == "shelf-a"
    assert lg.paths[1].state == "LEFT"
    assert lg.counts["PAID"] == 1 and lg.counts["LEFT"] == 1 and lg.counts["REVIEW"] == 0


def test_exit_without_checkout_reviews():
    lg = StoreLogic(SCENE, dwell=4.0)
    ev, _ = walk(lg, 1, [(FLOOR, 1), (SHELF, 5), (FLOOR, 1), (EXIT, 1)])
    assert [e.kind for e in ev] == ["product", "exit_without_checkout"]
    assert lg.paths[1].state == "REVIEW" and lg.counts["REVIEW"] == 1


def test_product_after_paying_reviews():
    lg = StoreLogic(SCENE, dwell=4.0)
    ev, _ = walk(lg, 1, [(FLOOR, 1), (CHECKOUT, 2), (FLOOR, 1), (SHELF, 5), (FLOOR, 1), (EXIT, 1)])
    assert [e.kind for e in ev] == ["paid", "product", "exit_without_checkout"]


def test_browsing_without_product_leaves():
    lg = StoreLogic(SCENE, dwell=4.0)
    ev, _ = walk(lg, 1, [(FLOOR, 1), (SHELF, 2), (FLOOR, 1), (EXIT, 1)])
    assert [e.kind for e in ev] == ["left"]
    assert lg.paths[1].state == "LEFT"


def test_staff_excluded():
    lg = StoreLogic(SCENE, dwell=4.0)
    ev, _ = walk(lg, 1, [(FLOOR, 1), (STAFF, 0.5), (SHELF, 6), (FLOOR, 1), (EXIT, 1)])
    assert [e.kind for e in ev] == ["staff"]
    assert lg.paths[1].state == "STAFF" and lg.counts["REVIEW"] == 0
    assert "shelf-a" not in lg.zone_seconds  # staff time is not shop dwell time


def test_short_shelf_pass_is_not_product():
    lg = StoreLogic(SCENE, dwell=4.0)
    ev, _ = walk(lg, 1, [(SHELF, 3), (FLOOR, 2), (SHELF, 3), (EXIT, 1)])
    assert [e.kind for e in ev] == ["left"]  # 3 s + 3 s with a 2 s break is two short passes, not 6 s


def test_short_detection_gap_keeps_shelf_timer():
    lg = StoreLogic(SCENE, dwell=4.0)
    ev, _ = walk(lg, 1, [(SHELF, 2.5), (None, 0.5), (SHELF, 2)])
    assert [e.kind for e in ev] == ["product"]


def test_first_seen_in_exit_is_not_leaving():
    lg = StoreLogic(SCENE)
    ev, _ = walk(lg, 1, [(EXIT, 1), (FLOOR, 1)])  # someone coming in through the door
    assert ev == [] and lg.paths[1].state == "TRACKING"


def test_id_switch_starts_a_new_path():
    """A known limit: a new tracker id loses the product, so the exit is a plain LEFT, not REVIEW."""
    lg = StoreLogic(SCENE, dwell=4.0)
    _, t = walk(lg, 1, [(FLOOR, 1), (SHELF, 5), (FLOOR, 1)])
    ev, _ = walk(lg, 2, [(FLOOR, 0.5), (EXIT, 1)], start=t)
    assert [e.kind for e in ev] == ["left"] and lg.paths[1].state == "PRODUCT"


def test_dwell_seconds_per_zone():
    lg = StoreLogic(SCENE)
    walk(lg, 1, [(FLOOR, 1), (CHECKOUT, 3), (FLOOR, 1)])
    walk(lg, 2, [(CHECKOUT, 1)], start=10.0)
    assert abs(lg.zone_seconds["checkout-1"] - 4.0) < 0.25
    assert lg.zone_visitors["checkout-1"] == 2


def test_untracked_and_other_labels_ignored():
    lg = StoreLogic(SCENE)
    lg.update(0.0, [Track(-1, (130, 100, 170, 200), "person", 0.9), Track(3, (130, 100, 170, 200), "cart", 0.9)])
    assert lg.paths == {}


def test_nested_box_merged_but_child_kept():
    import supervision as sv

    from vigia.apps.store_flow.run import merge_nested, nested_in

    whole = [100, 100, 200, 500]
    above_cart = [102, 104, 198, 300]  # same person cut at the cart handle: same top edge
    child = [120, 330, 180, 500]       # inside the adult's box but with its own head lower down
    other = [400, 100, 500, 500]
    xyxy = np.array([whole, above_cart, child, other], float)
    assert nested_in(xyxy).tolist() == [-1, 0, -1, -1]
    out = merge_nested(sv.Detections(xyxy=xyxy, confidence=np.array([0.45, 0.9, 0.7, 0.8])))
    assert out.xyxy.tolist() == [whole, child, other] and out.confidence.tolist() == [0.9, 0.7, 0.8]
    assert nested_in(np.zeros((0, 4))).tolist() == []


def test_store_empty_frames():
    lg = StoreLogic(SCENE)
    _, t = walk(lg, 1, [(SHELF, 2)])
    ev, _ = stay(lg, 1, None, 100.0, start=t)  # 100 s with nobody: no crash, old paths are dropped
    assert ev == [] and lg.paths == {}
    assert lg.counts == {"PRODUCT": 0, "PAID": 0, "REVIEW": 0, "STAFF": 0, "LEFT": 0}
