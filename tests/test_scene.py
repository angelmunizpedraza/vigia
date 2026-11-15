import json
import warnings

import pytest

from vigia.core.scene import box_fraction_in_polygon, load_scene, point_in_polygon, side_of_line
from vigia.core.types import Track

L_SHAPE = [(0, 0), (10, 0), (10, 4), (4, 4), (4, 10), (0, 10)]


def test_point_in_polygon_concave():
    assert point_in_polygon((2, 8), L_SHAPE)
    assert point_in_polygon((8, 2), L_SHAPE)
    assert not point_in_polygon((8, 8), L_SHAPE)  # the notch
    assert not point_in_polygon((-1, 5), L_SHAPE)


def test_side_of_line_sign_flips():
    a, b = (0, 5), (10, 5)
    assert side_of_line((5, 0), a, b) * side_of_line((5, 10), a, b) < 0
    assert side_of_line((5, 5), a, b) == 0


def test_box_fraction_full_half_none():
    square = [(0, 0), (10, 0), (10, 10), (0, 10)]
    assert box_fraction_in_polygon((2, 2, 4, 4), square) == pytest.approx(1.0)
    assert box_fraction_in_polygon((8, 0, 12, 10), square) == pytest.approx(0.5)
    assert box_fraction_in_polygon((20, 20, 30, 30), square) == pytest.approx(0.0)


def test_box_fraction_concave_polygon():
    # box covers the notch corner: 6x6 box from (4,4) to (10,10) lies entirely in the notch
    assert box_fraction_in_polygon((4, 4, 10, 10), L_SHAPE) == pytest.approx(0.0)
    # box (2,2)-(6,6): area 16, the part inside the L is 16 - 4 (the 2x2 corner in the notch)
    assert box_fraction_in_polygon((2, 2, 6, 6), L_SHAPE) == pytest.approx(12 / 16)


def _write(tmp_path, data):
    p = tmp_path / "scene.json"
    p.write_text(json.dumps(data))
    return p


def test_scene_scales_to_frame(tmp_path):
    p = _write(tmp_path, {"aspect": 16 / 9, "lines": {"door": [[0, 0.5], [1, 0.5]]},
                          "zones": {"table-1": [[0.5, 0.5], [1, 0.5], [1, 1]]}, "params": {"k": 1}})
    s = load_scene(p, 1920, 1080)
    assert s.lines["door"] == ((0.0, 540.0), (1920.0, 540.0))
    assert s.zones["table-1"][0] == (960.0, 540.0)
    assert s.params == {"k": 1}


def test_scene_warns_on_aspect_mismatch(tmp_path):
    p = _write(tmp_path, {"aspect": 16 / 9, "lines": {}, "zones": {}})
    with pytest.warns(UserWarning, match="aspect"):
        load_scene(p, 640, 480)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        load_scene(p, 1280, 720)


def test_track_anchors():
    t = Track(1, (10, 20, 30, 60), "person", 0.9)
    assert t.center == (20, 40)
    assert t.bottom_center == (20, 60)
    assert (t.w, t.h) == (20, 40)
