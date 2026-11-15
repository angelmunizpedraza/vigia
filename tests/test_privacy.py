from vigia.core.privacy import head_boxes, plate_box


def test_upright_head_band_spans_full_width():
    # a person whose box also takes in a bag on the right: the head is off-centre, so the band covers the full width
    [b] = head_boxes((100, 50, 300, 450))
    assert b[0] == 100 and b[2] == 300
    assert b[1] < 50 < 90 < b[3] < 450 * 0.5


def test_lying_person_gets_both_ends():
    ends = head_boxes((0, 100, 400, 200))  # wider than tall: head at either end
    assert len(ends) == 2
    assert ends[0][0] == 0 and ends[1][2] == 400
    assert all(e[1] == 100 and e[3] == 200 for e in ends)


def test_plate_box_is_lower_part_of_vehicle():
    x1, y1, x2, y2 = plate_box((10, 100, 210, 300))
    assert (x1, x2, y2) == (10, 210, 300) and 180 < y1 < 220


def test_keypoint_head_box_around_face_points():
    import numpy as np

    from vigia.core.privacy import keypoint_head

    kp = np.zeros((17, 3), np.float32)
    kp[0] = (200, 300, 0.9)  # nose of someone bent over, low in a tall box
    kp[3] = (190, 295, 0.8)
    x1, y1, x2, y2 = keypoint_head(kp, (100, 100, 300, 500))
    assert x1 < 190 and x2 > 200 and y1 < 295 and y2 > 300
    assert keypoint_head(np.zeros((17, 3), np.float32), (0, 0, 10, 10)) is None


def test_blur_holds_regions_for_a_moment():
    import numpy as np

    from vigia.core.privacy import FaceBlur

    fb = object.__new__(FaceBlur)  # no YuNet model needed
    fb.grow, fb.hold, fb._recent = 0.25, 1.0, []
    fb.faces = lambda frame: np.zeros((0, 4))
    pattern = (np.indices((100, 100)).sum(0) % 2 * 255).astype(np.uint8)[..., None].repeat(3, 2)

    def blurred(t, heads):
        out = fb(pattern.copy(), heads=heads, t=t)
        return not np.array_equal(out[40:60, 40:60], pattern[40:60, 40:60])

    assert blurred(0.0, [(40, 40, 60, 60)])
    assert blurred(0.5, [])          # detector missed the person: the last region is still hidden
    assert not blurred(2.0, [])      # after the hold it is released
