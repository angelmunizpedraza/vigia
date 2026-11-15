import json
import math
from pathlib import Path

import numpy as np
import pytest

from vigia.apps.gesture_leds.logic import GestureLogic, fingers_up

# Synthetic right hand, palm to the camera, image pixels (x right, y down, z negative = towards the camera).
# The wrist is at the origin, the palm (wrist to middle knuckle) is 100 px long, the thumb is on the image right.
MCP = {5: (35, -95), 9: (12, -100), 13: (-10, -95), 17: (-30, -85)}
BONES = {5: (45, 28, 22), 9: (50, 32, 24), 13: (46, 30, 23), 17: (36, 22, 20)}
THUMB_OUT = [(25, -20, 0), (45, -45, 0), (70, -65, 0), (90, -80, 0)]
THUMB_ACROSS_FIST = [(25, -20, -5), (40, -40, -15), (30, -60, -40), (8, -72, -48)]
THUMB_ACROSS_PALM = [(25, -20, -3), (35, -45, -10), (15, -70, -12), (-5, -85, -8)]


def hand(fingers: str, thumb=THUMB_OUT) -> np.ndarray:
    """`fingers`: index..pinky, "1" straight, "0" curled into the palm (90, 100 and 70 degrees at the joints)."""
    p = np.zeros((21, 3))
    p[1:5] = thumb
    for up, m in zip(fingers, MCP):
        p[m] = (*MCP[m], 0)
        phi = 0.0
        for k, (bend, length) in enumerate(zip((0, 0, 0) if up == "1" else (90, 100, 70), BONES[m])):
            phi += math.radians(bend)  # curl towards the camera, the palm side
            p[m + k + 1] = p[m + k] + length * np.array((0, -math.cos(phi), -math.sin(phi)))
    return p


def rotate(p: np.ndarray, deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return p @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]).T + (640, 360, 0)


MIRROR = np.array([-1, 1, 1])  # a left hand is a mirrored right hand


def test_open_hand_all_up():
    assert fingers_up(hand("1111"), "Right") == (1, 1, 1, 1, 1)
    assert fingers_up(hand("1111")[:, :2], "Right") == (1, 1, 1, 1, 1)  # works without z too


def test_fist_all_down():
    assert fingers_up(hand("0000", THUMB_ACROSS_FIST), "Right") == (0, 0, 0, 0, 0)


def test_thumb_up_on_a_fist():
    assert fingers_up(hand("0000"), "Right") == (1, 0, 0, 0, 0)


def test_thumb_across_the_palm_is_down():
    assert fingers_up(hand("1111", THUMB_ACROSS_PALM), "Right") == (0, 1, 1, 1, 1)


def test_rotated_hand_still_works():
    for deg in range(0, 360, 30):  # 90 degrees as in the plan, and every other 30 degrees
        assert fingers_up(rotate(hand("1111"), deg), "Right") == (1, 1, 1, 1, 1), deg
        assert fingers_up(rotate(hand("0110"), deg), "Right") == (1, 0, 1, 1, 0), deg
        assert fingers_up(rotate(hand("0000", THUMB_ACROSS_FIST), deg), "Right") == (0, 0, 0, 0, 0), deg


def test_left_hand_thumb():
    assert fingers_up(hand("1111") * MIRROR, "Left") == (1, 1, 1, 1, 1)
    assert fingers_up(hand("1111", THUMB_ACROSS_PALM) * MIRROR, "Left") == (0, 1, 1, 1, 1)
    assert fingers_up(hand("0000", THUMB_ACROSS_FIST) * MIRROR, "Left") == (0, 0, 0, 0, 0)
    assert fingers_up(rotate(hand("0001") * MIRROR, 90), "Left") == (1, 0, 0, 0, 1)


FRAMES = json.loads((Path(__file__).parent / "test_hands_landmarks.json").read_text())["frames"]


@pytest.mark.parametrize("fr", FRAMES, ids=lambda f: f"frame{f['frame']}-{f['expected']}")
def test_real_landmarks_from_the_sample(fr):
    p = np.array(fr["landmarks"])
    want = tuple(int(c) for c in fr["expected"])
    assert fingers_up(p, fr["handedness"]) == want, fr["note"]
    assert fingers_up(rotate(p, 90), fr["handedness"]) == want, fr["note"]
    assert fingers_up(p * MIRROR, "Left") == want, fr["note"]


def feed(lg, masks, start=0.0, fps=10):
    events = []
    for i, m in enumerate(masks):
        events += lg.update(start + i / fps, m)
    return events


def test_debounce():
    lg = GestureLogic(stable_frames=3)
    two = (0, 1, 1, 0, 0)
    ev = feed(lg, [two, two, (0, 0, 0, 0, 0), two, two])
    assert lg.mask == "00000" and ev == []          # never 3 frames in a row yet
    ev = feed(lg, [two], start=0.5)
    assert lg.mask == "01100" and [e.kind for e in ev] == ["leds"]
    assert ev[0].data == {"mask": "01100", "fingers": ["index", "middle"]} and ev[0].t == 0.5
    ev = feed(lg, [(0, 1, 0, 0, 0), two, two], start=0.6)  # one-frame flicker
    assert lg.mask == "01100" and ev == []


def test_hand_leaving_turns_leds_off():
    lg = GestureLogic(stable_frames=3)
    feed(lg, [(1, 1, 1, 1, 1)] * 3)
    assert lg.mask == "11111" and lg.hand
    ev = feed(lg, [None, None], start=1.0)
    assert lg.mask == "11111" and not lg.hand and ev == []
    ev = feed(lg, [None], start=1.2)
    assert lg.mask == "00000" and ev[0].data["mask"] == "00000"


def test_hands_empty_frames():
    lg = GestureLogic()
    assert feed(lg, [None] * 100) == []
    assert lg.mask == "00000" and not lg.hand


def test_serial_protocol_over_loop():
    pytest.importorskip("serial")
    from vigia.apps.gesture_leds.board import SerialBoard

    board = SerialBoard("loop://")
    board.update("10110", now=0.0)   # first call: sent at once
    board.update("10110", now=0.2)   # same mask, heartbeat not due: nothing
    board.update("00000", now=0.3)   # change: sent at once
    board.update("00000", now=0.9)   # 0.5 s since the last line: heartbeat repeats the state
    assert board.ser.read(100) == b"LEDS 10110\nLEDS 00000\nLEDS 00000\n"
    board.close()


def test_virtual_board_draws_leds():
    from vigia.apps.gesture_leds.board import VirtualBoard

    frame = np.zeros((1920, 1080, 3), np.uint8)
    board = VirtualBoard()
    board.update("10100")
    board.draw(frame)
    assert frame.any()
