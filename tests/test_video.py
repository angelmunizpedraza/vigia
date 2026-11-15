import cv2
import numpy as np
import pytest
from PIL import Image

from vigia.core import video
from vigia.core.video import SourceError, Writer, frames


def make_video(path, n=30, fps=10, size=(64, 48)):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    for i in range(n):
        w.write(np.full((size[1], size[0], 3), i * 8 % 255, np.uint8))
    w.release()
    return path


def test_frames_respects_every_and_seconds(tmp_path):
    p = make_video(tmp_path / "v.avi")
    got = list(frames(str(p), every=3, seconds=2.0))
    ts = [round(t, 3) for t, _ in got]
    assert ts == [0.0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8]
    assert got[0][1].shape == (48, 64, 3)


def test_frames_missing_source_raises(tmp_path):
    with pytest.raises(SourceError, match="cannot open"):
        list(frames(str(tmp_path / "nope.mp4")))


class FakeLive:
    def __init__(self, *_):
        self.n = 0

    def isOpened(self):
        return True

    def get(self, prop):
        return 0.0  # webcams often report 0 fps

    def read(self):
        self.n += 1
        return (self.n <= 4), np.zeros((4, 4, 3), np.uint8)

    def grab(self):
        return True

    def release(self):
        pass


def test_frames_live_source_uses_monotonic_time(monkeypatch):
    clock = iter([100.0, 100.0, 100.04, 100.09, 100.15])
    monkeypatch.setattr(video.cv2, "VideoCapture", FakeLive)
    monkeypatch.setattr(video.time, "monotonic", lambda: next(clock))
    ts = [t for t, _ in frames(0)]
    assert ts == pytest.approx([0.0, 0.04, 0.09, 0.15])


def test_gif_writer_downscales_and_caps_fps(tmp_path):
    p = tmp_path / "o.gif"
    w = Writer(p, fps=30, size=(1440, 810), gif_width=720, gif_fps=12)
    for i in range(30):
        w.write(np.full((810, 1440, 3), i * 8, np.uint8))
    w.close()
    im = Image.open(p)
    assert im.size == (720, 405)
    assert im.n_frames <= 12  # 1 s of 30 fps video at <= 12 fps


def test_gif_keeps_ui_colours_drawn_only_late(tmp_path):
    from vigia.core.draw import PALETTE

    p = tmp_path / "late.gif"
    w = Writer(p, fps=10, size=(200, 100))
    for i in range(20):
        f = np.full((100, 200, 3), (40 + i, 60, 80), np.uint8)
        if i == 19:
            f[20:80, 50:150] = PALETTE["ok"]  # appears only in the last frame
        w.write(f)
    w.close()
    im = Image.open(p)
    im.seek(im.n_frames - 1)
    b, g, r = PALETTE["ok"]
    assert im.convert("RGB").getpixel((100, 50)) == (r, g, b)
