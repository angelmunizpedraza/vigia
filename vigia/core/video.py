"""Frames with their timestamp, and MP4/GIF writers.

Files are timed by frame index / fps, so results are the same on any machine. Live sources (webcam, RTSP, or a
file that reports no fps) are timed with the monotonic clock.
"""
from __future__ import annotations

import math
import time
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


class SourceError(RuntimeError):
    pass


def _parse(source: str | int) -> str | int:
    return int(source) if isinstance(source, str) and source.isdigit() else source


def _is_stream(source: str | int) -> bool:
    return isinstance(source, int) or str(source).lower().startswith(("rtsp://", "http://", "https://"))


def _open(source: str | int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SourceError(f"cannot open {source}")
    return cap


def source_fps(source: str | int) -> float:
    source = _parse(source)
    cap = _open(source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    cap.release()
    return fps if 0 < fps < 240 else 30.0


def frames(source: str | int, every: int = 1, seconds: float | None = None) -> Iterator[tuple[float, np.ndarray]]:
    source = _parse(source)
    cap = _open(source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    live = _is_stream(source) or not 0 < fps < 240
    start = time.monotonic() if live else 0.0
    i = 0
    try:
        while True:
            if i % every:
                if not cap.grab():
                    break
                i += 1
                continue
            ok, frame = cap.read()
            if not ok:
                break
            t = time.monotonic() - start if live else i / fps
            if seconds is not None and t > seconds + 1e-9:
                break
            yield t, frame
            i += 1
    finally:
        cap.release()


class Writer:
    """`.mp4` with OpenCV, `.gif` with Pillow (<= 640 px wide, <= 10 fps, one palette for all frames)."""

    def __init__(self, path: Path | str, fps: float, size: tuple[int, int], gif_width: int = 640, gif_fps: float = 10,
                 gif_max_frames: int = 600):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.gif = self.path.suffix.lower() == ".gif"
        self.fps = fps
        if self.gif:
            self._step = max(1, math.ceil(fps / gif_fps - 1e-9))
            self._size = (min(gif_width, size[0]), round(size[1] * min(gif_width, size[0]) / size[0]))
            self._frames: list[np.ndarray] = []  # RGB, quantized at close() with one palette for the whole clip
            self._max = gif_max_frames  # ~1 min at 10 fps, ~400 MB of RAM: a GIF is not for long or live recordings
            self._n = 0
        else:
            self._vw = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

    def write(self, frame_bgr: np.ndarray) -> None:
        if not self.gif:
            self._vw.write(frame_bgr)
            return
        if self._n % self._step == 0 and len(self._frames) < self._max:
            self._frames.append(cv2.resize(frame_bgr, self._size, interpolation=cv2.INTER_AREA)[:, :, ::-1].copy())
        self._n += 1

    def close(self) -> None:
        if not self.gif:
            self._vw.release()
        elif self._frames:
            pal = gif_palette(self._frames)
            frames = [Image.fromarray(f).quantize(palette=pal, dither=Image.Dither.NONE) for f in self._frames]
            ms = round(1000 * self._step / self.fps)
            frames[0].save(self.path, save_all=True, append_images=frames[1:], duration=ms, loop=0, optimize=True)
            self._frames = []


def gif_palette(frames: list[np.ndarray], samples: int = 12) -> Image.Image:
    """One palette for the whole GIF (smaller file, no colour flicker), learnt from frames spread over the clip so
    colours that appear late survive, plus the exact UI colours so states keep their colour."""
    from .draw import DARK, PALETTE, WHITE

    ui = [(c[2], c[1], c[0]) for c in (*PALETTE.values(), WHITE, DARK)]  # BGR -> RGB
    idx = np.linspace(0, len(frames) - 1, min(samples, len(frames))).astype(int)
    mosaic = Image.fromarray(np.vstack([frames[i] for i in idx]))
    n = 256 - len(ui)
    learnt = mosaic.quantize(colors=n, method=Image.Quantize.MEDIANCUT).getpalette()[: n * 3]
    learnt += [0] * (n * 3 - len(learnt))
    pal = Image.new("P", (1, 1))
    pal.putpalette(learnt + [v for rgb in ui for v in rgb])
    return pal
