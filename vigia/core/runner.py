"""The loop every demo shares: read frames, let the app decide, draw, blur faces, save, show, log."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Protocol

import numpy as np

from .events import EventLog
from .scene import Scene, load_scene
from .types import Event
from .video import SourceError, Writer, frames, source_fps

ROOT = Path(__file__).resolve().parents[2]


class App(Protocol):
    name: str           # CLI name, e.g. "tables"; also the default scene file scenes/<name>.json
    blur_faces: bool    # False only where the face itself is the subject (drowsiness)
    tracks: list        # latest Tracks; people's head areas (and any Detector's raw people) are pixelated

    @staticmethod
    def add_args(p: argparse.ArgumentParser) -> None: ...

    def setup(self, scene: Scene | None, fps: float, size: tuple[int, int], args: argparse.Namespace) -> None: ...

    def process(self, t: float, frame: np.ndarray) -> list[Event]: ...

    def draw(self, frame: np.ndarray) -> np.ndarray: ...

    def summary(self) -> dict: ...


VEHICLES = {"car", "motorbike", "bus", "truck"}


def _to_blur(app, labels: set[str]) -> list[tuple]:
    """Boxes of these classes from the app's tracks plus the latest raw detections of every Detector it holds:
    a tracker leaves objects out for their first frames until it confirms them."""
    from .detect import Detector

    boxes = [tr.box for tr in getattr(app, "tracks", []) if tr.label in labels]
    for det in (v for v in vars(app).values() if isinstance(v, Detector)):
        names = det.last.data.get("class_name", [])
        boxes += [tuple(float(v) for v in b) for b, n in zip(det.last.xyxy, names) if n in labels]
    return boxes


def people_to_blur(app) -> list[tuple]:
    return _to_blur(app, {"person"})


def vehicles_to_blur(app) -> list[tuple]:
    return _to_blur(app, VEHICLES)


def heads_to_blur(app) -> list[tuple]:
    """Heads placed by pose keypoints, for apps that run pose (bent-over or lying people)."""
    from .privacy import keypoint_head

    heads = [keypoint_head(tr.keypoints, tr.box) for tr in getattr(app, "tracks", [])
             if tr.label == "person" and tr.keypoints is not None]
    return [h for h in heads if h is not None]


def _scene(app: App, args, size) -> Scene | None:
    path = Path(args.scene) if args.scene else ROOT / "scenes" / f"{app.name}.json"
    if not path.exists():
        if args.scene:
            raise SourceError(f"cannot open scene {path}")
        return None
    if not args.scene:  # drawn for the sample video: say so, it may not fit another camera
        print(f"using scene {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path} "
              f"(draw your own with: vigia zones <video> --out scenes/<name>.json)", file=sys.stderr)
    return load_scene(path, *size)


def run(app: App, args: argparse.Namespace) -> int:
    writer = log = None
    window = False
    try:
        fps = source_fps(args.source) / max(1, args.every)
        it = frames(args.source, args.every, args.seconds)
        t, frame = next(it, (None, None))
        if frame is None:
            raise SourceError(f"cannot open {args.source}: no frames")
        size = (frame.shape[1], frame.shape[0])
        app.setup(_scene(app, args, size), fps, size, args)
        log = EventLog(Path(args.events) if getattr(args, "events", None) else ROOT / "out" / f"{app.name}-events.jsonl")
        blur = None
        if args.out:
            writer = Writer(args.out, fps, size)
            if app.blur_faces and not args.no_blur:
                from .privacy import FaceBlur
                blur = FaceBlur()
        while frame is not None:
            for ev in app.process(t, frame):
                log.write(ev)
            if writer:
                base = frame.copy()
                if blur:
                    blur(base, people_to_blur(app), vehicles_to_blur(app), heads_to_blur(app), t)
                writer.write(app.draw(base))
            if args.show:
                import cv2

                window = True
                cv2.imshow(f"vigia {app.name} (q to quit)", app.draw(frame.copy()))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            t, frame = next(it, (None, None))
    except (SourceError, OSError) as e:  # missing file or model, failed download, camera gone
        print(f"error: {e}", file=sys.stderr)
        return 2
    except ImportError as e:
        print(f"error: {e.name or e} is not installed; run: uv sync --all-extras", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        pass
    finally:
        if writer:
            writer.close()
        if log:
            log.close()
        if window:
            import cv2

            cv2.destroyAllWindows()
    print(json.dumps(app.summary(), indent=2, ensure_ascii=False, default=float))
    return 0
