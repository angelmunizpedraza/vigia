"""vigia <demo> --source <video|webcam index|rtsp://...> [options]"""
from __future__ import annotations

import argparse
import importlib
import sys

DEMOS = {
    "tables": ("vigia.apps.table_occupancy.run", "Which cafe tables are free or occupied"),
    "count": ("vigia.apps.people_counter.run", "People in, out and inside now through a doorway"),
    "crowd": ("vigia.apps.crowd_density.run", "Crowd density in people per square metre, with crush-risk alerts"),
    "fall": ("vigia.apps.fall_detector.run", "Falls, told apart from sitting or lying down on purpose"),
    "exam": ("vigia.apps.exam_monitor.run", "Exam moments worth a human review (phone in hand, sustained side look)"),
    "desk": ("vigia.apps.desk_activity.run", "Per-desk state: working, away, phone"),
    "store": ("vigia.apps.store_flow.run", "Shop paths: product picked, checkout, exit without checkout"),
    "road": ("vigia.apps.road_damage.run", "Potholes and cracks, each counted once, with a report"),
    "drowsy": ("vigia.apps.drowsiness.run", "Eyes-closed alarm for drivers and operators, with an Arduino"),
    "hands": ("vigia.apps.gesture_leds.run", "Five LEDs driven by raised fingers, with an Arduino"),
}


def _usage() -> str:
    lines = ["usage: vigia <demo> --source <video|0|rtsp://...> [options]", "", "demos:"]
    lines += [f"  {k:<8} {d}" for k, (_, d) in DEMOS.items()]
    lines += ["", "tools:", "  zones    draw zones and lines on a video's first frame: vigia zones <video> --out scenes/x.json",
              "  models   download models: vigia models [name ...]", "", "vigia <demo> --help for its options"]
    return "\n".join(lines)


def common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--source", required=True, help="video file, webcam index (0) or stream URL")
    p.add_argument("--scene", help="zones/lines JSON (default: scenes/<demo>.json if it exists)")
    p.add_argument("--out", help="save annotated output (.mp4 or .gif)")
    p.add_argument("--show", action="store_true", help="show a live window (q quits)")
    p.add_argument("--model", default="auto", choices=["auto", "s", "m"], help="D-FINE size (auto: m on an NPU, s on CPU)")
    p.add_argument("--device", default="auto", help="auto (NPU if present, else CPU), CPU or NPU")
    p.add_argument("--every", type=int, default=1, help="process one frame out of N")
    p.add_argument("--seconds", type=float, help="stop after this many seconds of video")
    p.add_argument("--no-blur", action="store_true", help="do not pixelate faces in saved output")
    p.add_argument("--events", help="JSONL event log path (default: out/<demo>-events.jsonl)")


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help"):
        print(_usage())
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd == "zones":
        from vigia.tools.zones import main as zones_main
        return zones_main(rest)
    if cmd == "models":
        from vigia.core.models import REGISTRY, ensure
        for name in rest or [k for k, v in REGISTRY.items() if v.url]:
            print(ensure(name))
        return 0
    if cmd not in DEMOS:
        print(f"unknown demo {cmd!r}\n\n{_usage()}", file=sys.stderr)
        return 2
    module = importlib.import_module(DEMOS[cmd][0])
    p = argparse.ArgumentParser(prog=f"vigia {cmd}", description=DEMOS[cmd][1])
    common_args(p)
    module.APP.add_args(p)
    args = p.parse_args(rest)
    if args.every < 1:
        p.error("--every must be 1 or more")
    from vigia.core.runner import run
    return run(module.APP(), args)


if __name__ == "__main__":
    sys.exit(main())
