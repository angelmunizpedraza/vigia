"""How often does `vigia fall` catch a fall, and how often does it cry wolf? Measured on GMDCSA24.

GMDCSA24: Alam et al., CC BY 4.0, https://zenodo.org/records/12921216 (4 subjects, 79 fall clips, 81 clips of
activities of daily living, ADL). Extract it anywhere and point this script at the folder holding `Subject 1..4`:

    uv run python scripts/eval_fall.py samples/fall_detector/GMDCSA24/<extracted folder> --every 2 --subjects 3 4

It runs the demo's own pipeline (detector, tracker, pose, FallLogic) headless on every clip. A clip counts as a
detected fall if any `fall_detected` event fires. Prints Markdown tables: recall on fall clips, false alarms on ADL
clips (overall and by kind of activity), a separate table for the subjects not used to tune the thresholds, and the
clips it got wrong. Each clip's result is appended to `--results` as soon as it is known; a rerun with the same
settings skips the clips already there, so an interrupted run resumes.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

from vigia.apps.fall_detector.run import FallApp
from vigia.core.video import frames, source_fps

# For the false-alarm breakdown; a clip goes to the first kind whose keywords appear in its CSV description.
KINDS = [
    ("lying down or getting up (bed or floor)", ("sleep", "lying")),
    ("exercise, yoga, push-ups", ("exercis", "yoga", "push-up")),
    ("picking something up from the floor", ("pick",)),
    ("sitting on or getting up from the floor", ("floor", "ground")),
    ("sitting on a bed or chair", ("sit",)),
    ("walking, standing, other", ()),
]


def descriptions(csv_path: Path) -> dict[str, str]:
    """File name -> lower-case description. Subject 1's CSV has (length, description), the others (length, time of
    recording, attire, description); descriptions can contain commas, so everything after those fields is kept."""
    lines = csv_path.read_text(encoding="utf-8", errors="replace").splitlines()
    skip = 3 if lines and "attire" in lines[0].lower() else 1
    out = {}
    for line in lines[1:]:
        name, _, rest = line.partition(",")
        if name.strip().endswith(".mp4"):
            out[name.strip()] = rest.split(",", skip)[-1].strip().lower()
    return out


def kind_of(text: str) -> str:
    return next(k for k, words in KINDS if not words or any(w in text for w in words))


def clips(dataset: Path, subjects: list[int]) -> list[tuple[int, str, Path, str]]:
    """(subject, "Fall" | "ADL", path, description) for every clip of the chosen subjects."""
    out = []
    for s in subjects:
        for cls in ("Fall", "ADL"):
            folder = dataset / f"Subject {s}" / cls
            desc = descriptions(dataset / f"Subject {s}" / f"{cls}.csv")
            if not folder.is_dir():
                raise SystemExit(f"error: {folder} not found; point the script at the folder holding 'Subject 1..4'")
            out += [(s, cls, p, desc.get(p.name, "")) for p in sorted(folder.glob("*.mp4"))]
    return out


def fires(app: FallApp, path: Path, args: argparse.Namespace) -> bool:
    """Run the demo pipeline on one clip; True if it raised fall_detected."""
    fired = False
    for i, (t, frame) in enumerate(frames(str(path), args.every)):
        if i == 0:
            app.setup(None, source_fps(str(path)) / args.every, (frame.shape[1], frame.shape[0]), args)
        fired |= any(e.kind == "fall_detected" for e in app.process(t, frame))
    return fired


def pct(a: int, b: int) -> str:
    return f"{a}/{b} ({a / b:.0%})" if b else "0/0"


def report(rows: list[tuple[int, str, Path, str, bool]], title: str) -> str:
    falls = [r for r in rows if r[1] == "Fall"]
    adl = [r for r in rows if r[1] == "ADL"]
    tp = sum(r[4] for r in falls)
    fp = sum(r[4] for r in adl)
    prec = f"{tp / (tp + fp):.0%}" if tp + fp else "n/a"
    lines = [f"**{title}**", "", "| Fall clips detected (recall) | ADL clips with a false alarm | Precision (clips) |",
             "|---|---|---|", f"| {pct(tp, len(falls))} | {pct(fp, len(adl))} | {prec} |", "",
             "| ADL kind | Clips with a false alarm |", "|---|---|"]
    for k, _ in KINDS:
        sub = [r for r in adl if kind_of(r[3]) == k]
        if sub:
            lines.append(f"| {k} | {pct(sum(r[4] for r in sub), len(sub))} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("dataset", type=Path, help="folder holding 'Subject 1' .. 'Subject 4'")
    p.add_argument("--every", type=int, default=2, help="process one frame out of N (the clips are 15 or 30 fps)")
    p.add_argument("--subjects", type=int, nargs="+", default=[1, 2, 3, 4])
    p.add_argument("--tuned-on", type=int, nargs="*", default=[1, 2],
                   help="subjects the thresholds were tuned on; the others also get a held-out table")
    p.add_argument("--model", default="auto", choices=["auto", "s", "m"])
    p.add_argument("--device", default="auto")
    p.add_argument("--results", type=Path, default=Path("out/fall-eval-clips.csv"),
                   help="per-clip results, appended as the run goes and reused by a rerun with the same settings")
    FallApp.add_args(p)
    args = p.parse_args(argv)

    todo = clips(args.dataset, args.subjects)
    # shortcut: the key covers the CLI settings, not the code; delete --results after changing logic.py or run.py
    settings = (f"every={args.every} model={args.model} device={args.device} descent={args.descent_speed} "
                f"angle={args.lying_angle} confirm={args.confirm_after} recover={args.recover_window}")
    done = {}
    if args.results.exists():
        with args.results.open(newline="") as f:
            done = {(int(r["subject"]), r["class"], r["file"]): r["fired"] == "1"
                    for r in csv.DictReader(f) if r["settings"] == settings}
    args.results.parent.mkdir(parents=True, exist_ok=True)
    new = not args.results.exists()
    app, rows, start = FallApp(), [], time.monotonic()
    with args.results.open("a", newline="") as f:
        out = csv.writer(f)
        if new:
            out.writerow(["subject", "class", "file", "fired", "settings"])
        for i, (s, cls, path, desc) in enumerate(todo, 1):
            fired = done.get((s, cls, path.name))
            if fired is None:
                fired = fires(app, path, args)
                out.writerow([s, cls, path.name, int(fired), settings])
                f.flush()
            rows.append((s, cls, path, desc, fired))
            print(f"[{i}/{len(todo)}] Subject {s}/{cls}/{path.name}: {'FALL' if fired else '-'}", file=sys.stderr)
    setup = f"one frame in {args.every}, {app.det.name if hasattr(app, 'det') else 'D-FINE (' + args.model + ')'}"
    held = [s for s in args.subjects if s not in args.tuned_on]
    if held and held != args.subjects:
        print(report([r for r in rows if r[0] in held], f"GMDCSA24, held-out subjects {', '.join(map(str, held))} "
                                                         f"(not used for tuning), {setup}") + "\n")
    print(report(rows, f"GMDCSA24, subjects {', '.join(map(str, args.subjects))}, {setup}"))
    wrong = [r for r in rows if (r[1] == "Fall") != r[4]]
    print("\n| Clip | Got | Description |\n|---|---|---|")
    for s, cls, path, desc, fired in wrong:
        print(f"| Subject {s}/{cls}/{path.name} | {'false alarm' if fired else 'missed'} | {desc.strip()[:90]} |")
    print(f"\n{len(todo)} clips in {time.monotonic() - start:.0f} s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
