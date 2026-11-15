"""Download each demo's sample videos from their original source into samples/<demo>/ (git-ignored).

The videos are not redistributed in this repo (the Pexels licence forbids it); only this list is.

uv run python scripts/get_samples.py            all demos
uv run python scripts/get_samples.py tables     one demo
"""
from __future__ import annotations

import json
import sys
import tempfile
import urllib.request
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130 Safari/537.36"


def manifests(demos: list[str]) -> dict[str, list[dict]]:
    out = {}
    for m in sorted((ROOT / "vigia" / "apps").glob("*/samples.json")):
        demo = m.parent.name
        if not demos or demo in demos:
            out[demo] = json.loads(m.read_text(encoding="utf-8"))
    return out


def fetch(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)


def transcode(src: Path, dest: Path, start: float, end: float | None, max_width: int = 1920) -> tuple:
    """Trim to [start, end] seconds and cap the width, so samples stay small and quick to process."""
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open download {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    r = min(1.0, max_width / max(w, h))
    size = (round(w * r) // 2 * 2, round(h * r) // 2 * 2)
    vw = cv2.VideoWriter(str(dest), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    i = n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = i / fps
        i += 1
        if t < start:
            continue
        if end is not None and t > end:
            break
        vw.write(cv2.resize(frame, size, interpolation=cv2.INTER_AREA) if r < 1 else frame)
        n += 1
    cap.release()
    vw.release()
    return size, fps, n / fps


def main(argv: list[str]) -> int:
    failed = 0
    for demo, items in manifests(argv).items():
        for it in items:
            dest = ROOT / "samples" / demo / f"{it['name']}.mp4"
            if dest.exists():
                print(f"ok      {dest.relative_to(ROOT)}")
                continue
            if not it.get("file_url"):
                print(f"manual  {demo}/{it['name']}: download from {it['page']} ({it['license']})")
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory() as tmp:
                raw = Path(tmp) / "raw"
                try:
                    fetch(it["file_url"], raw)
                    start, end = (it.get("seconds") or [0, None])
                    size, fps, dur = transcode(raw, dest, start, end, it.get("max_width", 1920))
                except Exception as e:  # noqa: BLE001 - report every failure, keep going
                    print(f"FAILED  {demo}/{it['name']}: {e}")
                    dest.unlink(missing_ok=True)
                    failed += 1
                    continue
            print(f"saved   {dest.relative_to(ROOT)}  {size[0]}x{size[1]} {fps:.0f} fps {dur:.1f} s  ({it['license']}, {it['credit']})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
