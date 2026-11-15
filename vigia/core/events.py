"""Append-only JSONL log of what each demo detects, flushed line by line so a crash loses nothing."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .types import Event


class EventLog:
    def __init__(self, path: Path | None):
        self._f = None
        if path is not None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self._f = open(path, "w", encoding="utf-8")

    def write(self, ev: Event) -> None:
        if self._f:
            self._f.write(json.dumps(asdict(ev), ensure_ascii=False, default=float) + "\n")
            self._f.flush()

    def close(self) -> None:
        if self._f:
            self._f.close()
            self._f = None
