"""Milliseconds per frame of each D-FINE size on this machine, on the same OpenVINO path vigia uses.
Prints a Markdown table.

uv run python scripts/bench.py --models s m --device cpu npu
"""
from __future__ import annotations

import argparse
import platform
import time

import numpy as np
import openvino as ov

from vigia.core.models import ensure


def _openvino(path, device):
    core = ov.Core()
    model = core.read_model(str(path))
    model.reshape([1, 3, 640, 640])  # the NPU needs static shapes; the ONNX export is dynamic
    m = core.compile_model(model, device, {"PERFORMANCE_HINT": "LATENCY"})
    return lambda x: m(x)


def bench(fn, runs: int = 20) -> float:
    x = np.random.rand(1, 3, 640, 640).astype(np.float32)
    for _ in range(3):
        fn(x)
    t = time.perf_counter()
    for _ in range(runs):
        fn(x)
    return (time.perf_counter() - t) / runs * 1000


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["s", "m"])
    p.add_argument("--device", nargs="+", default=["cpu"], help="OpenVINO devices: cpu, npu, gpu")
    a = p.parse_args()
    print(f"{platform.processor() or platform.machine()} - OpenVINO {ov.__version__}\n")
    print("| model | device | ms/frame | fps |\n|---|---|---|---|")
    for m in a.models:
        path = ensure(f"dfine-{m}")
        for d in a.device:
            try:
                ms = bench(_openvino(path, d.upper()))
                print(f"| D-FINE-{m.upper()} | {d} | {ms:.0f} | {1000 / ms:.1f} |")
            except Exception as e:  # noqa: BLE001 - report and keep benchmarking the rest
                print(f"| D-FINE-{m.upper()} | {d} | failed: {type(e).__name__} | |")


if __name__ == "__main__":
    main()
