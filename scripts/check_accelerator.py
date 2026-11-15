"""Before trusting an accelerator, check it finds the same people as the CPU reference (onnxruntime).

uv run python scripts/check_accelerator.py --models s m --device NPU GPU

On a Core Ultra 7 258V with OpenVINO 2026.4 the NPU matched the CPU on 99.5-100 % of people; the GPU plugin found
only 32-77 %, so vigia never picks the GPU on its own.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import openvino as ov
import supervision as sv

from vigia.core.detect import postprocess
from vigia.core.models import ensure

ROOT = Path(__file__).resolve().parents[1]


def sample_frames(per_video: int = 8) -> list[np.ndarray]:
    out = []
    for v in sorted((ROOT / "samples").rglob("*.mp4"))[:6]:
        cap = cv2.VideoCapture(str(v))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for k in range(per_video):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * (k + 0.5) / per_video))
            ok, f = cap.read()
            if ok:
                out.append(f)
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["s", "m"])
    p.add_argument("--device", nargs="+", default=["NPU"])
    a = p.parse_args()
    frames = sample_frames()
    if not frames:
        print("no sample videos: run scripts/get_samples.py first")
        return 2
    core = ov.Core()
    for size in a.models:
        path = str(ensure(f"dfine-{size}"))
        ref = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        for dev in a.device:
            net = core.read_model(path)
            net.reshape([1, 3, 640, 640])
            m = core.compile_model(net, dev, {"PERFORMANCE_HINT": "LATENCY"})
            matched = total = extra = 0
            for f in frames:
                x = (cv2.resize(f, (640, 640))[:, :, ::-1].transpose(2, 0, 1)[None] / np.float32(255)).astype(np.float32)
                h, w = f.shape[:2]
                ra = postprocess(*ref.run(None, {"pixel_values": x}), w, h, 0.5, {"person"})
                o = m(x)
                rb = postprocess(o[0], o[1], w, h, 0.5, {"person"})
                total += len(ra)
                if len(ra) and len(rb):
                    iou = sv.box_iou_batch(ra.xyxy, rb.xyxy)
                    matched += int((iou.max(1) > 0.5).sum())
                    extra += int((iou.max(0) <= 0.5).sum())
                else:
                    extra += len(rb)
            print(f"D-FINE-{size.upper()} {dev}: finds {matched / max(1, total):.1%} of the CPU's people "
                  f"({matched}/{total}), {extra} boxes the CPU does not have")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
