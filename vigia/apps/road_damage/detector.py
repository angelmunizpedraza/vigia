"""RF-DETR Small fine-tuned on RDD2022 (cracks and potholes), run with OpenVINO.

Model: dronefreak/rdd2022-rfdetr-small (Apache-2.0), trained on RDD2022 (CC BY-SA 4.0), exported once to
models/road-rfdetr-s.onnx with scripts/export_road_model.py.
  input  `input` (1, 3, 512, 512) float32: RGB, stretch-resized (bilinear, no antialias), x/255, ImageNet mean/std
  output `dets` (1, 300, 4) cx, cy, w, h in 0-1; `labels` (1, 300, 5) logits: the 4 classes of NAMES, in the
         checkpoint's class_names order, plus an unused no-object slot (rfdetr's own predict() drops it too)
Decoding is the same as D-FINE's (sigmoid, top 300 over queries x classes), so it reuses core `postprocess`.

Checked against rfdetr 1.12.1's own RFDETRSmall(...).predict() on 9 sample images: every box scoring >= 0.15 comes
out with the same class, IoU > 0.9 and a score within 0.006, on the CPU and on the Intel NPU alike, so "auto" picks
the NPU (85 ms per 512 px inference on a Core Ultra 7 258V vs 550-640 ms on its CPU).
"""
from __future__ import annotations

import cv2
import numpy as np
import supervision as sv

from vigia.core.detect import empty, pick_device, postprocess, tiles
from vigia.core.models import MODELS_DIR, ensure

NAMES = ["longitudinal crack", "transverse crack", "alligator crack", "pothole"]
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def preprocess(bgr: np.ndarray, size: int = 512) -> np.ndarray:
    x = cv2.resize(bgr, (size, size), interpolation=cv2.INTER_LINEAR)[:, :, ::-1].astype(np.float32) / 255
    return ((x - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)


class RoadDetector:
    def __init__(self, conf: float = 0.25, device: str = "auto", size: int = 512):
        import openvino as ov

        self.conf, self.size, self.device = conf, size, pick_device(device)
        core = ov.Core()
        core.set_property({"CACHE_DIR": str(MODELS_DIR / "openvino-cache")})
        self._net = core.compile_model(core.read_model(str(ensure("road-rfdetr-s"))), self.device,
                                       {"PERFORMANCE_HINT": "LATENCY"})

    @property
    def name(self) -> str:
        return f"RF-DETR-S RDD2022 on {self.device}"

    def _one(self, bgr: np.ndarray) -> sv.Detections:
        out = self._net(preprocess(bgr, self.size))
        boxes, logits = out[0], out[1]
        h, w = bgr.shape[:2]
        return postprocess(logits[..., :len(NAMES)], boxes, w, h, self.conf, names=NAMES)

    def __call__(self, frame_bgr: np.ndarray, roi: tuple[int, int, int, int] | None = None,
                 tile: int = 0) -> sv.Detections:
        """Damage in `roi` (x1, y1, x2, y2; whole frame by default). `tile` > 0 cuts the roi into overlapping
        squares of that many pixels, each run at 512 px: a zoom for roads filmed from the side or far away."""
        H, W = frame_bgr.shape[:2]
        rx1, ry1, rx2, ry2 = roi or (0, 0, W, H)
        rects = [(rx1, ry1, rx2, ry2)] if tile <= 0 else \
            [(rx1 + a, ry1 + b, rx1 + c, ry1 + d) for a, b, c, d in tiles(rx2 - rx1, ry2 - ry1, tile, 0.25)]
        parts = []
        for x1, y1, x2, y2 in rects:
            d = self._one(frame_bgr[y1:y2, x1:x2])
            if len(d):
                d.xyxy += np.array([x1, y1, x1, y1], dtype=d.xyxy.dtype)
                parts.append(d)
        if not parts:
            return empty()
        # one query can score on two classes, and tiles overlap: keep one box per damage
        return sv.Detections.merge(parts).with_nms(threshold=0.5, class_agnostic=True)
