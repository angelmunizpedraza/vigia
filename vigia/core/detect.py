"""D-FINE (Apache-2.0, COCO-only weights) through OpenVINO: Intel NPU when there is one, CPU otherwise.

Why not the Intel GPU: with OpenVINO 2026.4 the GPU plugin returns wrong D-FINE outputs on an Arc 140V (it found
32-77 % of the people the CPU finds, even at f32), while the NPU matched the CPU on 99.5-100 % of them.
`scripts/check_accelerator.py` measures that agreement on any machine before you trust a device.

ONNX from onnx-community/dfine_{n,s,m}_coco-ONNX:
  input  pixel_values (batch, 3, H, W) float32, RGB, x/255, no mean/std (preprocessor_config.json)
  output logits (batch, 300, 80), pred_boxes (batch, 300, 4) as cx, cy, w, h in 0-1
"""
from __future__ import annotations

import math

import cv2
import numpy as np
import supervision as sv

from .models import MODELS_DIR, ensure

# Names exactly as in the model's config.json (VOC-style spellings kept: "motorbike", "diningtable"...).
COCO = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa',
        'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard',
        'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
        'teddy bear', 'hair drier', 'toothbrush']


def empty() -> sv.Detections:
    d = sv.Detections.empty()
    d.data["class_name"] = np.array([], dtype=object)
    return d


def postprocess(logits: np.ndarray, boxes: np.ndarray, w: int, h: int, conf: float,
                labels: set[str] | None = None, names: list[str] = COCO) -> sv.Detections:
    """RT-DETR style: sigmoid, top 300 over queries x classes, cxcywh (0-1) -> xyxy pixels."""
    scores = 1 / (1 + np.exp(-logits[0].astype(np.float32)))
    n_cls = scores.shape[1]
    flat = scores.ravel()
    k = min(300, flat.size)
    top = np.argpartition(-flat, k - 1)[:k]
    top = top[flat[top] >= conf]
    if labels is not None:
        top = top[np.isin(top % n_cls, [names.index(n) for n in labels if n in names])]
    if top.size == 0:
        return empty()
    q, c = top // n_cls, top % n_cls
    cx, cy, bw, bh = boxes[0, q].T.astype(np.float32)
    xyxy = np.stack([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h], 1)
    xyxy = np.clip(xyxy, 0, [w, h, w, h])
    d = sv.Detections(xyxy=xyxy, confidence=flat[top], class_id=c.astype(int))
    d.data["class_name"] = np.array([names[i] for i in c], dtype=object)
    return d


def tiles(W: int, H: int, tile: int, ov: float) -> list[tuple[int, int, int, int]]:
    """Overlapping square tiles covering the image, evenly spread, the last ones flush with the edges."""
    nx = max(1, math.ceil((W - tile * ov) / (tile * (1 - ov)))) if W > tile else 1
    ny = max(1, math.ceil((H - tile * ov) / (tile * (1 - ov)))) if H > tile else 1
    xs = [0] if nx == 1 else [round(i * (W - tile) / (nx - 1)) for i in range(nx)]
    ys = [0] if ny == 1 else [round(i * (H - tile) / (ny - 1)) for i in range(ny)]
    return [(x, y, min(W, x + tile), min(H, y + tile)) for y in ys for x in xs]


def pick_device(device: str = "auto") -> str:
    """NPU if present (validated against the CPU), else CPU. An explicit device is used as given."""
    import openvino as ov

    if device.upper() != "AUTO":
        return device.upper()
    return "NPU" if "NPU" in ov.Core().available_devices else "CPU"


class Detector:
    def __init__(self, model: str = "auto", labels: set[str] | None = None, conf: float = 0.4,
                 size: int = 640, device: str = "auto"):
        import openvino as ov

        self.device = pick_device(device)
        if model == "auto":
            model = "m" if self.device == "NPU" else "s"
        self.model, self.labels, self.conf, self.size = model, labels, conf, size
        core = ov.Core()
        core.set_property({"CACHE_DIR": str(MODELS_DIR / "openvino-cache")})  # NPU compile: ~27 s once, then ~1 s
        net = core.read_model(str(ensure(f"dfine-{model}")))
        net.reshape([1, 3, size, size])  # the NPU needs static shapes
        self._net = core.compile_model(net, self.device, {"PERFORMANCE_HINT": "LATENCY"})
        self.last = empty()  # latest detections, read by the runner to pixelate people the tracker has not confirmed

    @property
    def name(self) -> str:
        return f"D-FINE-{self.model.upper()} on {self.device}"

    def _infer(self, bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = cv2.resize(bgr, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        x = (x[:, :, ::-1].transpose(2, 0, 1)[None] / np.float32(255)).astype(np.float32)
        out = self._net(x)
        return out[0], out[1]  # logits, pred_boxes

    def __call__(self, frame_bgr: np.ndarray) -> sv.Detections:
        h, w = frame_bgr.shape[:2]
        logits, boxes = self._infer(frame_bgr)
        self.last = postprocess(logits, boxes, w, h, self.conf, self.labels)
        return self.last

    def tiled(self, frame_bgr: np.ndarray, tile: int = 640, overlap: float = 0.25) -> sv.Detections:
        """Detect small objects (people seen from high up) by running on overlapping tiles plus the full frame."""
        H, W = frame_bgr.shape[:2]
        parts = [self(frame_bgr)]
        for x0, y0, x1, y1 in tiles(W, H, tile, overlap):
            crop = frame_bgr[y0:y1, x0:x1]
            logits, boxes = self._infer(crop)
            d = postprocess(logits, boxes, x1 - x0, y1 - y0, self.conf, self.labels)
            if len(d):
                d.xyxy += np.array([x0, y0, x0, y0], dtype=d.xyxy.dtype)
                parts.append(d)
        merged = sv.Detections.merge(parts)
        self.last = empty() if len(merged) == 0 else merged.with_nms(threshold=0.5, class_agnostic=False)
        return self.last
