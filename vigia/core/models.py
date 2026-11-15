"""Where every model comes from, its licence, and a downloader that never runs what it fetches."""
from __future__ import annotations

import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"

_HF = "https://huggingface.co/onnx-community/dfine_{}_coco-ONNX/resolve/main/onnx/model.onnx"
_MP = "https://storage.googleapis.com/mediapipe-models/{0}/{0}/float16/latest/{0}.task"


@dataclass(frozen=True)
class ModelSpec:
    url: str | None
    filename: str
    license: str
    size_mb: float
    source: str


REGISTRY: dict[str, ModelSpec] = {
    # COCO-only weights: the obj365/obj2coco variants may carry Objects365 terms.
    # No nano: the onnx-community nano export tops out at ~0.4 confidence on obvious people (S/M give ~0.86).
    "dfine-s": ModelSpec(_HF.format("s"), "dfine-s.onnx", "Apache-2.0", 41.5, "ustc-community/dfine-small-coco"),
    "dfine-m": ModelSpec(_HF.format("m"), "dfine-m.onnx", "Apache-2.0", 78.0, "ustc-community/dfine-medium-coco"),
    "face-landmarker": ModelSpec(_MP.format("face_landmarker"), "face_landmarker.task", "Apache-2.0", 3.8,
                                 "Google MediaPipe Face Landmarker"),
    "hand-landmarker": ModelSpec(_MP.format("hand_landmarker"), "hand_landmarker.task", "Apache-2.0", 7.8,
                                 "Google MediaPipe Hand Landmarker"),
    "yunet": ModelSpec("https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
                       "face_detection_yunet/face_detection_yunet_2023mar.onnx",
                       "yunet.onnx", "MIT", 0.3, "OpenCV Zoo YuNet"),
    "road-rfdetr-s": ModelSpec(None, "road-rfdetr-s.onnx", "Apache-2.0", 120.0,
                               "dronefreak/rdd2022-rfdetr-small (trained on RDD2022, CC BY-SA 4.0)"),
}


def _download(url: str, dest: Path) -> None:
    part = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "vigia"})
    with urllib.request.urlopen(req, timeout=60) as r, open(part, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    part.replace(dest)


def ensure(name: str) -> Path:
    """Path to model `name`, downloading it from its official source the first time."""
    spec = REGISTRY[name]
    path = MODELS_DIR / spec.filename
    if path.exists():
        return path
    if spec.url is None:
        raise FileNotFoundError(
            f"{path} is missing. It is exported once, in a separate environment (needs torch): "
            "uv venv .export --python 3.12 && uv pip install --python .export \"rfdetr[onnx]==1.12.1\" "
            "--index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple "
            "--index-strategy unsafe-best-match && then run scripts/export_road_model.py with .export's python "
            "(.export/Scripts/python.exe on Windows, .export/bin/python elsewhere)")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"downloading {name} ({spec.size_mb:.0f} MB, {spec.license}) from {spec.source}", file=sys.stderr)
    _download(spec.url, path)
    return path
