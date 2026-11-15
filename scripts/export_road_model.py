"""One-off: export the road-damage detector to ONNX (models/road-rfdetr-s.onnx). Needs torch, so it runs in a
throwaway environment and torch never becomes a dependency of vigia:

uv venv .export --python 3.12
uv pip install --python .export/Scripts/python.exe "rfdetr[onnx]==1.12.1" ^
    --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match
.export/Scripts/python.exe scripts/export_road_model.py

Model: dronefreak/rdd2022-rfdetr-small (Apache-2.0), RF-DETR Small fine-tuned on RDD2022 (CC BY-SA 4.0):
longitudinal crack, transverse crack, alligator crack, pothole. The checkpoint is loaded with rfdetr's safe loader
(trust_checkpoint=False: tensors only, no pickled code).
"""
from __future__ import annotations

import shutil
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
URL = "https://huggingface.co/dronefreak/rdd2022-rfdetr-small/resolve/main/checkpoint_best_total.pth"
CLASSES = ["longitudinal_crack", "transverse_crack", "alligator_crack", "pothole"]


def main() -> None:
    from rfdetr import RFDETRSmall

    models = ROOT / "models"
    models.mkdir(exist_ok=True)
    pth = models / "road-rfdetr-s.pth"
    if not pth.exists():
        print(f"downloading {URL} (127 MB, Apache-2.0)")
        urllib.request.urlretrieve(URL, pth.with_suffix(".part"))
        pth.with_suffix(".part").replace(pth)
    model = RFDETRSmall(pretrain_weights=str(pth), num_classes=len(CLASSES), resolution=512)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(model.export(output_dir=tmp, shape=(512, 512), opset_version=17))
        onnx = out if out.suffix == ".onnx" else next(Path(tmp).rglob("*.onnx"))
        shutil.copy(onnx, models / "road-rfdetr-s.onnx")
    print("saved", models / "road-rfdetr-s.onnx")


if __name__ == "__main__":
    main()
