from pathlib import Path

import numpy as np
import pytest

from vigia.core.detect import COCO, postprocess

PERSON, CAR = COCO.index("person"), COCO.index("car")


def _outputs(score_logit: float, cls: int, box=(0.5, 0.5, 0.2, 0.4)):
    logits = np.full((1, 300, 80), -10.0, dtype=np.float32)
    boxes = np.zeros((1, 300, 4), dtype=np.float32)
    logits[0, 7, cls] = score_logit
    boxes[0, 7] = box
    return logits, boxes


def test_postprocess_converts_and_filters():
    logits, boxes = _outputs(4.0, PERSON)  # sigmoid(4) ~ 0.98
    d = postprocess(logits, boxes, 1000, 500, conf=0.4)
    assert len(d) == 1
    np.testing.assert_allclose(d.xyxy[0], [400, 150, 600, 350], atol=1e-3)
    assert d.data["class_name"][0] == "person"
    assert d.confidence[0] == pytest.approx(0.982, abs=1e-3)


def test_postprocess_drops_low_scores_and_other_labels():
    logits, boxes = _outputs(-2.2, PERSON)  # ~0.10
    assert len(postprocess(logits, boxes, 1000, 500, conf=0.4)) == 0
    logits, boxes = _outputs(4.0, PERSON)
    assert len(postprocess(logits, boxes, 1000, 500, conf=0.4, labels={"car"})) == 0


def test_postprocess_empty_is_valid():
    logits = np.full((1, 300, 80), -10.0, dtype=np.float32)
    d = postprocess(logits, np.zeros((1, 300, 4), np.float32), 640, 480, conf=0.4)
    assert len(d) == 0 and d.xyxy.shape == (0, 4) and len(d.data["class_name"]) == 0


@pytest.mark.slow
def test_detector_finds_people_in_cafe():
    from vigia.core.detect import Detector

    vid = Path("samples/table_occupancy/cafe-from-above.mp4")
    if not vid.exists() or not Path("models/dfine-s.onnx").exists():
        pytest.skip("needs samples/ and models/")
    import cv2

    ok, frame = cv2.VideoCapture(str(vid)).read()
    assert ok
    d = Detector("s", labels={"person"}, device="CPU")(frame)
    assert len(d) >= 8
