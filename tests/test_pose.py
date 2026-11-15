from pathlib import Path

import numpy as np
import pytest

from vigia.core.pose import midpoint


def test_midpoint_uses_confident_points():
    kp = np.zeros((17, 3), np.float32)
    kp[5] = (10, 20, 0.9)
    kp[6] = (30, 40, 0.9)
    assert midpoint(kp, 5, 6) == (20, 30)
    kp[6, 2] = 0.1
    assert midpoint(kp, 5, 6) == (10, 20)
    kp[5, 2] = 0.1
    assert midpoint(kp, 5, 6) is None


@pytest.mark.slow
def test_pose_shapes():
    vid = Path("samples/exam_monitor/exam-room.mp4")
    if not vid.exists():
        pytest.skip("needs samples/")
    import cv2

    from vigia.core.pose import Pose

    ok, frame = cv2.VideoCapture(str(vid)).read()
    pose = Pose()
    assert pose(frame, np.zeros((0, 4))).shape == (0, 17, 3)
    kp = pose(frame, np.array([[300, 300, 700, 1000]]))
    assert kp.shape == (1, 17, 3) and 0 <= kp[..., 2].min() and kp[..., 2].max() <= 1
