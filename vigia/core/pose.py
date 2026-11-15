"""17 COCO body keypoints per person box, with RTMPose (rtmlib, Apache-2.0) on the boxes D-FINE already found.

Weights: OpenMMLab RTMPose-m "body7" (Apache-2.0); its training mix includes datasets licensed for
non-commercial use only, which THIRD_PARTY.md records. Keypoint order: 0 nose, 1-2 eyes, 3-4 ears, 5-6 shoulders,
7-8 elbows, 9-10 wrists, 11-12 hips, 13-14 knees, 15-16 ankles (left before right).
"""
from __future__ import annotations

import numpy as np

RTMPOSE_M = ("https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
             "rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip")

NOSE, L_EYE, R_EYE, L_EAR, R_EAR, L_SHOULDER, R_SHOULDER = 0, 1, 2, 3, 4, 5, 6
L_ELBOW, R_ELBOW, L_WRIST, R_WRIST, L_HIP, R_HIP = 7, 8, 9, 10, 11, 12
L_KNEE, R_KNEE, L_ANKLE, R_ANKLE = 13, 14, 15, 16


class Pose:
    """Runs on the CPU (~26 ms per person on a Core Ultra 7 258V). With OpenVINO 2026.4 the Arc GPU gave keypoints
    up to 69 px off and compiling for the NPU crashed the process, so neither is used."""

    def __init__(self):
        from rtmlib import RTMPose

        self._model = RTMPose(RTMPOSE_M, model_input_size=(192, 256), backend="onnxruntime", device="cpu")

    def __call__(self, frame_bgr: np.ndarray, boxes_xyxy: np.ndarray) -> np.ndarray:
        """(N, 17, 3): x, y in frame pixels and a 0-1 confidence."""
        boxes = np.asarray(boxes_xyxy, dtype=np.float32).reshape(-1, 4)
        if len(boxes) == 0:
            return np.zeros((0, 17, 3), np.float32)
        kps, scores = self._model(frame_bgr, bboxes=boxes.tolist())
        conf = np.clip(np.asarray(scores, np.float32), 0, 1)
        return np.concatenate([np.asarray(kps, np.float32).reshape(-1, 17, 2), conf.reshape(-1, 17, 1)], axis=2)


def midpoint(kp: np.ndarray, a: int, b: int, min_conf: float = 0.3) -> tuple[float, float] | None:
    """Midpoint of two keypoints if both are confident, else the confident one, else None."""
    ok = [i for i in (a, b) if kp[i, 2] >= min_conf]
    if not ok:
        return None
    return (float(np.mean([kp[i, 0] for i in ok])), float(np.mean([kp[i, 1] for i in ok])))
