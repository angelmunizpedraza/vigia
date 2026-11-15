# Third-party software, models and data

vigia's own code is MIT. Nothing below is copied into this repository: libraries install from PyPI, models and
videos download from their original source (`vigia models` or the first run of a demo; RTMPose goes to rtmlib's own
cache; `scripts/get_samples.py` for the videos). The road model is exported once with `scripts/export_road_model.py`. No AGPL or GPL component is used.

## Libraries

| Package | Licence | Used for |
|---|---|---|
| [OpenVINO](https://github.com/openvinotoolkit/openvino) | Apache-2.0 | D-FINE inference on Intel NPU or CPU |
| [ONNX Runtime](https://github.com/microsoft/onnxruntime) | MIT | RTMPose and RF-DETR inference on CPU |
| [OpenCV](https://github.com/opencv/opencv-python) (contrib build) | Apache-2.0 (bundled FFmpeg: LGPL-2.1) | video I/O, drawing, YuNet |
| [supervision](https://github.com/roboflow/supervision) **0.29.1** | MIT | ByteTrack tracking, detection containers. Pinned: 0.30+ requires PyAV, whose wheels bundle FFmpeg built with GPL x264/x265 |
| [rtmlib](https://github.com/Tau-J/rtmlib) | Apache-2.0 | RTMPose wrapper |
| [MediaPipe](https://github.com/google-ai-edge/mediapipe) | Apache-2.0 | face and hand landmarks |
| [NumPy](https://numpy.org), [Pillow](https://python-pillow.org) | BSD / MIT-CMU | arrays, GIFs |
| [pyserial](https://github.com/pyserial/pyserial) | BSD-3-Clause | Arduino link |
| [rfdetr](https://github.com/roboflow/rf-detr), [PyTorch](https://pytorch.org) | Apache-2.0 / BSD | only in the throwaway env of `scripts/export_road_model.py`, never a vigia dependency |

Deliberately not used: Ultralytics YOLO (AGPL-3.0), PyAV wheels (GPL FFmpeg build), boxmot (AGPL-3.0).

## Models

| Model | Licence | Source | Notes |
|---|---|---|---|
| D-FINE S / M, COCO | Apache-2.0 | [ustc-community](https://huggingface.co/ustc-community) via the [onnx-community](https://huggingface.co/onnx-community/dfine_s_coco-ONNX) ONNX exports | COCO-only weights; the Objects365 variants were avoided because the D-FINE authors warn they may carry Objects365 terms |
| RTMPose-m body7 | Apache-2.0 | [OpenMMLab](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose) | Its training mix (body7) includes datasets released for non-commercial research use; check before any commercial use |
| RF-DETR Small fine-tuned on RDD2022 | Apache-2.0 | [dronefreak/rdd2022-rfdetr-small](https://huggingface.co/dronefreak/rdd2022-rfdetr-small) | Trained on RDD2022 (CC BY-SA 4.0) |
| MediaPipe Face Landmarker, Hand Landmarker | Apache-2.0 | [Google](https://ai.google.dev/edge/mediapipe/solutions/guide) | |
| YuNet face detector | MIT | [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) | face pixelation in saved output |

## Datasets

| Dataset | Licence | Use |
|---|---|---|
| [GMDCSA24](https://zenodo.org/records/12921216) (Alam et al.) | CC BY 4.0 | fall detector evaluation and GIF |
| [RDD2022](https://github.com/sekilab/RoadDamageDetector) (Arya et al.) | CC BY-SA 4.0 (figshare lists CC BY 4.0; the stricter one is assumed) | training data of the road model (not downloaded here) |

## Sample videos

Pexels and Pixabay videos are free to use but may not be redistributed, so only their links live here
(`vigia/apps/*/samples.json`); the GIFs in `docs/gifs/` are derived works with faces, heads and number plates pixelated.

| Demo | Video | Licence |
|---|---|---|
| tables | [Lively Turkish café scene from above](https://www.pexels.com/video/lively-turkish-cafe-scene-from-above-34569882/), Talha Uğuz | Pexels License |
| count | [Black and white video of people](https://www.pexels.com/video/black-and-white-video-of-people-853889/), Coverr | Pexels License |
| crowd | [Aerial view of people walking in the streets](https://www.pexels.com/video/aerial-view-of-people-walking-in-the-streets-2088290/), Tom Fisk | Pexels License |
| exam | [Students during an exam](https://www.pexels.com/video/students-cheating-during-an-exam-7092083/), RDNE Stock project (staged scene) | Pexels License |
| desk | [Men working in the office](https://www.pexels.com/video/men-working-in-the-office-5484894/), cottonbro studio | Pexels License |
| store | [People paying at the counter in the supermarket](https://www.pexels.com/video/people-paying-at-the-counter-in-the-supermarket-4249560/), Nazim Zafri | Pexels License |
| drowsy | [A man covering his mouth while yawning](https://www.pexels.com/video/a-man-covering-his-mouth-while-yawning-4588464/), Ketut Subiyanto | Pexels License |
| hands | [A person doing hand signs](https://www.pexels.com/video/a-person-doing-hand-signs-9019617/) | Pexels License |
| road | [Pothole.ogv](https://commons.wikimedia.org/wiki/File:Pothole.ogv), [Cars Moving into pothole.webm](https://commons.wikimedia.org/wiki/File:Cars_Moving_into_pothole.webm), Amuzujoe | CC BY-SA 4.0 (the road GIF is CC BY-SA 4.0 too) |
| fall | GMDCSA24 clips (above) | CC BY 4.0 |
