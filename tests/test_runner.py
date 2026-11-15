import argparse

from PIL import Image

from tests.test_video import make_video
from vigia.core.runner import run
from vigia.core.types import Event


class FakeApp:
    name = "fake"
    blur_faces = True

    @staticmethod
    def add_args(p):
        pass

    def setup(self, scene, fps, size, args):
        self.size, self.n = size, 0

    def process(self, t, frame):
        self.n += 1
        return [Event(t, "tick")]

    def draw(self, frame):
        return frame

    def summary(self):
        return {"frames": self.n}


def _args(tmp_path, source, out=None):
    return argparse.Namespace(source=source, scene=None, out=out, show=False, every=1, seconds=None,
                              no_blur=True, model="n", device="CPU", events=str(tmp_path / "ev.jsonl"))


def test_runner_missing_source(tmp_path, capsys):
    assert run(FakeApp(), _args(tmp_path, str(tmp_path / "missing.mp4"))) == 2
    assert "cannot open" in capsys.readouterr().err


def test_runner_source_ends_early_finalizes_output(tmp_path, capsys):
    src = make_video(tmp_path / "v.avi", n=5)
    out = tmp_path / "o.gif"
    app = FakeApp()
    assert run(app, _args(tmp_path, str(src), str(out))) == 0
    assert app.n == 5
    assert Image.open(out).n_frames >= 1
    assert (tmp_path / "ev.jsonl").read_text().count("tick") == 5
    assert '"frames": 5' in capsys.readouterr().out


def test_blur_covers_untracked_detections():
    import numpy as np
    import supervision as sv

    from vigia.core.detect import Detector
    from vigia.core.runner import people_to_blur
    from vigia.core.types import Track

    app = FakeApp()
    app.tracks = [Track(1, (0, 0, 10, 20), "person", 0.9), Track(2, (5, 5, 9, 9), "cell phone", 0.9)]
    app.det = object.__new__(Detector)  # no model needed: only its last detections are read
    app.det.last = sv.Detections(xyxy=np.array([[50, 50, 60, 80], [0, 0, 100, 50]], float),
                                 data={"class_name": np.array(["person", "car"], dtype=object)})
    assert people_to_blur(app) == [(0, 0, 10, 20), (50.0, 50.0, 60.0, 80.0)]


def test_vehicles_to_blur_reads_detectors():
    import numpy as np
    import supervision as sv

    from vigia.core.detect import Detector
    from vigia.core.runner import vehicles_to_blur

    app = FakeApp()
    app.objects = object.__new__(Detector)
    app.objects.last = sv.Detections(xyxy=np.array([[0, 0, 10, 10], [5, 5, 50, 40]], float),
                                     data={"class_name": np.array(["person", "car"], dtype=object)})
    assert vehicles_to_blur(app) == [(5.0, 5.0, 50.0, 40.0)]


def test_heads_to_blur_uses_keypoints():
    import numpy as np

    from vigia.core.runner import heads_to_blur
    from vigia.core.types import Track

    kp = np.zeros((17, 3), np.float32)
    kp[0] = (50, 60, 0.9)
    app = FakeApp()
    app.tracks = [Track(1, (0, 0, 100, 100), "person", 0.9, kp), Track(2, (0, 0, 9, 9), "person", 0.9)]
    [box] = heads_to_blur(app)
    assert box[0] < 50 < box[2] and box[1] < 60 < box[3]
