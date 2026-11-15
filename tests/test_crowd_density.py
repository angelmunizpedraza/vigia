import numpy as np

from vigia.apps.crowd_density.logic import DensityLogic, apply_homography, homography
from vigia.core.scene import Scene
from vigia.core.types import Track

SCENE = Scene(1000, 1000)  # px_per_m=100 and cell_m=2 -> a 5x5 grid of 200 px cells


def person_at(x, y, w=20.0, h=20.0, i=-1):
    return Track(i, (x - w / 2, y - h / 2, x + w / 2, y + h / 2), "person", 0.9)


def crowd(n, x, y, w=20.0):
    return [person_at(x + (i % 4) * 30 - 45, y + (i // 4) * 30 - 45, w) for i in range(n)]


def test_density_counts_per_cell():
    lg = DensityLogic(SCENE, cell_m=2.0, px_per_m=100.0)
    lg.update(0.0, crowd(8, 100, 100) + crowd(2, 500, 300) + [Track(-1, (0, 0, 9, 9), "car", 0.9)])
    assert lg.grid.shape == (5, 5)
    assert lg.grid[0, 0] == 8 / 4 and lg.grid[1, 2] == 2 / 4
    assert np.nansum(lg.grid) == 10 / 4 and lg.people == 10
    assert lg.peak == 2.0


def test_partial_edge_cells():
    # 1100 px wide: the last column is half inside (counts with half the area); 1050: a quarter (ignored)
    lg = DensityLogic(Scene(1100, 1000), px_per_m=100.0)
    lg.update(0.0, [person_at(1050, 100)])
    assert lg.grid.shape == (5, 6) and lg.grid[0, 5] == 1 / 2
    lg = DensityLogic(Scene(1050, 1000), px_per_m=100.0)
    lg.update(0.0, [person_at(1025, 100)])
    assert np.isnan(lg.grid[0, 5]) and lg.peak == 0.0 and (0, 5) not in lg.level and lg.people == 1


def test_levels_thresholds():
    lg = DensityLogic(SCENE, px_per_m=100.0, amber=2.0, red=4.0)
    lg.update(0.0, crowd(7, 100, 100) + crowd(8, 300, 100) + crowd(15, 500, 100) + crowd(16, 700, 100))
    assert [lg.level[(0, c)] for c in range(5)] == ["ok", "warn", "warn", "alert", "ok"]
    assert lg.peak == 4.0 and lg.peak_level == "alert"


def test_alert_needs_hold():
    lg = DensityLogic(SCENE, px_per_m=100.0, red=4.0, hold=3.0)
    packed = crowd(16, 100, 100)
    assert [lg.update(t, packed) for t in (0.0, 1.0, 2.0)] == [[], [], []]
    ev = lg.update(3.0, packed)
    assert [e.kind for e in ev] == ["density_alert"] and ev[0].data["cell"] == [0, 0]
    assert ev[0].data["density"] == 4.0 and ev[0].data["since"] == 0.0
    assert lg.update(4.0, packed) == []                 # once per stretch
    assert lg.update(5.0, crowd(8, 100, 100)) == []     # drops below red: re-armed
    assert lg.update(6.0, packed) == [] and lg.update(8.0, packed) == []  # 2 s is not enough
    assert [e.kind for e in lg.update(9.0, packed)] == ["density_alert"]


def test_alert_with_sparse_frames():
    # processed at 1 frame every 2 s: time comes from t, not from counting frames
    lg = DensityLogic(SCENE, px_per_m=100.0, hold=3.0)
    packed = crowd(16, 100, 100)
    assert lg.update(10.0, packed) == [] and lg.update(12.0, packed) == []
    assert len(lg.update(14.0, packed)) == 1


def test_estimated_scale_flagged():
    lg = DensityLogic(SCENE, min_people=10)
    assert lg.estimated and lg.px_per_m is None
    lg.update(0.0, crowd(3, 100, 100, w=45))            # too few people to estimate: no grid yet
    assert lg.px_per_m is None and lg.grid.size == 0 and lg.people == 3
    lg.update(1.0, crowd(12, 100, 100, w=45))           # 45 px / 0.45 m
    assert abs(lg.px_per_m - 100.0) < 1e-6 and lg.grid[0, 0] == 12 / 4
    lg.update(2.0, crowd(12, 100, 100, w=90))           # the drone came down: scale follows
    assert abs(lg.px_per_m - 200.0) < 1e-6
    lg.update(3.0, crowd(2, 100, 100, w=10))            # too few: keep the last estimate
    assert abs(lg.px_per_m - 200.0) < 1e-6
    fixed = DensityLogic(SCENE, px_per_m=50.0)
    fixed.update(0.0, crowd(12, 100, 100, w=45))
    assert not fixed.estimated and fixed.px_per_m == 50.0
    assert not DensityLogic(SCENE, homography=np.eye(3)).estimated


# Oblique camera: a 10 m x 10 m square on the ground seen as a trapezoid (far side shorter, at the top).
IMG = [(300, 200), (700, 200), (1000, 900), (0, 900)]
GROUND = [(0, 0), (10, 0), (10, 10), (0, 10)]


def test_homography_maps_calibration_points():
    H = homography(IMG, GROUND)
    assert np.allclose(apply_homography(H, IMG), GROUND, atol=1e-6)
    # the centre of the square is where the trapezoid's diagonals cross, well above its mid-height (y=550)
    assert np.allclose(apply_homography(np.linalg.inv(H), [(5, 5)])[0], (500, 400), atol=1e-6)


def test_homography_density_on_ground_grid():
    H = homography(IMG, GROUND)
    lg = DensityLogic(Scene(1000, 1000, zones={"area": IMG}), cell_m=2.0, homography=H)
    feet = apply_homography(np.linalg.inv(H), [(1, 1)] * 4 + [(9, 9)] * 2)   # feet on the ground, in pixels
    lg.update(0.0, [Track(-1, (x - 5, y - 40, x + 5, y), "person", 0.9) for x, y in feet])
    assert lg.grid.shape == (5, 5) and not np.isnan(lg.grid).any()
    assert np.isclose(lg.grid[0, 0], 1.0) and np.isclose(lg.grid[4, 4], 0.5)
    assert np.isclose(np.nansum(lg.grid), 1.5)
    corners = lg.cell_polygon(0, 0)                           # drawn as a perspective quad in the image
    assert np.allclose(corners[0], IMG[0], atol=1e-6)


def test_homography_grid_defaults_to_calibration_quad():
    # no "area" zone: the whole image of an oblique camera can reach the horizon, so the grid covers the calibration
    H = homography(IMG, GROUND)
    lg = DensityLogic(Scene(1000, 1000, zones={"calibration": IMG}), cell_m=2.0, homography=H)
    lg.update(0.0, [])
    assert lg.grid.shape == (5, 5)


def test_crowd_empty_frames():
    lg = DensityLogic(SCENE)
    assert all(lg.update(i / 10, []) == [] for i in range(100))
    assert lg.grid.size == 0 and lg.peak == 0.0 and lg.peak_level == "ok" and lg.people == 0
    lg = DensityLogic(SCENE, px_per_m=100.0)
    assert all(lg.update(i / 10, []) == [] for i in range(100))
    assert np.nansum(lg.grid) == 0 and lg.level and set(lg.level.values()) == {"ok"}
