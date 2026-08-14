import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from focuswatch_dataset.physics import (
    angle_deg, detect_quaternion_order, gravity_from_quaternion, norm_stats, still_mask,
)


def _random_rotations(n=500, seed=0):
    return Rotation.random(n, random_state=seed)


def test_gravity_from_quaternion_matches_the_rotated_down_vector():
    rots = _random_rotations()
    q = rots.as_quat()                       # scipy emits xyzw
    expected = rots.inv().apply([0.0, 0.0, -1.0])
    assert np.allclose(gravity_from_quaternion(q), expected, atol=1e-12)


def test_gravity_from_quaternion_is_unit_norm():
    g = gravity_from_quaternion(_random_rotations().as_quat())
    assert np.allclose(np.linalg.norm(g, axis=1), 1.0, atol=1e-12)


def test_detect_quaternion_order_finds_xyzw():
    rots = _random_rotations()
    q_xyzw = rots.as_quat()
    gravity = rots.inv().apply([0.0, 0.0, -1.0])
    assert detect_quaternion_order(q_xyzw, gravity) == "xyzw"


def test_detect_quaternion_order_finds_wxyz():
    rots = _random_rotations()
    q_xyzw = rots.as_quat()
    gravity = rots.inv().apply([0.0, 0.0, -1.0])
    q_wxyz = np.roll(q_xyzw, 1, axis=1)      # stored scalar-first
    assert detect_quaternion_order(q_wxyz, gravity) == "wxyz"


def test_angle_deg_endpoints():
    a = np.array([[0.0, 0.0, -1.0]])
    assert angle_deg(a, a)[0] == pytest.approx(0.0, abs=1e-9)
    assert angle_deg(a, -a)[0] == pytest.approx(180.0, abs=1e-9)


def test_norm_stats_on_a_unit_vector_field():
    v = np.tile([0.0, 0.0, 1.0], (100, 1))
    s = norm_stats(v)
    assert s.median == pytest.approx(1.0)
    assert s.iqr == pytest.approx(0.0)
    assert s.n == 100


def test_still_mask_selects_the_quiet_stretch():
    rng = np.random.default_rng(0)
    gyro = np.vstack([rng.normal(0, 0.5, (300, 3)), rng.normal(0, 0.001, (300, 3))])
    mask = still_mask(gyro, fs_hz=100.0, window_s=1.0, threshold=0.05)
    assert mask[350:550].mean() > 0.9
    assert mask[:250].mean() < 0.1
