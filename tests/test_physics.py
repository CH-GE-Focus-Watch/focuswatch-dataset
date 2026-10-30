import math

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


def _reconstruction_error_deg(q, order, gravity):
    # Why: turns "which reading scored lower" into "the chosen reading is
    # actually right" — this fails whenever the returned label itself is
    # wrong (e.g. a hardcoded constant), but a bug confined to the internal
    # comparison that still happens to select the correct label is not
    # observable this way; see task-4-fix-report.md Finding 2.
    q_xyzw = q if order == "xyzw" else np.roll(q, -1, axis=1)
    return float(np.median(angle_deg(gravity_from_quaternion(q_xyzw), gravity)))


def test_detect_quaternion_order_finds_xyzw():
    rots = _random_rotations()
    q_xyzw = rots.as_quat()
    gravity = rots.inv().apply([0.0, 0.0, -1.0])
    order = detect_quaternion_order(q_xyzw, gravity)
    assert order == "xyzw"
    assert _reconstruction_error_deg(q_xyzw, order, gravity) < 1.0


def test_detect_quaternion_order_finds_wxyz():
    rots = _random_rotations()
    q_xyzw = rots.as_quat()
    gravity = rots.inv().apply([0.0, 0.0, -1.0])
    q_wxyz = np.roll(q_xyzw, 1, axis=1)      # stored scalar-first
    order = detect_quaternion_order(q_wxyz, gravity)
    assert order == "wxyz"
    assert _reconstruction_error_deg(q_wxyz, order, gravity) < 1.0


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


def test_norm_stats_on_an_entirely_non_finite_column_returns_nan_not_a_crash():
    v = np.full((10, 3), np.nan)
    s = norm_stats(v)
    assert math.isnan(s.median)
    assert math.isnan(s.p05)
    assert math.isnan(s.p95)
    assert math.isnan(s.iqr)
    assert math.isnan(s.mean)
    assert math.isnan(s.std)
    assert s.n == 0


def test_still_mask_selects_the_quiet_stretch():
    rng = np.random.default_rng(0)
    gyro = np.vstack([rng.normal(0, 0.5, (300, 3)), rng.normal(0, 0.001, (300, 3))])
    mask = still_mask(gyro, fs_hz=100.0, window_s=1.0, threshold=0.05)
    assert mask[350:550].mean() > 0.9
    assert mask[:250].mean() < 0.1


def test_still_mask_requires_a_quiet_stretch_at_least_as_long_as_the_window():
    # Why: pins window_s * fs_hz, not just the centred-vs-trailing choice — a
    # 0.5 s quiet gap inside a 2 s window is never fully covered by the
    # rolling window, so it must stay flagged as not-still.
    rng = np.random.default_rng(1)
    loud = rng.normal(0, 0.5, (300, 3))
    quiet_gap = rng.normal(0, 0.001, (50, 3))          # 0.5 s at 100 Hz
    gyro = np.vstack([loud[:300], quiet_gap, loud[:300]])
    mask = still_mask(gyro, fs_hz=100.0, window_s=2.0, threshold=0.05)
    assert mask[300:350].mean() < 0.1


def test_gravity_from_quaternion_raises_on_zero_norm_quaternion():
    # Why: documents the failure mode norm_stats' non-finite filter exists to
    # avoid feeding into detect_quaternion_order.
    q = np.array([[0.0, 0.0, 0.0, 0.0]])
    with pytest.raises(ValueError):
        gravity_from_quaternion(q)
