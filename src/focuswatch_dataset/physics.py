"""Vector-level measurements that expose semantic errors column names cannot.

Gravity is a unit vector, angular velocity is not, and an attitude quaternion
reconstructs the gravity direction exactly. Those three facts identify the
quantity a column actually holds, independent of what it is called.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

DOWN = np.array([0.0, 0.0, -1.0])


@dataclass(frozen=True)
class NormStats:
    median: float
    p05: float
    p95: float
    iqr: float
    mean: float
    std: float
    n: int


def norm_stats(vectors: np.ndarray) -> NormStats:
    v = np.asarray(vectors, dtype=float)
    v = v[np.isfinite(v).all(axis=1)]
    n = np.linalg.norm(v, axis=1)
    q25, q75 = np.percentile(n, [25, 75])
    return NormStats(
        median=float(np.median(n)), p05=float(np.percentile(n, 5)),
        p95=float(np.percentile(n, 95)), iqr=float(q75 - q25),
        mean=float(n.mean()), std=float(n.std()), n=int(n.size),
    )


def gravity_from_quaternion(q_xyzw: np.ndarray) -> np.ndarray:
    """Rotate the world-frame down vector into the device frame."""
    return Rotation.from_quat(np.asarray(q_xyzw, dtype=float)).inv().apply(DOWN)


def angle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, float) / np.linalg.norm(a, axis=1, keepdims=True)
    b = np.asarray(b, float) / np.linalg.norm(b, axis=1, keepdims=True)
    return np.degrees(np.arccos(np.clip((a * b).sum(axis=1), -1.0, 1.0)))


def still_mask(gyro: np.ndarray, fs_hz: float, window_s: float = 2.0,
               threshold: float = 0.05) -> np.ndarray:
    """Samples whose surrounding window stays below `threshold` rad/s."""
    win = max(int(round(window_s * fs_hz)), 1)
    n = np.linalg.norm(np.asarray(gyro, dtype=float), axis=1)
    rolling_max = pd.Series(n).rolling(win, center=True, min_periods=1).max().to_numpy()
    return rolling_max < threshold


def detect_quaternion_order(q_first_last: np.ndarray, gravity: np.ndarray) -> str:
    """Decide whether stored components are scalar-last or scalar-first.

    Norm checks cannot tell these apart; the reconstructed gravity direction can.
    """
    q = np.asarray(q_first_last, dtype=float)
    as_xyzw = float(np.median(angle_deg(gravity_from_quaternion(q), gravity)))
    as_wxyz = float(np.median(angle_deg(gravity_from_quaternion(np.roll(q, -1, axis=1)), gravity)))
    return "xyzw" if as_xyzw <= as_wxyz else "wxyz"
