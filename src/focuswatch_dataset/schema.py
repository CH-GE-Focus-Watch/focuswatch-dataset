"""Canonical column names, units and the physical thresholds the validator enforces.

Both the CI tests and the build gate import from here so the two cannot drift.
"""
from __future__ import annotations

from enum import StrEnum

SCHEMA_VERSION = "1.0"
G_TO_MS2 = 9.80665


class Quantity(StrEnum):
    ACCEL_USER = "accel_user"
    ACCEL_TOTAL = "accel_total"
    GYRO = "gyro"
    GRAVITY = "gravity"
    QUAT = "quat"


COLUMNS: dict[Quantity, tuple[str, ...]] = {
    Quantity.ACCEL_USER: ("accel_user_x", "accel_user_y", "accel_user_z"),
    Quantity.ACCEL_TOTAL: ("accel_total_x", "accel_total_y", "accel_total_z"),
    Quantity.GYRO: ("gyro_x", "gyro_y", "gyro_z"),
    Quantity.GRAVITY: ("gravity_x", "gravity_y", "gravity_z"),
    # Why: scalar-last, matching scipy.spatial.transform.Rotation.from_quat.
    Quantity.QUAT: ("quat_x", "quat_y", "quat_z", "quat_w"),
}

UNITS: dict[Quantity, str] = {
    Quantity.ACCEL_USER: "g",
    Quantity.ACCEL_TOTAL: "g",
    Quantity.GYRO: "rad/s",
    Quantity.GRAVITY: "g",
    Quantity.QUAT: "1",
}

TIME_COLUMN = "t_ns"

ACCEL_USER_BAND = (0.0, 0.2)
ACCEL_TOTAL_BAND = (0.9, 1.1)
# Why: a user/total confusion averages into the gap; a missed /9.80665 lands near 9.8.
ACCEL_FORBIDDEN_BANDS = ((0.2, 0.9), (5.0, 15.0))

GRAVITY_NORM_BAND = (0.99, 1.01)
GRAVITY_NORM_MAX_IQR = 0.01
GYRO_MEDIAN_BAND = (0.005, 2.0)
GYRO_P95_MAX = 20.0          # deg/s data would sit far above this
QUAT_NORM_BAND = (0.999, 1.001)
QUAT_GRAVITY_ANGLE_MAX_DEG = 2.0
QUAT_STILL_ANGLE_MAX_DEG = 5.0
RATE_TOLERANCE = 0.20
SPILL_GUARD_S = 60.0

MODALITIES = ("watch", "watch_rawaccel", "headimu", "pen", "markers", "attention")
DOT_TYPES = ("PEN_DOWN", "PEN_MOVE", "PEN_UP", "PEN_HOVER")
