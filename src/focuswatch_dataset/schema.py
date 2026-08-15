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
# Why: g-scale gravity sits near 1.0, SI (m/s2) sits near 9.80665 - a wide
# margin between the two makes a median-norm check unambiguous.
GRAVITY_SI_NORM_THRESHOLD = 2.0
GYRO_MEDIAN_BAND = (0.005, 2.0)
GYRO_P95_MAX = 20.0          # deg/s data would sit far above this
QUAT_NORM_BAND = (0.999, 1.001)
QUAT_GRAVITY_ANGLE_MAX_DEG = 2.0
QUAT_STILL_ANGLE_MAX_DEG = 5.0
RATE_TOLERANCE = 0.20
SPILL_GUARD_S = 60.0
# Why: the AirPods per-sample label column is an expansion of the interval
# ground truth over an unstated clock anchor; re-expanding the parsed
# intervals and comparing against that column proves the anchor before it is
# dropped. Below this agreement the anchor, not the data, is wrong.
ATTENTION_EXPANSION_MIN_AGREEMENT = 0.99

MODALITIES = ("watch", "watch_rawaccel", "headimu", "pen", "markers", "attention")
DOT_TYPES = ("PEN_DOWN", "PEN_MOVE", "PEN_UP", "PEN_HOVER")

# Which modality's Finding stream a manifest capability flag gates. The single
# source both validate.check_coverage (which physical checks a flag requires)
# and manifest.build_manifest (which flags are recomputed from table/column
# presence rather than trusted from an adapter's meta) import - so a flag
# check_coverage gates on can never silently fall out of sync with what
# build_manifest actually derives, and vice versa. MODALITY_FLAGS is the
# has_<modality> subset alone (used for the coverage-wide time_magnitude
# requirement); CAPABILITY_FLAG_MODALITY extends it with the sub-flags that
# describe a capability *within* a modality (gravity/quaternion/gyro).
MODALITY_FLAGS: dict[str, str] = {f"has_{m}": m for m in MODALITIES}
CAPABILITY_FLAG_MODALITY: dict[str, str] = {
    **MODALITY_FLAGS,
    "has_gravity": "watch",
    "has_quaternion": "watch",
    "has_head_gravity": "headimu",
    "has_head_quaternion": "headimu",
    "has_head_gyro": "headimu",
}

# Which real column manifest.build_manifest checks for, within the table
# CAPABILITY_FLAG_MODALITY names, to structurally derive a capability
# sub-flag. Paired with CAPABILITY_FLAG_MODALITY rather than folded into it,
# because the six has_<modality> flags in MODALITY_FLAGS need no column check
# (table presence alone is the fact they describe) and keeping their value
# type a plain modality string is what lets validate.check_coverage use
# CAPABILITY_FLAG_MODALITY[flag] directly as a modality name. A flag added
# here (and to CAPABILITY_FLAG_MODALITY) is derived by build_manifest with no
# further code change - see that module's structural-flag loop.
CAPABILITY_FLAG_COLUMN: dict[str, str] = {
    "has_gravity": "gravity_x",
    "has_quaternion": "quat_x",
    "has_head_gravity": "gravity_x",
    "has_head_quaternion": "quat_x",
    "has_head_gyro": "gyro_x",
}

# Why: both tables are keyed by the same sub-flags and can only drift apart in
# silence - one named here but not there is derived and never required; the
# reverse is required and never derived. Checking the key sets at import turns
# that into an immediate failure rather than a check that stops firing.
assert set(CAPABILITY_FLAG_COLUMN) == set(CAPABILITY_FLAG_MODALITY) - set(MODALITY_FLAGS), (
    "CAPABILITY_FLAG_COLUMN must name exactly the capability sub-flags: "
    f"{sorted(set(CAPABILITY_FLAG_MODALITY) - set(MODALITY_FLAGS))}"
)
