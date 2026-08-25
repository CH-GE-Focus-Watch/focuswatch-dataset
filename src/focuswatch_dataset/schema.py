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

# Closed vocabulary: validation rejects empty, unknown, or future values.
ACCEL_SEMANTICS_QUANTITY: dict[str, Quantity] = {
    "total": Quantity.ACCEL_TOTAL,
    "user": Quantity.ACCEL_USER,
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

# Closed observer-label vocabulary; validation rejects future typos.
ATTENTION_LABELS = ("distracted", "focused")

# The three motion-table modalities a recording can carry (build.py's
# validation dispatch and validate.detect_dropouts both key off this - one
# table, not two kept in agreement by memory).
MOTION_MODALITIES = ("watch", "watch_rawaccel", "headimu")

# A stream below 1% of the recording span is a dropout, not a valid partial
# stream. The threshold separates observed disconnected streams from fixtures.
DROPOUT_COVERAGE_RATIO_MIN = 0.01

# Shared ETH web-app event mappings. Export generation A uses `pen_dot` and
# generation B uses `pen_move`; both adapters import these constants.
PEN_EVENTS_GEN_A: dict[str, str] = {"pen_down": "PEN_DOWN", "pen_dot": "PEN_MOVE", "pen_up": "PEN_UP"}
PEN_EVENTS_GEN_B: dict[str, str] = {"pen_down": "PEN_DOWN", "pen_move": "PEN_MOVE", "pen_up": "PEN_UP"}
# Non-stroke pen events have no position and belong in markers/, not pen/.
PEN_NON_STROKE_EVENTS = ("pen_paper_info", "pen_session_sync")

# Coordinate and pressure labels are keyed by export generation. Their scales
# are not known to be equivalent, so they remain distinct.
PEN_XY_UNIT_GEN_A = "webapp_raw"
PEN_PRESSURE_SCALE_GEN_A = "webapp_force"
PEN_XY_UNIT_GEN_B = "sl_webapp_raw"
PEN_PRESSURE_SCALE_GEN_B = "sl_webapp_force"

# Payload allow-list: unknown fields are dropped and counted. Excluded browser,
# display, and free-text fields can fingerprint a device or expose private text.
MARKER_PAYLOAD_ALLOWED_KEYS = frozenset({
    "color", "duration_ms", "force", "from", "get_ready_ms", "ground_truth",
    "handedness", "idx", "look_down", "mode", "operator_mode", "participant_id",
    "pen_connected_at_start", "pen_connected_t_ms", "pen_minus_session_ms",
    "pen_name", "phase", "phase_count", "phase_duration_ms", "phase_elapsed_ms",
    "session_number", "session_start_t_ms", "tilt", "time_origin_ms",
    "timestamp", "tip", "total_events", "total_pen_events", "x", "y",
})

# Typed handedness vocabulary; unknown payload text must not reach the manifest.
HANDEDNESS_VALUES = ("left", "right", "unknown")

# Provenance columns may use a clock other than their table's canonical t_ns
# axis. These shared values are applied through time_domain_by_column.
TIME_DOMAIN_PEN_DEVICE_CLOCK = "pen_device_clock"
# Why: not a clock reading at all - milliseconds SINCE session start, so no
# wall-clock name would be honest. Emitted by every adapter that carries
# `src_t_session_ms` (ML4SCS pen/markers, both ETH pipelines).
TIME_DOMAIN_SESSION_RELATIVE_OFFSET_MS = "session_relative_offset_ms"
TIME_DOMAIN_PHONE_WALL_CLOCK = "phone_wall_clock"
# Time-valued provenance columns name their own clock; other columns inherit
# their row's modality clock. AirPods sensor timestamps are device uptime.
TIME_DOMAIN_DEVICE_MONOTONIC_CLOCK = "device_monotonic_clock"

# A pairing based on overlapping ranges from independently maintained wall
# clocks. The clocks have not been synchronised and no offset is applied, so
# sample-level cross-modal joins are approximate even though both axes use a
# Unix epoch representation.
TIME_ALIGNMENT_OVERLAP_ONLY = "overlap_only"
TIME_ALIGNMENT_VALUES = ("shared_clock", "estimated_delta", TIME_ALIGNMENT_OVERLAP_ONLY)

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
    "has_watch_gyro": "watch",
    "has_head_gravity": "headimu",
    "has_head_quaternion": "headimu",
    "has_head_gyro": "headimu",
}

# Which complete vector manifest.build_manifest checks for, within the table
# CAPABILITY_FLAG_MODALITY names, to structurally derive a capability
# sub-flag. A partial vector is malformed rather than evidence that the
# capability is absent, so each value intentionally comes from COLUMNS rather
# than naming only an x component. Paired with CAPABILITY_FLAG_MODALITY rather
# than folded into it, because the six has_<modality> flags in MODALITY_FLAGS
# need no column check (table presence alone is the fact they describe) and
# keeping their value type a plain modality string is what lets
# validate.check_coverage use CAPABILITY_FLAG_MODALITY[flag] directly as a
# modality name.
CAPABILITY_FLAG_COLUMNS: dict[str, tuple[str, ...]] = {
    "has_gravity": COLUMNS[Quantity.GRAVITY],
    "has_quaternion": COLUMNS[Quantity.QUAT],
    "has_watch_gyro": COLUMNS[Quantity.GYRO],
    "has_head_gravity": COLUMNS[Quantity.GRAVITY],
    "has_head_quaternion": COLUMNS[Quantity.QUAT],
    "has_head_gyro": COLUMNS[Quantity.GYRO],
}

# Why: both tables are keyed by the same sub-flags and can only drift apart in
# silence - one named here but not there is derived and never required; the
# reverse is required and never derived. Checking the key sets at import turns
# that into an immediate failure rather than a check that stops firing.
assert set(CAPABILITY_FLAG_COLUMNS) == set(CAPABILITY_FLAG_MODALITY) - set(MODALITY_FLAGS), (
    "CAPABILITY_FLAG_COLUMNS must name exactly the capability sub-flags: "
    f"{sorted(set(CAPABILITY_FLAG_MODALITY) - set(MODALITY_FLAGS))}"
)
