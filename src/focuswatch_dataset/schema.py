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

# Closed vocabulary for the manifest's accel_semantics field (I6). Anything
# not a key here - an empty string, a typo, a future third value - is
# unrecognised and validate.validate_recording must fail loudly on it rather
# than silently skip the cross-check (see that function's
# accel_semantics_matches_columns finding).
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

# M8: the AirPods attention label vocabulary, measured across all 26
# ground-truth files (51 `focused` intervals, 53 `distracted`) - closed now
# that it is known, so a typo in a future observer `.txt` is caught rather
# than silently becoming its own class.
ATTENTION_LABELS = ("distracted", "focused")

# The three motion-table modalities a recording can carry (build.py's
# validation dispatch and validate.detect_dropouts both key off this - one
# table, not two kept in agreement by memory).
MOTION_MODALITIES = ("watch", "watch_rawaccel", "headimu")

# C4: a motion modality whose own span covers less than this fraction of the
# recording's overall span is a DROPOUT (sensor disconnected early/never
# returned), not a genuine stream - see validate.detect_dropouts. Measured
# case (ETH-SL-E3_session1's headimu: 58 samples over 2.1 s inside an
# 8713.6 s recording) sits at ratio ~0.00024, two orders of magnitude below
# this threshold. 0.01 (1%) is chosen to sit comfortably above that measured
# floor while staying safely below the smallest legitimate partial-coverage
# case seen in the fixtures (Ege's headimu fixture: 1490 ms inside a 29990 ms
# recording, ratio ~0.0497) - a merely imperfect stream must never be wrongly
# exempted from its physics checks.
DROPOUT_COVERAGE_RATIO_MIN = 0.01

# Pen event vocabulary shared by both ETH pipelines (DESIGN §8.1). Both come
# from the same web app but store its two export generations differently:
# generation A (Ege's own CSV export; SensorLogger's S3/T8/T9/T10, which
# carry it under a separate `pen_events` JSON key) samples the pen at
# "pen_dot"; generation B (SensorLogger's E1/E2/E3, embedded in `events`)
# samples it at "pen_move" instead. One table per generation, imported by
# both adapters (C3), so they cannot independently drift on the mapping.
PEN_EVENTS_GEN_A: dict[str, str] = {"pen_down": "PEN_DOWN", "pen_dot": "PEN_MOVE", "pen_up": "PEN_UP"}
PEN_EVENTS_GEN_B: dict[str, str] = {"pen_down": "PEN_DOWN", "pen_move": "PEN_MOVE", "pen_up": "PEN_UP"}
# Non-stroke pen event classes, present in both generations: neither carries
# a position, so neither belongs in pen/ - both are routed to markers/
# instead (DESIGN §8.1's pen_session_sync treatment, extended to
# pen_paper_info in this fix - see C3/I8 in whole-branch-review-findings.md).
PEN_NON_STROKE_EVENTS = ("pen_paper_info", "pen_session_sync")

# Pen coordinate/pressure scale labels, keyed by EXPORT GENERATION - not by
# which pipeline directory a recording happens to live in (C3). Generation A
# gets one label regardless of which pipeline produced it (Ege's own export
# and SensorLogger's S3/T8/T9/T10 are the same generation); generation B
# (SensorLogger's E1/E2/E3) is kept distinct because whole-branch-review-
# findings.md's open question 6 leaves whether the two generations share one
# underlying coordinate system unconfirmed - conflating them would publish an
# unverified equivalence as fact.
PEN_XY_UNIT_GEN_A = "webapp_raw"
PEN_PRESSURE_SCALE_GEN_A = "webapp_force"
PEN_XY_UNIT_GEN_B = "sl_webapp_raw"
PEN_PRESSURE_SCALE_GEN_B = "sl_webapp_force"

# C5: the ETH web app's own event payload vocabulary, closed and reviewed -
# see whole-branch-review-findings.md §C5. An ALLOW-list, not a deny-list: a
# deny-list fails open (the next export adds a field and it publishes
# unreviewed), an allow-list fails closed (a new field is dropped, and
# counted, until someone reviews and adds it here). user_agent (browser
# build) and screen (display geometry) are a device fingerprint; notes is
# free text an experimenter can type anything into (measured: "S3_Sensor_
# logger_dl-studying") - none of the three may ever reach the public bundle.
MARKER_PAYLOAD_ALLOWED_KEYS = frozenset({
    "color", "duration_ms", "force", "from", "get_ready_ms", "ground_truth",
    "handedness", "idx", "look_down", "mode", "operator_mode", "participant_id",
    "pen_connected_at_start", "pen_connected_t_ms", "pen_minus_session_ms",
    "pen_name", "phase", "phase_count", "phase_duration_ms", "phase_elapsed_ms",
    "session_number", "session_start_t_ms", "tilt", "time_origin_ms",
    "timestamp", "tip", "total_events", "total_pen_events", "x", "y",
})

# C5: handedness, promoted from a buried, unread payload field to a typed
# manifest column. Closed vocabulary - an unrecognised value must fail loudly
# rather than publish free text verbatim into a typed column (the same
# payload that carries it also carries `notes`, an experimenter free-text
# field, so nothing about this payload can be trusted to already be clean).
HANDEDNESS_VALUES = ("left", "right", "unknown")

# Item 1 (fix round C): time_domain vocabulary for src_-prefixed provenance
# columns whose clock is NOT their modality's default (manifest.build_channels'
# time_domain_by_column override, the same pattern as
# unit_conversion_factor_by_column). Each adapter's per-modality default
# (watch_capture_clock, server_wall_clock, backend_wall_clock,
# device_wall_clock - declared inline in time_domain_by_modality, not
# centralised here) names the clock the modality's PRIMARY t_ns axis is on;
# these name the clock a specific provenance column is actually on, measured
# against real corpus data (whole-branch-review-2.md finding 1):
# pen.src_timestamp published 749-923 days off its modality's declared
# clock. Centralised (not restated per adapter) because more than one
# adapter's provenance columns are the identical physical thing under the
# identical name.
TIME_DOMAIN_PEN_DEVICE_CLOCK = "pen_device_clock"
# Why: not a clock reading at all - milliseconds SINCE session start, so no
# wall-clock name would be honest. Emitted by every adapter that carries
# `src_t_session_ms` (ML4SCS pen/markers, both ETH pipelines).
TIME_DOMAIN_SESSION_RELATIVE_OFFSET_MS = "session_relative_offset_ms"
TIME_DOMAIN_PHONE_WALL_CLOCK = "phone_wall_clock"
# Why: a batch sequence counter (ML4SCS watch.src_sequence) carries no time
# information at all - declaring it "on" any clock, including its own
# modality's, would be a category error, not merely an imprecise one.
TIME_DOMAIN_NOT_A_CLOCK = "not_a_clock"
# Why: AirPods' src_sensor_timestamp_s is CMDeviceMotion's free-running
# uptime clock (seconds since device boot), never reset to wall-clock epoch -
# distinct from device_wall_clock, which headimu/attention's canonical t_ns
# axis (from timestamp_iso) is actually on.
TIME_DOMAIN_DEVICE_MONOTONIC_CLOCK = "device_monotonic_clock"

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
