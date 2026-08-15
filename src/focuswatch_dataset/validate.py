"""Physical and structural gate applied to every table before it is written."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from . import schema as S
from .physics import angle_deg, gravity_from_quaternion, norm_stats, still_mask
from .time_axis import median_rate_hz


@dataclass(frozen=True)
class Finding:
    check: str
    recording_id: str
    modality: str
    column: str
    observed: float | str
    expected: str
    passed: bool


@dataclass
class ValidationReport:
    findings: list[Finding] = field(default_factory=list)

    @property
    def failed(self) -> list[Finding]:
        return [f for f in self.findings if not f.passed]

    def to_json(self) -> str:
        return json.dumps([asdict(f) for f in self.findings], indent=2)


def _in_band(value: float, band: tuple[float, float]) -> bool:
    return band[0] <= value <= band[1]


def _has(df: pd.DataFrame, q: S.Quantity) -> bool:
    return all(c in df.columns for c in S.COLUMNS[q])


def _vec(df: pd.DataFrame, q: S.Quantity) -> np.ndarray:
    return df[list(S.COLUMNS[q])].to_numpy(dtype=float)


def _diagnose_accel(median: float, band: tuple[float, float]) -> str:
    """Name the likely cause when an acceleration norm lands outside its band."""
    if S.ACCEL_FORBIDDEN_BANDS[0][0] <= median <= S.ACCEL_FORBIDDEN_BANDS[0][1]:
        return f"median in {band}; observed value suggests a user/total mix"
    if S.ACCEL_FORBIDDEN_BANDS[1][0] <= median <= S.ACCEL_FORBIDDEN_BANDS[1][1]:
        return f"median in {band}; observed value suggests m/s2 instead of g"
    return f"median in {band}"


def validate_motion_table(df: pd.DataFrame, recording_id: str, modality: str,
                          nominal_hz: float | None) -> list[Finding]:
    out: list[Finding] = []

    def add(check, column, observed, expected, passed):
        out.append(Finding(check, recording_id, modality, column, observed, expected, passed))

    t = df[S.TIME_COLUMN].to_numpy(dtype=np.int64)
    add("time_monotonic", S.TIME_COLUMN, int(np.sum(np.diff(t) < 0)),
        "no backward steps", bool(np.all(np.diff(t) >= 0)))

    measured = median_rate_hz(t)
    # Why: pd.notna, not `if nominal_hz:` - a manifest row read back out of a
    # DataFrame turns a source's `None` (no nominal rate declared) into
    # float NaN, and `bool(float("nan"))` is True. A truthiness check would
    # silently take the declared-rate branch on a NaN and always fail the
    # comparison below (NaN < tolerance is False), misreporting an honestly
    # undeclared rate as a bad one. See tests/test_manifest.py for the proof.
    if pd.notna(nominal_hz):
        add("sample_rate", S.TIME_COLUMN, round(measured, 3), f"{nominal_hz} Hz +-20%",
            abs(measured - nominal_hz) / nominal_hz < S.RATE_TOLERANCE)
    else:
        # Why: check_coverage requires sample_rate unconditionally; a source that
        # declares no nominal rate for this table (e.g. Ege's head-IMU) must still
        # produce a finding, or coverage fails on every build of that recording.
        add("sample_rate", S.TIME_COLUMN, round(measured, 3),
            "no nominal rate declared; measured only", True)

    for q, band in ((S.Quantity.ACCEL_USER, S.ACCEL_USER_BAND),
                    (S.Quantity.ACCEL_TOTAL, S.ACCEL_TOTAL_BAND)):
        if not _has(df, q):
            continue
        st = norm_stats(_vec(df, q))
        add("accel_semantic_band", q.value, round(st.median, 5),
            _diagnose_accel(st.median, band), _in_band(st.median, band))

    if _has(df, S.Quantity.GRAVITY):
        st = norm_stats(_vec(df, S.Quantity.GRAVITY))
        add("gravity_norm", S.Quantity.GRAVITY.value, round(st.median, 5),
            f"median in {S.GRAVITY_NORM_BAND}, IQR < {S.GRAVITY_NORM_MAX_IQR}",
            _in_band(st.median, S.GRAVITY_NORM_BAND) and st.iqr < S.GRAVITY_NORM_MAX_IQR)
        flat = np.abs(_vec(df, S.Quantity.GRAVITY)[:, :2]).max(axis=1) < 0.1
        if flat.sum() > 10:
            gz = float(np.median(_vec(df, S.Quantity.GRAVITY)[flat, 2]))
            add("gravity_sign", S.Quantity.GRAVITY.value, round(gz, 4),
                "gz near -1 when the device lies flat", gz < 0)

    if _has(df, S.Quantity.GYRO):
        st = norm_stats(_vec(df, S.Quantity.GYRO))
        add("gyro_range", S.Quantity.GYRO.value, round(st.median, 5),
            f"median in {S.GYRO_MEDIAN_BAND}, p95 < {S.GYRO_P95_MAX}",
            _in_band(st.median, S.GYRO_MEDIAN_BAND) and st.p95 < S.GYRO_P95_MAX)

    out += _validate_quaternion(df, recording_id, modality)
    return out


def _validate_quaternion(df: pd.DataFrame, recording_id: str, modality: str) -> list[Finding]:
    """Quaternion checks, including the fallback for sources without a gravity column.

    A scalar-first storage order leaves every norm at 1.0 and raises no exception;
    only the reconstructed gravity direction exposes it. Sources that ship no
    gravity channel are compared against the acceleration direction while the
    device is still, which is the gravity direction to within the sensor bias.
    """
    out: list[Finding] = []

    def add(check, observed, expected, passed):
        out.append(Finding(check, recording_id, modality, S.Quantity.QUAT.value,
                           observed, expected, passed))

    if not _has(df, S.Quantity.QUAT):
        return out
    q_arr = _vec(df, S.Quantity.QUAT)
    finite_q = np.isfinite(q_arr).all(axis=1)
    # Why: ML4SCS quaternion capture is forward-only, so older sessions carry
    # partly empty columns. Rotation.from_quat raises "Found zero norm
    # quaternions" on those rows, so an unfiltered array aborts the build.
    if finite_q.sum() < 100:
        return out

    st = norm_stats(q_arr[finite_q])
    add("quat_norm", round(st.median, 6), f"median in {S.QUAT_NORM_BAND}",
        _in_band(st.median, S.QUAT_NORM_BAND))

    reconstructed = gravity_from_quaternion(q_arr[finite_q])

    if _has(df, S.Quantity.GRAVITY):
        grav = _vec(df, S.Quantity.GRAVITY)[finite_q]
        rows = np.isfinite(grav).all(axis=1)
        if rows.sum() >= 100:
            ang = float(np.median(angle_deg(reconstructed[rows], grav[rows])))
            add("quat_gravity_agreement", round(ang, 4),
                f"median angle < {S.QUAT_GRAVITY_ANGLE_MAX_DEG} deg",
                ang < S.QUAT_GRAVITY_ANGLE_MAX_DEG)
        return out

    accel_q = S.Quantity.ACCEL_TOTAL if _has(df, S.Quantity.ACCEL_TOTAL) else None
    if accel_q is None or not _has(df, S.Quantity.GYRO):
        return out
    gyro = _vec(df, S.Quantity.GYRO)[finite_q]
    accel = _vec(df, accel_q)[finite_q]
    fs = median_rate_hz(df[S.TIME_COLUMN].to_numpy(dtype=np.int64)[finite_q])
    still = still_mask(gyro, fs_hz=fs if np.isfinite(fs) else 100.0)
    if still.sum() < 100:
        add("quat_still_agreement", f"{int(still.sum())} still samples",
            "at least 100 still samples for the fallback check", False)
        return out
    ang = float(np.median(angle_deg(accel[still], reconstructed[still])))
    add("quat_still_agreement", round(ang, 4),
        f"median angle < {S.QUAT_STILL_ANGLE_MAX_DEG} deg over still windows",
        ang < S.QUAT_STILL_ANGLE_MAX_DEG)
    return out


def validate_pen_table(df: pd.DataFrame, recording_id: str) -> list[Finding]:
    unknown = sorted(set(df["dot_type"]) - set(S.DOT_TYPES))
    return [Finding("dot_type_vocabulary", recording_id, "pen", "dot_type",
                    ", ".join(unknown) or "-", f"subset of {S.DOT_TYPES}", not unknown)]


_INTERVAL_START, _INTERVAL_END = "t_start_ns", "t_end_ns"
# A contemporary wall clock in nanoseconds; anything far below is session-relative.
_EPOCH_NS_MIN, _EPOCH_NS_MAX = 1e18, 2e18


def _table_span(df: pd.DataFrame) -> tuple[int, int] | None:
    if _INTERVAL_START in df.columns and _INTERVAL_END in df.columns and len(df):
        return int(df[_INTERVAL_START].min()), int(df[_INTERVAL_END].max())
    if S.TIME_COLUMN in df.columns and len(df):
        return int(df[S.TIME_COLUMN].min()), int(df[S.TIME_COLUMN].max())
    return None


def validate_recording(recording_id: str, tables: dict[str, pd.DataFrame],
                       meta: dict) -> list[Finding]:
    """Checks that only make sense across the tables of one recording."""
    out: list[Finding] = []

    def add(check, modality, observed, expected, passed):
        out.append(Finding(check, recording_id, modality, "-", observed, expected, passed))

    spans = {m: s for m, s in ((m, _table_span(df)) for m, df in tables.items()) if s}

    for modality, (lo, _hi) in spans.items():
        add("time_magnitude", modality, lo,
            "timestamps on the declared wall clock", _EPOCH_NS_MIN <= lo <= _EPOCH_NS_MAX)

    if len(spans) > 1:
        latest_start = max(lo for lo, _ in spans.values())
        earliest_end = min(hi for _, hi in spans.values())
        overlap_s = (earliest_end - latest_start) / 1e9
        add("streams_overlap", "+".join(sorted(spans)), round(overlap_s, 3),
            "modality time ranges overlap", overlap_s > 0)

    # Why: the declared session start, taken from the source's own session
    # metadata - not the computed span, which would make the check tautological.
    start = meta.get("session_start_ns")
    if start is not None:
        for modality, (lo, _hi) in spans.items():
            lag_s = (int(start) - lo) / 1e9
            add("spill_guard", modality, round(lag_s, 3),
                f"no sample more than {S.SPILL_GUARD_S} s before session start",
                lag_s <= S.SPILL_GUARD_S)

    # I6: accel_semantics must agree with which acceleration columns the
    # motion table actually carries - "total" implies accel_total_*, "user"
    # implies accel_user_*. Checked against watch when present, else headimu
    # (AirPods' only motion stream) - the same modality accel_semantics
    # describes per DESIGN §7. Makes the column-names-carry-semantics
    # invariant structural, not just declarative. Emitted whenever a motion
    # table exists, regardless of whether `semantics` resolves - an
    # unrecognised value (empty string, a typo, a future third value) must
    # fail this finding rather than silently produce none at all (Finding 5,
    # correction round 1): a skipped check and a passed check are otherwise
    # indistinguishable, same principle as check_coverage below.
    semantics = meta.get("accel_semantics")
    quantity = S.ACCEL_SEMANTICS_QUANTITY.get(semantics)
    modality = "watch" if "watch" in tables else ("headimu" if "headimu" in tables else None)
    if modality is not None:
        add("accel_semantics_matches_columns", modality, semantics or "-",
            f"accel_semantics must be one of {sorted(S.ACCEL_SEMANTICS_QUANTITY)} and its "
            "implied accel_total_*/accel_user_* columns must be present",
            quantity is not None and all(c in tables[modality].columns for c in S.COLUMNS[quantity]))
    return out


# Which checks a recording must have produced, derived from its manifest flags.
# A skipped check and a passed check are otherwise indistinguishable. Keyed by
# (recording_id, modality) - not just recording_id - so a check that ran for
# one modality of a recording (e.g. headimu's own time_magnitude finding)
# cannot silently satisfy a requirement declared for a different modality of
# the same recording (e.g. a dropped attention table that still claims
# has_attention). Every manifest flag maps to the modality name its findings
# carry in `Finding.modality`, matching the table key every adapter and
# `validate_recording` already use.
_REQUIRED_MOTION_CHECKS = ("time_monotonic", "sample_rate", "accel_semantic_band", "gyro_range")
# Why: watch_rawaccel is accel-only by construction - the name says so, and no
# export of a raw-accelerometer stream will ever carry a gyroscope. That is a
# structural fact of the modality, not a property of any particular
# recording, so gyro_range is excluded from its required set outright rather
# than gated by a flag: a requirement no instance could ever satisfy is a
# permanent false alarm, not a safety net.
_MOTION_CHECKS_NO_GYRO = tuple(c for c in _REQUIRED_MOTION_CHECKS if c != "gyro_range")
# has_headimu's base set also excludes gyro_range - whether a head table
# carries a gyroscope is a DATA fact that varies by source (Ege's does not;
# SensorLogger's and AirPods' do, and AirPods' head gyro is that cohort's
# only motion signal), so it is required separately, gated by has_head_gyro,
# below - never assumed either way from has_headimu alone.
_MOTION_REQUIRED_CHECKS = {
    "has_watch": _REQUIRED_MOTION_CHECKS,
    "has_headimu": _MOTION_CHECKS_NO_GYRO,
    "has_watch_rawaccel": _MOTION_CHECKS_NO_GYRO,
}
# Why: S.MODALITY_FLAGS is the single source for which modality a
# has_<modality> flag gates - manifest.py's structural derivation reads the
# same table (via S.CAPABILITY_FLAG_MODALITY, which extends it), so a flag
# gated here can never silently fall out of sync with what build_manifest
# actually derives from the tables.
_MODALITY_FLAGS = S.MODALITY_FLAGS
_MOTION_MODALITY_FLAGS = ("has_watch", "has_headimu", "has_watch_rawaccel")
# Capability sub-flags (a property *within* a modality, not the modality's
# own presence) and the single physical check each gates. The modality each
# flag maps to comes from S.CAPABILITY_FLAG_MODALITY below, not restated here
# - has_quaternion's extra OR-requirement (quat_gravity_agreement vs
# quat_still_agreement) is handled separately, right after this loop.
_CAPABILITY_CHECKS = (
    ("has_gravity", "gravity_norm"),
    ("has_quaternion", "quat_norm"),
    ("has_head_gravity", "gravity_norm"),
    ("has_head_quaternion", "quat_norm"),
    ("has_head_gyro", "gyro_range"),
)


def check_coverage(manifest: pd.DataFrame, findings: list[Finding]) -> list[str]:
    problems: list[str] = []
    by_recording_modality: dict[tuple[str, str], set[str]] = {}
    for f in findings:
        by_recording_modality.setdefault((f.recording_id, f.modality), set()).add(f.check)

    for _, row in manifest.iterrows():
        rid = row["recording_id"]

        def seen(modality: str) -> set[str]:
            return by_recording_modality.get((rid, modality), set())

        def require(modality: str, check: str, reason: str) -> None:
            if check not in seen(modality):
                problems.append(f"{rid}/{modality}: {check} never ran ({reason})")

        for flag in _MOTION_MODALITY_FLAGS:
            if row.get(flag):
                modality = _MODALITY_FLAGS[flag]
                for check in _MOTION_REQUIRED_CHECKS[flag]:
                    require(modality, check, "motion table present")
        for flag, modality in _MODALITY_FLAGS.items():
            if row.get(flag):
                require(modality, "time_magnitude", f"{flag} is true")

        # Why: has_gravity/has_quaternion are the Watch-Capabilities fields
        # (docs/DESIGN.md) and describe the watch/ stream specifically;
        # has_head_gravity/has_head_quaternion/has_head_gyro are the separate
        # Head-Capabilities trio and describe the headimu stream.
        for flag, check in _CAPABILITY_CHECKS:
            if row.get(flag):
                require(S.CAPABILITY_FLAG_MODALITY[flag], check, f"{flag} is true")
        if row.get("has_quaternion"):
            modality = S.CAPABILITY_FLAG_MODALITY["has_quaternion"]
            if not ({"quat_gravity_agreement", "quat_still_agreement"} & seen(modality)):
                problems.append(
                    f"{rid}/{modality}: neither quat_gravity_agreement nor quat_still_agreement ran "
                    "(has_quaternion is true)")
        if row.get("has_pen"):
            require("pen", "dot_type_vocabulary", "has_pen is true")
    return problems
