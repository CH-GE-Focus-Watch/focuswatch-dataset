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
    if nominal_hz:
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
    return out


# Which checks a recording must have produced, derived from its manifest flags.
# A skipped check and a passed check are otherwise indistinguishable.
_REQUIRED_MOTION_CHECKS = ("time_monotonic", "sample_rate", "accel_semantic_band", "gyro_range")
# Every manifest flag backed by its own table in `validate_recording`'s `tables`
# dict, and therefore expected to carry a time_magnitude finding.
_MODALITY_FLAGS = ("has_watch", "has_watch_rawaccel", "has_headimu", "has_pen",
                   "has_markers", "has_attention")


def check_coverage(manifest: pd.DataFrame, findings: list[Finding]) -> list[str]:
    problems: list[str] = []
    by_recording: dict[str, set[str]] = {}
    for f in findings:
        by_recording.setdefault(f.recording_id, set()).add(f.check)

    for _, row in manifest.iterrows():
        rid = row["recording_id"]
        seen = by_recording.get(rid, set())

        def require(check: str, reason: str) -> None:
            if check not in seen:
                problems.append(f"{rid}: {check} never ran ({reason})")

        if row.get("has_watch") or row.get("has_headimu") or row.get("has_watch_rawaccel"):
            for check in _REQUIRED_MOTION_CHECKS:
                require(check, "motion table present")
        if any(row.get(flag) for flag in _MODALITY_FLAGS):
            require("time_magnitude", "recording declares at least one modality")
        if row.get("has_gravity"):
            require("gravity_norm", "has_gravity is true")
        if row.get("has_quaternion"):
            require("quat_norm", "has_quaternion is true")
            if not ({"quat_gravity_agreement", "quat_still_agreement"} & seen):
                problems.append(
                    f"{rid}: neither quat_gravity_agreement nor quat_still_agreement ran "
                    "(has_quaternion is true)")
        if row.get("has_pen"):
            require("dot_type_vocabulary", "has_pen is true")
    return problems
