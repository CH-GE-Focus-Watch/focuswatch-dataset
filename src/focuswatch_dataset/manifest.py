"""The manifest is the single source of truth for capability flags.

Parquet key-value metadata and any downstream descriptor are generated from
it; nothing else is maintained independently.

Two guarantees this module exists to provide:

1. Capability flags cannot silently drop out of the built manifest. The
   modality/gravity/quaternion/gyro booleans that `check_coverage` gates on
   are recomputed here directly from `bundle.tables` column presence -
   never merely trusted from an adapter's own `meta` dict, which could in
   principle disagree with the data it describes. Every other field an
   adapter puts in `meta` still reaches the manifest even if a future
   adapter introduces a key `MANIFEST_COLUMNS` has not been told about yet
   (see `build_manifest`'s column union at the end) - a hand-maintained
   column list can drift from the adapters that actually populate it, and
   the failure mode of that drift is a check that quietly never ran.
2. Optional numeric fields (`watch_hz_nominal`, `head_hz_nominal`) survive
   the round trip through a `pandas.DataFrame`. A `None` in an all-numeric
   column becomes `NaN` on the way out, and `NaN` is truthy in Python - a
   naive `if nominal_hz:` downstream would treat "no nominal rate declared"
   as if a rate of 0 divided into it. `check_manifest_consistency` and every
   accessor in this module read those fields with `pandas.notna`, and
   `validate.validate_motion_table` was fixed to do the same (see that
   module's history).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import schema as S
from .adapters.base import RecordingBundle
from .time_axis import median_rate_hz

MANIFEST_COLUMNS = (
    "recording_id", "participant_id", "cohort", "pipeline",
    "has_watch", "has_watch_rawaccel", "has_headimu", "has_pen", "has_markers", "has_attention",
    "watch_hz_nominal", "watch_hz_measured", "has_gravity", "has_quaternion",
    "accel_semantics", "accel_calibration", "accel_still_bias", "gravity_source",
    "head_hz_nominal", "head_hz_measured", "has_head_gravity", "has_head_quaternion", "has_head_gyro",
    "time_domain", "time_alignment", "t_start_ns", "t_end_ns", "duration_s",
    "protocol_id", "study_mode", "subject_index",
    "watch_wrist_side",
    "pen_xy_unit", "pen_pressure_scale", "pen_delta_s", "pen_delta_sigma",
    "delta_applied", "alignment_note",
    "n_samples_watch", "n_samples_pen", "n_samples_head", "issue_codes",
    "source_pipeline", "schema_version", "redaction_policy", "build_git_sha",
)

# Why: these are the flags check_coverage gates required physical checks on
# (validate.py's _MOTION_MODALITY_FLAGS / _MODALITY_FLAGS / the has_head_*
# trio). Recomputing them from table/column presence here - rather than
# trusting whatever an adapter's meta dict says - is what makes "the flag an
# adapter forgot to set" structurally impossible rather than merely tested
# per-adapter. Excluded from the meta merge below so a stale or missing meta
# value can never override the structural truth.
_STRUCTURAL_FLAGS = frozenset(
    {f"has_{m}" for m in S.MODALITIES}
    | {"has_gravity", "has_quaternion", "has_head_gravity", "has_head_quaternion", "has_head_gyro"}
)

_DEFAULTS: dict[str, object] = {
    "watch_wrist_side": "unknown", "delta_applied": False,
    "pen_delta_s": np.nan, "pen_delta_sigma": np.nan, "alignment_note": "",
    "accel_still_bias": np.nan, "issue_codes": "", "redaction_policy": "none",
    "build_git_sha": "", "subject_index": -1, "study_mode": "",
    "watch_hz_nominal": np.nan, "watch_hz_measured": np.nan,
    "head_hz_nominal": np.nan, "head_hz_measured": np.nan,
    "pen_xy_unit": "", "pen_pressure_scale": "",
    "accel_semantics": "", "accel_calibration": "", "gravity_source": "none",
    "time_domain": "", "time_alignment": "", "protocol_id": "",
    "source_pipeline": "",
    **{flag: False for flag in _STRUCTURAL_FLAGS},
}


def _span(bundle: RecordingBundle) -> tuple[int, int]:
    starts, ends = [], []
    for df in bundle.tables.values():
        if not len(df):
            continue
        # Why: interval tables carry start and end separately. Reading the end
        # from t_start_ns would drop the final interval from the duration.
        if {"t_start_ns", "t_end_ns"} <= set(df.columns):
            starts.append(int(df["t_start_ns"].min()))
            ends.append(int(df["t_end_ns"].max()))
        elif S.TIME_COLUMN in df.columns:
            starts.append(int(df[S.TIME_COLUMN].min()))
            ends.append(int(df[S.TIME_COLUMN].max()))
    return (min(starts), max(ends)) if starts else (0, 0)


def _rate(df: pd.DataFrame | None) -> float:
    if df is None or S.TIME_COLUMN not in df.columns or not len(df):
        return np.nan
    hz = median_rate_hz(df[S.TIME_COLUMN].to_numpy(dtype=np.int64))
    return round(hz, 3) if np.isfinite(hz) else np.nan


def build_manifest(bundles: list[RecordingBundle]) -> pd.DataFrame:
    rows = []
    for b in bundles:
        t0, t1 = _span(b)
        row: dict[str, object] = dict(_DEFAULTS)
        row.update({
            "recording_id": b.ref.recording_id, "participant_id": b.ref.participant_id,
            "cohort": b.ref.cohort, "pipeline": b.ref.pipeline,
            "source_pipeline": b.ref.pipeline, "schema_version": S.SCHEMA_VERSION,
            "t_start_ns": t0, "t_end_ns": t1, "duration_s": round((t1 - t0) / 1e9, 3),
            "n_samples_watch": len(b.tables.get("watch", [])),
            "n_samples_pen": len(b.tables.get("pen", [])),
            "n_samples_head": len(b.tables.get("headimu", [])),
        })

        # Structural flags: table/column presence, not meta - see module docstring.
        for m in S.MODALITIES:
            row[f"has_{m}"] = m in b.tables
        watch = b.tables.get("watch")
        if watch is not None:
            row["watch_hz_measured"] = _rate(watch)
            row["has_gravity"] = "gravity_x" in watch.columns
            row["has_quaternion"] = "quat_x" in watch.columns
        head = b.tables.get("headimu")
        if head is not None:
            row["head_hz_measured"] = _rate(head)
            row["has_head_gravity"] = "gravity_x" in head.columns
            row["has_head_quaternion"] = "quat_x" in head.columns
            row["has_head_gyro"] = "gyro_x" in head.columns

        # Everything else the adapter declared - including keys this module's
        # own MANIFEST_COLUMNS list has never heard of; see the column union
        # below. Structural flags are excluded so a redundant (and always
        # consistent, in every adapter today) meta declaration can never
        # override the value just derived from the actual tables.
        row.update({k: v for k, v in b.meta.items() if k not in _STRUCTURAL_FLAGS})
        rows.append(row)

    df = pd.DataFrame(rows)
    # Why: union, not `df[list(MANIFEST_COLUMNS)]` - the latter silently drops
    # any meta key this list has not been updated for. Declared columns first
    # for a stable, readable layout; anything extra sorted after them.
    ordered = list(MANIFEST_COLUMNS) + sorted(c for c in df.columns if c not in MANIFEST_COLUMNS)
    for c in ordered:
        if c not in df.columns:
            df[c] = _DEFAULTS.get(c, np.nan)
    return df[ordered]


_TIME_COLUMNS = {S.TIME_COLUMN, "t_start_ns", "t_end_ns"}

# Non-physical signal columns that are not part of a schema.Quantity. Curated
# for readability; any column this table has never heard of - including a
# future one - still gets a non-empty (unit, semantics) from the fallback in
# _describe_column, so build_channels can never emit a blank cell.
_KNOWN_METADATA_COLUMNS: dict[str, tuple[str, str]] = {
    "dot_type": ("category", "pen_contact_state"),
    "x": ("device_native", "pen_position"),
    "y": ("device_native", "pen_position"),
    "pressure": ("device_native", "pen_pressure"),
    "tilt_x": ("device_native", "pen_tilt"),
    "tilt_y": ("device_native", "pen_tilt"),
    "label": ("category", "attention_state"),
    "event": ("category", "study_event"),
    "task_id": ("category", "study_task_id"),
    "task_name": ("category", "study_task_name"),
    "task_index": ("count", "study_task_index"),
    "task_category": ("category", "study_task_category"),
    "protocol_id": ("category", "study_protocol_id"),
}


def _describe_column(column: str, quantity: S.Quantity | None) -> tuple[str, str, str]:
    """(unit, semantics, frame) for one column - never an empty unit or semantics.

    Physical quantities come from schema.UNITS. Time columns (the primary
    axis and interval endpoints) are nanoseconds. Known non-physical columns
    (pen/marker/attention metadata) have a hand-picked entry. Anything else -
    chiefly src_-prefixed provenance passthrough columns, but also any future
    column no branch above recognises - falls back to an explicit
    "n/a"/"metadata" pair rather than an empty string.
    """
    if quantity is not None:
        return S.UNITS[quantity], quantity.value, "device"
    if column in _TIME_COLUMNS:
        return "ns", "timestamp", ""
    if column.startswith("src_"):
        return "source_native", "source_provenance", ""
    unit, semantics = _KNOWN_METADATA_COLUMNS.get(column, ("n/a", "metadata"))
    return unit, semantics, ""


def build_channels(bundles: list[RecordingBundle]) -> pd.DataFrame:
    col_to_quantity = {c: q for q, cols in S.COLUMNS.items() for c in cols}
    rows = []
    for b in bundles:
        for modality, df in b.tables.items():
            hz = _rate(df)
            for column in df.columns:
                q = col_to_quantity.get(column)
                unit, semantics, frame = _describe_column(column, q)
                rows.append({
                    "recording_id": b.ref.recording_id, "modality": modality, "column": column,
                    "quantity": q.value if q else ("time" if column in _TIME_COLUMNS else "other"),
                    "unit": unit, "semantics": semantics, "frame": frame,
                    "sample_rate_hz": hz, "unit_conversion_factor": 1.0,
                })
    return pd.DataFrame(rows, columns=[
        "recording_id", "modality", "column", "quantity", "unit", "semantics", "frame",
        "sample_rate_hz", "unit_conversion_factor",
    ])


def check_manifest_consistency(manifest: pd.DataFrame, root: Path) -> list[str]:
    """Cross-check declared has_<modality> flags against files actually on disk."""
    problems = []
    for _, row in manifest.iterrows():
        for m in S.MODALITIES:
            path = root / m / f"{row['recording_id']}.parquet"
            declared, exists = bool(row[f"has_{m}"]), path.exists()
            if declared and not exists:
                problems.append(f"{m}/{row['recording_id']}.parquet declared but missing")
            if exists and not declared:
                problems.append(f"{m}/{row['recording_id']}.parquet present but not declared")
    return problems
