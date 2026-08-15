"""The manifest is the single source of truth for capability flags.

Parquet key-value metadata and any downstream descriptor are generated from
it; nothing else is maintained independently.

Three guarantees this module exists to provide:

1. Capability flags cannot silently drop out of the built manifest. The
   modality/gravity/quaternion/gyro booleans that `check_coverage` gates on
   are recomputed here directly from `bundle.tables` column presence -
   never merely trusted from an adapter's own `meta` dict, which could in
   principle disagree with the data it describes. Which flags these are,
   which table each looks in, and which column each looks for comes from
   `schema.CAPABILITY_FLAG_MODALITY`/`schema.CAPABILITY_FLAG_COLUMN`, the
   same tables `validate.check_coverage` reads its modality bindings from -
   so a flag `check_coverage` starts gating on can never silently fall out
   of sync with what this module derives from the tables (see
   `_DERIVED_FIELDS` and the structural-flag loop in `build_manifest`,
   which needs no per-flag line of its own). The two measured-rate fields
   (`watch_hz_measured`, `head_hz_measured`) get the identical exclusion
   treatment for the identical reason, even though they are not booleans
   and so are not part of the shared flag tables.
2. Every `bundle.meta` key is either published (`MANIFEST_COLUMNS`),
   deliberately kept internal (`_INTERNAL_META_KEYS`, e.g. the
   spill-guard-only `session_start_ns`), or fails the build loudly. A
   hand-maintained column list can drift from the adapters that actually
   populate it; the old failure mode of that drift was `df[list(
   MANIFEST_COLUMNS)]` silently dropping an unlisted key (this project's
   own `has_head_gyro` very nearly shipped that way). The fix is not to
   swing the other way and publish every unknown key unexamined either -
   that would leak internal-only bookkeeping (a spill-guard timestamp with
   no declared unit or purpose) into the published dataset. Only a
   classified key gets through, one way or the other; an unclassified one
   raises.
3. Optional numeric fields (`watch_hz_nominal`, `head_hz_nominal`,
   `n_writing_tasks`, `n_idle_tasks`) survive the round trip through a
   `pandas.DataFrame`. A `None` in an all-numeric column becomes `NaN` on
   the way out, and `NaN` is truthy in Python - a naive `if nominal_hz:`
   downstream would treat "not applicable" as if a real value divided into
   it. `check_manifest_consistency` and every accessor in this module read
   those fields with `pandas.notna`, and `validate.validate_motion_table`
   was fixed to do the same (see that module's history).
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
    "protocol_id", "study_mode", "subject_index", "n_writing_tasks", "n_idle_tasks",
    "watch_wrist_side",
    "pen_xy_unit", "pen_pressure_scale", "pen_delta_s", "pen_delta_sigma",
    "delta_applied", "alignment_note",
    "n_samples_watch", "n_samples_pen", "n_samples_head", "issue_codes",
    "source_pipeline", "schema_version", "redaction_policy", "build_git_sha",
)

# Meta keys that are legitimate for an adapter to emit but are never
# published: internal bookkeeping consumed elsewhere in the pipeline before
# the manifest is built (session_start_ns feeds validate_recording's spill
# guard; src_standardisation is SensorLogger's own audit trail for the unit
# harmonisation it already applied). A key that is neither published nor
# listed here fails the build loudly - see build_manifest.
_INTERNAL_META_KEYS = frozenset({"session_start_ns", "src_standardisation"})

# Why: these are the flags check_coverage gates required physical checks on -
# schema.CAPABILITY_FLAG_MODALITY is the single table both modules read them
# from. Recomputing them from table/column presence here - rather than
# trusting whatever an adapter's meta dict says - is what makes "the flag an
# adapter forgot to set" structurally impossible rather than merely tested
# per-adapter. The two measured-rate fields get the same treatment: they are
# derived from the same tables the structural flags are, so they are excluded
# from the meta merge alongside them, even though they are not gated by
# check_coverage and so are not part of the shared flag/modality table.
_STRUCTURAL_FLAGS = frozenset(S.CAPABILITY_FLAG_MODALITY)
_DERIVED_MEASURED_FIELDS = frozenset({"watch_hz_measured", "head_hz_measured"})
_DERIVED_TASK_COUNT_FIELDS = frozenset({"n_writing_tasks", "n_idle_tasks"})
_DERIVED_FIELDS = _STRUCTURAL_FLAGS | _DERIVED_MEASURED_FIELDS | _DERIVED_TASK_COUNT_FIELDS

_DEFAULTS: dict[str, object] = {
    "watch_wrist_side": "unknown", "delta_applied": False,
    "pen_delta_s": np.nan, "pen_delta_sigma": np.nan, "alignment_note": "",
    "accel_still_bias": np.nan, "issue_codes": "", "redaction_policy": "none",
    "build_git_sha": "", "subject_index": -1, "study_mode": "",
    "watch_hz_nominal": np.nan, "watch_hz_measured": np.nan,
    "head_hz_nominal": np.nan, "head_hz_measured": np.nan,
    "n_writing_tasks": None, "n_idle_tasks": None,
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


def _task_counts(markers: pd.DataFrame | None) -> tuple[int | None, int | None]:
    """Count distinct writing/idle task instances from the markers table.

    Each task instance writes one `task_start` and one `task_end` row
    sharing a `task_category`; counting `task_start` rows avoids double
    counting. Only ml4scs.py populates `task_category` with real values
    (`writing`/`idle`, per Study Mode's protocol taxonomy) - ege.py and
    sensorlogger.py's markers tables are a plain session-event stream and
    always write `""`. Returns `(None, None)`, not `(0, 0)`, when no task
    taxonomy is present at all: a 0 would misread as "this recording had a
    protocol with zero writing tasks", which is false: it is a recording
    whose source never had a task-labelled protocol in the first place.
    """
    if markers is None or not len(markers) or "task_category" not in markers.columns:
        return None, None
    cats = markers.loc[markers.get("event") == "task_start", "task_category"]
    if not (cats.astype(str) != "").any():
        return None, None
    return int((cats == "writing").sum()), int((cats == "idle").sum())


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
        # Capability sub-flags: same structural treatment, driven by
        # S.CAPABILITY_FLAG_COLUMN (which column) paired with
        # S.CAPABILITY_FLAG_MODALITY (which table) - not five literal lines.
        # A flag added to those shared tables later is derived correctly here
        # with no matching edit in this module; before this loop existed, a
        # flag missing its own literal line fell through to _DEFAULTS' False
        # regardless of what the data actually said (see fix round 2's
        # mutation in the task report).
        for flag, column in S.CAPABILITY_FLAG_COLUMN.items():
            table = b.tables.get(S.CAPABILITY_FLAG_MODALITY[flag])
            row[flag] = table is not None and column in table.columns
        watch = b.tables.get("watch")
        if watch is not None:
            row["watch_hz_measured"] = _rate(watch)
        head = b.tables.get("headimu")
        if head is not None:
            row["head_hz_measured"] = _rate(head)
        row["n_writing_tasks"], row["n_idle_tasks"] = _task_counts(b.tables.get("markers"))

        # Everything else the adapter declared, excluding the fields just
        # derived above (a redundant meta declaration - today always
        # consistent - can never override the value derived from the actual
        # tables) and the internal-only keys classified in
        # _INTERNAL_META_KEYS (excluded here so they never occupy a manifest
        # column at all - see the loud-failure check below for why an
        # unclassified key is not simply dropped the same way).
        row.update({
            k: v for k, v in b.meta.items()
            if k not in _DERIVED_FIELDS and k not in _INTERNAL_META_KEYS
        })
        rows.append(row)

    df = pd.DataFrame(rows)
    # Why: a column that is neither declared nor explicitly internal is not
    # silently published (the old df[list(MANIFEST_COLUMNS)] bug this module
    # exists to prevent) NOR silently dropped (the opposite failure - an
    # unexamined new flag would then never reach check_coverage either).
    # Fails the build loudly instead, so a new adapter meta key forces a
    # deliberate MANIFEST_COLUMNS-or-_INTERNAL_META_KEYS decision.
    unclassified = sorted(c for c in df.columns if c not in MANIFEST_COLUMNS
                          and c not in _INTERNAL_META_KEYS)
    if unclassified:
        raise ValueError(
            f"unclassified manifest column(s) {unclassified}: add to MANIFEST_COLUMNS to "
            "publish them, or to _INTERNAL_META_KEYS to keep them internal-only"
        )
    for c in MANIFEST_COLUMNS:
        if c not in df.columns:
            df[c] = _DEFAULTS.get(c, np.nan)
    return df[list(MANIFEST_COLUMNS)]


_TIME_COLUMNS = {S.TIME_COLUMN, "t_start_ns", "t_end_ns"}

# Non-physical signal columns whose actual unit is declared per-recording in
# the manifest, rather than fixed by schema - Moleskine's ncode grid and the
# ETH web app's own pixel/force scale are genuinely different units for the
# same column name, so a fixed placeholder would hide real information the
# manifest already carries. column -> (semantics, manifest meta key holding
# the actual per-recording unit).
_PEN_SCALE_COLUMNS: dict[str, tuple[str, str]] = {
    "x": ("pen_position", "pen_xy_unit"),
    "y": ("pen_position", "pen_xy_unit"),
    "pressure": ("pen_pressure", "pen_pressure_scale"),
}

# Non-physical signal columns with a fixed, source-independent unit. Curated
# for readability; any column this table has never heard of - including a
# future one - still gets a non-empty (unit, semantics) from the fallback in
# _describe_column, so build_channels can never emit a blank cell. Vocabulary
# glossary (see also docs/DESIGN.md's channels.parquet section):
#   category  - one of a fixed, enumerated set of string values
#   ordinal   - the position of an item within a sequence, not a magnitude
#   device_native - raw values in whatever unit the source stream itself
#                   uses; look up the per-recording pen_xy_unit/
#                   pen_pressure_scale manifest fields (or, for tilt, the
#                   source's own docs) for what that unit actually is
#   source_native - a src_-prefixed provenance passthrough column: the
#                   source device's own value, kept for audit, never
#                   reinterpreted or rescaled
#   n/a       - no physical or source unit applies (e.g. a categorical id)
_KNOWN_METADATA_COLUMNS: dict[str, tuple[str, str]] = {
    "dot_type": ("category", "pen_contact_state"),
    "tilt_x": ("device_native", "pen_tilt"),
    "tilt_y": ("device_native", "pen_tilt"),
    "label": ("category", "attention_state"),
    "event": ("category", "study_event"),
    "task_id": ("category", "study_task_id"),
    "task_name": ("category", "study_task_name"),
    "task_index": ("ordinal", "study_task_index"),
    "task_category": ("category", "study_task_category"),
    "protocol_id": ("category", "study_protocol_id"),
}


def _describe_column(column: str, quantity: S.Quantity | None,
                     meta: dict[str, object]) -> tuple[str, str, str]:
    """(unit, semantics, frame) for one column - never an empty unit or semantics.

    Physical quantities come from schema.UNITS. Time columns (the primary
    axis and interval endpoints) are nanoseconds. Pen x/y/pressure take their
    unit from this recording's own manifest declaration (_PEN_SCALE_COLUMNS).
    Other known non-physical columns (pen tilt, marker/attention metadata)
    have a hand-picked entry. Anything else - chiefly src_-prefixed
    provenance passthrough columns, but also any future column no branch
    above recognises - falls back to an explicit "n/a"/"metadata" pair
    rather than an empty string.
    """
    if quantity is not None:
        return S.UNITS[quantity], quantity.value, "device"
    if column in _TIME_COLUMNS:
        return "ns", "timestamp", ""
    if column.startswith("src_"):
        return "source_native", "source_provenance", ""
    if column in _PEN_SCALE_COLUMNS:
        semantics, meta_key = _PEN_SCALE_COLUMNS[column]
        return (meta.get(meta_key) or "device_native"), semantics, ""
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
                unit, semantics, frame = _describe_column(column, q, b.meta)
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
