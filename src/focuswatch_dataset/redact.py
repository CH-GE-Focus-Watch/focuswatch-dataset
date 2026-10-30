"""Optional, reversible removal of pen coordinates.

Blanking ``x``/``y`` prevents literal glyph reconstruction but preserves timing
and other behavioural signals. The bundle's manifest always records the policy.
"""
from __future__ import annotations

import json
from enum import StrEnum

import numpy as np
import pandas as pd

from . import schema as S
from .adapters.base import RecordingBundle

XY_COLUMNS = ("x", "y")
PAYLOAD_COLUMN = "src_payload"
FREE_WRITING_TASKS = frozenset({"free_writing", "think_pause_writing"})


class RedactionPolicy(StrEnum):
    NONE = "none"
    FREE_WRITING_XY = "free_writing_xy"
    ALL_XY = "all_xy"


def _coerce_policy(policy: RedactionPolicy | str) -> RedactionPolicy:
    try:
        return RedactionPolicy(policy)
    except ValueError as exc:
        raise ValueError(f"unknown redaction policy {policy!r}") from exc


def _free_writing_spans(markers: pd.DataFrame) -> list[tuple[int, int]]:
    spans = []
    for task_index, block in markers[markers["task_id"].isin(FREE_WRITING_TASKS)].groupby("task_index"):
        starts = block.loc[block["event"] == "task_start", S.TIME_COLUMN]
        ends = block.loc[block["event"] == "task_end", S.TIME_COLUMN]
        if len(starts) and len(ends):
            spans.append((int(starts.min()), int(ends.max())))
    return spans


def apply_redaction(pen: pd.DataFrame, markers: pd.DataFrame | None,
                    policy: RedactionPolicy | str) -> pd.DataFrame:
    policy = _coerce_policy(policy)
    # Why: == not is - a caller wired from a CLI flag or config value passes a
    # plain str, and StrEnum equality (unlike identity) still matches it. An
    # `is` check that silently misses would fall through to the next branch
    # instead of raising, quietly downgrading e.g. "all_xy" to a partial redaction.
    if policy == RedactionPolicy.NONE:
        return pen
    out = pen.copy()
    present = [c for c in XY_COLUMNS if c in out.columns]
    if policy == RedactionPolicy.ALL_XY:
        out[present] = np.nan
        return out
    spans = _free_writing_spans(markers) if markers is not None and not markers.empty else []
    if not _has_task_structure(markers):
        # Why: an absent marker table and a marker table without task ids are the
        # same situation - we cannot tell free writing apart, so we do not guess.
        # The ZHAW sources emit phase events without task ids and land here.
        out[present] = np.nan
        return out
    mask = pd.Series(False, index=out.index)
    for lo, hi in spans:
        mask |= out[S.TIME_COLUMN].between(lo, hi)
    out.loc[mask, present] = np.nan
    return out


def _payload_fields(payload: str) -> dict:
    fields = json.loads(payload)
    if not isinstance(fields, dict):
        raise ValueError(f"marker payload is not a JSON object: {payload[:60]!r}")
    return fields


# Why: recursive, because an allow-listed key (ground_truth, tip, ...) may hold
# an object, and a coordinate one level down is still a coordinate.
def _has_coordinates(value) -> bool:
    if isinstance(value, dict):
        return any(k in XY_COLUMNS or _has_coordinates(v) for k, v in value.items())
    if isinstance(value, list):
        return any(_has_coordinates(v) for v in value)
    return False


def _without_coordinates(value):
    if isinstance(value, dict):
        return {k: _without_coordinates(v) for k, v in value.items() if k not in XY_COLUMNS}
    if isinstance(value, list):
        return [_without_coordinates(v) for v in value]
    return value


def payload_has_coordinates(payload: str) -> bool:
    return bool(payload) and _has_coordinates(_payload_fields(payload))


def _payload_without_coordinates(payload: str) -> str:
    if not payload_has_coordinates(payload):
        return payload
    return json.dumps(_without_coordinates(_payload_fields(payload)))


def strip_payload_coordinates(markers: pd.DataFrame) -> pd.DataFrame:
    # Why: under every policy, not only within free-writing spans. No analysis
    # needs a coordinate from a marker payload, and the pen table is where the
    # policy's span logic applies.
    if PAYLOAD_COLUMN not in markers.columns:
        return markers
    out = markers.copy()
    out[PAYLOAD_COLUMN] = out[PAYLOAD_COLUMN].map(_payload_without_coordinates)
    return out


def _has_task_structure(markers: pd.DataFrame | None) -> bool:
    if markers is None or markers.empty or "task_id" not in markers.columns:
        return False
    return markers["task_id"].astype(str).str.strip().ne("").any()


def redact_bundle(bundle: RecordingBundle,
                  policy: RedactionPolicy | str = RedactionPolicy.NONE) -> RecordingBundle:
    """Apply `policy` to a bundle's pen table and marker payloads, and record it in meta.

    Default parameter is NONE, not merely a documented convention - a caller
    that builds a bundle without deciding on a policy publishes unredacted
    data and an honest "none" in the manifest, never a silent redaction.
    """
    policy = _coerce_policy(policy)
    tables = dict(bundle.tables)
    if "pen" in tables:
        tables["pen"] = apply_redaction(tables["pen"], tables.get("markers"), policy)
    if "markers" in tables and policy != RedactionPolicy.NONE:
        tables["markers"] = strip_payload_coordinates(tables["markers"])
    meta = dict(bundle.meta)
    meta["redaction_policy"] = policy.value
    return RecordingBundle(bundle.ref, tables, meta)
