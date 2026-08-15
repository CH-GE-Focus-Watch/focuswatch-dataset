"""Optional removal of pen coordinates.

Stroke geometry reconstructs the written text. Structured tasks have prescribed
content; free-writing blocks do not. The policy runs on the canonical pen table
so it covers every source, and it blanks values rather than dropping rows.

Default is NONE - the dataset owner's decision is to publish full coordinates
for now, keeping the option to obscure them later. `redact_bundle` is the seam
a build pipeline calls to keep that choice reversible: it applies the policy
and stamps the resulting `RecordingBundle.meta["redaction_policy"]` in the
same call, so the published manifest (which trusts this meta key verbatim,
see `manifest.py`) can never disagree with what was actually done to the data.
"""
from __future__ import annotations

from enum import StrEnum

import numpy as np
import pandas as pd

from . import schema as S
from .adapters.base import RecordingBundle

XY_COLUMNS = ("x", "y")
FREE_WRITING_TASKS = frozenset({"free_writing", "think_pause_writing"})


class RedactionPolicy(StrEnum):
    NONE = "none"
    FREE_WRITING_XY = "free_writing_xy"
    ALL_XY = "all_xy"


def _free_writing_spans(markers: pd.DataFrame) -> list[tuple[int, int]]:
    spans = []
    for task_index, block in markers[markers["task_id"].isin(FREE_WRITING_TASKS)].groupby("task_index"):
        starts = block.loc[block["event"] == "task_start", S.TIME_COLUMN]
        ends = block.loc[block["event"] == "task_end", S.TIME_COLUMN]
        if len(starts) and len(ends):
            spans.append((int(starts.min()), int(ends.max())))
    return spans


def apply_redaction(pen: pd.DataFrame, markers: pd.DataFrame | None,
                    policy: RedactionPolicy) -> pd.DataFrame:
    if policy is RedactionPolicy.NONE:
        return pen
    out = pen.copy()
    present = [c for c in XY_COLUMNS if c in out.columns]
    if policy is RedactionPolicy.ALL_XY:
        out[present] = np.nan
        return out
    spans = _free_writing_spans(markers) if markers is not None and not markers.empty else []
    if not _has_task_structure(markers):
        # Why: an absent marker table and a marker table without task ids are the
        # same situation - we cannot tell free writing apart, so we do not guess.
        # The ETH sources emit phase events without task ids and land here.
        out[present] = np.nan
        return out
    mask = pd.Series(False, index=out.index)
    for lo, hi in spans:
        mask |= out[S.TIME_COLUMN].between(lo, hi)
    out.loc[mask, present] = np.nan
    return out


def _has_task_structure(markers: pd.DataFrame | None) -> bool:
    if markers is None or markers.empty or "task_id" not in markers.columns:
        return False
    return markers["task_id"].astype(str).str.strip().ne("").any()


def redact_bundle(bundle: RecordingBundle,
                  policy: RedactionPolicy = RedactionPolicy.NONE) -> RecordingBundle:
    """Apply `policy` to a bundle's pen table and record the choice in meta.

    Default parameter is NONE, not merely a documented convention - a caller
    that builds a bundle without deciding on a policy publishes unredacted
    data and an honest "none" in the manifest, never a silent redaction.
    """
    tables = dict(bundle.tables)
    if "pen" in tables:
        tables["pen"] = apply_redaction(tables["pen"], tables.get("markers"), policy)
    meta = dict(bundle.meta)
    meta["redaction_policy"] = str(policy)
    return RecordingBundle(bundle.ref, tables, meta)
