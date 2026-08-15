"""Reader API for a published bundle.

Every other module in this package serves the people who built the dataset;
this one serves the people using it - a reuser who downloads the archive and
has never seen `docs/DESIGN.md`. Errors here name the fact and where to look
it up (`recording X has no watch table (has_watch is false)`), not a bare
`FileNotFoundError` on a path nobody outside this repo would recognise.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import schema as S
from .write import read_table


def load_manifest(root: Path | str) -> pd.DataFrame:
    """The one-row-per-recording capability table (`sessions.parquet`)."""
    return pd.read_parquet(Path(root) / "sessions.parquet")


def load_channels(root: Path | str) -> pd.DataFrame:
    """The one-row-per-(recording, modality, column) channel description."""
    return pd.read_parquet(Path(root) / "channels.parquet")


def _missing_table_reason(root: Path, recording_id: str, modality: str) -> str:
    # Why: best-effort only. A missing table is diagnosed with the manifest
    # when one is reachable; a bundle with no manifest (e.g. a hand-built
    # fixture that writes only modality tables) still gets a plain message
    # rather than an unrelated crash while trying to explain the first one.
    try:
        manifest = load_manifest(root)
    except Exception:
        # Why: broad on purpose. This path only ever runs after a table is
        # already known to be missing, purely to add a better reason to that
        # failure - a truncated or corrupted sessions.parquet from a partial
        # download raises pyarrow.lib.ArrowInvalid (not an OSError subclass),
        # and any such secondary failure here must lose to the message it was
        # trying to improve, not replace it with an unrelated traceback.
        return ""
    rows = manifest.loc[manifest["recording_id"] == recording_id]
    if rows.empty:
        return f" ('{recording_id}' is not a recording_id in {root / 'sessions.parquet'})"
    flag = f"has_{modality}"
    if flag in rows.columns and not bool(rows.iloc[0][flag]):
        return f" ({flag} is false)"
    return ""


def load_recording(root: Path | str, recording_id: str, modality: str = "watch") -> pd.DataFrame:
    """The single motion/event table for one recording and modality."""
    root = Path(root)
    if modality not in S.MODALITIES:
        raise ValueError(f"unknown modality {modality!r}; expected one of {S.MODALITIES}")
    path = root / modality / f"{recording_id}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"recording {recording_id!r} has no {modality} table"
            f"{_missing_table_reason(root, recording_id, modality)}"
        )
    return read_table(path)


def load_recordings(root: Path | str, recording_ids: list[str],
                    modality: str = "watch") -> pd.DataFrame:
    """Concatenate several recordings' tables, tagging each row with `recording_id`.

    Refuses to mix `accel_semantics` when `modality == "watch"`: acceleration
    with gravity in it (`total`) and acceleration with gravity removed
    (`user`) can carry near-identical values after per-session normalisation,
    so a silent concatenation would hide a real difference in what the
    numbers mean. Call `convert.to_user_acceleration` first, or restrict the
    selection to one semantics. The guard is scoped to `watch` because
    `accel_semantics` documents that stream specifically (see
    `docs/DESIGN.md`'s manifest section) - it would misfire on, say, two
    recordings' `pen` tables that happen to disagree about watch semantics
    neither table carries.
    """
    root = Path(root)
    if not recording_ids:
        # Why: the natural path, not an abuse - a by_flags()/query() filter
        # that matched nothing pipes an empty recording_id column straight in
        # here. pd.concat([]) would otherwise raise its own bare "No objects
        # to concatenate", which points at neither the empty input nor the
        # corpus, and reads as this function's bug rather than the caller's
        # filter.
        raise ValueError(
            f"recording_ids is empty; nothing to load from {root / 'sessions.parquet'} "
            "- check the filter that produced this selection"
        )
    manifest = load_manifest(root)
    known = set(manifest["recording_id"])
    missing = [rid for rid in recording_ids if rid not in known]
    if missing:
        raise KeyError(
            f"recording id(s) {missing} not found in {root / 'sessions.parquet'}; "
            "see load_manifest(root)['recording_id'] for the ids that exist"
        )
    if modality == "watch" and "accel_semantics" in manifest.columns:
        by_id = manifest.set_index("recording_id")["accel_semantics"]
        present = {rid: s for rid, s in by_id.loc[recording_ids].items() if pd.notna(s) and s != ""}
        semantics = set(present.values())
        if len(semantics) > 1:
            # Why: naming ids per semantics value, not just the value set, so a
            # stranger can spot the offender directly instead of re-deriving
            # the grouping from a selection that may span dozens of ids.
            grouped = {sem: sorted(rid for rid, s in present.items() if s == sem)
                      for sem in sorted(semantics)}
            raise ValueError(
                f"mixed acceleration semantics {grouped}; "
                "convert with to_user_acceleration() or restrict the selection"
            )
    frames = []
    for rid in recording_ids:
        df = load_recording(root, rid, modality)
        df.insert(0, "recording_id", rid)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)
