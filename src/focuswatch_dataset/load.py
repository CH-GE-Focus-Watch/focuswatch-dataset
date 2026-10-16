"""Reader API for published bundles with user-facing missing-table errors."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import schema as S
from .write import read_table

# Why: native rates stay native (no resampling), so concatenating a 50 Hz and
# a 100 Hz recording silently doubles every sample-count window in one of
# them. Measured rates of one nominal stream jitter, though - the AirPods head
# IMU spans 27.0-28.6 Hz in the 2026-08-24 bundle - so only a max/min spread
# above this ratio counts as mixed.
MAX_RATE_SPREAD = 1.10

_PEN_UNIT_FIELDS = ("pen_xy_unit", "pen_pressure_scale")


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


def _sample_rates(root: Path, recording_ids: list[str], modality: str) -> pd.Series | None:
    """Per-recording measured rate of one motion modality, from `channels.parquet`."""
    path = root / "channels.parquet"
    if not path.exists():
        # Why: every built bundle ships channels.parquet (build writes it and
        # validate requires it), so only hand-built fixtures that write a
        # manifest plus modality tables land here; they declare no rate to check.
        return None
    channels = pd.read_parquet(path, columns=["recording_id", "modality", "sample_rate_hz"])
    rows = channels.loc[(channels["modality"] == modality)
                        & channels["recording_id"].isin(recording_ids)]
    return rows.groupby("recording_id")["sample_rate_hz"].first().reindex(recording_ids)


def _group_ids(values: pd.Series) -> dict[object, list[str]]:
    # Why: naming ids per value, not just the value set, so a stranger can
    # spot the offender directly instead of re-deriving the grouping from a
    # selection that may span dozens of ids.
    return {value: sorted(ids.index) for value, ids in values.groupby(values)}


def load_recordings(root: Path | str, recording_ids: list[str],
                    modality: str = "watch", *,
                    allow_mixed_rates: bool = False,
                    allow_mixed_pen_units: bool = False) -> pd.DataFrame:
    """Concatenate several recordings' tables, tagging each row with `recording_id`.

    Refuses to mix `accel_semantics` when `modality == "watch"`: acceleration
    with gravity in it (`total`) and acceleration with gravity removed
    (`user`) can carry near-identical values after per-session normalisation,
    so a silent concatenation would hide a real difference in what the
    numbers mean. Call `convert.to_user_acceleration` first, or restrict the
    selection to one semantics. The guard is scoped to `watch` because
    `accel_semantics` documents that stream specifically - it would misfire on, say, two
    recordings' `pen` tables that happen to disagree about watch semantics
    neither table carries.

    Also refuses, for the motion modalities, recordings whose sample rates
    differ by more than `MAX_RATE_SPREAD`, and, for `pen`, recordings whose
    `pen_xy_unit` or `pen_pressure_scale` differ. Both are legitimate inputs
    for per-recording processing, so each guard has an opt-in:
    `allow_mixed_rates=True` adds a `sample_rate_hz` column and
    `allow_mixed_pen_units=True` adds the two unit columns, so the
    information travels with every row instead of being lost in the concat.
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
    # Why: unique ids, so the per-recording lookups below stay scalar even when
    # a caller lists one recording twice.
    unique_ids = list(dict.fromkeys(recording_ids))
    by_id = manifest.set_index("recording_id").loc[unique_ids]
    if modality == "watch" and "accel_semantics" in by_id.columns:
        semantics = by_id["accel_semantics"]
        semantics = semantics[semantics.notna() & (semantics != "")]
        if semantics.nunique() > 1:
            raise ValueError(
                f"mixed acceleration semantics {_group_ids(semantics)}; "
                "convert with to_user_acceleration() or restrict the selection"
            )

    rates = _sample_rates(root, unique_ids, modality) if modality in S.MOTION_MODALITIES else None
    if rates is not None and not allow_mixed_rates:
        measured = rates.dropna()
        if len(measured) > 1 and measured.max() / measured.min() > MAX_RATE_SPREAD:
            raise ValueError(
                f"mixed {modality} sample rates {_group_ids(measured.round())}; "
                "restrict the selection to one rate (e.g. watch_hz_nominal), or pass "
                "allow_mixed_rates=True to keep a per-row sample_rate_hz column"
            )

    pen_units = by_id[[f for f in _PEN_UNIT_FIELDS if f in by_id.columns]] if modality == "pen" else None
    if pen_units is not None and not allow_mixed_pen_units:
        for field in pen_units.columns:
            values = pen_units[field].dropna()
            if values.nunique() > 1:
                raise ValueError(
                    f"mixed {field} {_group_ids(values)}; pen coordinates and pressure "
                    "are device-native, so restrict the selection to one unit, or pass "
                    "allow_mixed_pen_units=True to keep the unit columns on every row"
                )

    frames = []
    for rid in recording_ids:
        df = load_recording(root, rid, modality)
        df.insert(0, "recording_id", rid)
        if allow_mixed_rates and rates is not None:
            df.insert(1, "sample_rate_hz", rates[rid])
        if allow_mixed_pen_units and pen_units is not None:
            for position, field in enumerate(pen_units.columns, start=1):
                df.insert(position, field, pen_units.at[rid, field])
        frames.append(df)
    return pd.concat(frames, ignore_index=True)
