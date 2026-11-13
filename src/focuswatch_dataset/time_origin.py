"""Optional rebasing of every absolute timestamp onto the recording start.

A Unix timestamp tells anyone the day and minute a participant was recorded.
Under ``recording_start`` the same origin is subtracted from every absolute
clock value of a recording, so the earliest sample sits at 0 while every
difference - between samples, between modalities, between clocks - survives.
The bundle's manifest always records which origin was used.
"""
from __future__ import annotations

import json
from enum import StrEnum

import numpy as np
import pandas as pd

from . import schema as S
from .adapters.base import RecordingBundle

NS_COLUMNS = (S.TIME_COLUMN, "t_start_ns", "t_end_ns")
# src_ provenance columns that hold Unix milliseconds on their source's clock.
MS_COLUMNS = ("src_timestamp", "src_local_ts_ms", "src_phone_received_at", "src_server_received_ms")
# Marker payload fields that hold Unix milliseconds on the backend clock.
MS_PAYLOAD_KEYS = ("time_origin_ms", "pen_connected_t_ms", "session_start_t_ms")
PAYLOAD_COLUMN = "src_payload"

# Unix time between 2001 and 2033, in milliseconds, microseconds and
# nanoseconds. Seconds are left out on purpose: an uptime counter in
# milliseconds reaches 1e9 after 12 days and would read as a date.
_EPOCH_RANGES = ((1e12, 2e12), (1e15, 2e15), (1e18, 2e18))
# A nanosecond column is only checked against the nanosecond range: 17 minutes
# after the recording start it reaches 1e12, which would read as Unix ms.
_EPOCH_NS_RANGE = ((1e18, 2e18),)


class TimeOrigin(StrEnum):
    UNIX_EPOCH = "unix_epoch"
    RECORDING_START = "recording_start"


def _coerce_origin(origin: TimeOrigin | str) -> TimeOrigin:
    try:
        return TimeOrigin(origin)
    except ValueError as exc:
        raise ValueError(f"unknown time origin {origin!r}") from exc


def recording_span(tables: dict[str, pd.DataFrame]) -> tuple[int, int]:
    starts, ends = [], []
    for df in tables.values():
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


def _shift_ms(value, origin_ns: int):
    # Integers stay integers: the origin is floored to whole milliseconds.
    if isinstance(value, (int, np.integer)):
        return value - origin_ns // 1_000_000
    return value - origin_ns / 1e6


def _shift_table(df: pd.DataFrame, origin_ns: int) -> pd.DataFrame:
    out = df.copy()
    for column in NS_COLUMNS:
        if column in out.columns:
            out[column] = out[column] - origin_ns
    for column in MS_COLUMNS:
        if column in out.columns:
            if pd.api.types.is_integer_dtype(out[column]):
                out[column] = out[column] - origin_ns // 1_000_000
            else:
                out[column] = out[column] - origin_ns / 1e6
    return out


def _shift_payload(payload: str, origin_ns: int) -> str:
    if not payload:
        return payload
    fields = json.loads(payload)
    if not isinstance(fields, dict):
        raise ValueError(f"marker payload is not a JSON object: {payload[:60]!r}")
    # Why: only values that are a Unix time. Some exports already log these
    # fields relative to the session, and shifting those would invent a
    # timestamp decades before the recording.
    shifted = {k: _shift_ms(v, origin_ns) if k in MS_PAYLOAD_KEYS and _is_unix_ms(v) else v
               for k, v in fields.items()}
    return payload if shifted == fields else json.dumps(shifted)


def _is_unix_ms(value) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and _is_epoch(np.asarray([value], dtype=float)))


def rebase_bundle(bundle: RecordingBundle,
                  origin: TimeOrigin | str = TimeOrigin.UNIX_EPOCH) -> RecordingBundle:
    """Apply `origin` to every absolute timestamp of a bundle and record it in meta.

    The default is UNIX_EPOCH, which changes nothing - the same honest default
    as redact_bundle: a caller that never chose an origin publishes the source
    clocks and says so in the manifest.
    """
    origin = _coerce_origin(origin)
    tables = dict(bundle.tables)
    meta = dict(bundle.meta)
    meta["time_origin"] = origin.value
    if origin == TimeOrigin.UNIX_EPOCH:
        return RecordingBundle(bundle.ref, tables, meta)

    origin_ns, _ = recording_span(tables)
    tables = {modality: _shift_table(df, origin_ns) for modality, df in tables.items()}
    markers = tables.get("markers")
    if markers is not None and PAYLOAD_COLUMN in markers.columns:
        markers[PAYLOAD_COLUMN] = markers[PAYLOAD_COLUMN].map(
            lambda payload: _shift_payload(payload, origin_ns))
    # Why: validate_recording compares it against the table spans, so it must
    # move with them or the spill guard would read every sample as decades early.
    if meta.get("session_start_ns") is not None:
        meta["session_start_ns"] = int(meta["session_start_ns"]) - origin_ns
    return RecordingBundle(bundle.ref, tables, meta)


def _is_epoch(values: np.ndarray, ranges=_EPOCH_RANGES) -> bool:
    magnitude = np.abs(values[np.isfinite(values)])
    return any(((magnitude >= lo) & (magnitude < hi)).any() for lo, hi in ranges)


def _payload_numbers(value) -> list[float]:
    if isinstance(value, dict):
        return [n for v in value.values() for n in _payload_numbers(v)]
    if isinstance(value, list):
        return [n for v in value for n in _payload_numbers(v)]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return [float(value)]
    return []


def epoch_columns(df: pd.DataFrame) -> list[str]:
    """Columns of `df`, payload fields included, that still hold a Unix timestamp."""
    found = []
    for column in df.columns:
        if column == PAYLOAD_COLUMN:
            numbers = [n for payload in df[column] if payload
                       for n in _payload_numbers(json.loads(payload))]
            if numbers and _is_epoch(np.asarray(numbers, dtype=float)):
                found.append(column)
        elif pd.api.types.is_numeric_dtype(df[column]) and not pd.api.types.is_bool_dtype(df[column]):
            ranges = _EPOCH_NS_RANGE if column in NS_COLUMNS else _EPOCH_RANGES
            if _is_epoch(df[column].to_numpy(dtype=float), ranges):
                found.append(column)
    return found
