"""Adapter for Apple Watch accelerometer CSVs paired with a Moleskine DB backup.

Each watch file defines one recording window. Pen points are assigned only
when their SQLite wall-clock timestamps fall inside that window, and exactly
one notebook page must overlap. The two devices were not explicitly clock
synchronised, so the published alignment is deliberately ``overlap_only``.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from .. import schema as S
from ..time_axis import parse_iso_to_unix_ns, sort_stable_by_time, to_unix_ns
from .base import RecordingBundle, RecordingRef, register

COHORT = "MOLESKINE"

_TIME = "loggingTime(txt)"
_MONOTONIC = "accelerometerTimestamp_sinceReboot(s)"
_ACCEL = (
    "accelerometerAccelerationX(G)",
    "accelerometerAccelerationY(G)",
    "accelerometerAccelerationZ(G)",
)
_DRAW_TYPES = {17: "PEN_DOWN", 18: "PEN_MOVE", 20: "PEN_UP"}


def _database_path(watch_path: Path) -> Path:
    return watch_path.parent.parent / "moleskine" / "backup-db" / "note.db"


def _recording_id(t_ns: int) -> str:
    timestamp = pd.Timestamp(t_ns, unit="ns", tz="UTC")
    return f"{COHORT}-{timestamp.strftime('%Y%m%dT%H%M%S')}{timestamp.microsecond // 1000:03d}Z"


class MoleskineAdapter:
    name = "moleskine"

    def discover(self, root: Path) -> list[RecordingRef]:
        watch_files = sorted((root / "sensorlog").glob("*.csv"))
        if not watch_files:
            return []
        database = root / "moleskine" / "backup-db" / "note.db"
        if not database.is_file():
            raise FileNotFoundError(f"Moleskine database not found: {database}")

        refs = []
        for path in watch_files:
            first = pd.read_csv(path, usecols=[_TIME], nrows=1)
            if first.empty:
                raise ValueError(f"{path}: watch CSV is empty")
            t_ns = int(parse_iso_to_unix_ns(first[_TIME])[0])
            # Participant attribution is absent for five files and filenames
            # contain informal labels for three others. Publishing those labels
            # would leak source names and invent identity links, so every row is
            # explicitly unattributed rather than guessed.
            refs.append(RecordingRef(
                _recording_id(t_ns), f"{COHORT}-UNKNOWN", COHORT, self.name, path,
            ))
        return refs

    def load(self, ref: RecordingRef) -> RecordingBundle:
        watch = self._watch(ref.path)
        pen = self._pen(
            _database_path(ref.path), int(watch["t_ns"].iloc[0]), int(watch["t_ns"].iloc[-1]),
        )
        meta = {
            "watch_hz_nominal": 50.0,
            "accel_semantics": "total",
            "accel_calibration": "raw",
            "gravity_source": "none",
            "time_domain_by_modality": {
                "watch": "watch_device_wall_clock",
                "pen": "moleskine_app_wall_clock",
            },
            "time_domain_by_column": {
                ("watch", "src_accelerometer_timestamp_s"):
                    S.TIME_DOMAIN_DEVICE_MONOTONIC_CLOCK,
                ("pen", "src_timestamp"): S.TIME_DOMAIN_PEN_DEVICE_CLOCK,
                ("pen", "src_t_session_ms"): S.TIME_DOMAIN_SESSION_RELATIVE_OFFSET_MS,
            },
            "time_alignment": S.TIME_ALIGNMENT_OVERLAP_ONLY,
            "pen_xy_unit": "ncode_grid",
            "pen_pressure_scale": "moleskine_raw",
            "session_start_ns": int(watch["t_ns"].iloc[0]),
        }
        return RecordingBundle(ref, {"watch": watch, "pen": pen}, meta)

    @staticmethod
    def _watch(path: Path) -> pd.DataFrame:
        raw = pd.read_csv(path)
        required = {_TIME, _MONOTONIC, *_ACCEL}
        missing = sorted(required - set(raw.columns))
        if missing:
            raise ValueError(f"{path}: missing required watch column(s) {missing}")
        if raw.empty:
            raise ValueError(f"{path}: watch CSV is empty")

        out = pd.DataFrame({
            "t_ns": parse_iso_to_unix_ns(raw[_TIME]),
            "src_accelerometer_timestamp_s": raw[_MONOTONIC].astype(float),
        })
        out[list(S.COLUMNS[S.Quantity.ACCEL_TOTAL])] = raw[list(_ACCEL)].astype(float).to_numpy()
        return sort_stable_by_time(out)

    @staticmethod
    def _pen(database: Path, start_ns: int, end_ns: int) -> pd.DataFrame:
        # SQLite values are integer milliseconds. Rounding inward prevents a
        # point just outside the nanosecond recording window from being pulled
        # in by a lossy boundary conversion.
        start_ms = (start_ns + 999_999) // 1_000_000
        end_ms = end_ns // 1_000_000
        uri = f"{database.resolve().as_uri()}?mode=ro"
        with sqlite3.connect(uri, uri=True) as connection:
            raw = pd.read_sql_query(
                """
                SELECT d.date, d.drawType, d.pageX, d.pageY, d.penPressure, s.pageID
                FROM draw_point AS d
                JOIN Stroke AS s ON s.id = d.stokeID
                WHERE d.date BETWEEN ? AND ?
                ORDER BY d.date, d.id
                """,
                connection,
                params=(int(start_ms), int(end_ms)),
            )
        if raw.empty:
            raise ValueError(
                f"{database}: no pen points overlap watch range [{start_ns}, {end_ns}]"
            )
        pages = sorted(raw["pageID"].unique().tolist())
        if len(pages) != 1:
            raise ValueError(
                f"{database}: watch range overlaps {len(pages)} notebook pages {pages}; "
                "recording membership is ambiguous"
            )
        unknown = sorted(set(raw["drawType"]) - set(_DRAW_TYPES))
        if unknown:
            raise ValueError(f"{database}: unknown Moleskine drawType value(s) {unknown}")

        out = pd.DataFrame({
            "t_ns": to_unix_ns(raw["date"].to_numpy(dtype=np.int64), "ms"),
            "dot_type": raw["drawType"].map(_DRAW_TYPES),
            "x": raw["pageX"].astype(float),
            "y": raw["pageY"].astype(float),
            "pressure": raw["penPressure"].astype(float),
            # This backup contains no tilt channel. Keep the canonical pen
            # columns and represent the unavailable measurements honestly.
            "tilt_x": np.nan,
            "tilt_y": np.nan,
            # No distinct pen-device or session-relative timestamp is exposed
            # by this backup. Nulls preserve the cross-cohort pen schema
            # without duplicating the canonical wall-clock t_ns value.
            "src_timestamp": np.nan,
            "src_t_session_ms": np.nan,
        })
        return sort_stable_by_time(out)


register(MoleskineAdapter())
