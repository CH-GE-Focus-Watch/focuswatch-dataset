"""Adapter for the ML4SCS capture pipeline (watch + Moleskine pen + study markers)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .. import schema as S
from ..time_axis import sort_stable_by_time, to_unix_ns
from .base import RecordingBundle, RecordingRef, register

COHORT = "ML4SCS"

_WATCH_MAP = {
    "ax": "accel_user_x", "ay": "accel_user_y", "az": "accel_user_z",
    "rx": "gyro_x", "ry": "gyro_y", "rz": "gyro_z",
    "gx": "gravity_x", "gy": "gravity_y", "gz": "gravity_z",
    "qx": "quat_x", "qy": "quat_y", "qz": "quat_z", "qw": "quat_w",
}
_KEEP_AS_SOURCE = ("local_ts_ms", "sequence", "server_received_ms", "phone_received_at")


class Ml4scsAdapter:
    name = "ml4scs"

    def discover(self, root: Path) -> list[RecordingRef]:
        sessions = pd.read_csv(root / "sessions.csv").set_index("session_id")
        refs = []
        for f in sorted((root / "watch").glob("*_watch.csv")):
            sid = f.name.removesuffix("_watch.csv")
            if sid not in sessions.index:
                raise ValueError(f"{sid} has no row in sessions.csv")
            person = str(sessions.loc[sid, "person_id"]).strip()
            # Why: a placeholder here would collapse several recordings onto one
            # participant, and the article's participant count would silently lie.
            if not person or person.lower() in {"nan", "none"}:
                raise ValueError(f"{sid} has no person_id in sessions.csv")
            refs.append(RecordingRef(f"{COHORT}-{sid}", f"{COHORT}-{person}", COHORT, self.name, root))
        return refs

    def load(self, ref: RecordingRef) -> RecordingBundle:
        sid = ref.recording_id.removeprefix(f"{COHORT}-")
        root = ref.path
        sessions = pd.read_csv(root / "sessions.csv").set_index("session_id")
        row = sessions.loc[sid]

        watch = self._watch(root / "watch" / f"{sid}_watch.csv")
        tables = {"watch": watch}

        pen_path = root / "pen" / f"{sid}_pen.csv"
        if pen_path.exists():
            tables["pen"] = self._pen(pen_path)
        marker_path = root / "markers" / f"{sid}_markers.csv"
        if marker_path.exists():
            tables["markers"] = self._markers(marker_path)

        meta = {
            "watch_hz_nominal": 50.0 if str(row.get("watch_profile")) == "50hz" else 100.0,
            "has_gravity": "gravity_x" in watch.columns,
            "has_quaternion": "quat_x" in watch.columns,
            "accel_semantics": "user",
            "accel_calibration": "fused",
            "gravity_source": "measured" if "gravity_x" in watch.columns else "none",
            "time_domain": "watch_capture_clock",
            "time_alignment": "estimated_delta",
            "protocol_id": f"ml4scs_{row.get('protocol_id')}",
            "study_mode": row.get("study_mode"),
            "subject_index": row.get("subject_index"),
            "pen_xy_unit": "ncode_grid",
            "pen_pressure_scale": "moleskine_raw",
            "session_start_ns": self._session_start_ns(row.get("start_time")),
        }
        return RecordingBundle(ref, tables, meta)

    @staticmethod
    def _session_start_ns(start_time: object) -> int | None:
        """Session start from the server clock, used only for the spill guard.

        It sits within NTP distance of the watch capture clock, well inside the
        60 s tolerance, so no conversion between the two is needed here.
        """
        if start_time is None or pd.isna(start_time):
            return None
        # utc=True accepts both the offset-carrying and the naive form found in
        # sessions.csv and returns nanoseconds since the epoch either way.
        return int(pd.to_datetime(start_time, utc=True).value)

    def _watch(self, path: Path) -> pd.DataFrame:
        raw = pd.read_csv(path)
        # Why: `ts` is the per-sample capture clock; `local_ts_ms` is batch arrival
        # and lags by minutes when the watch drains a spill buffer.
        out = pd.DataFrame({"t_ns": to_unix_ns(raw["ts"].to_numpy(), "ms")})
        for src, dst in _WATCH_MAP.items():
            if src in raw.columns and raw[src].notna().any():
                out[dst] = raw[src].astype(float)
        for c in _KEEP_AS_SOURCE:
            if c in raw.columns:
                out[f"src_{c}"] = raw[c]
        return sort_stable_by_time(out)

    def _pen(self, path: Path) -> pd.DataFrame:
        raw = pd.read_csv(path)
        out = pd.DataFrame({
            "t_ns": to_unix_ns(raw["local_ts_ms"].to_numpy(), "ms"),
            "dot_type": raw["dot_type"].astype(str),
            "x": raw["x"].astype(float), "y": raw["y"].astype(float),
            "pressure": raw["pressure"].astype(float),
            "tilt_x": raw["tilt_x"].astype(float), "tilt_y": raw["tilt_y"].astype(float),
            "src_timestamp": raw["timestamp"],
        })
        return sort_stable_by_time(out)

    def _markers(self, path: Path) -> pd.DataFrame:
        raw = pd.read_csv(path)
        out = raw.rename(columns={"timestamp_ms": "_ms"})
        out["t_ns"] = to_unix_ns(out.pop("_ms").to_numpy(), "ms")
        cols = ["t_ns", "event", "task_id", "task_name", "task_index", "task_category", "protocol_id"]
        return sort_stable_by_time(out[cols])


register(Ml4scsAdapter())
