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
            row = self._session_row(sessions, sid)
            person = str(row["person_id"]).strip()
            # Why: a placeholder here would collapse several recordings onto one
            # participant, and the article's participant count would silently lie.
            if not person or person.lower() in {"nan", "none"}:
                raise ValueError(f"{sid} has no person_id in sessions.csv")
            refs.append(RecordingRef(f"{COHORT}-{sid}", f"{COHORT}-{person}", COHORT, self.name, root))
        return refs

    @staticmethod
    def _session_row(sessions: pd.DataFrame, sid: str) -> pd.Series:
        if sid not in sessions.index:
            raise ValueError(f"{sid} has no row in sessions.csv")
        return sessions.loc[sid]

    def load(self, ref: RecordingRef) -> RecordingBundle:
        sid = ref.recording_id.removeprefix(f"{COHORT}-")
        root = ref.path
        sessions = pd.read_csv(root / "sessions.csv").set_index("session_id")
        # Why: a stale or hand-built RecordingRef can point at a session no longer
        # in sessions.csv; without this guard `.loc[sid]` raises a raw KeyError.
        row = self._session_row(sessions, sid)

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
            # Why (C2): watch samples are timestamped from the watch's own
            # capture clock (`ts`, see _watch below); pen and markers come
            # from `local_ts_ms`/`timestamp_ms` - both the SERVER clock, a
            # different clock entirely. Declaring one flat time_domain for
            # the whole recording (the pre-fix bug) silently mislabels pen
            # and marker timestamps as if they shared the watch's clock.
            "time_domain_by_modality": {
                "watch": "watch_capture_clock",
                "pen": "server_wall_clock",
                "markers": "server_wall_clock",
            },
            # Why (item 1, fix round C): the modality defaults above are each
            # the PRIMARY t_ns axis's clock; these provenance columns are
            # provably not on it. src_local_ts_ms/src_server_received_ms are
            # both server-stamped (see _watch's comment on `ts` vs.
            # `local_ts_ms`), not the watch capture clock the rest of
            # `watch/` is on. src_phone_received_at is the iPhone bridge's
            # own clock - a third device, not the server. pen.src_timestamp
            # is the raw Moleskine device's own clock (see _pen below),
            # ~923 days off server_wall_clock on this corpus (finding 1).
            # src_t_session_ms is a session-relative offset, not a wall
            # clock reading, on both pen and markers. src_sequence needs NO
            # override: it holds no time, so under the field's rule it
            # declares the clock its row is stamped on, which is the default.
            "time_domain_by_column": {
                ("watch", "src_local_ts_ms"): "server_wall_clock",
                ("watch", "src_server_received_ms"): "server_wall_clock",
                ("watch", "src_phone_received_at"): S.TIME_DOMAIN_PHONE_WALL_CLOCK,
                ("pen", "src_timestamp"): S.TIME_DOMAIN_PEN_DEVICE_CLOCK,
                ("pen", "src_t_session_ms"): S.TIME_DOMAIN_SESSION_RELATIVE_OFFSET_MS,
                ("markers", "src_t_session_ms"): S.TIME_DOMAIN_SESSION_RELATIVE_OFFSET_MS,
            },
            "time_alignment": "estimated_delta",
            "protocol_id": f"ml4scs_{row.get('protocol_id')}",
            "study_mode": row.get("study_mode"),
            "subject_index": row.get("subject_index"),
            "pen_xy_unit": "ncode_grid",
            "pen_pressure_scale": "moleskine_raw",
            "session_start_ns": self._session_start_ns(row.get("start_time")),
            "pen_delta_sigma": self._pen_delta_sigma(row.get("alignment_sigma")),
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

    @staticmethod
    def _pen_delta_sigma(alignment_sigma: object) -> float:
        # Why: the offset delta itself is not in the export - only its confidence
        # score is. Baking in an estimated delta would risk irreparable labels.
        if alignment_sigma is None or pd.isna(alignment_sigma):
            return float("nan")
        return float(alignment_sigma)

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
            # Why (I15): float64 across every cohort - see the note in
            # sensorlogger._from_session_json. Millisecond magnitudes are exact
            # in float64, so nothing is lost by not publishing this as int64 in
            # the cohorts that happen to have it.
            "src_timestamp": raw["timestamp"].astype(float),
            # Why (I15): no session-relative offset exists for this cohort - its
            # recordings are not web-app sessions - but one modality carries one
            # column set, so the column is present and empty rather than absent.
            "src_t_session_ms": np.nan,
        })
        return sort_stable_by_time(out)

    def _markers(self, path: Path) -> pd.DataFrame:
        raw = pd.read_csv(path)
        out = raw.rename(columns={"timestamp_ms": "_ms"})
        out["t_ns"] = to_unix_ns(out.pop("_ms").to_numpy(), "ms")
        # Why (I15): float64 with NaN = "not applicable" is the encoding both
        # ETH adapters now also use for this column - an explicit cast makes
        # that a structural fact of this column rather than an accident of
        # whether a given CSV happens to contain a blank cell.
        out["task_index"] = out["task_index"].astype("float64")
        # Why (I15): this cohort's marker CSV carries no source payload and no
        # session-relative offset, but one modality must present one column set
        # - a reuser concatenating the cohorts otherwise gets a ragged frame,
        # and the package descriptor cannot state a single schema for markers.
        # Present and empty, not absent.
        out["src_payload"] = ""
        out["src_t_session_ms"] = np.nan
        cols = ["t_ns", "event", "task_id", "task_name", "task_index", "task_category",
                "protocol_id", "src_payload", "src_t_session_ms"]
        return sort_stable_by_time(out[cols])


register(Ml4scsAdapter())
