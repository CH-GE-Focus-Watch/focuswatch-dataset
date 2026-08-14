"""Adapter for the custom watchOS pipeline that streams to a Supabase backend.

Two source quirks drive this module: acceleration includes gravity, and the
columns named g* hold the gyroscope. Both are asserted, not assumed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .. import schema as S
from ..physics import norm_stats, still_mask
from ..time_axis import median_rate_hz, sort_stable_by_time, to_unix_ns
from .base import RecordingBundle, RecordingRef, register

COHORT = "ETH-EGE"

_DOT_TYPES = {"pen_down": "PEN_DOWN", "pen_dot": "PEN_MOVE", "pen_up": "PEN_UP"}
_NON_STROKE_EVENTS = ("pen_paper_info", "pen_session_sync")


def _assert_gyroscope_like(values: np.ndarray) -> None:
    st = norm_stats(values)
    # Why: gravity would sit at 1.000 with near-zero spread; a gyro does not.
    if 0.99 <= st.median <= 1.01 and st.iqr < 0.01:
        raise ValueError(
            f"g* columns are not gyroscope-like (norm median {st.median:.4f}, "
            f"IQR {st.iqr:.4f}) - they look like a gravity unit vector"
        )


class EgeAdapter:
    name = "ege"

    def discover(self, root: Path) -> list[RecordingRef]:
        refs = []
        for d in sorted(p for p in root.iterdir() if (p / "imu_samples_rows.csv").exists()):
            refs.append(RecordingRef(f"{COHORT}-{d.name}", f"ETH-{d.name}", "ETH", self.name, d))
        return refs

    def load(self, ref: RecordingRef) -> RecordingBundle:
        d = ref.path
        watch = self._motion(d / "imu_samples_rows.csv")
        tables = {"watch": watch}
        head = d / "head_motion_samples_rows.csv"
        if head.exists():
            tables["headimu"] = self._motion(head)
        pen = d / "pen_events.csv"
        if pen.exists():
            tables["pen"] = self._pen(pen)
        events = d / "events.csv"
        if events.exists():
            tables["markers"] = self._markers(events)

        meta = {
            # Why: this source states no nominal wrist rate anywhere (no rate
            # column in imu_samples_rows.csv or sensor_session.csv); declaring
            # one anyway would fabricate a fact the source doesn't carry. The
            # validator falls back to the measured rate.
            "watch_hz_nominal": None,
            # Why: derived from the emitted table, not asserted independently -
            # a hardcoded flag and the actual columns can drift out of sync.
            "has_gravity": "gravity_x" in watch.columns,
            "has_quaternion": "quat_x" in watch.columns,
            "accel_semantics": "total",
            # Established by a still-window test on the real corpus: the norm sits at
            # 0.9954 (T6) / 0.9932 (T7), a persistent per-device bias. A recombination
            # of userAcceleration and gravity would sit at exactly 1.000.
            "accel_calibration": "raw_uncalibrated",
            "accel_still_bias": self._still_bias(watch),
            "gravity_source": "none",
            "time_domain": "backend_wall_clock",
            "time_alignment": "shared_clock",
            "protocol_id": "eth_web",
            "study_mode": "study",
            "pen_xy_unit": "webapp_raw",
            "pen_pressure_scale": "webapp_force",
            "session_start_ns": self._session_start_ns(d / "sensor_session.csv"),
        }
        return RecordingBundle(ref, tables, meta)

    @staticmethod
    def _session_start_ns(path: Path) -> int | None:
        if not path.exists():
            return None
        started = pd.read_csv(path)["started_at_ms"].iloc[0]
        return int(started) * 1_000_000 if pd.notna(started) else None

    @staticmethod
    def _still_bias(watch: pd.DataFrame) -> float:
        """Mean deviation of the acceleration norm from 1 g while the wrist is still.

        Reported, never corrected. It is the evidence for accel_calibration and it
        differs per session, so folding it in would bake a guess into the data.
        """
        cols = list(S.COLUMNS[S.Quantity.GYRO])
        if not set(cols) <= set(watch.columns):
            return float("nan")
        fs = median_rate_hz(watch["t_ns"].to_numpy(dtype=np.int64))
        mask = still_mask(watch[cols].to_numpy(dtype=float), fs_hz=fs if np.isfinite(fs) else 100.0)
        if mask.sum() < 100:
            return float("nan")
        norms = np.linalg.norm(
            watch.loc[mask, list(S.COLUMNS[S.Quantity.ACCEL_TOTAL])].to_numpy(dtype=float), axis=1)
        return round(float(norms.mean() - 1.0), 6)

    def _motion(self, path: Path) -> pd.DataFrame:
        raw = pd.read_csv(path)
        out = pd.DataFrame({"t_ns": to_unix_ns(raw["t_ms"].to_numpy(), "ms")})
        out[list(S.COLUMNS[S.Quantity.ACCEL_TOTAL])] = raw[["ax", "ay", "az"]].astype(float).to_numpy()
        if {"gx", "gy", "gz"} <= set(raw.columns):
            gyro = raw[["gx", "gy", "gz"]].astype(float).to_numpy()
            _assert_gyroscope_like(gyro)
            out[list(S.COLUMNS[S.Quantity.GYRO])] = gyro
        if {"qx", "qy", "qz", "qw"} <= set(raw.columns):
            out[list(S.COLUMNS[S.Quantity.QUAT])] = raw[["qx", "qy", "qz", "qw"]].astype(float).to_numpy()
        if "t_session_ms" in raw.columns:
            out["src_t_session_ms"] = raw["t_session_ms"]
        return sort_stable_by_time(out)

    def _pen(self, path: Path) -> pd.DataFrame:
        raw = pd.read_csv(path)
        raw = raw[~raw["type"].isin(_NON_STROKE_EVENTS)]
        out = pd.DataFrame({
            "t_ns": to_unix_ns(raw["t_ms"].to_numpy(), "ms"),
            "dot_type": raw["type"].map(_DOT_TYPES).to_numpy(),
            "x": raw["x"].astype(float).to_numpy(), "y": raw["y"].astype(float).to_numpy(),
            "pressure": raw["force"].astype(float).to_numpy(),
            "src_t_session_ms": raw["t_session_ms"].to_numpy(),
        })
        if out["dot_type"].isna().any():
            raise ValueError(f"unmapped pen event types in {path}")
        return sort_stable_by_time(out)

    def _markers(self, path: Path) -> pd.DataFrame:
        raw = pd.read_csv(path)
        out = pd.DataFrame({
            "t_ns": to_unix_ns(raw["t_ms"].to_numpy(), "ms"),
            "event": raw["event_type"].astype(str),
            "task_id": "", "task_name": "", "task_index": -1,
            "task_category": "", "protocol_id": "eth_web",
            "src_payload": raw.get("payload", ""),
            # Why: t_session_ms is a fixed column of events.csv in the documented
            # source schema; its absence means the format changed and that must
            # fail loudly here, not degrade to a silent None in the output.
            "src_t_session_ms": raw["t_session_ms"],
        })
        return sort_stable_by_time(out)


register(EgeAdapter())
