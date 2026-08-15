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
        pen_markers = None
        pen = d / "pen_events.csv"
        if pen.exists():
            stroke_table, pen_markers = self._pen(pen)
            if stroke_table is not None:
                tables["pen"] = stroke_table
        events = d / "events.csv"
        session_markers = self._markers(events) if events.exists() else None
        # Why (C3/I8): pen_paper_info and pen_session_sync carry no position,
        # so neither belongs in pen/ - both route to markers/ instead,
        # matching what the SensorLogger adapter already does with
        # pen_session_sync (DESIGN §8.1). They come from a different source
        # file (pen_events.csv, not events.csv) so are merged here rather
        # than produced by _markers directly.
        combined = pd.concat([t for t in (session_markers, pen_markers) if t is not None],
                             ignore_index=True)
        if len(combined):
            tables["markers"] = sort_stable_by_time(combined)

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
            # Why: whether headimu carries a gyroscope is a DATA fact, not
            # structural - this source's head table never has gx/gy/gz (the
            # fixture confirms; head_motion_samples_rows.csv has no g*
            # columns), so this declares false and check_coverage requires
            # nothing of it, rather than exempting headimu wholesale.
            "has_head_gyro": "headimu" in tables and "gyro_x" in tables["headimu"].columns,
            "accel_semantics": "total",
            # Established by a still-window test on the real corpus: the norm sits at
            # 0.9954 (T6) / 0.9932 (T7), a persistent per-device bias. A recombination
            # of userAcceleration and gravity would sit at exactly 1.000.
            "accel_calibration": "raw_uncalibrated",
            "accel_still_bias": self._still_bias(watch),
            "gravity_source": "none",
            # Why (C2): the backend stamps every modality on the same wall
            # clock for this cohort (DESIGN §5.0's shared_clock regime), so
            # every table this adapter emits gets the identical domain -
            # declared per modality anyway, so a future modality with a
            # genuinely different clock cannot inherit this one by omission.
            "time_domain_by_modality": {
                "watch": "backend_wall_clock", "headimu": "backend_wall_clock",
                "pen": "backend_wall_clock", "markers": "backend_wall_clock",
            },
            "time_alignment": "shared_clock",
            "protocol_id": "eth_web",
            "study_mode": "study",
            # Why (C3): Ege is generation A (see schema.py's PEN_EVENTS_GEN_A);
            # the label follows the export generation, not this pipeline
            # directory - SensorLogger's own generation-A recordings
            # (S3/T8/T9/T10) get the identical value.
            "pen_xy_unit": S.PEN_XY_UNIT_GEN_A,
            "pen_pressure_scale": S.PEN_PRESSURE_SCALE_GEN_A,
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

    def _pen(self, path: Path) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
        """Strokes -> pen/, non-stroke events (pen_paper_info/pen_session_sync) -> markers/.

        Both live in this same CSV (C3/I8) - split here rather than have two
        methods each re-read the file.
        """
        raw = pd.read_csv(path)
        strokes = raw[~raw["type"].isin(S.PEN_NON_STROKE_EVENTS)]
        pen = None
        if len(strokes):
            pen = pd.DataFrame({
                "t_ns": to_unix_ns(strokes["t_ms"].to_numpy(), "ms"),
                "dot_type": strokes["type"].map(S.PEN_EVENTS_GEN_A).to_numpy(),
                "x": strokes["x"].astype(float).to_numpy(), "y": strokes["y"].astype(float).to_numpy(),
                "pressure": strokes["force"].astype(float).to_numpy(),
                "src_t_session_ms": strokes["t_session_ms"].to_numpy(),
            })
            if pen["dot_type"].isna().any():
                raise ValueError(f"unmapped pen event types in {path}")
            pen = sort_stable_by_time(pen)

        non_stroke = raw[raw["type"].isin(S.PEN_NON_STROKE_EVENTS)]
        markers = None
        if len(non_stroke):
            markers = sort_stable_by_time(pd.DataFrame({
                "t_ns": to_unix_ns(non_stroke["t_ms"].to_numpy(), "ms"),
                "event": non_stroke["type"].astype(str),
                "task_id": "", "task_name": "", "task_index": np.nan,
                "task_category": "", "protocol_id": "eth_web",
                "src_t_session_ms": non_stroke["t_session_ms"].to_numpy(),
            }))
        return pen, markers

    def _markers(self, path: Path) -> pd.DataFrame:
        raw = pd.read_csv(path)
        out = pd.DataFrame({
            "t_ns": to_unix_ns(raw["t_ms"].to_numpy(), "ms"),
            "event": raw["event_type"].astype(str),
            # Why (I15): NaN, not -1 - one encoding for "not applicable" across
            # cohorts. -1 looks like a valid index to a reuser filtering
            # `task_index >= 0`; NaN cannot be mistaken for one.
            "task_id": "", "task_name": "", "task_index": np.nan,
            "task_category": "", "protocol_id": "eth_web",
            "src_payload": raw.get("payload", ""),
            # Why: t_session_ms is a fixed column of events.csv in the documented
            # source schema; its absence means the format changed and that must
            # fail loudly here, not degrade to a silent None in the output.
            "src_t_session_ms": raw["t_session_ms"],
        })
        return sort_stable_by_time(out)


register(EgeAdapter())
