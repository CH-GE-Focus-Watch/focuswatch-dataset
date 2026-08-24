"""Adapter for the custom watchOS pipeline that streams to a Supabase backend.

Two source quirks drive this module: acceleration includes gravity, and the
columns named g* hold the gyroscope. Both are asserted, not assumed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .. import schema as S
from ..physics import norm_stats, still_mask
from ..time_axis import median_rate_hz, sort_stable_by_time, to_unix_ns
from .base import RecordingBundle, RecordingRef, register
from .eth_web import handedness_from_payload, redact_payload_json

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
        session_markers, dropped_keys, handedness = (
            self._markers(events) if events.exists() else (None, 0, "unknown"))
        # pen_paper_info and pen_session_sync carry no position,
        # so neither belongs in pen/ - both route to markers/ instead,
        # matching what the SensorLogger adapter already does with
        # pen_session_sync. They come from a different source
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
            "accel_semantics": "total",
            # Shared session_start payload keeps the ETH covariate consistent.
            "handedness": handedness,
            "src_payload_dropped_key_count": dropped_keys,
            # Established by a still-window test on the real corpus: the norm sits at
            # 0.9954 (T6) / 0.9932 (T7), a persistent per-device bias. A recombination
            # of userAcceleration and gravity would sit at exactly 1.000.
            "accel_calibration": "raw_uncalibrated",
            "accel_still_bias": self._still_bias(watch),
            "gravity_source": "none",
            # The backend stamps every modality on the same wall
            # clock for this cohort, so
            # every table this adapter emits gets the identical domain -
            # declared per modality anyway, so a future modality with a
            # genuinely different clock cannot inherit this one by omission.
            "time_domain_by_modality": {
                "watch": "backend_wall_clock", "headimu": "backend_wall_clock",
                "pen": "backend_wall_clock", "markers": "backend_wall_clock",
            },
            # src_t_session_ms (wherever it
            # appears - pen_events.csv and events.csv both carry it) is a
            # session-relative millisecond offset, not a wall-clock reading -
            # no clock name would be honest for it. pen.src_timestamp is
            # structurally present but always NaN for this generation-A
            # source (no pen-device clock in the export); the
            # domain it WOULD be on, were it ever populated, is the same
            # physical thing SensorLogger's generation-B src_timestamp is -
            # declared for schema consistency across the two pen adapters,
            # not because Ege ever measures it.
            "time_domain_by_column": {
                ("watch", "src_t_session_ms"): S.TIME_DOMAIN_SESSION_RELATIVE_OFFSET_MS,
                ("headimu", "src_t_session_ms"): S.TIME_DOMAIN_SESSION_RELATIVE_OFFSET_MS,
                ("pen", "src_t_session_ms"): S.TIME_DOMAIN_SESSION_RELATIVE_OFFSET_MS,
                ("pen", "src_timestamp"): S.TIME_DOMAIN_PEN_DEVICE_CLOCK,
                ("markers", "src_t_session_ms"): S.TIME_DOMAIN_SESSION_RELATIVE_OFFSET_MS,
            },
            "time_alignment": "shared_clock",
            "protocol_id": "eth_web",
            "study_mode": "study",
            # Ege is generation A (see schema.py's PEN_EVENTS_GEN_A);
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

        Both live in this CSV; split them here rather than have two
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
                # Generation A has no tilt or pen-device clock; NaN preserves
                # the shared pen schema.
                "tilt_x": np.nan, "tilt_y": np.nan, "src_timestamp": np.nan,
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
                # These rows come from pen_events.csv, which has no
                # payload column - but one modality must carry one column set,
                # so the column exists here and is empty rather than absent.
                "src_payload": "",
                "src_t_session_ms": non_stroke["t_session_ms"].to_numpy(),
            }))
        return pen, markers

    def _markers(self, path: Path) -> tuple[pd.DataFrame, int, str]:
        raw = pd.read_csv(path)
        # This export carries the same session_start payload as the
        # SensorLogger one - `user_agent` and `screen` included - so it needs the
        # same allow-list. Cleaning only one adapter left the fingerprint in the
        # published Ege markers.
        redacted = [redact_payload_json(v) for v in raw.get("payload", pd.Series([""] * len(raw)))]
        handedness = "unknown"
        starts = raw.index[raw["event_type"].astype(str) == "session_start"]
        if len(starts):
            payload = raw.get("payload", pd.Series([""] * len(raw))).iloc[starts[0]]
            if isinstance(payload, str) and payload.strip():
                handedness = handedness_from_payload(json.loads(payload))
        out = pd.DataFrame({
            "t_ns": to_unix_ns(raw["t_ms"].to_numpy(), "ms"),
            "event": raw["event_type"].astype(str),
            # NaN, not -1: use one encoding for "not applicable" across
            # cohorts. -1 looks like a valid index to a reuser filtering
            # `task_index >= 0`; NaN cannot be mistaken for one.
            "task_id": "", "task_name": "", "task_index": np.nan,
            "task_category": "", "protocol_id": "eth_web",
            "src_payload": [p for p, _ in redacted],
            # Why: t_session_ms is a fixed column of events.csv in the documented
            # source schema; its absence means the format changed and that must
            # fail loudly here, not degrade to a silent None in the output.
            "src_t_session_ms": raw["t_session_ms"],
        })
        return sort_stable_by_time(out), sum(n for _, n in redacted), handedness


register(EgeAdapter())
