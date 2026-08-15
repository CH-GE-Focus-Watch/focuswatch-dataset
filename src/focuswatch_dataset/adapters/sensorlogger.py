"""Adapter for the Sensor Logger iOS app export.

Unit handling follows the app's documented behaviour: acceleration channels
are in g when `standardisation` is off and in SI when it is on, but headphone
gravity is always written in m/s2 regardless of the flag - the same recording
can carry both. See
https://github.com/tszheichoi/awesome-sensor-logger/blob/main/UNITS.md. Pen
strokes and phase markers live in the session JSON, not in the (empty)
`Annotation.csv`. `WatchAccelerometerUncalibrated.csv` is a second, raw
total-acceleration stream with its own rate and goes to its own table.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .. import schema as S
from ..time_axis import sort_stable_by_time, to_unix_ns
from .base import RecordingBundle, RecordingRef, register

COHORT = "ETH-SL"

_WRIST_SENSOR_NAME = "Wrist Motion"


def _standardisation(meta_path: Path) -> bool:
    meta = pd.read_csv(meta_path)
    return str(meta["standardisation"].iloc[0]).strip().lower() == "true"


def _watch_hz_nominal(meta_path: Path) -> float | None:
    """Nominal wrist rate from Metadata.csv's two parallel pipe-lists.

    `sensors` and `sampleRateMs` are indexed the same way (stream name to its
    period in ms); some streams, like Annotation, carry no rate at all. If
    the Wrist Motion entry is missing, blank or unparseable, no nominal rate
    is declared and the validator falls back to the measured rate.
    """
    meta = pd.read_csv(meta_path)
    sensors = str(meta.get("sensors", pd.Series([""])).iloc[0]).split("|")
    rates = str(meta.get("sampleRateMs", pd.Series([""])).iloc[0]).split("|")
    if _WRIST_SENSOR_NAME not in sensors:
        return None
    idx = sensors.index(_WRIST_SENSOR_NAME)
    if idx >= len(rates) or not rates[idx].strip():
        return None
    try:
        ms = float(rates[idx])
    except ValueError:
        return None
    return 1000.0 / ms if ms > 0 else None


def _harmonise_gravity_to_g(xyz: np.ndarray) -> tuple[np.ndarray, float]:
    """Convert a gravity column to g if its measured magnitude shows it is m/s2.

    Gated on the measured norm, never on which table produced the column: the
    wrist stream is in g or m/s2 depending on `standardisation`, the headphone
    stream is documented as always m/s2. A magnitude check treats both the
    same way and is idempotent, so an already-g column is never divided twice.

    Returns the harmonised array AND the factor actually applied (1.0 or
    1/G_TO_MS2) - this is a per-recording runtime decision (the same adapter
    code can go either way depending on the measured magnitude), so C1's
    published unit_conversion_factor has to come from here, not be
    re-declared as a constant beside it.
    """
    median_norm = float(np.median(np.linalg.norm(xyz, axis=1)))
    if median_norm > S.GRAVITY_SI_NORM_THRESHOLD:
        return xyz / S.G_TO_MS2, 1.0 / S.G_TO_MS2
    return xyz, 1.0


class SensorLoggerAdapter:
    name = "sensorlogger"

    def discover(self, root: Path) -> list[RecordingRef]:
        refs = []
        for d in sorted(p for p in root.iterdir() if (p / "WristMotion.csv").exists()):
            refs.append(RecordingRef(f"{COHORT}-{d.name}", f"ETH-{d.name}", "ETH", self.name, d))
        return refs

    def load(self, ref: RecordingRef) -> RecordingBundle:
        d = ref.path
        si = _standardisation(d / "Metadata.csv")
        # Why: standardisation on means the source already wrote SI units, so
        # dividing by G_TO_MS2 harmonises back to g; off means it is already g.
        accel_factor = 1.0 / S.G_TO_MS2 if si else 1.0

        watch_df, watch_factors = self._motion(d / "WristMotion.csv", accel_factor)
        tables = {"watch": watch_df}
        # Why (C1): keyed by (modality, column) - the factor actually applied,
        # per the brief's shape. Populated as each table is built, right next
        # to the division it records, so it cannot drift from what happened.
        column_factors: dict[tuple[str, str], float] = {
            ("watch", c): f for c, f in watch_factors.items()
        }

        head = d / "Headphone.csv"
        if head.exists():
            head_df, head_factors = self._motion(head, accel_factor)
            tables["headimu"] = head_df
            column_factors.update({("headimu", c): f for c, f in head_factors.items()})

        rawaccel = d / "WatchAccelerometerUncalibrated.csv"
        if rawaccel.exists():
            rawaccel_df, rawaccel_factors = self._rawaccel(rawaccel, accel_factor)
            tables["watch_rawaccel"] = rawaccel_df
            column_factors.update({("watch_rawaccel", c): f for c, f in rawaccel_factors.items()})

        pen, markers, generation = self._from_session_json(d)
        if pen is not None:
            tables["pen"] = pen
        if markers is not None:
            tables["markers"] = markers

        meta = {
            "watch_hz_nominal": _watch_hz_nominal(d / "Metadata.csv"),
            "has_gravity": "gravity_x" in tables["watch"].columns,
            "has_quaternion": "quat_x" in tables["watch"].columns,
            # Why: the Head-Capabilities pair, kept distinct from the
            # Watch-Capabilities has_gravity/has_quaternion above - the
            # headphone stream's gravity is real (measured, harmonised from
            # m/s2) and must feed its own coverage requirement, not the
            # watch's.
            "has_head_gravity": "headimu" in tables and "gravity_x" in tables["headimu"].columns,
            "has_head_quaternion": "headimu" in tables and "quat_x" in tables["headimu"].columns,
            "has_head_gyro": "headimu" in tables and "gyro_x" in tables["headimu"].columns,
            "has_watch_rawaccel": "watch_rawaccel" in tables,
            "has_pen": "pen" in tables,
            "accel_semantics": "user", "accel_calibration": "fused",
            "gravity_source": "measured",
            # Why (C2): this backend stamps every modality on the same wall
            # clock (DESIGN §5.0's shared_clock regime) - declared per
            # modality, like the ege adapter, not once for the whole recording.
            "time_domain_by_modality": {
                "watch": "backend_wall_clock", "headimu": "backend_wall_clock",
                "watch_rawaccel": "backend_wall_clock",
                "pen": "backend_wall_clock", "markers": "backend_wall_clock",
            },
            "time_alignment": "shared_clock",
            "protocol_id": "eth_web", "study_mode": "study",
            # Why (C3): follows the EXPORT GENERATION, not this pipeline
            # directory - genA (S3/T8/T9/T10) gets the same value as Ege's
            # own genA export; genB (E1/E2/E3) is kept distinct (see
            # schema.py's PEN_XY_UNIT_GEN_* docstring for why).
            "pen_xy_unit": S.PEN_XY_UNIT_GEN_A if generation == "A" else S.PEN_XY_UNIT_GEN_B,
            "pen_pressure_scale": (
                S.PEN_PRESSURE_SCALE_GEN_A if generation == "A" else S.PEN_PRESSURE_SCALE_GEN_B),
            "src_standardisation": si,
            "unit_conversion_factor_by_column": column_factors,
            "session_start_ns": self._session_start_ns(d / "Metadata.csv"),
        }
        return RecordingBundle(ref, tables, meta)

    @staticmethod
    def _session_start_ns(meta_path: Path) -> int | None:
        epoch_ms = pd.read_csv(meta_path)["recording epoch time"].iloc[0]
        return int(epoch_ms) * 1_000_000 if pd.notna(epoch_ms) else None

    def _motion(self, path: Path, accel_factor: float) -> tuple[pd.DataFrame, dict[str, float]]:
        raw = pd.read_csv(path)
        # Why: `time` is already int64 Unix nanoseconds - to_unix_ns takes the
        # integer path here and performs no float64 scaling.
        out = pd.DataFrame({"t_ns": to_unix_ns(raw["time"].to_numpy(), "ns")})
        out[list(S.COLUMNS[S.Quantity.ACCEL_USER])] = (
            raw[["accelerationX", "accelerationY", "accelerationZ"]].astype(float).to_numpy()
            * accel_factor)
        out[list(S.COLUMNS[S.Quantity.GYRO])] = (
            raw[["rotationRateX", "rotationRateY", "rotationRateZ"]].astype(float).to_numpy())
        gravity, gravity_factor = _harmonise_gravity_to_g(
            raw[["gravityX", "gravityY", "gravityZ"]].astype(float).to_numpy())
        out[list(S.COLUMNS[S.Quantity.GRAVITY])] = gravity
        out[list(S.COLUMNS[S.Quantity.QUAT])] = (
            raw[["quaternionX", "quaternionY", "quaternionZ", "quaternionW"]].astype(float).to_numpy())
        # Why (C1): gyro and quaternion columns are never scaled - 1.0 for them
        # is the true identity factor, not a placeholder. Only acceleration and
        # gravity carry a possibly-non-1.0 factor.
        factors = {c: accel_factor for c in S.COLUMNS[S.Quantity.ACCEL_USER]}
        factors.update({c: gravity_factor for c in S.COLUMNS[S.Quantity.GRAVITY]})
        return sort_stable_by_time(out), factors

    def _rawaccel(self, path: Path, factor: float) -> tuple[pd.DataFrame, dict[str, float]]:
        raw = pd.read_csv(path)
        out = pd.DataFrame({"t_ns": to_unix_ns(raw["time"].to_numpy(), "ns")})
        out[list(S.COLUMNS[S.Quantity.ACCEL_TOTAL])] = raw[["x", "y", "z"]].astype(float).to_numpy() * factor
        factors = {c: factor for c in S.COLUMNS[S.Quantity.ACCEL_TOTAL]}
        return sort_stable_by_time(out), factors

    @staticmethod
    def _read_session_json(d: Path) -> dict:
        candidates = list(d.glob("*.json"))
        if len(candidates) != 1:
            raise ValueError(f"expected exactly one session JSON in {d}, found {candidates}")
        return json.loads(candidates[0].read_text())

    def _from_session_json(self, d: Path) -> tuple[pd.DataFrame | None, pd.DataFrame | None, str]:
        payload = self._read_session_json(d)
        events = payload["events"]

        # Why (C3): generation A (S3/T8/T9/T10) stores strokes under a
        # separate, flat `pen_events` key (type/x/y/force, no payload
        # wrapper, no tilt, no pen-device clock) instead of inside `events`
        # (generation B: E1/E2/E3). Detected structurally by key presence,
        # not from the recording id - the two generations never mix within
        # one recording. Normalised here to the same event/payload shape
        # `events` already uses, so genA strokes need no second code path
        # below; `events` itself never carries stroke data for a genA
        # session, only session/phase markers and pen_session_sync.
        if "pen_events" in payload:
            generation = "A"
            dot_types = S.PEN_EVENTS_GEN_A
            normalised_pen_events = []
            for e in payload["pen_events"]:
                # Why: fail loudly and specifically rather than a bare
                # KeyError - the flat generation-A pen_events shape
                # (t_ms/type/x/y/force, no payload wrapper) is inferred from
                # whole-branch-review-findings.md's C3, not verified against a
                # real S3/T8/T9/T10 export in this environment.
                missing = [k for k in ("t_ms", "type") if k not in e]
                if missing:
                    raise ValueError(
                        f"{d}: pen_events entry missing required key(s) {missing} - "
                        f"expected a flat {{'t_ms', 'type', 'x', 'y', 'force', ...}} "
                        f"record, got keys {sorted(e)}"
                    )
                normalised_pen_events.append({
                    "t_ms": e["t_ms"], "t_session_ms": e.get("t_session_ms", np.nan),
                    "event": e["type"],
                    "payload": {k: v for k, v in e.items() if k not in ("t_ms", "t_session_ms", "type")},
                })
            all_events = events + normalised_pen_events
        else:
            generation = "B"
            dot_types = S.PEN_EVENTS_GEN_B
            all_events = events

        strokes = [e for e in all_events if e["event"] in dot_types]
        pen = None
        if strokes:
            pen = sort_stable_by_time(pd.DataFrame({
                # Why: no float cast. float64 resolves only to 256 ns at wall-clock
                # magnitude, which would shift every timestamp and can collapse
                # neighbouring samples onto the same value.
                "t_ns": to_unix_ns(np.array([e["t_ms"] for e in strokes], dtype=np.int64), "ms"),
                "dot_type": [dot_types[e["event"]] for e in strokes],
                "x": [e["payload"].get("x", np.nan) for e in strokes],
                "y": [e["payload"].get("y", np.nan) for e in strokes],
                "pressure": [e["payload"].get("force", np.nan) for e in strokes],
                # Why (C3): generation A carries no tilt and no pen-device
                # clock (open question 4) - NaN here is honest, not a bug.
                "tilt_x": [e["payload"].get("tilt", {}).get("x", np.nan) for e in strokes],
                "tilt_y": [e["payload"].get("tilt", {}).get("y", np.nan) for e in strokes],
                # Why: the pen's own clock, roughly 749 days behind wall clock. Metadata only.
                "src_timestamp": [e["payload"].get("timestamp", np.nan) for e in strokes],
            }))

        others = [e for e in all_events if e["event"] not in dot_types]
        markers = None
        if others:
            markers = sort_stable_by_time(pd.DataFrame({
                "t_ns": to_unix_ns(np.array([e["t_ms"] for e in others], dtype=np.int64), "ms"),
                "event": [e["event"] for e in others],
                # Why (I15): NaN, not -1 - see ege.py's identical fix for the rationale.
                "task_id": "", "task_name": "", "task_index": np.nan,
                "task_category": "", "protocol_id": "eth_web",
                "src_payload": [json.dumps(e.get("payload", {})) for e in others],
                "src_t_session_ms": [e.get("t_session_ms", np.nan) for e in others],
            }))
        return pen, markers, generation


register(SensorLoggerAdapter())
