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
from .eth_web import handedness_from_payload, redact_payload

COHORT = "ETH-SL"

_WRIST_SENSOR_NAME = "Wrist Motion"


def _standardisation(meta: pd.DataFrame) -> bool:
    return str(meta["standardisation"].iloc[0]).strip().lower() == "true"


def _watch_hz_nominal(meta: pd.DataFrame) -> float | None:
    """Nominal wrist rate from Metadata.csv's two parallel pipe-lists.

    `sensors` and `sampleRateMs` are indexed the same way (stream name to its
    period in ms); some streams, like Annotation, carry no rate at all. If
    the Wrist Motion entry is missing, blank or unparseable, no nominal rate
    is declared and the validator falls back to the measured rate.
    """
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
    code can go either way depending on the measured magnitude), so the
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
            token = self._participant_token(d)
            refs.append(RecordingRef(f"{COHORT}-{d.name}", f"ETH-{token}", "ETH", self.name, d))
        return refs

    @classmethod
    def _participant_token(cls, d: Path) -> str:
        """Return the participant id from session_start's payload.

        Never parsed from the directory name - real directories carry
        session suffixes (`E1_session3`, `focuswatch_T10_s1_d7498f51`) that
        are not DESIGN §2.2's `ETH-T8` token form. Fails loudly rather than
        falling back to a directory-name guess: the two ETH participant
        namespaces are disjoint, which only holds if the
        published token is the one the session itself states.
        """
        payload = cls._read_session_json(d)
        start = cls._find_event(payload["events"], "session_start")
        token = (start or {}).get("payload", {}).get("participant_id")
        if not token:
            raise ValueError(
                f"{d}: no events[session_start].payload.participant_id - "
                "cannot derive a participant id without guessing from the directory name"
            )
        return str(token)

    @staticmethod
    def _find_event(events: list[dict], name: str) -> dict | None:
        return next((e for e in events if e["event"] == name), None)

    def load(self, ref: RecordingRef) -> RecordingBundle:
        d = ref.path
        metadata = pd.read_csv(d / "Metadata.csv")
        si = _standardisation(metadata)
        # Why: standardisation on means the source already wrote SI units, so
        # dividing by G_TO_MS2 harmonises back to g; off means it is already g.
        accel_factor = 1.0 / S.G_TO_MS2 if si else 1.0

        watch_df, watch_factors = self._motion(d / "WristMotion.csv", accel_factor)
        tables = {"watch": watch_df}
        # Keyed by (modality, column): record the factor actually applied
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

        pen, markers, generation, handedness, dropped_keys = self._from_session_json(d)
        if pen is not None:
            tables["pen"] = pen
        if markers is not None:
            tables["markers"] = markers

        meta = {
            "watch_hz_nominal": _watch_hz_nominal(metadata),
            "accel_semantics": "user", "accel_calibration": "fused",
            "gravity_source": "measured",
            # This backend stamps every modality on the same wall
            # clock (DESIGN §5.0's shared_clock regime) - declared per
            # modality, like the ege adapter, not once for the whole recording.
            "time_domain_by_modality": {
                "watch": "backend_wall_clock", "headimu": "backend_wall_clock",
                "watch_rawaccel": "backend_wall_clock",
                "pen": "backend_wall_clock", "markers": "backend_wall_clock",
            },
            # pen.src_timestamp is the pen
            # hardware's own free-running clock (~749 days off backend_wall_
            # clock, populated for generation-B recordings and structurally NaN
            # for generation A (same physical column as ml4scs.py's
            # pen.src_timestamp and ege.py's). src_t_session_ms is a
            # session-relative millisecond offset, not a wall-clock reading.
            "time_domain_by_column": {
                ("pen", "src_timestamp"): S.TIME_DOMAIN_PEN_DEVICE_CLOCK,
                ("pen", "src_t_session_ms"): S.TIME_DOMAIN_SESSION_RELATIVE_OFFSET_MS,
                ("markers", "src_t_session_ms"): S.TIME_DOMAIN_SESSION_RELATIVE_OFFSET_MS,
            },
            "time_alignment": "shared_clock",
            "protocol_id": "eth_web", "study_mode": "study",
            # Follows the export generation, not this pipeline
            # directory - genA (S3/T8/T9/T10) gets the same value as Ege's
            # own genA export; genB (E1/E2/E3) is kept distinct (see
            # schema.py's PEN_XY_UNIT_GEN_* docstring for why).
            "pen_xy_unit": S.PEN_XY_UNIT_GEN_A if generation == "A" else S.PEN_XY_UNIT_GEN_B,
            "pen_pressure_scale": (
                S.PEN_PRESSURE_SCALE_GEN_A if generation == "A" else S.PEN_PRESSURE_SCALE_GEN_B),
            "src_standardisation": si,
            "unit_conversion_factor_by_column": column_factors,
            "session_start_ns": self._session_start_ns(metadata),
            # Promoted from a payload field to a
            # typed manifest column - nothing read it while watch_wrist_side
            # published "unknown" for all 9 ETH recordings even though the
            # source states it.
            "handedness": handedness,
            # Internal-only (see manifest._INTERNAL_META_KEYS):
            # surfaced as a validate.py Finding, not a manifest column, so a
            # future export's new payload field is visible in
            # validation_report.json rather than silently dropped.
            "src_payload_dropped_key_count": dropped_keys,
        }
        return RecordingBundle(ref, tables, meta)

    @staticmethod
    def _session_start_ns(meta: pd.DataFrame) -> int | None:
        epoch_ms = meta["recording epoch time"].iloc[0]
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
        # Gyro and quaternion are unscaled; only acceleration and gravity vary.
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

    # Both ETH pipelines share payload redaction and handedness extraction.
    _redact_payload = staticmethod(redact_payload)

    @staticmethod
    def _handedness(events: list[dict]) -> str:
        start = SensorLoggerAdapter._find_event(events, "session_start")
        return handedness_from_payload((start or {}).get("payload", {}))

    def _from_session_json(
        self, d: Path
    ) -> tuple[pd.DataFrame | None, pd.DataFrame | None, str, str, int]:
        payload = self._read_session_json(d)
        events = payload["events"]
        handedness = self._handedness(events)

        # Generation A stores strokes under a
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
                # the documented export shape.
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
                # float64 even though this source states whole
                # numbers. Pressure and tilt are physical quantities, and a
                # cohort whose values happen to be integers must not publish a
                # different dtype for the same column. src_timestamp is
                # millisecond-magnitude, far inside float64's exact range.
                "pressure": np.array([e["payload"].get("force", np.nan)
                                      for e in strokes], dtype=float),
                # Generation A carries no tilt or pen-device clock; NaN is
                # therefore an honest value, not a bug.
                "tilt_x": np.array([e["payload"].get("tilt", {}).get("x", np.nan)
                                    for e in strokes], dtype=float),
                "tilt_y": np.array([e["payload"].get("tilt", {}).get("y", np.nan)
                                    for e in strokes], dtype=float),
                # Why: the pen's own clock, roughly 749 days behind wall clock. Metadata only.
                "src_timestamp": np.array([e["payload"].get("timestamp", np.nan)
                                           for e in strokes], dtype=float),
                # The session-relative offset that Ege also
                # publishes - one column set per modality, so it is present here
                # rather than leaving that cohort's pen tables a different shape.
                "src_t_session_ms": [e.get("t_session_ms", np.nan) for e in strokes],
            }))

        others = [e for e in all_events if e["event"] not in dot_types]
        markers = None
        dropped_keys = 0
        if others:
            redacted = [self._redact_payload(e.get("payload", {})) for e in others]
            dropped_keys = sum(n for _, n in redacted)
            markers = sort_stable_by_time(pd.DataFrame({
                "t_ns": to_unix_ns(np.array([e["t_ms"] for e in others], dtype=np.int64), "ms"),
                "event": [e["event"] for e in others],
                # NaN, not -1, represents an inapplicable task index.
                "task_id": "", "task_name": "", "task_index": np.nan,
                "task_category": "", "protocol_id": "eth_web",
                # Allow-listed, not the raw payload: user_agent/
                # screen/notes must never reach the public bundle.
                "src_payload": [json.dumps(p) for p, _ in redacted],
                "src_t_session_ms": [e.get("t_session_ms", np.nan) for e in others],
            }))
        return pen, markers, generation, handedness, dropped_keys


register(SensorLoggerAdapter())
