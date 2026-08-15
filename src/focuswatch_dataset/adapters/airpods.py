"""Adapter for the AirPods attention study (head IMU with observer annotation).

Different research question than the other three sources: no pen, no watch,
observer-annotated focused/distracted intervals against head IMU alone. The
per-sample `label` column in the source is an expansion of the interval
protocol over an unstated clock anchor; the intervals are the raw annotation
and go into their own `attention` table, never into `pen`. The expansion
column itself is dropped once it has served as the crosscheck for that anchor.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .. import schema as S
from ..time_axis import median_rate_hz, sort_stable_by_time, to_unix_ns
from .base import RecordingBundle, RecordingRef, register

# Why: this cohort's own participant ids (P1..P26) collide with ML4SCS's;
# every id leaving this module is cohort-prefixed so the two never merge.
COHORT = "AIRPODS"

_RECORDING = re.compile(r"^(P\d+)_.*_labeled\.csv$")
_GT_LINE = re.compile(r"^\s*(\d+):(\d{2})\s+(\S+)\s*$")
_GT_DIR = "1_Data_Protocols+GroundTruth"

# Source states plain values (accel norm 0.0156 -> userAcceleration; gravity
# norm 1.0000 -> already in g), so this is a rename, not a unit conversion.
_MOTION_MAP = {
    "user_acceleration_x_g": "accel_user_x", "user_acceleration_y_g": "accel_user_y",
    "user_acceleration_z_g": "accel_user_z",
    "rotation_rate_x_rad_s": "gyro_x", "rotation_rate_y_rad_s": "gyro_y",
    "rotation_rate_z_rad_s": "gyro_z",
    "gravity_x_g": "gravity_x", "gravity_y_g": "gravity_y", "gravity_z_g": "gravity_z",
    "quaternion_x": "quat_x", "quaternion_y": "quat_y",
    "quaternion_z": "quat_z", "quaternion_w": "quat_w",
}


def parse_ground_truth(text: str) -> list[tuple[float, str]]:
    """Parse `M:SS label` lines into (offset_seconds, label) pairs."""
    out = []
    for line in text.splitlines():
        m = _GT_LINE.match(line)
        if m:
            out.append((int(m.group(1)) * 60 + int(m.group(2)), m.group(3)))
    return [(float(s), lab) for s, lab in out]


class AirPodsAdapter:
    name = "airpods"

    def discover(self, root: Path) -> list[RecordingRef]:
        refs = []
        for f in sorted(root.glob("*_labeled.csv")):
            m = _RECORDING.match(f.name)
            if m:
                rid = f"{COHORT}-{m.group(1)}"
                refs.append(RecordingRef(rid, rid, COHORT, self.name, f))
        return refs

    def load(self, ref: RecordingRef) -> RecordingBundle:
        raw = pd.read_csv(ref.path)
        # Why: pandas' default datetime64 resolution from to_datetime is not
        # pinned across versions (pandas 3 defaults to us, not ns) - `as_unit`
        # fixes a known unit before the int64 view, then to_unix_ns does the
        # final integer scaling so this never depends on the ambient default.
        parsed = pd.to_datetime(raw["timestamp_iso"], format="ISO8601", utc=True).dt.as_unit("us")
        t_ns = to_unix_ns(parsed.astype("int64").to_numpy(), "us")

        head = pd.DataFrame({"t_ns": t_ns})
        for src, dst in _MOTION_MAP.items():
            head[dst] = raw[src].astype(float)
        # Why: the device's own free-running clock, kept as metadata only -
        # t_ns (from timestamp_iso) is the canonical axis for this table.
        head["src_sensor_timestamp_s"] = raw["sensor_timestamp_s"]
        head = sort_stable_by_time(head)

        tables = {"headimu": head}
        attention = self._attention(ref, t_ns, raw["label"])
        if attention is not None:
            tables["attention"] = attention

        meta = {
            # Why: derived from the emitted tables/columns, not asserted
            # independently - these feed check_coverage, and a flag that
            # drifts from the data it describes defeats the coverage matrix.
            "has_watch": "watch" in tables,
            "has_watch_rawaccel": "watch_rawaccel" in tables,
            "has_headimu": "headimu" in tables,
            "has_pen": "pen" in tables,
            "has_markers": "markers" in tables,
            "has_attention": "attention" in tables,
            "has_gravity": "gravity_x" in head.columns,
            "has_quaternion": "quat_x" in head.columns,
            "accel_semantics": "user",
            "accel_calibration": "fused",
            "gravity_source": "measured",
            # Why: the export states no nominal head rate anywhere (no rate
            # column, no fixed-Hz claim in the corpus docs) and the measured
            # rate itself varies recording to recording - declaring a nominal
            # value would fabricate a fact the source doesn't carry. The
            # validator falls back to the measured rate when this is None.
            "head_hz_nominal": None,
            "head_hz_measured": round(median_rate_hz(head["t_ns"].to_numpy(dtype=np.int64)), 3),
            "time_domain": "device_wall_clock",
            "time_alignment": "shared_clock",
            "protocol_id": "airpods_attention",
            "study_mode": "study",
            # Why: no source file records an independent session-start marker
            # for this export (the filename date/time is export time, not a
            # verified capture start); fabricating one from it risks a spill
            # guard reading a fact the source never asserted. None skips the
            # check rather than asserting a guessed value (see validate.py's
            # spill_guard skip-when-absent path).
            "session_start_ns": None,
        }
        return RecordingBundle(ref, tables, meta)

    def _attention(self, ref: RecordingRef, t_ns: np.ndarray,
                   source_labels: pd.Series) -> pd.DataFrame | None:
        pid = ref.recording_id.removeprefix(f"{COHORT}-")
        gt_dir = ref.path.parent / _GT_DIR
        matches = sorted(gt_dir.glob(f"{pid}_ground_truth_*.txt")) if gt_dir.exists() else []
        if not matches:
            return None
        intervals = parse_ground_truth(matches[0].read_text())
        if not intervals:
            return None

        t0, t_end = int(t_ns.min()), int(t_ns.max())
        starts = [t0 + int(round(s * 1e9)) for s, _ in intervals]
        ends = starts[1:] + [t_end]
        table = pd.DataFrame({
            "t_start_ns": np.array(starts, dtype=np.int64),
            "t_end_ns": np.array(ends, dtype=np.int64),
            "label": [lab for _, lab in intervals],
        })
        self._verify_expansion(table, t_ns, source_labels)
        return table

    @staticmethod
    def _verify_expansion(table: pd.DataFrame, t_ns: np.ndarray,
                          source_labels: pd.Series) -> None:
        """Check the interval anchor against the source's own per-sample labels.

        The interval offsets are relative to an unstated clock. Before the
        per-sample column is dropped it is used to prove the anchor: expanding
        the parsed intervals back onto the sample timeline and comparing
        against the source's own expansion turns the discarded derivative
        into a test of the assumption (anchor = first head-IMU sample) that
        `_attention` makes when it builds `t_start_ns`/`t_end_ns`.
        """
        expanded = np.empty(len(t_ns), dtype=object)
        for row in table.itertuples(index=False):
            expanded[(t_ns >= row.t_start_ns) & (t_ns <= row.t_end_ns)] = row.label
        agreement = float((expanded == source_labels.to_numpy()).mean())
        if agreement < S.ATTENTION_EXPANSION_MIN_AGREEMENT:
            raise ValueError(
                f"interval expansion matches only {agreement:.3f} of the source labels; "
                "the ground-truth clock anchor is wrong"
            )


register(AirPodsAdapter())
