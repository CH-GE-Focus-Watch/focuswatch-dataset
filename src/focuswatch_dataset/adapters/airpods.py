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
from ..time_axis import parse_iso_to_unix_ns, sort_stable_by_time
from .base import RecordingBundle, RecordingRef, register

# Why: this cohort's own participant ids (P1..P26) collide with ML4SCS's;
# every id leaving this module is cohort-prefixed so the two never merge.
COHORT = "AIRPODS"

_RECORDING = re.compile(r"^(P\d+)_.*_labeled\.csv$")
_GT_LINE = re.compile(r"^\s*(\d+):(\d{2})\s+(\S+)\s*$")
_GT_DIR = "1_Data_Protocols+GroundTruth"

# I13: the `*_protokoll_*.txt` block table beside each `*_ground_truth_*.txt`
# ("  1 |   0:00 |   3:00 |   3:00 | focused     | Abschreiben ...") -
# idx | start mm:ss | end mm:ss | duration mm:ss | label | free-text activity.
_PROTOCOL_ROW = re.compile(
    r"^\s*\d+\s*\|\s*(\d+):(\d{2})\s*\|\s*(\d+):(\d{2})\s*\|\s*\d+:\d{2}\s*\|\s*(\S+)\s*\|\s*(.+?)\s*$"
)
# I14: "Gesamtdauer" (total duration), stated elsewhere in the same file -
# cross-checked when it parses, never depended on (the core tail computation
# uses the last block's own end instead, since blocks are contiguous).
_GESAMTDAUER = re.compile(r"Gesamtdauer\D*(\d+):(\d{2})")

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


def parse_protocol(text: str) -> list[tuple[float, float, str, str]]:
    """Parse `idx | start | end | duration | label | activity` block rows.

    Returns (start_s, end_s, label, activity) per block, file order. The
    German source text is umlaut-free ASCII already ("Loesen", "Uebergaenge")
    and is published verbatim - a translated label would no longer be the
    observer's own record.
    """
    out = []
    for line in text.splitlines():
        m = _PROTOCOL_ROW.match(line)
        if m:
            start = int(m.group(1)) * 60 + int(m.group(2))
            end = int(m.group(3)) * 60 + int(m.group(4))
            out.append((float(start), float(end), m.group(5), m.group(6)))
    return out


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
        t_ns = parse_iso_to_unix_ns(raw["timestamp_iso"])

        head = pd.DataFrame({"t_ns": t_ns})
        for src, dst in _MOTION_MAP.items():
            head[dst] = raw[src].astype(float)
        # Why: the device's own free-running clock, kept as metadata only -
        # t_ns (from timestamp_iso) is the canonical axis for this table.
        head["src_sensor_timestamp_s"] = raw["sensor_timestamp_s"]
        head = sort_stable_by_time(head)

        tables = {"headimu": head}
        attention, tail_s = self._attention(ref, t_ns, raw["label"])
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
            # Why: has_gravity/has_quaternion are the Watch-Capabilities fields
            # (docs/DESIGN.md:306-308) and describe the watch/ stream, which
            # this source does not have at all. Head capabilities are a
            # separate field pair - conflating them would have every AirPods
            # manifest row claim watch gravity for a recording with no watch.
            "has_head_gravity": "gravity_x" in head.columns,
            "has_head_quaternion": "quat_x" in head.columns,
            # Why: this cohort's head gyro is its only motion signal (no
            # watch table exists at all) - declared like the other two head
            # capabilities so check_coverage actually requires gyro_range to
            # have run on headimu, rather than assuming it.
            "has_head_gyro": "gyro_x" in head.columns,
            # Why: accel_semantics/accel_calibration and gravity_source are
            # not symmetric here, despite both being Watch-Capabilities
            # fields. docs/DESIGN.md:317-318 scopes accel_semantics to
            # watch/ only as a DISAMBIGUATION rule for recordings with
            # several accel streams; AirPods has exactly one motion stream,
            # so there is nothing to disambiguate and it honestly describes
            # that stream. gravity_source has no such exemption - it is
            # has_gravity's direct companion, and has_gravity is correctly
            # False here (no watch/ stream at all), so "measured" would
            # openly contradict it in the same row. Derived the way
            # ml4scs.py derives it, from the watch table's columns; this
            # source never builds one, so it always resolves to "none" - the
            # head stream's own gravity is carried by has_head_gravity. Do
            # not "fix" one of these two fields into agreement with the
            # other; they answer genuinely different questions.
            "accel_semantics": "user",
            "accel_calibration": "fused",
            "gravity_source": "measured" if "gravity_x" in tables.get("watch", pd.DataFrame()).columns
                              else "none",
            # Why: the export states no nominal head rate anywhere (no rate
            # column, no fixed-Hz claim in the corpus docs) and the measured
            # rate itself varies recording to recording - declaring a nominal
            # value would fabricate a fact the source doesn't carry. The
            # validator falls back to the measured rate when this is None.
            # head_hz_measured is NOT declared here - manifest.build_manifest
            # recomputes it from the headimu table itself (same treatment as
            # watch_hz_measured), so a value asserted here would only ever be
            # dead weight at best or a silently-overridden lie at worst.
            "head_hz_nominal": None,
            # Why (C2): the attention intervals are anchored to this same
            # head-IMU sample clock (_attention's t0 = t_ns.min(), verified
            # against the source's own per-sample label column by
            # _verify_expansion before that column is dropped) - so both
            # modalities this adapter emits share device_wall_clock, declared
            # per modality rather than once for the whole recording.
            "time_domain_by_modality": {
                "headimu": "device_wall_clock", "attention": "device_wall_clock",
            },
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
            # Why (I14): internal-only (manifest._INTERNAL_META_KEYS) -
            # surfaced as a validate.py Finding (attention_protocol_tail),
            # not a manifest column; the measured value is the point, and it
            # never gates a build (P14's +44.55 s is a real recording).
            "attention_protocol_tail_s": tail_s,
        }
        return RecordingBundle(ref, tables, meta)

    def _attention(self, ref: RecordingRef, t_ns: np.ndarray,
                   source_labels: pd.Series) -> tuple[pd.DataFrame | None, float | None]:
        pid = ref.recording_id.removeprefix(f"{COHORT}-")
        gt_dir = ref.path.parent / _GT_DIR
        matches = sorted(gt_dir.glob(f"{pid}_ground_truth_*.txt")) if gt_dir.exists() else []
        if not matches:
            return None, None
        intervals = parse_ground_truth(matches[0].read_text())
        if not intervals:
            return None, None
        rows, tail_s = self._resolve_blocks(gt_dir, pid, intervals, t_ns)

        t0, t_end = int(t_ns.min()), int(t_ns.max())
        starts = [t0 + int(round(s * 1e9)) for s, _, _ in rows]
        ends = starts[1:] + [t_end]
        table = pd.DataFrame({
            "t_start_ns": np.array(starts, dtype=np.int64),
            "t_end_ns": np.array(ends, dtype=np.int64),
            "label": [lab for _, lab, _ in rows],
            "activity": [act for _, _, act in rows],
        })
        # Why: the ground truth's offsets are relative to the *protocol*, not
        # this recording's own stream length - if the head-IMU stream ends
        # before the last offset, the closing interval's end (clamped to
        # t_end) lands before its own start. That is a genuinely short
        # recording, not a formatting quirk, and must fail loudly rather
        # than publish a malformed (or silently clamped) interval.
        bad = table[table["t_start_ns"] > table["t_end_ns"]]
        if not bad.empty:
            row = bad.iloc[0]
            raise ValueError(
                f"{len(bad)} attention interval(s) end before they start "
                f"(e.g. label={row.label!r} t_start_ns={row.t_start_ns} > "
                f"t_end_ns={row.t_end_ns}) - the head-IMU stream ends before its "
                "own ground truth; this recording is shorter than its protocol"
            )
        self._verify_expansion(table, t_ns, source_labels)
        return table, tail_s

    @staticmethod
    def _resolve_blocks(gt_dir: Path, pid: str, intervals: list[tuple[float, str]],
                        t_ns: np.ndarray) -> tuple[list[tuple[float, str, str]], float | None]:
        """I13/I14: rows for the attention table, at protocol-block granularity.

        The observer's `*_ground_truth_*.txt` is the sibling `*_protokoll_*.txt`
        with adjacent same-label blocks merged - verified across all 26 pairs in
        the corpus (boundaries a subset, labels identical, merge reproduces the
        file exactly). Publishing the blocks is therefore lossless in the
        direction that matters: merging adjacent same-label rows reproduces the
        ground truth, while the reverse cannot recover the activities. So each
        row carries exactly one activity instead of several joined into a cell.

        The merge equality is the check that ties the two files together, and it
        fails loudly rather than silently preferring one. Without a protocol the
        ground-truth intervals stand as rows with an empty activity - the column
        exists either way, since a column that appears only sometimes within one
        modality is the schema drift I15 fixes elsewhere.

        The tail (I14) is reported, never enforced. Measured across the 25
        recordings it runs -0.47 s to +44.55 s, 18 of them within a second: the
        negatives are sub-second, the stream ending a fraction before the
        protocol's nominal end rather than starting late. P14's +44.55 s is a
        real recording and must not fail a strict build.
        """
        plain = [(s, lab, "") for s, lab in intervals]
        matches = sorted(gt_dir.glob(f"{pid}_protokoll_*.txt")) if gt_dir.exists() else []
        if not matches:
            return plain, None
        path = matches[0]
        text = path.read_text()
        blocks = parse_protocol(text)
        if not blocks:
            return plain, None

        merged: list[tuple[float, str]] = []
        for start, _end, label, _activity in blocks:
            if not merged or merged[-1][1] != label:
                merged.append((start, label))
        if len(merged) != len(intervals) or any(
                abs(m_start - gt_start) > 0.5 or m_label != gt_label
                for (m_start, m_label), (gt_start, gt_label) in zip(merged, intervals)):
            raise ValueError(
                f"{path}: merging adjacent same-label protocol blocks yields {merged}, "
                f"but the ground truth states {intervals} - the two files disagree"
            )

        protocol_total_s = blocks[-1][1]   # blocks are contiguous - last end = declared total
        m = _GESAMTDAUER.search(text)
        if m:
            stated_s = int(m.group(1)) * 60 + int(m.group(2))
            if abs(stated_s - protocol_total_s) > 1.0:
                raise ValueError(
                    f"{path}: block end ({protocol_total_s}s) disagrees with its own "
                    f"stated Gesamtdauer ({stated_s}s)"
                )
        recording_s = (int(t_ns.max()) - int(t_ns.min())) / 1e9
        return ([(s, label, activity) for s, _e, label, activity in blocks],
                recording_s - protocol_total_s)

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
