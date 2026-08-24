"""Physical and structural gate applied to every table before it is written."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import schema as S
from .adapters.base import RecordingBundle, RecordingRef
from .physics import angle_deg, gravity_from_quaternion, norm_stats, still_mask
from .time_axis import median_rate_hz
from .write import read_table, table_metadata


@dataclass(frozen=True)
class Finding:
    check: str
    recording_id: str
    modality: str
    column: str
    observed: float | str
    expected: str
    passed: bool


@dataclass
class ValidationReport:
    findings: list[Finding] = field(default_factory=list)

    @property
    def failed(self) -> list[Finding]:
        return [f for f in self.findings if not f.passed]

    def to_json(self) -> str:
        return json.dumps([asdict(f) for f in self.findings], indent=2)


def _in_band(value: float, band: tuple[float, float]) -> bool:
    return band[0] <= value <= band[1]


def _has(df: pd.DataFrame, q: S.Quantity) -> bool:
    return all(c in df.columns for c in S.COLUMNS[q])


def _vec(df: pd.DataFrame, q: S.Quantity) -> np.ndarray:
    return df[list(S.COLUMNS[q])].to_numpy(dtype=float)


def _diagnose_accel(median: float, band: tuple[float, float]) -> str:
    """Name the likely cause when an acceleration norm lands outside its band."""
    if S.ACCEL_FORBIDDEN_BANDS[0][0] <= median <= S.ACCEL_FORBIDDEN_BANDS[0][1]:
        return f"median in {band}; observed value suggests a user/total mix"
    if S.ACCEL_FORBIDDEN_BANDS[1][0] <= median <= S.ACCEL_FORBIDDEN_BANDS[1][1]:
        return f"median in {band}; observed value suggests m/s2 instead of g"
    return f"median in {band}"


def validate_motion_table(df: pd.DataFrame, recording_id: str, modality: str,
                          nominal_hz: float | None) -> list[Finding]:
    out: list[Finding] = []

    def add(check, column, observed, expected, passed):
        out.append(Finding(check, recording_id, modality, column, observed, expected, passed))

    t = df[S.TIME_COLUMN].to_numpy(dtype=np.int64)
    add("time_monotonic", S.TIME_COLUMN, int(np.sum(np.diff(t) < 0)),
        "no backward steps", bool(np.all(np.diff(t) >= 0)))

    measured = median_rate_hz(t)
    # Why: pd.notna, not `if nominal_hz:` - a manifest row read back out of a
    # DataFrame turns a source's `None` (no nominal rate declared) into
    # float NaN, and `bool(float("nan"))` is True. A truthiness check would
    # silently take the declared-rate branch on a NaN and always fail the
    # comparison below (NaN < tolerance is False), misreporting an honestly
    # undeclared rate as a bad one. See tests/test_manifest.py for the proof.
    if pd.notna(nominal_hz):
        add("sample_rate", S.TIME_COLUMN, round(measured, 3), f"{nominal_hz} Hz +-20%",
            abs(measured - nominal_hz) / nominal_hz < S.RATE_TOLERANCE)
    else:
        # Why: check_coverage requires sample_rate unconditionally; a source that
        # declares no nominal rate for this table (e.g. Ege's head-IMU) must still
        # produce a finding, or coverage fails on every build of that recording.
        add("sample_rate", S.TIME_COLUMN, round(measured, 3),
            "no nominal rate declared; measured only", True)

    for q, band in ((S.Quantity.ACCEL_USER, S.ACCEL_USER_BAND),
                    (S.Quantity.ACCEL_TOTAL, S.ACCEL_TOTAL_BAND)):
        if not _has(df, q):
            continue
        st = norm_stats(_vec(df, q))
        add("accel_semantic_band", q.value, round(st.median, 5),
            _diagnose_accel(st.median, band), _in_band(st.median, band))

    if _has(df, S.Quantity.GRAVITY):
        st = norm_stats(_vec(df, S.Quantity.GRAVITY))
        add("gravity_norm", S.Quantity.GRAVITY.value, round(st.median, 5),
            f"median in {S.GRAVITY_NORM_BAND}, IQR < {S.GRAVITY_NORM_MAX_IQR}",
            _in_band(st.median, S.GRAVITY_NORM_BAND) and st.iqr < S.GRAVITY_NORM_MAX_IQR)
        flat = np.abs(_vec(df, S.Quantity.GRAVITY)[:, :2]).max(axis=1) < 0.1
        if flat.sum() > 10:
            gz = float(np.median(_vec(df, S.Quantity.GRAVITY)[flat, 2]))
            add("gravity_sign", S.Quantity.GRAVITY.value, round(gz, 4),
                "gz near -1 when the device lies flat", gz < 0)

    if _has(df, S.Quantity.GYRO):
        st = norm_stats(_vec(df, S.Quantity.GYRO))
        add("gyro_range", S.Quantity.GYRO.value, round(st.median, 5),
            f"median in {S.GYRO_MEDIAN_BAND}, p95 < {S.GYRO_P95_MAX}",
            _in_band(st.median, S.GYRO_MEDIAN_BAND) and st.p95 < S.GYRO_P95_MAX)

    out += _validate_quaternion(df, recording_id, modality)
    return out


def _validate_quaternion(df: pd.DataFrame, recording_id: str, modality: str) -> list[Finding]:
    """Quaternion checks, including the fallback for sources without a gravity column.

    A scalar-first storage order leaves every norm at 1.0 and raises no exception;
    only the reconstructed gravity direction exposes it. Sources that ship no
    gravity channel are compared against the acceleration direction while the
    device is still, which is the gravity direction to within the sensor bias.
    """
    out: list[Finding] = []

    def add(check, observed, expected, passed):
        out.append(Finding(check, recording_id, modality, S.Quantity.QUAT.value,
                           observed, expected, passed))

    if not _has(df, S.Quantity.QUAT):
        return out
    q_arr = _vec(df, S.Quantity.QUAT)
    finite_q = np.isfinite(q_arr).all(axis=1)
    # Why: ML4SCS quaternion capture is forward-only, so older sessions carry
    # partly empty columns. Rotation.from_quat raises "Found zero norm
    # quaternions" on those rows, so an unfiltered array aborts the build.
    if finite_q.sum() < 100:
        return out

    st = norm_stats(q_arr[finite_q])
    add("quat_norm", round(st.median, 6), f"median in {S.QUAT_NORM_BAND}",
        _in_band(st.median, S.QUAT_NORM_BAND))

    reconstructed = gravity_from_quaternion(q_arr[finite_q])

    if _has(df, S.Quantity.GRAVITY):
        grav = _vec(df, S.Quantity.GRAVITY)[finite_q]
        rows = np.isfinite(grav).all(axis=1)
        if rows.sum() >= 100:
            ang = float(np.median(angle_deg(reconstructed[rows], grav[rows])))
            add("quat_gravity_agreement", round(ang, 4),
                f"median angle < {S.QUAT_GRAVITY_ANGLE_MAX_DEG} deg",
                ang < S.QUAT_GRAVITY_ANGLE_MAX_DEG)
        return out

    accel_q = S.Quantity.ACCEL_TOTAL if _has(df, S.Quantity.ACCEL_TOTAL) else None
    if accel_q is None or not _has(df, S.Quantity.GYRO):
        return out
    gyro = _vec(df, S.Quantity.GYRO)[finite_q]
    accel = _vec(df, accel_q)[finite_q]
    fs = median_rate_hz(df[S.TIME_COLUMN].to_numpy(dtype=np.int64)[finite_q])
    still = still_mask(gyro, fs_hz=fs if np.isfinite(fs) else 100.0)
    if still.sum() < 100:
        add("quat_still_agreement", f"{int(still.sum())} still samples",
            "at least 100 still samples for the fallback check", False)
        return out
    ang = float(np.median(angle_deg(accel[still], reconstructed[still])))
    add("quat_still_agreement", round(ang, 4),
        f"median angle < {S.QUAT_STILL_ANGLE_MAX_DEG} deg over still windows",
        ang < S.QUAT_STILL_ANGLE_MAX_DEG)
    return out


def validate_pen_table(df: pd.DataFrame, recording_id: str) -> list[Finding]:
    unknown = sorted(set(df["dot_type"]) - set(S.DOT_TYPES))
    return [Finding("dot_type_vocabulary", recording_id, "pen", "dot_type",
                    ", ".join(unknown) or "-", f"subset of {S.DOT_TYPES}", not unknown)]


def validate_attention_table(df: pd.DataFrame, recording_id: str) -> list[Finding]:
    """M8: the attention label vocabulary is now known and closed (schema.ATTENTION_LABELS)."""
    unknown = sorted(set(df["label"]) - set(S.ATTENTION_LABELS))
    return [Finding("attention_label_vocabulary", recording_id, "attention", "label",
                    ", ".join(unknown) or "-", f"subset of {S.ATTENTION_LABELS}", not unknown)]


_INTERVAL_START, _INTERVAL_END = "t_start_ns", "t_end_ns"
# A contemporary wall clock in nanoseconds; anything far below is session-relative.
_EPOCH_NS_MIN, _EPOCH_NS_MAX = 1e18, 2e18


def _table_span(df: pd.DataFrame) -> tuple[int, int] | None:
    if _INTERVAL_START in df.columns and _INTERVAL_END in df.columns and len(df):
        return int(df[_INTERVAL_START].min()), int(df[_INTERVAL_END].max())
    if S.TIME_COLUMN in df.columns and len(df):
        return int(df[S.TIME_COLUMN].min()), int(df[S.TIME_COLUMN].max())
    return None


@dataclass(frozen=True)
class Dropout:
    """A motion modality whose stream covers only a sliver of the recording (C4).

    Not a defect to hide: the sensor genuinely disconnected early (or never
    connected) and the samples that exist are real - see
    ETH-SL-E3_session1's headimu (58 samples spanning 2.1 s inside an
    8713.6 s recording). Declaring it exempts the modality from checks that
    need a substantial span to mean anything (cross-modality overlap, and
    the physics checks whose minimum-sample floors a sliver cannot clear)
    without deleting the samples or hiding that it happened - see
    detect_dropouts, validate_recording's streams_overlap exclusion, and
    check_coverage's capability-check exemption.
    """
    modality: str
    n_samples: int
    coverage_ratio: float


def detect_dropouts(tables: dict[str, pd.DataFrame]) -> dict[str, Dropout]:
    """Motion modalities (schema.MOTION_MODALITIES) below the coverage floor.

    The reference span is the enclosing span across the OTHER motion
    modalities only, not every table in the recording - pen/markers/
    attention timestamps can be wildly misaligned exactly when a merge is
    genuinely broken (the streams_overlap failure this function must not
    quietly absorb), and letting one of those inflate the denominator would
    misclassify an ordinary motion table as a "dropout" instead of surfacing
    the real alignment failure. Needs at least two motion modalities to
    compare - with only one, there is nothing to be a sliver relative to.
    """
    motion_spans = {m: s for m, s in
                    ((m, _table_span(tables[m])) for m in S.MOTION_MODALITIES if m in tables) if s}
    if len(motion_spans) < 2:
        return {}
    lo = min(s[0] for s in motion_spans.values())
    hi = max(s[1] for s in motion_spans.values())
    total = hi - lo
    if total <= 0:
        return {}
    out: dict[str, Dropout] = {}
    for modality, (s_lo, s_hi) in motion_spans.items():
        ratio = (s_hi - s_lo) / total
        if ratio < S.DROPOUT_COVERAGE_RATIO_MIN:
            out[modality] = Dropout(modality, len(tables[modality]), round(ratio, 6))
    return out


def validate_recording(recording_id: str, tables: dict[str, pd.DataFrame],
                       meta: dict) -> list[Finding]:
    """Checks that only make sense across the tables of one recording."""
    out: list[Finding] = []

    def add(check, modality, observed, expected, passed):
        out.append(Finding(check, recording_id, modality, "-", observed, expected, passed))

    spans = {m: s for m, s in ((m, _table_span(df)) for m, df in tables.items()) if s}

    for modality, (lo, _hi) in spans.items():
        add("time_magnitude", modality, lo,
            "timestamps on the declared wall clock", _EPOCH_NS_MIN <= lo <= _EPOCH_NS_MAX)

    # C4: a declared dropout (schema.DROPOUT_COVERAGE_RATIO_MIN) is exempt
    # from the overlap requirement - a sensor that disconnected 2 s into an
    # 8713.6 s recording was never going to overlap the rest, and that is
    # the dropout, not a second failure to report. Still gets its own
    # informational Finding (never omitted, per DESIGN's "publish with the
    # truth attached") so the fact is visible without failing the build.
    dropouts = detect_dropouts(tables)
    overlap_spans = {m: s for m, s in spans.items() if m not in dropouts}
    if len(overlap_spans) > 1:
        latest_start = max(lo for lo, _ in overlap_spans.values())
        earliest_end = min(hi for _, hi in overlap_spans.values())
        overlap_s = (earliest_end - latest_start) / 1e9
        add("streams_overlap", "+".join(sorted(overlap_spans)), round(overlap_s, 3),
            "modality time ranges overlap", overlap_s > 0)

    for modality, d in dropouts.items():
        add("modality_dropout", modality, f"{d.n_samples} samples, ratio {d.coverage_ratio}",
            f"coverage ratio < {S.DROPOUT_COVERAGE_RATIO_MIN} - exempt from streams_overlap "
            "and minimum-length capability checks (see check_coverage)", True)

    # Why: the declared session start, taken from the source's own session
    # metadata - not the computed span, which would make the check tautological.
    start = meta.get("session_start_ns")
    if start is not None:
        for modality, (lo, _hi) in spans.items():
            lag_s = (int(start) - lo) / 1e9
            add("spill_guard", modality, round(lag_s, 3),
                f"no sample more than {S.SPILL_GUARD_S} s before session start",
                lag_s <= S.SPILL_GUARD_S)

    # C5: informational, always passed - the count itself is the point. An
    # unknown payload key is dropped before publication (schema.py's
    # MARKER_PAYLOAD_ALLOWED_KEYS), not published; this Finding is how many
    # were dropped, so a future export's new field is visible in
    # validation_report.json rather than silently absorbed.
    dropped_keys = meta.get("src_payload_dropped_key_count")
    if dropped_keys is not None:
        add("payload_keys_redacted", "markers", dropped_keys,
            "payload keys outside schema.MARKER_PAYLOAD_ALLOWED_KEYS are dropped before "
            "publication, not silently kept", True)

    # I14: the observer timeline is checkable. The attention intervals are
    # constructed anchored to headimu's own first/last sample (airpods.py's
    # _attention), so this should always hold by construction - it is a
    # cross-modality structural invariant worth checking directly, not an
    # assumption about how the adapter built the table.
    if "attention" in spans and "headimu" in spans:
        a_lo, a_hi = spans["attention"]
        h_lo, h_hi = spans["headimu"]
        add("attention_within_headimu_range", "attention", f"[{a_lo}, {a_hi}]",
            f"inside headimu range [{h_lo}, {h_hi}]", h_lo <= a_lo and a_hi <= h_hi)

    # I14: reports a quantity rather than gating one, so it always passes -
    # P14's +44 s is a real recording and must not fail a strict build. The
    # tail is computed by airpods.py (it alone has the parsed protocol total)
    # and passed through meta, the same pattern as session_start_ns.
    tail_s = meta.get("attention_protocol_tail_s")
    if tail_s is not None:
        add("attention_protocol_tail", "attention", round(tail_s, 3),
            "recording length minus the protocol's declared Gesamtdauer - measured "
            "-0.47 s to +44.55 s across the 25 recordings, 18 within +/-1 s", True)

    # I6: accel_semantics must agree with which acceleration columns the
    # motion table actually carries - "total" implies accel_total_*, "user"
    # implies accel_user_*. Checked against watch when present, else headimu
    # (AirPods' only motion stream) - the same modality accel_semantics
    # describes per DESIGN §7. Makes the column-names-carry-semantics
    # invariant structural, not just declarative. Emitted whenever a motion
    # table exists, regardless of whether `semantics` resolves - an
    # unrecognised value (empty string, a typo, a future third value) must
    # fail this finding rather than silently produce none at all (Finding 5,
    # correction round 1): a skipped check and a passed check are otherwise
    # indistinguishable, same principle as check_coverage below.
    semantics = meta.get("accel_semantics")
    quantity = S.ACCEL_SEMANTICS_QUANTITY.get(semantics)
    modality = "watch" if "watch" in tables else ("headimu" if "headimu" in tables else None)
    if modality is not None:
        add("accel_semantics_matches_columns", modality, semantics or "-",
            f"accel_semantics must be one of {sorted(S.ACCEL_SEMANTICS_QUANTITY)} and its "
            "implied accel_total_*/accel_user_* columns must be present",
            quantity is not None and all(c in tables[modality].columns for c in S.COLUMNS[quantity]))
    return out


# Which checks a recording must have produced, derived from its manifest flags.
# A skipped check and a passed check are otherwise indistinguishable. Keyed by
# (recording_id, modality) - not just recording_id - so a check that ran for
# one modality of a recording (e.g. headimu's own time_magnitude finding)
# cannot silently satisfy a requirement declared for a different modality of
# the same recording (e.g. a dropped attention table that still claims
# has_attention). Every manifest flag maps to the modality name its findings
# carry in `Finding.modality`, matching the table key every adapter and
# `validate_recording` already use.
_REQUIRED_MOTION_CHECKS = ("time_monotonic", "sample_rate", "accel_semantic_band", "gyro_range")
# Why: watch_rawaccel is accel-only by construction - the name says so, and no
# export of a raw-accelerometer stream will ever carry a gyroscope. That is a
# structural fact of the modality, not a property of any particular
# recording, so gyro_range is excluded from its required set outright rather
# than gated by a flag: a requirement no instance could ever satisfy is a
# permanent false alarm, not a safety net.
_MOTION_CHECKS_NO_GYRO = tuple(c for c in _REQUIRED_MOTION_CHECKS if c != "gyro_range")
# has_headimu's base set also excludes gyro_range - whether a head table
# carries a gyroscope is a DATA fact that varies by source (Ege's does not;
# SensorLogger's and AirPods' do, and AirPods' head gyro is that cohort's
# only motion signal), so it is required separately, gated by has_head_gyro,
# below - never assumed either way from has_headimu alone.
_MOTION_REQUIRED_CHECKS = {
    "has_watch": _REQUIRED_MOTION_CHECKS,
    "has_headimu": _MOTION_CHECKS_NO_GYRO,
    "has_watch_rawaccel": _MOTION_CHECKS_NO_GYRO,
}
# Why: S.MODALITY_FLAGS is the single source for which modality a
# has_<modality> flag gates - manifest.py's structural derivation reads the
# same table (via S.CAPABILITY_FLAG_MODALITY, which extends it), so a flag
# gated here can never silently fall out of sync with what build_manifest
# actually derives from the tables.
_MODALITY_FLAGS = S.MODALITY_FLAGS
_MOTION_MODALITY_FLAGS = ("has_watch", "has_headimu", "has_watch_rawaccel")
# Capability sub-flags (a property *within* a modality, not the modality's
# own presence) and the single physical check each gates. The modality each
# flag maps to comes from S.CAPABILITY_FLAG_MODALITY below, not restated here
# - has_quaternion's extra OR-requirement (quat_gravity_agreement vs
# quat_still_agreement) is handled separately, right after this loop.
_CAPABILITY_CHECKS = (
    ("has_gravity", "gravity_norm"),
    ("has_quaternion", "quat_norm"),
    ("has_head_gravity", "gravity_norm"),
    ("has_head_quaternion", "quat_norm"),
    ("has_head_gyro", "gyro_range"),
)


def check_coverage(manifest: pd.DataFrame, findings: list[Finding],
                   dropouts: dict[str, dict[str, Dropout]] | None = None) -> list[str]:
    """`dropouts`: recording_id -> detect_dropouts(bundle.tables) (C4).

    Optional and keyed by recording_id, not derived from `manifest` here -
    `manifest` alone cannot recompute it (that needs the tables, which this
    function never receives), so the caller (build.py) passes what it
    already computed once per bundle. Absent or omitted for a recording
    means no dropout, i.e. today's exact behaviour - existing callers that
    never pass it are unaffected.
    """
    dropouts = dropouts or {}
    problems: list[str] = []
    by_recording_modality: dict[tuple[str, str], set[str]] = {}
    for f in findings:
        by_recording_modality.setdefault((f.recording_id, f.modality), set()).add(f.check)

    for _, row in manifest.iterrows():
        rid = row["recording_id"]
        rec_dropouts = dropouts.get(rid, {})

        def seen(modality: str) -> set[str]:
            return by_recording_modality.get((rid, modality), set())

        def require(modality: str, check: str, reason: str) -> None:
            if check not in seen(modality):
                problems.append(f"{rid}/{modality}: {check} never ran ({reason})")

        for flag in _MOTION_MODALITY_FLAGS:
            if row.get(flag):
                modality = _MODALITY_FLAGS[flag]
                for check in _MOTION_REQUIRED_CHECKS[flag]:
                    require(modality, check, "motion table present")
        for flag, modality in _MODALITY_FLAGS.items():
            if row.get(flag):
                require(modality, "time_magnitude", f"{flag} is true")

        # Why: has_gravity/has_quaternion are the Watch-Capabilities fields
        # (docs/DESIGN.md) and describe the watch/ stream specifically;
        # has_head_gravity/has_head_quaternion/has_head_gyro are the separate
        # Head-Capabilities trio and describe the headimu stream.
        for flag, check in _CAPABILITY_CHECKS:
            if row.get(flag):
                modality = S.CAPABILITY_FLAG_MODALITY[flag]
                # Why (C4): a declared dropout's physics checks need a
                # sample count no sliver can clear (quat_norm's explicit
                # 100-sample floor today; the principle generalises to
                # gravity_norm/gyro_range) - the narrow, "minimum-length"
                # exemption named in the finding, not a blanket pass on the
                # modality's base motion checks above.
                if modality in rec_dropouts:
                    continue
                require(modality, check, f"{flag} is true")
        if row.get("has_quaternion"):
            modality = S.CAPABILITY_FLAG_MODALITY["has_quaternion"]
            if (modality not in rec_dropouts
                    and not ({"quat_gravity_agreement", "quat_still_agreement"} & seen(modality))):
                problems.append(
                    f"{rid}/{modality}: neither quat_gravity_agreement nor quat_still_agreement ran "
                    "(has_quaternion is true)")
        if row.get("has_pen"):
            require("pen", "dot_type_vocabulary", "has_pen is true")
        if row.get("has_attention"):
            require("attention", "attention_label_vocabulary", "has_attention is true")
            if row.get("has_headimu"):
                require("attention", "attention_within_headimu_range",
                       "has_attention and has_headimu are true")
    return problems


def validate_bundle(bundle: RecordingBundle) -> list[Finding]:
    """Run every validation check applicable to a bundle's tables and metadata."""
    findings: list[Finding] = []
    for modality in S.MOTION_MODALITIES:
        if modality in bundle.tables:
            nominal_key = "head_hz_nominal" if modality == "headimu" else "watch_hz_nominal"
            findings += validate_motion_table(
                bundle.tables[modality], bundle.ref.recording_id, modality,
                bundle.meta.get(nominal_key),
            )
    if "pen" in bundle.tables:
        findings += validate_pen_table(bundle.tables["pen"], bundle.ref.recording_id)
    if "attention" in bundle.tables:
        findings += validate_attention_table(bundle.tables["attention"], bundle.ref.recording_id)
    return findings + validate_recording(bundle.ref.recording_id, bundle.tables, bundle.meta)


def _archive_meta(row: pd.Series) -> dict[str, object]:
    """Return only recording metadata that the archive itself publishes."""
    return row.to_dict()


def _archive_bundle(manifest_row: pd.Series, root: Path) -> RecordingBundle:
    recording_id = str(manifest_row["recording_id"])
    declared_schema = str(manifest_row["schema_version"])
    if declared_schema != S.SCHEMA_VERSION:
        raise ValueError(
            f"sessions.parquet: {recording_id} declares schema_version "
            f"{declared_schema!r}, expected {S.SCHEMA_VERSION!r}"
        )

    tables: dict[str, pd.DataFrame] = {}
    for modality in S.MODALITIES:
        if not bool(manifest_row[f"has_{modality}"]):
            continue
        path = root / modality / f"{recording_id}.parquet"
        try:
            metadata = table_metadata(path)
            table = read_table(path)
        except Exception as exc:
            raise ValueError(f"{path}: could not read declared Parquet table: {exc}") from exc

        for key, expected in (("recording_id", recording_id),
                              ("schema_version", declared_schema)):
            observed = metadata.get(key)
            if observed != expected:
                raise ValueError(
                    f"{path}: embedded {key} is {observed!r}, expected {expected!r}"
                )

        tables[modality] = table

    return RecordingBundle(
        RecordingRef(
            recording_id,
            str(manifest_row["participant_id"]),
            str(manifest_row["cohort"]),
            str(manifest_row["pipeline"]),
            root,
        ),
        tables,
        _archive_meta(manifest_row),
    )


def _same_archive_value(observed: object, expected: object) -> bool:
    if pd.isna(observed) and pd.isna(expected):
        return True
    if isinstance(observed, (float, np.floating)) or isinstance(expected, (float, np.floating)):
        # Archive summaries are deliberately rounded before publication, so a
        # relative tolerance would grow with the value and let a changed
        # duration/rate through (e.g. 8713.6 -> 8713.65). Keep only enough
        # absolute room for binary floating-point round trips.
        return bool(np.isclose(observed, expected, rtol=0.0, atol=1e-12, equal_nan=True))
    return bool(observed == expected)


def _declared_table_keys(manifest: pd.DataFrame) -> set[tuple[str, str]]:
    return {
        (str(row["recording_id"]), modality)
        for _, row in manifest.iterrows()
        for modality in S.MODALITIES
        if bool(row[f"has_{modality}"])
    }


def _validate_archive_inventory(root: Path, manifest: pd.DataFrame,
                                channels: pd.DataFrame) -> None:
    declared = _declared_table_keys(manifest)
    stored: set[tuple[str, str]] = set()
    for modality in S.MODALITIES:
        directory = root / modality
        if directory.exists() and not directory.is_dir():
            raise ValueError(f"{directory}: modality path is not a directory")
        if directory.is_dir():
            stored.update((path.stem, modality) for path in directory.glob("*.parquet"))
    orphans = sorted(stored - declared)
    if orphans:
        raise ValueError(f"orphan modality Parquet files: {orphans}")

    channel_keys = set(zip(channels["recording_id"].astype(str), channels["modality"].astype(str)))
    undeclared = sorted(channel_keys - declared)
    if undeclared:
        raise ValueError(f"channels.parquet rows for unknown or undeclared tables: {undeclared}")


def _validate_archive_descriptors(bundle: RecordingBundle, channels: pd.DataFrame,
                                  descriptor_values) -> None:
    recording_id = bundle.ref.recording_id
    primary = "watch" if "watch" in bundle.tables else ("headimu" if "headimu" in bundle.tables else None)
    for modality, table in bundle.tables.items():
        described = channels.loc[
            (channels["recording_id"] == recording_id) & (channels["modality"] == modality)
        ]
        duplicate_columns = described.loc[described["column"].duplicated(), "column"].tolist()
        if duplicate_columns:
            raise ValueError(
                f"channels.parquet: duplicate descriptors for {recording_id}/{modality}: "
                f"{sorted(duplicate_columns)}"
            )
        by_column = described.set_index("column")
        declared_columns = set(by_column.index)
        stored_columns = set(table.columns)
        if declared_columns != stored_columns:
            raise ValueError(
                f"channels.parquet: descriptors for {recording_id}/{modality} do not match "
                f"the stored table (missing={sorted(stored_columns - declared_columns)}, "
                f"unexpected={sorted(declared_columns - stored_columns)})"
            )
        for column in table.columns:
            descriptor = by_column.loc[column]
            expected = descriptor_values(table, column, bundle.meta)
            for field, expected_value in expected.items():
                if not _same_archive_value(descriptor[field], expected_value):
                    raise ValueError(
                        f"channels.parquet: {recording_id}/{modality}/{column} {field} is "
                        f"{descriptor[field]!r}, expected {expected_value!r}"
                    )
            factor = descriptor["unit_conversion_factor"]
            if not isinstance(factor, (int, float, np.number)) or not np.isfinite(factor):
                raise ValueError(
                    f"channels.parquet: {recording_id}/{modality}/{column} has invalid "
                    f"unit_conversion_factor {factor!r}"
                )
            domain = descriptor["time_domain"]
            if not isinstance(domain, str) or not domain:
                raise ValueError(
                    f"channels.parquet: {recording_id}/{modality}/{column} has no time_domain"
                )
        if modality == primary and S.TIME_COLUMN in by_column.index:
            observed_domain = by_column.loc[S.TIME_COLUMN, "time_domain"]
            expected_domain = bundle.meta["time_domain"]
            if not _same_archive_value(observed_domain, expected_domain):
                raise ValueError(
                    f"channels.parquet: {recording_id}/{modality}/{S.TIME_COLUMN} time_domain is "
                    f"{observed_domain!r}, expected manifest value {expected_domain!r}"
                )


def validate_dataset(root: Path | str) -> ValidationReport:
    """Revalidate a published archive from its manifest, channels, and Parquet tables.

    This intentionally does not consume ``validation_report.json`` or recreate
    adapter-only metadata: neither is an authoritative description of the
    published archive. Checks that depend on unpublished adapter metadata are
    naturally skipped by ``validate_recording`` because that metadata is absent.
    """
    root = Path(root)
    try:
        manifest = pd.read_parquet(root / "sessions.parquet")
        channels = pd.read_parquet(root / "channels.parquet")
    except Exception as exc:
        raise ValueError(f"could not read archive manifest or channels at {root}: {exc}") from exc

    # Imported lazily because manifest imports this module's dropout helper.
    from .manifest import (
        CHANNEL_COLUMNS, MANIFEST_COLUMNS, TABLE_DERIVED_MANIFEST_COLUMNS,
        check_manifest_consistency, table_column_descriptor_values,
        table_derived_manifest_values,
    )

    for name, frame, expected_columns in (
        ("sessions.parquet", manifest, MANIFEST_COLUMNS),
        ("channels.parquet", channels, CHANNEL_COLUMNS),
    ):
        missing = sorted(set(expected_columns) - set(frame.columns))
        unexpected = sorted(set(frame.columns) - set(expected_columns))
        if missing or unexpected:
            raise ValueError(f"{name}: contract columns mismatch (missing={missing}, unexpected={unexpected})")
    duplicate_ids = sorted(
        str(recording_id) for recording_id in manifest.loc[
            manifest["recording_id"].duplicated(keep=False), "recording_id"
        ].unique()
    )
    if duplicate_ids:
        raise ValueError(f"sessions.parquet: duplicate recording_id values: {', '.join(duplicate_ids)}")

    report = ValidationReport()
    for problem in check_manifest_consistency(manifest, root):
        report.findings.append(Finding(
            "manifest_consistency", "-", "-", "-", problem,
            "has_<modality> flag matches file presence", False,
        ))

    _validate_archive_inventory(root, manifest, channels)
    bundles = [_archive_bundle(row, root) for _, row in manifest.iterrows()]
    dropouts: dict[str, dict[str, Dropout]] = {}
    for bundle in bundles:
        expected_manifest = table_derived_manifest_values(bundle)
        for column in TABLE_DERIVED_MANIFEST_COLUMNS:
            observed = bundle.meta[column]
            if not _same_archive_value(observed, expected_manifest[column]):
                raise ValueError(
                    f"sessions.parquet: {bundle.ref.recording_id} {column} is {observed!r}, "
                    f"expected {expected_manifest[column]!r} from stored tables"
                )
        _validate_archive_descriptors(bundle, channels, table_column_descriptor_values)
        report.findings += validate_bundle(bundle)
        dropouts[bundle.ref.recording_id] = detect_dropouts(bundle.tables)

    for problem in check_coverage(manifest, report.findings, dropouts):
        report.findings.append(Finding(
            "coverage_gap", "-", "-", "-", problem,
            "check ran and produced a finding", False,
        ))
    return report
