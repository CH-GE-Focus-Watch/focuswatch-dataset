from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from focuswatch_dataset import schema as S
from focuswatch_dataset.adapters.base import RecordingBundle, RecordingRef
from focuswatch_dataset.manifest import (
    MANIFEST_COLUMNS, build_channels, build_manifest, check_manifest_consistency,
)
from focuswatch_dataset.validate import check_coverage, validate_motion_table


def bundle(rid="ML4SCS-S096", with_pen=True, **meta):
    n = 200
    watch = pd.DataFrame({
        "t_ns": np.arange(n, dtype=np.int64) * 10_000_000,
        **{c: np.zeros(n) for c in S.COLUMNS[S.Quantity.ACCEL_USER]},
        **{c: np.zeros(n) for c in S.COLUMNS[S.Quantity.GYRO]},
    })
    tables = {"watch": watch}
    if with_pen:
        tables["pen"] = pd.DataFrame({"t_ns": [0, 1], "dot_type": ["PEN_DOWN", "PEN_UP"],
                                      "x": [1.0, 2.0], "y": [1.0, 2.0]})
    base = {"watch_hz_nominal": 100.0, "accel_semantics": "user",
            "accel_calibration": "fused", "gravity_source": "none",
            "time_domain": "watch_capture_clock", "time_alignment": "estimated_delta",
            "protocol_id": "ml4scs_v2"}
    return RecordingBundle(RecordingRef(rid, "ML4SCS-P76", "ML4SCS", "ml4scs", Path(".")),
                           tables, base | meta)


def _head_bundle(rid="AIRPODS-P99", has_head_gyro_meta=True, n=200):
    """AirPods-shaped bundle: headimu only, gyro really present in the table."""
    t = np.arange(n, dtype=np.int64) * 40_000_000
    head = pd.DataFrame({
        "t_ns": t,
        **{c: np.zeros(n) for c in S.COLUMNS[S.Quantity.ACCEL_USER]},
        **{c: np.zeros(n) for c in S.COLUMNS[S.Quantity.GYRO]},
    })
    return RecordingBundle(
        RecordingRef(rid, rid, "AIRPODS", "airpods", Path(".")),
        {"headimu": head},
        {
            "head_hz_nominal": None,
            # Why: deliberately WRONG - proves build_manifest does not trust
            # this value, it recomputes has_head_gyro from the table itself.
            "has_head_gyro": has_head_gyro_meta,
            "accel_semantics": "user", "accel_calibration": "fused",
            "gravity_source": "none", "time_domain": "device_wall_clock",
            "time_alignment": "shared_clock", "protocol_id": "airpods_attention",
        },
    )


def test_manifest_has_all_declared_columns():
    m = build_manifest([bundle()])
    assert set(MANIFEST_COLUMNS) <= set(m.columns)


def test_modality_flags_follow_the_tables():
    m = build_manifest([bundle(with_pen=False)]).iloc[0]
    assert m["has_watch"] is np.True_ or m["has_watch"] is True
    assert not m["has_pen"]
    assert not m["has_attention"]


def test_measured_rate_is_derived_not_copied():
    m = build_manifest([bundle()]).iloc[0]
    assert m["watch_hz_measured"] == pytest.approx(100.0)


def test_channels_declare_a_unit_for_every_signal_column():
    ch = build_channels([bundle()])
    signal = ch[ch["column"] != "t_ns"]
    assert (signal["unit"] != "").all()
    assert (signal["semantics"] != "").all()
    assert set(ch[ch["column"] == "accel_user_x"]["unit"]) == {"g"}


def test_consistency_flags_a_missing_file(tmp_path):
    m = build_manifest([bundle()])
    problems = check_manifest_consistency(m, tmp_path)
    assert any("watch/ML4SCS-S096.parquet" in p for p in problems)


def test_consistency_is_clean_when_files_exist(tmp_path):
    m = build_manifest([bundle(with_pen=False)])
    (tmp_path / "watch").mkdir()
    (tmp_path / "watch" / "ML4SCS-S096.parquet").write_bytes(b"x")
    assert check_manifest_consistency(m, tmp_path) == []


# --- Finding 1: capability flags must not be able to drift from the data ---

def test_structural_flag_overrides_a_wrong_meta_declaration():
    """has_head_gyro is recomputed from the headimu table, not trusted from meta.

    The fixture's meta claims has_head_gyro=False while the headimu table
    genuinely carries gyro_x/y/z. If build_manifest ever started trusting
    meta over the table, this would flip to False and the coverage
    requirement below would stop firing without anyone touching
    check_coverage.
    """
    m = build_manifest([_head_bundle(has_head_gyro_meta=False)]).iloc[0]
    assert bool(m["has_head_gyro"]) is True


def test_manifest_built_from_a_real_bundle_drives_coverage_for_head_gyro():
    """Reproduces the exact bug class the brief describes for AirPods.

    A hand-built manifest DataFrame in a coverage test can (and once did)
    simply omit a flag it was meant to require, and the test stays green for
    rounds because nothing ever re-derives that flag from real adapter
    output. Here the manifest comes from build_manifest() on an actual
    RecordingBundle - the production path - and no gyro_range finding was
    ever produced for headimu. check_coverage must catch that, proving the
    flag actually reached the manifest and is actually driving the gate.
    """
    m = build_manifest([_head_bundle()])
    problems = check_coverage(m, findings=[])
    assert any("gyro_range" in p and "headimu" in p for p in problems)


def test_head_measured_rate_is_derived_not_copied():
    """Fix-round-1 item 1: head_hz_measured gets the same guarantee as
    watch_hz_measured. The fixture's meta claims a nonsense 999.0 Hz; the
    200-sample, 40ms-spaced headimu table is genuinely 25 Hz. Before this
    fix, head_hz_measured was excluded from neither _STRUCTURAL_FLAGS nor
    any equivalent set, so the meta merge silently overwrote the value
    computed from the real table - this test would have observed 999.0.
    """
    b = RecordingBundle(
        RecordingRef("AIRPODS-P50", "AIRPODS-P50", "AIRPODS", "airpods", Path(".")),
        {"headimu": pd.DataFrame({
            "t_ns": np.arange(200, dtype=np.int64) * 40_000_000,
            **{c: np.zeros(200) for c in S.COLUMNS[S.Quantity.ACCEL_USER]},
            **{c: np.zeros(200) for c in S.COLUMNS[S.Quantity.GYRO]},
        })},
        {"head_hz_nominal": None, "head_hz_measured": 999.0,  # deliberately wrong
         "accel_semantics": "user", "accel_calibration": "fused", "gravity_source": "none",
         "time_domain": "device_wall_clock", "time_alignment": "shared_clock",
         "protocol_id": "airpods_attention"},
    )
    m = build_manifest([b]).iloc[0]
    assert m["head_hz_measured"] == pytest.approx(25.0)


# --- Finding 3 (fix round 1): unknown meta keys must be classified, not silently included ---

def test_unclassified_meta_key_fails_the_build_loudly():
    """A brand-new adapter meta key that is neither in MANIFEST_COLUMNS nor
    _INTERNAL_META_KEYS must raise, not silently leak into the publication
    (the pre-fix-round-1 behaviour) or silently vanish from it (the
    original brief's df[list(MANIFEST_COLUMNS)] bug). A deliberate
    MANIFEST_COLUMNS-or-_INTERNAL_META_KEYS decision is the only way through.
    """
    b = bundle(has_never_before_seen_flag=True)
    assert "has_never_before_seen_flag" not in MANIFEST_COLUMNS
    with pytest.raises(ValueError, match="has_never_before_seen_flag"):
        build_manifest([b])


def test_internal_meta_key_does_not_appear_in_the_published_manifest():
    """session_start_ns is real (every adapter sets it, for the spill guard)
    and legitimately internal - it must build without raising, but must not
    occupy a column in the published manifest.
    """
    b = bundle(session_start_ns=1_780_000_000_000_000_000)
    m = build_manifest([b])
    assert "session_start_ns" not in m.columns


# --- Fix round 1, item 5: n_writing_tasks/n_idle_tasks ---

def _markers(instances: list[tuple[str, str]]) -> pd.DataFrame:
    """One task_start + task_end pair per (task_id, task_category) instance."""
    rows = []
    for i, (task_id, category) in enumerate(instances):
        for event in ("task_start", "task_end"):
            rows.append({"t_ns": i, "event": event, "task_id": task_id, "task_name": task_id,
                        "task_index": i, "task_category": category, "protocol_id": "v2"})
    return pd.DataFrame(rows)


def test_task_counts_come_from_ml4scs_style_markers():
    b = bundle()
    b.tables["markers"] = _markers([("abschreiben", "writing"), ("math", "writing"),
                                    ("pause1", "idle"), ("pause2", "idle"), ("pause3", "idle")])
    m = build_manifest([b]).iloc[0]
    assert m["n_writing_tasks"] == 2
    assert m["n_idle_tasks"] == 3


def test_task_counts_are_none_not_zero_for_an_untaxonomised_markers_table():
    """ege.py/sensorlogger.py's markers tables are a session-event stream with
    task_category always "" - a real fact this cohort's protocol has no task
    taxonomy, not "zero writing tasks happened". None, not 0, must come out.
    """
    b = bundle()
    b.tables["markers"] = _markers([("pen_session_sync", ""), ("pen_paper_info", "")])
    m = build_manifest([b]).iloc[0]
    assert m["n_writing_tasks"] is None or (isinstance(m["n_writing_tasks"], float)
                                            and np.isnan(m["n_writing_tasks"]))
    assert m["n_idle_tasks"] is None or (isinstance(m["n_idle_tasks"], float)
                                         and np.isnan(m["n_idle_tasks"]))


def test_task_counts_are_none_with_no_markers_table_at_all():
    m = build_manifest([_head_bundle()]).iloc[0]  # AirPods-shaped: no markers table
    assert m["n_writing_tasks"] is None or (isinstance(m["n_writing_tasks"], float)
                                            and np.isnan(m["n_writing_tasks"]))


# --- Finding 2: None must survive the DataFrame round trip ---

def test_nominal_hz_none_survives_the_dataframe_round_trip():
    """watch_hz_nominal=None becomes NaN in the DataFrame, not a truthy value.

    bool(float("nan")) is True in plain Python, so any consumer that does
    `if nominal_hz:` on a value pulled out of this DataFrame would wrongly
    treat "no nominal rate declared" as a declared rate. This test goes
    through build_manifest -> a real pandas row -> validate_motion_table,
    not a hand-built dict, so it actually exercises the round trip the bug
    lives in. Two bundles, one with a real nominal rate and one without -
    matching the real corpus (Ege/AirPods None mixed with ML4SCS/SensorLogger
    floats) - is what forces pandas to unify the column to float64 and turn
    None into NaN, rather than leaving a single-row object column holding
    None untouched.
    """
    declared = bundle(rid="ML4SCS-S001")
    undeclared = bundle(rid="ML4SCS-S002", watch_hz_nominal=None)
    manifest = build_manifest([declared, undeclared])
    row = manifest[manifest["recording_id"] == "ML4SCS-S002"].iloc[0]
    value = row["watch_hz_nominal"]

    assert isinstance(value, float)
    assert np.isnan(value)
    assert bool(value) is True  # the trap: NaN is truthy in plain Python
    assert pd.notna(value) is False  # the safe check validate.py now uses

    n = 200
    watch = pd.DataFrame({
        "t_ns": np.arange(n, dtype=np.int64) * 10_000_000,
        **{c: np.zeros(n) for c in S.COLUMNS[S.Quantity.ACCEL_USER]},
        **{c: np.zeros(n) for c in S.COLUMNS[S.Quantity.GYRO]},
    })
    findings = validate_motion_table(watch, "R1", "watch", value)
    rate = next(f for f in findings if f.check == "sample_rate")
    assert rate.passed
    assert "no nominal rate declared" in rate.expected
