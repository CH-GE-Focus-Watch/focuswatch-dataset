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


def test_manifest_carries_a_meta_key_the_column_list_has_never_heard_of():
    """A future adapter's novel meta key must not be silently dropped.

    MANIFEST_COLUMNS is a hand-maintained list; it can lag the adapters that
    populate a bundle's meta. build_manifest must still surface any key an
    adapter actually emits, or that key's information (here modelling a
    hypothetical future flag) is lost without any test noticing - the same
    failure shape as the AirPods has_head_gyro omission, just one layer
    earlier (in the manifest builder itself, not in a hand-built test
    fixture).
    """
    b = bundle(has_never_before_seen_flag=True)
    m = build_manifest([b])
    assert "has_never_before_seen_flag" not in MANIFEST_COLUMNS
    assert "has_never_before_seen_flag" in m.columns
    assert bool(m.iloc[0]["has_never_before_seen_flag"]) is True


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
