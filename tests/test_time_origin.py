import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from focuswatch_dataset.adapters.base import RecordingBundle, RecordingRef
from focuswatch_dataset.time_origin import TimeOrigin, epoch_columns, rebase_bundle

T0 = 1_778_855_597_578_000_000  # 2026-05-15, a Unix time in ns


def _bundle():
    watch = pd.DataFrame({
        "t_ns": T0 + np.arange(3, dtype=np.int64) * 20_000_000,
        "src_server_received_ms": (T0 // 1_000_000 + np.array([5, 25, 45])).astype(np.int64),
    })
    pen = pd.DataFrame({
        "t_ns": T0 + np.array([10, 30], dtype=np.int64) * 1_000_000,
        "src_timestamp": [1_703_183_966_194.0, 1_703_183_966_206.0],  # pen clock, days off
    })
    attention = pd.DataFrame({
        "t_start_ns": [T0 - 1_000_000_000], "t_end_ns": [T0 + 1_000_000_000],
    })
    markers = pd.DataFrame({
        "t_ns": [T0], "event": ["session_start"],
        "src_payload": [json.dumps({"session_start_t_ms": T0 // 1_000_000, "phase": "a"})],
    })
    return RecordingBundle(
        RecordingRef("ZHAW-R1", "ZHAW-P1", "ZHAW", "ege", Path(".")),
        {"watch": watch, "pen": pen, "attention": attention, "markers": markers},
        {"session_start_ns": T0},
    )


def test_unix_epoch_changes_nothing_and_records_the_origin():
    bundle = _bundle()
    out = rebase_bundle(bundle)

    assert out.meta["time_origin"] == "unix_epoch"
    for modality, df in bundle.tables.items():
        pd.testing.assert_frame_equal(out.tables[modality], df)


def test_recording_start_puts_the_earliest_instant_at_zero():
    out = rebase_bundle(_bundle(), TimeOrigin.RECORDING_START)

    assert out.meta["time_origin"] == "recording_start"
    assert out.tables["attention"]["t_start_ns"].iloc[0] == 0
    assert out.tables["watch"]["t_ns"].tolist() == [1_000_000_000, 1_020_000_000, 1_040_000_000]
    assert out.meta["session_start_ns"] == 1_000_000_000
    for df in out.tables.values():
        assert epoch_columns(df) == []


def test_recording_start_keeps_every_difference_between_clocks():
    before, after = _bundle(), rebase_bundle(_bundle(), "recording_start")
    watch_lag = lambda w: w["src_server_received_ms"] - w["t_ns"] // 1_000_000  # noqa: E731

    assert watch_lag(after.tables["watch"]).tolist() == watch_lag(before.tables["watch"]).tolist()
    assert after.tables["watch"]["src_server_received_ms"].dtype == np.int64
    assert np.diff(after.tables["pen"]["src_timestamp"]) == pytest.approx(12.0)
    payload = json.loads(after.tables["markers"]["src_payload"].iloc[0])
    assert payload == {"session_start_t_ms": 1000, "phase": "a"}


def test_epoch_columns_finds_a_clock_nobody_listed():
    df = pd.DataFrame({
        "src_new_clock_ms": [1_778_855_597_578.0],
        "src_uptime_ms": [1_500_000_000.0],  # 17 days of uptime, not a date
        "src_payload": [json.dumps({"nested": {"sent_ms": 1_778_855_597_578}})],
    })
    assert epoch_columns(df) == ["src_new_clock_ms", "src_payload"]


def test_rebase_rejects_an_unknown_origin():
    with pytest.raises(ValueError, match="unknown time origin"):
        rebase_bundle(_bundle(), "first_sample")


def test_payload_fields_already_relative_to_the_session_stay_as_they_are():
    bundle = _bundle()
    bundle.tables["markers"]["src_payload"] = [json.dumps({"session_start_t_ms": 2.0})]
    out = rebase_bundle(bundle, TimeOrigin.RECORDING_START)

    assert json.loads(out.tables["markers"]["src_payload"].iloc[0]) == {"session_start_t_ms": 2.0}
