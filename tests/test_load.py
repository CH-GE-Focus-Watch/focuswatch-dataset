import numpy as np
import pandas as pd
import pytest

from focuswatch_dataset import select
from focuswatch_dataset.load import load_channels, load_manifest, load_recording, load_recordings
from focuswatch_dataset.write import write_table


def build_bundle(root):
    rows = []
    for rid, sem, hz, grav in [("A", "user", 100.0, True), ("B", "user", 50.0, False),
                               ("C", "total", 100.0, False)]:
        col = f"accel_{sem}_x"
        df = pd.DataFrame({"t_ns": np.arange(50, dtype=np.int64) * 10_000_000,
                           col: np.zeros(50)})
        write_table(df, root / "watch" / f"{rid}.parquet", rid)
        rows.append({"recording_id": rid, "has_watch": True, "has_pen": rid != "C",
                     "watch_hz_nominal": hz, "has_gravity": grav, "accel_semantics": sem})
    m = pd.DataFrame(rows)
    m.to_parquet(root / "sessions.parquet", index=False)
    m.to_csv(root / "sessions.csv", index=False)
    return root


def test_load_manifest(tmp_path):
    build_bundle(tmp_path)
    assert len(load_manifest(tmp_path)) == 3


def test_load_recording(tmp_path):
    build_bundle(tmp_path)
    assert len(load_recording(tmp_path, "A")) == 50


def test_load_channels(tmp_path):
    build_bundle(tmp_path)
    channels = pd.DataFrame([
        {"recording_id": "A", "modality": "watch", "column": "accel_user_x",
         "quantity": "accel_user", "unit": "g", "semantics": "accel_user",
         "frame": "device", "sample_rate_hz": 100.0, "unit_conversion_factor": 1.0},
    ])
    channels.to_parquet(tmp_path / "channels.parquet", index=False)
    got = load_channels(tmp_path)
    assert got["column"].tolist() == ["accel_user_x"]


def test_by_flags_filters(tmp_path):
    build_bundle(tmp_path)
    m = load_manifest(tmp_path)
    got = select.by_flags(m, has_watch=True, has_pen=True, watch_hz_nominal=100.0)
    assert got["recording_id"].tolist() == ["A"]


def test_query_string_works_too(tmp_path):
    build_bundle(tmp_path)
    m = load_manifest(tmp_path)
    assert m.query("has_watch and has_gravity")["recording_id"].tolist() == ["A"]


def test_load_recordings_concatenates_and_tags(tmp_path):
    build_bundle(tmp_path)
    out = load_recordings(tmp_path, ["A", "B"])
    assert len(out) == 100
    assert set(out["recording_id"]) == {"A", "B"}


def test_load_recordings_refuses_mixed_accel_semantics(tmp_path):
    build_bundle(tmp_path)
    with pytest.raises(ValueError, match="mixed acceleration semantics"):
        load_recordings(tmp_path, ["A", "C"])


# --- mutation (a): a loader that ignores the requested id --------------------
# build_bundle's own A/B fixtures are both all-zero "user" accel and would not
# distinguish a loader that always returns the first recording on disk from a
# correct one. This fixture gives each recording distinct content instead.

def build_distinct_bundle(root):
    rows = []
    for rid, value in [("X", 1.0), ("Y", 2.0), ("Z", 3.0)]:
        df = pd.DataFrame({"t_ns": np.arange(10, dtype=np.int64) * 10_000_000,
                           "accel_user_x": np.full(10, value)})
        write_table(df, root / "watch" / f"{rid}.parquet", rid)
        rows.append({"recording_id": rid, "has_watch": True, "accel_semantics": "user"})
    pd.DataFrame(rows).to_parquet(root / "sessions.parquet", index=False)
    return root


def test_load_recording_returns_the_requested_recording(tmp_path):
    build_distinct_bundle(tmp_path)
    got = load_recording(tmp_path, "Y")
    assert (got["accel_user_x"] == 2.0).all()
    assert not (got["accel_user_x"] == 1.0).any()


def test_load_recordings_tags_match_their_own_content(tmp_path):
    build_distinct_bundle(tmp_path)
    out = load_recordings(tmp_path, ["X", "Z"], modality="watch")
    by_tag = out.set_index("recording_id")["accel_user_x"]
    assert (by_tag.loc["X"] == 1.0).all()
    assert (by_tag.loc["Z"] == 3.0).all()


# --- mutation (b): a filter that silently matches everything -----------------

def test_by_flags_matching_nothing_is_empty_not_the_whole_manifest(tmp_path):
    build_bundle(tmp_path)
    m = load_manifest(tmp_path)
    got = select.by_flags(m, has_pen=True, watch_hz_nominal=999.0)
    assert len(got) == 0


def test_by_flags_unknown_column_raises(tmp_path):
    build_bundle(tmp_path)
    m = load_manifest(tmp_path)
    with pytest.raises(KeyError, match="unknown manifest column"):
        select.by_flags(m, not_a_real_flag=True)


# --- error-message quality ----------------------------------------------------

def test_load_recording_missing_modality_names_the_flag(tmp_path):
    build_bundle(tmp_path)
    # C is declared has_pen=False in the manifest and no pen/ file was ever
    # written for it - the message should point at that flag, not just say
    # the file is missing.
    with pytest.raises(FileNotFoundError, match=r"has_pen is false"):
        load_recording(tmp_path, "C", modality="pen")


def test_load_recording_unknown_modality_raises(tmp_path):
    build_bundle(tmp_path)
    with pytest.raises(ValueError, match="unknown modality"):
        load_recording(tmp_path, "A", modality="Watch")


def test_load_recordings_unknown_id_raises(tmp_path):
    build_bundle(tmp_path)
    with pytest.raises(KeyError, match=r"\['NOPE'\]"):
        load_recordings(tmp_path, ["A", "NOPE"])


def test_load_recordings_unknown_id_points_at_load_manifest(tmp_path):
    build_bundle(tmp_path)
    with pytest.raises(KeyError, match=r"load_manifest\(root\)\['recording_id'\]"):
        load_recordings(tmp_path, ["NOPE"])


# --- fix round 1: empty recording_ids ----------------------------------------
# Pre-fix, this reached pd.concat([]) and raised pandas' own bare
# "ValueError: No objects to concatenate" - no ids, no corpus path, no hint
# the input was empty. This is the natural path (a by_flags()/query() filter
# that matched nothing), not misuse, so it must name what happened.

def test_load_recordings_empty_ids_names_the_input_and_corpus(tmp_path):
    build_bundle(tmp_path)
    with pytest.raises(ValueError, match="recording_ids is empty") as exc_info:
        load_recordings(tmp_path, [])
    assert "sessions.parquet" in str(exc_info.value)
    assert "No objects to concatenate" not in str(exc_info.value)


def test_load_recordings_empty_ids_from_a_matching_nothing_filter(tmp_path):
    # The realistic trigger: a by_flags() filter matches nothing, and its
    # empty recording_id column is piped straight into load_recordings.
    build_bundle(tmp_path)
    m = load_manifest(tmp_path)
    got = select.by_flags(m, watch_hz_nominal=999.0)
    with pytest.raises(ValueError, match="recording_ids is empty"):
        load_recordings(tmp_path, got["recording_id"].tolist())


# --- fix round 1: diagnostic except must survive a corrupted manifest --------
# A truncated/corrupted sessions.parquet (a realistic partial-download state)
# raises pyarrow.lib.ArrowInvalid from _missing_table_reason's best-effort
# load_manifest() call, not an OSError - the original narrow except let that
# secondary failure replace the FileNotFoundError it was trying to improve.

def test_load_recording_missing_table_survives_a_corrupt_manifest(tmp_path):
    (tmp_path / "sessions.parquet").write_bytes(b"not actually parquet")
    with pytest.raises(FileNotFoundError, match=r"has no watch table"):
        load_recording(tmp_path, "A")


def test_load_recordings_semantics_guard_is_scoped_to_watch(tmp_path):
    # Two recordings disagree on watch accel_semantics ("user" vs "total"),
    # but pen tables carry no acceleration at all - the guard must not block
    # a pen-only pull just because the watch column happens to differ.
    root = build_bundle(tmp_path)
    for rid in ("A", "C"):
        pen = pd.DataFrame({"t_ns": np.arange(5, dtype=np.int64), "x": np.zeros(5)})
        write_table(pen, root / "pen" / f"{rid}.parquet", rid)
    out = load_recordings(root, ["A", "C"], modality="pen")
    assert set(out["recording_id"]) == {"A", "C"}


# --- sample-rate and pen-unit guards ------------------------------------------

def build_rate_bundle(root, rates):
    """Watch tables of equal semantics whose channels declare `rates`."""
    rows, channels = [], []
    for rid, hz in rates.items():
        df = pd.DataFrame({"t_ns": np.arange(10, dtype=np.int64), "accel_user_x": np.zeros(10)})
        write_table(df, root / "watch" / f"{rid}.parquet", rid)
        rows.append({"recording_id": rid, "has_watch": True, "accel_semantics": "user"})
        channels.append({"recording_id": rid, "modality": "watch",
                         "column": "accel_user_x", "sample_rate_hz": hz})
    pd.DataFrame(rows).to_parquet(root / "sessions.parquet", index=False)
    pd.DataFrame(channels).to_parquet(root / "channels.parquet", index=False)
    return root


def test_load_recordings_refuses_mixed_sample_rates(tmp_path):
    build_rate_bundle(tmp_path, {"A": 100.0, "B": 50.0})
    with pytest.raises(ValueError, match=r"mixed watch sample rates \{50.0: \['B'\], 100.0: \['A'\]\}"):
        load_recordings(tmp_path, ["A", "B"])


def test_load_recordings_tolerates_measured_rate_jitter(tmp_path):
    # The AirPods head IMU spans 27.0-28.6 Hz for one nominal stream.
    build_rate_bundle(tmp_path, {"A": 27.03, "B": 28.57})
    assert len(load_recordings(tmp_path, ["A", "B"])) == 20


def test_load_recordings_mixed_rates_opt_in_tags_rows(tmp_path):
    build_rate_bundle(tmp_path, {"A": 100.0, "B": 50.0})
    out = load_recordings(tmp_path, ["A", "B"], allow_mixed_rates=True)
    assert out.groupby("recording_id")["sample_rate_hz"].first().to_dict() == {"A": 100.0, "B": 50.0}


def test_load_recordings_lists_a_recording_twice_without_breaking_the_guards(tmp_path):
    build_rate_bundle(tmp_path, {"A": 100.0})
    out = load_recordings(tmp_path, ["A", "A"], allow_mixed_rates=True)
    assert len(out) == 20
    assert set(out["sample_rate_hz"]) == {100.0}


def build_pen_bundle(root, units):
    rows = []
    for rid, (xy, pressure) in units.items():
        pen = pd.DataFrame({"t_ns": np.arange(5, dtype=np.int64), "x": np.zeros(5)})
        write_table(pen, root / "pen" / f"{rid}.parquet", rid)
        rows.append({"recording_id": rid, "has_pen": True,
                     "pen_xy_unit": xy, "pen_pressure_scale": pressure})
    pd.DataFrame(rows).to_parquet(root / "sessions.parquet", index=False)
    return root


def test_load_recordings_refuses_mixed_pen_units(tmp_path):
    build_pen_bundle(tmp_path, {"A": ("ncode_grid", "ncode_force"),
                                "B": ("webapp_raw", "webapp_force")})
    with pytest.raises(ValueError, match=r"mixed pen_xy_unit \{'ncode_grid': \['A'\], 'webapp_raw': \['B'\]\}"):
        load_recordings(tmp_path, ["A", "B"], modality="pen")


def test_load_recordings_mixed_pen_units_opt_in_tags_rows(tmp_path):
    build_pen_bundle(tmp_path, {"A": ("ncode_grid", "ncode_force"),
                                "B": ("webapp_raw", "webapp_force")})
    out = load_recordings(tmp_path, ["A", "B"], modality="pen", allow_mixed_pen_units=True)
    units = out.groupby("recording_id")[["pen_xy_unit", "pen_pressure_scale"]].first()
    assert units.loc["B"].tolist() == ["webapp_raw", "webapp_force"]


def test_load_recordings_same_pen_units_need_no_opt_in(tmp_path):
    build_pen_bundle(tmp_path, {"A": ("ncode_grid", "ncode_force"),
                                "B": ("ncode_grid", "ncode_force")})
    out = load_recordings(tmp_path, ["A", "B"], modality="pen")
    assert "pen_xy_unit" not in out.columns
