import math

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.transform import Rotation

from focuswatch_dataset import schema as S
from focuswatch_dataset.adapters.ml4scs import Ml4scsAdapter
from focuswatch_dataset.manifest import build_channels, build_manifest
from focuswatch_dataset.validate import validate_motion_table


def write_fixture(root, sid="S096", n=500, fs=100.0, gravity=True,
                   gravity_all_nan=False, alignment_sigma=-3.3):
    rng = np.random.default_rng(0)
    rots = Rotation.random(n, random_state=0)
    g = rots.inv().apply([0.0, 0.0, -1.0])
    ts = 1780577357025 + (np.arange(n) * (1000 / fs)).astype(np.int64)
    watch = pd.DataFrame({
        "local_ts_ms": ts + 40, "session_id": sid, "sequence": np.arange(n),
        "sample_rate_hz": fs, "server_received_ms": ts + 50, "source": "watch",
        "phone_received_at": ts + 45,
        "ts": ts,
        "ax": rng.normal(0, 0.04, n), "ay": rng.normal(0, 0.04, n), "az": rng.normal(0, 0.04, n),
        "rx": rng.normal(0, 0.15, n), "ry": rng.normal(0, 0.15, n), "rz": rng.normal(0, 0.15, n),
    })
    if gravity:
        watch[["gx", "gy", "gz"]] = g
        watch[["qx", "qy", "qz", "qw"]] = rots.as_quat()
    elif gravity_all_nan:
        # Why: a present-but-entirely-null column, distinct from an absent one -
        # exercises the `.notna().any()` branch of the mapping skip, not the
        # `src in raw.columns` branch.
        watch[["gx", "gy", "gz"]] = np.nan
    (root / "watch").mkdir(parents=True, exist_ok=True)
    watch.to_csv(root / "watch" / f"{sid}_watch.csv", index=False)

    pen = pd.DataFrame({
        "local_ts_ms": ts[:20], "timestamp": ts[:20] - 79_660_800_000,
        "x": np.r_[np.linspace(1, 5, 19), -1.0], "y": np.r_[np.linspace(1, 5, 19), -1.0],
        "pressure": 300, "dot_type": ["PEN_DOWN"] + ["PEN_MOVE"] * 18 + ["PEN_UP"],
        "tilt_x": 90, "tilt_y": 100, "section": 0, "owner": 0, "note": 0, "page": 1,
    })
    (root / "pen").mkdir(parents=True, exist_ok=True)
    pen.to_csv(root / "pen" / f"{sid}_pen.csv", index=False)

    markers = pd.DataFrame({
        "timestamp_ms": [ts[0], ts[-1]], "event": ["task_start", "task_end"],
        "task_id": "abschreiben", "task_name": "Abschreiben", "task_index": 0,
        "task_category": "writing", "protocol_id": "v2",
    })
    (root / "markers").mkdir(parents=True, exist_ok=True)
    markers.to_csv(root / "markers" / f"{sid}_markers.csv", index=False)

    session_row = {
        "session_id": sid, "person_id": "P76", "study_mode": "study", "protocol_id": "v2",
        "subject_index": 3, "watch_profile": "100hz_grav" if gravity else "50hz",
        "start_time": pd.Timestamp(int(ts[0]), unit="ms", tz="UTC").isoformat(),
    }
    if alignment_sigma is not None:
        session_row["alignment_sigma"] = alignment_sigma
    pd.DataFrame([session_row]).to_csv(root / "sessions.csv", index=False)
    return root


def test_discover_finds_the_recording(tmp_path):
    write_fixture(tmp_path)
    refs = Ml4scsAdapter().discover(tmp_path)
    assert [r.recording_id for r in refs] == ["ML4SCS-S096"]
    assert refs[0].participant_id == "ML4SCS-P76"


def test_watch_columns_are_canonical_and_semantic(tmp_path):
    write_fixture(tmp_path)
    a = Ml4scsAdapter()
    watch = a.load(a.discover(tmp_path)[0]).tables["watch"]
    for c in S.COLUMNS[S.Quantity.ACCEL_USER] + S.COLUMNS[S.Quantity.GYRO]:
        assert c in watch.columns
    assert "ax" not in watch.columns
    assert "accel_total_x" not in watch.columns
    assert watch["t_ns"].dtype == np.int64


def test_time_axis_uses_capture_clock_not_arrival(tmp_path):
    write_fixture(tmp_path)
    a = Ml4scsAdapter()
    ref = a.discover(tmp_path)[0]
    watch = a.load(ref).tables["watch"]
    raw = pd.read_csv(tmp_path / "watch" / "S096_watch.csv")
    assert watch["t_ns"].iloc[0] == int(raw["ts"].iloc[0]) * 1_000_000


def test_loaded_watch_passes_the_validator(tmp_path):
    write_fixture(tmp_path)
    a = Ml4scsAdapter()
    bundle = a.load(a.discover(tmp_path)[0])
    failed = [f for f in validate_motion_table(bundle.tables["watch"], "ML4SCS-S096", "watch", 100.0)
              if not f.passed]
    assert failed == []


def test_legacy_recording_has_no_gravity_columns(tmp_path):
    write_fixture(tmp_path, sid="S008", fs=50.0, gravity=False)
    a = Ml4scsAdapter()
    ref = next(r for r in a.discover(tmp_path) if r.recording_id == "ML4SCS-S008")
    bundle = a.load(ref)
    assert "gravity_x" not in bundle.tables["watch"].columns
    assert bundle.meta["has_gravity"] is False
    assert bundle.meta["watch_hz_nominal"] == 50.0
    # Why: quaternion capture is forward-only, governed by the same skip logic as
    # gravity - a legacy session must lose both together, not just the first one.
    assert "quat_x" not in bundle.tables["watch"].columns
    assert bundle.meta["has_quaternion"] is False


def test_gravity_column_present_but_entirely_null_is_dropped(tmp_path):
    write_fixture(tmp_path, sid="S050", gravity=False, gravity_all_nan=True)
    # Confirm the fixture actually reaches the "present but null" branch, not the
    # "absent" one: the raw CSV must carry gx/gy/gz with every value NaN.
    raw = pd.read_csv(tmp_path / "watch" / "S050_watch.csv")
    assert "gx" in raw.columns
    assert raw["gx"].isna().all()

    a = Ml4scsAdapter()
    ref = next(r for r in a.discover(tmp_path) if r.recording_id == "ML4SCS-S050")
    bundle = a.load(ref)
    assert "gravity_x" not in bundle.tables["watch"].columns
    assert bundle.meta["has_gravity"] is False


def test_pen_framing_sentinel_survives(tmp_path):
    write_fixture(tmp_path)
    a = Ml4scsAdapter()
    pen = a.load(a.discover(tmp_path)[0]).tables["pen"]
    assert (pen["x"] == -1.0).sum() == 1


def test_pen_delta_sigma_reaches_metadata(tmp_path):
    write_fixture(tmp_path, alignment_sigma=-3.3)
    a = Ml4scsAdapter()
    bundle = a.load(a.discover(tmp_path)[0])
    assert bundle.meta["pen_delta_sigma"] == pytest.approx(-3.3)


def test_pen_and_markers_are_declared_on_the_server_clock_not_the_watch_clock(tmp_path):
    """C2 part 1: watch samples are timestamped from `ts` (the watch's own
    capture clock); pen comes from `local_ts_ms` and markers from
    `timestamp_ms` - both the SERVER clock, a different clock entirely (see
    _watch/_pen/_markers). Declaring one flat time_domain for the whole
    recording (the pre-fix bug) silently mislabels pen/marker timestamps as
    if they shared the watch's clock. channels.parquet's per-(recording,
    modality) time_domain column must disagree between watch and pen/markers.
    """
    write_fixture(tmp_path)
    a = Ml4scsAdapter()
    bundle = a.load(a.discover(tmp_path)[0])
    ch = build_channels([bundle]).set_index(["modality", "column"])["time_domain"]

    watch_domain = ch.loc[("watch", "t_ns")]
    pen_domain = ch.loc[("pen", "t_ns")]
    markers_domain = ch.loc[("markers", "t_ns")]
    assert watch_domain == "watch_capture_clock"
    assert pen_domain == markers_domain == "server_wall_clock"
    assert watch_domain != pen_domain


def test_src_provenance_columns_are_declared_on_the_clock_they_are_actually_on(tmp_path):
    """Item 1 (fix round C): C2 fixed the MODALITY-level clock (previous test)
    but build_channels then stamped that same modality clock onto every
    src_-prefixed provenance column too - measured false on the real corpus
    (whole-branch-review-2.md finding 1): pen.src_timestamp is the raw
    Moleskine device clock, ~923 days off server_wall_clock; watch.src_
    local_ts_ms/src_server_received_ms are server-stamped, not the watch
    capture clock the rest of watch/ is on; src_phone_received_at is a third
    device's clock; src_sequence is not a clock at all.
    """
    write_fixture(tmp_path)
    a = Ml4scsAdapter()
    bundle = a.load(a.discover(tmp_path)[0])
    ch = build_channels([bundle]).set_index(["modality", "column"])["time_domain"]

    assert ch.loc[("watch", "src_local_ts_ms")] == "server_wall_clock"
    assert ch.loc[("watch", "src_server_received_ms")] == "server_wall_clock"
    assert ch.loc[("watch", "src_phone_received_at")] == "phone_wall_clock"
    assert ch.loc[("watch", "src_sequence")] == "not_a_clock"
    assert ch.loc[("pen", "src_timestamp")] == "pen_device_clock"
    assert ch.loc[("pen", "src_t_session_ms")] == "session_relative_offset_ms"
    assert ch.loc[("markers", "src_t_session_ms")] == "session_relative_offset_ms"
    # The modality default is unaffected for every other watch/ column - the
    # override is scoped to the specific (modality, column) pairs above.
    assert ch.loc[("watch", "t_ns")] == "watch_capture_clock"
    assert ch.loc[("watch", "accel_user_x")] == "watch_capture_clock"


def test_alignment_note_is_populated_for_the_estimated_delta_cohort(tmp_path):
    """C2 part 2: alignment_note was always "" (manifest.py's only mention of
    it is the default) - a reuser had no explicit warning that pen/marker
    timestamps sit on a different, unpublished-offset clock. Every
    estimated_delta recording (currently ML4SCS only) must get a non-empty
    note naming the differing modalities, and pen_delta_s stays NaN (D5:
    publishing an estimated delta risks irreparable mislabeling).
    """
    write_fixture(tmp_path)
    a = Ml4scsAdapter()
    bundle = a.load(a.discover(tmp_path)[0])
    row = build_manifest([bundle]).iloc[0]

    assert row["time_alignment"] == "estimated_delta"
    assert row["alignment_note"] != ""
    assert "pen" in row["alignment_note"] and "markers" in row["alignment_note"]
    assert "pen_delta_sigma" in row["alignment_note"]
    assert math.isnan(row["pen_delta_s"])


def test_missing_alignment_sigma_yields_nan_not_a_baked_in_delta(tmp_path):
    write_fixture(tmp_path, alignment_sigma=None)
    sessions = pd.read_csv(tmp_path / "sessions.csv")
    assert "alignment_sigma" not in sessions.columns
    a = Ml4scsAdapter()
    bundle = a.load(a.discover(tmp_path)[0])
    assert math.isnan(bundle.meta["pen_delta_sigma"])
    # Why: the offset itself must never be estimated here - only its confidence.
    assert "pen_delta_s" not in bundle.meta


def test_discover_raises_when_the_session_row_is_missing(tmp_path):
    write_fixture(tmp_path)
    sessions = pd.read_csv(tmp_path / "sessions.csv")
    sessions[sessions["session_id"] != "S096"].to_csv(tmp_path / "sessions.csv", index=False)
    with pytest.raises(ValueError, match="S096 has no row in sessions.csv"):
        Ml4scsAdapter().discover(tmp_path)


def test_discover_raises_when_person_id_is_empty(tmp_path):
    write_fixture(tmp_path)
    sessions = pd.read_csv(tmp_path / "sessions.csv")
    sessions["person_id"] = ""
    sessions.to_csv(tmp_path / "sessions.csv", index=False)
    with pytest.raises(ValueError, match="S096 has no person_id in sessions.csv"):
        Ml4scsAdapter().discover(tmp_path)


def test_load_raises_when_the_session_row_disappears_after_discover(tmp_path):
    write_fixture(tmp_path)
    ref = Ml4scsAdapter().discover(tmp_path)[0]
    sessions = pd.read_csv(tmp_path / "sessions.csv")
    sessions[sessions["session_id"] != "S096"].to_csv(tmp_path / "sessions.csv", index=False)
    with pytest.raises(ValueError, match="S096 has no row in sessions.csv"):
        Ml4scsAdapter().load(ref)
