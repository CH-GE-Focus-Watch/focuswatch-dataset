import numpy as np
import pandas as pd
import pytest
from scipy.spatial.transform import Rotation

from focuswatch_dataset import schema as S
from focuswatch_dataset.adapters.airpods import AirPodsAdapter, parse_ground_truth
from focuswatch_dataset.validate import check_coverage, validate_motion_table, validate_recording

GT = "0:00 focused\n4:00 distracted\n9:30 focused\n"


def write_fixture(root, pid="P1", n=1500, fs=25.0):
    rng = np.random.default_rng(3)
    rots = Rotation.random(n, random_state=3)
    q, grav = rots.as_quat(), rots.inv().apply([0.0, 0.0, -1.0])
    t0 = pd.Timestamp("2026-04-28T12:29:29.515Z")
    iso = (t0 + pd.to_timedelta(np.arange(n) / fs, unit="s")).strftime("%Y-%m-%dT%H:%M:%S.%f").str[:-3] + "Z"
    labels = np.where(np.arange(n) / fs < 240, "focused", "distracted")
    pd.DataFrame({
        "timestamp_iso": iso, "sensor_timestamp_s": 6679.42 + np.arange(n) / fs,
        "attitude_roll_rad": 0.0, "attitude_pitch_rad": 0.0, "attitude_yaw_rad": 0.0,
        "quaternion_x": q[:, 0], "quaternion_y": q[:, 1], "quaternion_z": q[:, 2], "quaternion_w": q[:, 3],
        "rotation_rate_x_rad_s": rng.normal(0, 0.04, n),
        "rotation_rate_y_rad_s": rng.normal(0, 0.04, n),
        "rotation_rate_z_rad_s": rng.normal(0, 0.04, n),
        "gravity_x_g": grav[:, 0], "gravity_y_g": grav[:, 1], "gravity_z_g": grav[:, 2],
        "user_acceleration_x_g": rng.normal(0, 0.015, n),
        "user_acceleration_y_g": rng.normal(0, 0.015, n),
        "user_acceleration_z_g": rng.normal(0, 0.015, n),
        "label": labels,
    }).to_csv(root / f"{pid}_17m30s_airpod_motion_2026-04-28_14-47-30_labeled.csv", index=False)
    gt = root / "1_Data_Protocols+GroundTruth"
    gt.mkdir(exist_ok=True)
    (gt / f"{pid}_ground_truth_2026-04-28_13-40-36.txt").write_text(GT)
    return root


def test_parse_ground_truth():
    assert parse_ground_truth(GT) == [(0.0, "focused"), (240.0, "distracted"), (570.0, "focused")]


def test_discover_prefixes_the_participant_id(tmp_path):
    write_fixture(tmp_path, pid="P17")
    ref = AirPodsAdapter().discover(tmp_path)[0]
    # Why: AirPods P17 and ML4SCS P17 are different people.
    assert ref.recording_id == "AIRPODS-P17"
    assert ref.participant_id == "AIRPODS-P17"


def test_attention_table_holds_intervals_not_samples(tmp_path):
    write_fixture(tmp_path)
    a = AirPodsAdapter()
    att = a.load(a.discover(tmp_path)[0]).tables["attention"]
    assert list(att.columns) == ["t_start_ns", "t_end_ns", "label"]
    assert len(att) == 3
    assert att["t_end_ns"].iloc[0] - att["t_start_ns"].iloc[0] == 240 * 1_000_000_000


def test_per_sample_label_is_dropped(tmp_path):
    write_fixture(tmp_path)
    a = AirPodsAdapter()
    head = a.load(a.discover(tmp_path)[0]).tables["headimu"]
    assert "label" not in head.columns


def test_expansion_crosscheck_rejects_a_wrong_clock_anchor(tmp_path):
    # The interval offsets are relative to an unstated clock. Before the
    # per-sample column is dropped it is used to prove the anchor.
    write_fixture(tmp_path)
    gt = next((tmp_path / "1_Data_Protocols+GroundTruth").glob("*.txt"))
    gt.write_text("0:00 distracted\n4:00 focused\n9:30 distracted\n")   # inverted
    a = AirPodsAdapter()
    with pytest.raises(ValueError, match="anchor is wrong"):
        a.load(a.discover(tmp_path)[0])


def test_no_watch_table_and_flags_say_so(tmp_path):
    write_fixture(tmp_path)
    a = AirPodsAdapter()
    bundle = a.load(a.discover(tmp_path)[0])
    assert "watch" not in bundle.tables
    assert bundle.meta["has_watch"] is False
    assert bundle.meta["has_attention"] is True


def test_measured_head_rate_is_reported(tmp_path):
    write_fixture(tmp_path, fs=25.0)
    a = AirPodsAdapter()
    bundle = a.load(a.discover(tmp_path)[0])
    assert bundle.meta["head_hz_measured"] == pytest.approx(25.0, rel=0.05)


def test_loaded_headimu_passes_the_validator(tmp_path):
    write_fixture(tmp_path)
    a = AirPodsAdapter()
    head = a.load(a.discover(tmp_path)[0]).tables["headimu"]
    assert [f for f in validate_motion_table(head, "AIRPODS-P1", "headimu", 25.0)
            if not f.passed] == []


# --- Coverage-matrix consistency ------------------------------------------------
#
# The manifest flags this adapter emits (has_headimu, has_attention, has_gravity,
# has_quaternion, ...) are what feeds check_coverage - a flag that says "true"
# without the matching table/columns actually present would let the coverage
# matrix silently wave a missing physical check through. This test runs the
# real bundle through the real gate end to end, not just the adapter's own
# opinion of itself.

def test_coverage_matrix_agrees_with_the_real_bundle(tmp_path):
    write_fixture(tmp_path)
    a = AirPodsAdapter()
    ref = a.discover(tmp_path)[0]
    bundle = a.load(ref)

    manifest = pd.DataFrame([{
        "recording_id": ref.recording_id,
        "has_watch": bundle.meta["has_watch"],
        "has_watch_rawaccel": bundle.meta["has_watch_rawaccel"],
        "has_headimu": bundle.meta["has_headimu"],
        "has_pen": bundle.meta["has_pen"],
        "has_markers": bundle.meta["has_markers"],
        "has_attention": bundle.meta["has_attention"],
        "has_gravity": bundle.meta["has_gravity"],
        "has_quaternion": bundle.meta["has_quaternion"],
    }])
    findings = validate_motion_table(bundle.tables["headimu"], ref.recording_id, "headimu",
                                     bundle.meta["head_hz_nominal"])
    findings += validate_recording(ref.recording_id, bundle.tables, bundle.meta)
    assert check_coverage(manifest, findings) == []


def test_coverage_matrix_flags_a_stale_has_attention():
    """A manifest that claims has_attention while no attention (or any other)
    table ever produced a time_magnitude finding must be rejected - this is
    the check that catches a flag drifting from the data it describes.

    Isolated from a full bundle deliberately: check_coverage's time_magnitude
    requirement is per-recording, not per-modality (any table's finding
    satisfies it), so a headimu table sitting alongside a dropped attention
    table would mask exactly the drift this test needs to expose.
    """
    manifest = pd.DataFrame([{
        "recording_id": "AIRPODS-P1",
        "has_watch": False, "has_watch_rawaccel": False, "has_headimu": False,
        "has_pen": False, "has_markers": False,
        "has_attention": True,                     # stale: nothing below backs it
        "has_gravity": False, "has_quaternion": False,
    }])
    problems = check_coverage(manifest, [])
    assert any("time_magnitude" in p for p in problems)
