import numpy as np
import pandas as pd
import pytest
from scipy.spatial.transform import Rotation

from focuswatch_dataset import schema as S
from focuswatch_dataset.adapters.airpods import AirPodsAdapter, parse_ground_truth, parse_protocol
from focuswatch_dataset.manifest import build_channels, build_manifest
from focuswatch_dataset.validate import (
    check_coverage, validate_attention_table, validate_motion_table, validate_recording,
)

GT = "0:00 focused\n4:00 distracted\n9:30 focused\n"
# Why (I13/I14): mirrors GT's own three offsets/labels exactly (0:00/4:00/9:30,
# focused/distracted/focused) - the boundary cross-check in
# _activities_and_tail requires that. Total block span (10:00 = 600s) sits
# below the default fixture's ~619.96s recording length, matching the
# measured corpus direction (recording outlasts protocol, tail > 0).
PROTOCOL = (
    "  1 |   0:00 |   4:00 |   4:00 | focused     | Abschreiben eines Textes vom Laptop\n"
    "  2 |   4:00 |   9:30 |   5:30 | distracted  | Handy benutzen\n"
    "  3 |   9:30 |  10:00 |   0:30 | focused     | Loesen von Matheaufgaben\n"
    "Gesamtdauer: 10:00\n"
)


def write_fixture(root, pid="P1", n=15500, fs=25.0, with_protocol=True):
    rng = np.random.default_rng(3)
    # Why: small perturbations around an upright head orientation, not a full
    # SO(3)-random sample - a head never tips upside down during a recording
    # session, and a full-sphere sample makes the validator's gravity_sign
    # "flat" subset genuinely bimodal (roughly half the flat-classified
    # samples land near +1, half near -1), so its median becomes an
    # unstable coin flip on n and seed rather than a real physical signal.
    rots = Rotation.from_rotvec(rng.normal(0, 0.3, (n, 3)))
    q, grav = rots.as_quat(), rots.inv().apply([0.0, 0.0, -1.0])
    t0 = pd.Timestamp("2026-04-28T12:29:29.515Z")
    t_rel = np.arange(n) / fs
    iso = (t0 + pd.to_timedelta(t_rel, unit="s")).strftime("%Y-%m-%dT%H:%M:%S.%f").str[:-3] + "Z"
    # Why: mirrors parse_ground_truth(GT)'s three segments (0-240 focused,
    # 240-570 distracted, 570+ focused), not just the first - the default
    # duration now runs past the last GT offset (570 s) so every emitted
    # attention interval is well-formed (t_start_ns <= t_end_ns).
    labels = np.select([t_rel < 240, t_rel < 570], ["focused", "distracted"], default="focused")
    pd.DataFrame({
        "timestamp_iso": iso, "sensor_timestamp_s": 6679.42 + t_rel,
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
    if with_protocol:
        (gt / f"{pid}_protokoll_2026-04-28_13-40-36.txt").write_text(PROTOCOL)
    return root


def test_parse_ground_truth():
    assert parse_ground_truth(GT) == [(0.0, "focused"), (240.0, "distracted"), (570.0, "focused")]


def test_parse_protocol():
    assert parse_protocol(PROTOCOL) == [
        (0.0, 240.0, "focused", "Abschreiben eines Textes vom Laptop"),
        (240.0, 570.0, "distracted", "Handy benutzen"),
        (570.0, 600.0, "focused", "Loesen von Matheaufgaben"),
    ]


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
    assert list(att.columns) == ["t_start_ns", "t_end_ns", "label", "activity"]
    assert len(att) == 3
    assert att["t_end_ns"].iloc[0] - att["t_start_ns"].iloc[0] == 240 * 1_000_000_000


def test_src_sensor_timestamp_declares_the_devices_own_monotonic_clock(tmp_path):
    """Item 1 (fix round C): src_sensor_timestamp_s is CMDeviceMotion's own
    free-running clock (seconds since device boot), never reset to a
    wall-clock epoch - distinct from headimu's canonical t_ns axis (from
    timestamp_iso), which genuinely is on device_wall_clock. The modality
    default must still apply to t_ns itself.
    """
    write_fixture(tmp_path)
    a = AirPodsAdapter()
    bundle = a.load(a.discover(tmp_path)[0])
    ch = build_channels([bundle]).set_index(["modality", "column"])["time_domain"]

    assert ch.loc[("headimu", "src_sensor_timestamp_s")] == "device_monotonic_clock"
    assert ch.loc[("headimu", "t_ns")] == "device_wall_clock"


# --- I13: activity, parsed from the sibling protocol file ------------------

def test_activity_is_parsed_from_the_protocol_and_matched_by_start_time(tmp_path):
    write_fixture(tmp_path)
    a = AirPodsAdapter()
    att = a.load(a.discover(tmp_path)[0]).tables["attention"]
    assert att["activity"].tolist() == [
        "Abschreiben eines Textes vom Laptop", "Handy benutzen", "Loesen von Matheaufgaben",
    ]


def test_activity_is_empty_string_when_no_protocol_file_exists(tmp_path):
    """activity is always present on the table (I15's discipline: no column
    that only sometimes exists within the same modality), "" when there is
    nothing to report - never a missing column."""
    write_fixture(tmp_path, with_protocol=False)
    a = AirPodsAdapter()
    att = a.load(a.discover(tmp_path)[0]).tables["attention"]
    assert "activity" in att.columns
    assert (att["activity"] == "").all()


def test_protocol_boundary_mismatch_against_ground_truth_fails_loudly(tmp_path):
    write_fixture(tmp_path)
    gt = tmp_path / "1_Data_Protocols+GroundTruth"
    proto = next(gt.glob("*_protokoll_*.txt"))
    # Shift the second block's start away from the ground truth's 4:00.
    proto.write_text(PROTOCOL.replace("2 |   4:00 |   9:30 |   5:30 | distracted",
                                      "2 |   4:05 |   9:30 |   5:25 | distracted"))
    a = AirPodsAdapter()
    with pytest.raises(ValueError, match="the two files disagree"):
        a.load(a.discover(tmp_path)[0])


def test_protocol_label_mismatch_against_ground_truth_fails_loudly(tmp_path):
    write_fixture(tmp_path)
    gt = tmp_path / "1_Data_Protocols+GroundTruth"
    proto = next(gt.glob("*_protokoll_*.txt"))
    proto.write_text(PROTOCOL.replace("1 |   0:00 |   4:00 |   4:00 | focused",
                                      "1 |   0:00 |   4:00 |   4:00 | distracted"))
    a = AirPodsAdapter()
    with pytest.raises(ValueError, match="the two files disagree"):
        a.load(a.discover(tmp_path)[0])


def test_protocol_missing_a_labelled_run_fails_loudly(tmp_path):
    """Dropping a block whose label differs from its neighbour's removes a run,
    so the merge yields two entries against the ground truth's three."""
    write_fixture(tmp_path)
    gt = tmp_path / "1_Data_Protocols+GroundTruth"
    proto = next(gt.glob("*_protokoll_*.txt"))
    lines = PROTOCOL.splitlines()
    proto.write_text("\n".join(lines[:2] + [lines[-1]]) + "\n")
    a = AirPodsAdapter()
    with pytest.raises(ValueError, match="the two files disagree"):
        a.load(a.discover(tmp_path)[0])


def test_adjacent_same_label_blocks_stay_separate_rows(tmp_path):
    """The corpus shape: 22 of 26 ground-truth files hold FEWER intervals than
    their protocol holds blocks, because adjacent same-label blocks are merged
    there. The published table keeps the blocks, so each row carries exactly one
    activity; merging adjacent same-label rows reproduces the ground truth,
    which the reverse direction cannot do — it would lose the activities.
    """
    write_fixture(tmp_path)
    proto = next((tmp_path / "1_Data_Protocols+GroundTruth").glob("*_protokoll_*.txt"))
    # Split GT's closing `focused` run into two blocks with distinct activities.
    proto.write_text(
        "  1 |   0:00 |   4:00 |   4:00 | focused     | Abschreiben eines Textes vom Laptop\n"
        "  2 |   4:00 |   9:30 |   5:30 | distracted  | Handy benutzen\n"
        "  3 |   9:30 |   9:45 |   0:15 | focused     | Loesen von Matheaufgaben\n"
        "  4 |   9:45 |  10:00 |   0:15 | focused     | Arbeiten / Schreiben am Laptop\n"
        "Gesamtdauer: 10:00\n"
    )
    a = AirPodsAdapter()
    att = a.load(a.discover(tmp_path)[0]).tables["attention"]

    assert len(att) == 4, "four protocol blocks must survive as four rows"
    assert att["label"].tolist() == ["focused", "distracted", "focused", "focused"]
    assert att["activity"].tolist()[-2:] == [
        "Loesen von Matheaufgaben", "Arbeiten / Schreiben am Laptop"]

    runs = [lab for i, lab in enumerate(att["label"]) if i == 0 or lab != att["label"][i - 1]]
    assert runs == [lab for _, lab in parse_ground_truth(GT)]


def test_activity_is_ascii_verbatim_not_translated(tmp_path):
    """Umlaut-free ASCII already ("Loesen", not "Lösen") - published as-is,
    not translated: a translated label is no longer the observer's record."""
    write_fixture(tmp_path)
    a = AirPodsAdapter()
    att = a.load(a.discover(tmp_path)[0]).tables["attention"]
    assert "Loesen von Matheaufgaben" in att["activity"].tolist()


# --- I14: the observer timeline is checkable --------------------------------

def test_protocol_tail_is_reported_not_enforced(tmp_path):
    """The tail is a reported quantity, not a gate. Measured across the 25 real
    recordings it runs -0.47 s to +44.55 s (18 within a second); the negatives
    are sub-second stream ends, not late starts, and P14's +44.55 s is a real
    recording that must not fail a strict build."""
    write_fixture(tmp_path)
    a = AirPodsAdapter()
    bundle = a.load(a.discover(tmp_path)[0])
    assert bundle.meta["attention_protocol_tail_s"] == pytest.approx(19.96, abs=0.1)

    finding = next(f for f in validate_recording(bundle.ref.recording_id, bundle.tables, bundle.meta)
                   if f.check == "attention_protocol_tail")
    assert finding.observed == pytest.approx(19.96, abs=0.1)
    assert finding.passed is True   # informational - must never fail a strict build


def test_gesamtdauer_mismatch_against_block_end_fails_loudly(tmp_path):
    write_fixture(tmp_path)
    gt = tmp_path / "1_Data_Protocols+GroundTruth"
    proto = next(gt.glob("*_protokoll_*.txt"))
    proto.write_text(PROTOCOL.replace("Gesamtdauer: 10:00", "Gesamtdauer: 11:00"))
    a = AirPodsAdapter()
    with pytest.raises(ValueError, match="Gesamtdauer"):
        a.load(a.discover(tmp_path)[0])


def test_attention_intervals_fall_inside_headimu_range(tmp_path):
    write_fixture(tmp_path)
    a = AirPodsAdapter()
    bundle = a.load(a.discover(tmp_path)[0])
    finding = next(f for f in validate_recording(bundle.ref.recording_id, bundle.tables, bundle.meta)
                   if f.check == "attention_within_headimu_range")
    assert finding.passed is True


def test_per_sample_label_is_dropped(tmp_path):
    write_fixture(tmp_path)
    a = AirPodsAdapter()
    head = a.load(a.discover(tmp_path)[0]).tables["headimu"]
    assert "label" not in head.columns


def test_expansion_crosscheck_rejects_a_wrong_clock_anchor(tmp_path):
    # The interval offsets are relative to an unstated clock. Before the
    # per-sample column is dropped it is used to prove the anchor.
    write_fixture(tmp_path)
    d = tmp_path / "1_Data_Protocols+GroundTruth"
    next(d.glob("*_ground_truth_*.txt")).write_text(
        "0:00 distracted\n4:00 focused\n9:30 distracted\n")   # inverted
    # Why: the protocol has to be inverted with it. The two files are
    # cross-checked before the anchor is, so inverting only the ground truth
    # would trip that check first and this test would no longer exercise the
    # anchor it names.
    next(d.glob("*_protokoll_*.txt")).write_text(
        PROTOCOL.replace("| focused    ", "| TMP        ")
                .replace("| distracted ", "| focused    ")
                .replace("| TMP        ", "| distracted "))
    a = AirPodsAdapter()
    with pytest.raises(ValueError, match="anchor is wrong"):
        a.load(a.discover(tmp_path)[0])


def test_inverted_interval_fails_loudly_instead_of_publishing_malformed_data(tmp_path):
    """A stream shorter than its ground truth's last offset produces a closing
    interval whose end (clamped to the stream's own last sample) lands before
    its start. That must raise, not silently clamp or publish it - Task 16
    needs to hear about a genuinely short recording, not receive a malformed
    row."""
    write_fixture(tmp_path, n=1500, fs=25.0)   # 60 s of data; GT runs to 570 s
    a = AirPodsAdapter()
    with pytest.raises(ValueError, match="shorter than its protocol"):
        a.load(a.discover(tmp_path)[0])


def test_no_watch_table_and_flags_say_so(tmp_path):
    write_fixture(tmp_path)
    a = AirPodsAdapter()
    bundle = a.load(a.discover(tmp_path)[0])
    assert "watch" not in bundle.tables
    assert bundle.meta["has_watch"] is False
    assert bundle.meta["has_attention"] is True


def test_gravity_source_agrees_with_the_absent_watch_stream(tmp_path):
    """Fix-round-2 item 1: gravity_source is has_gravity's direct companion -
    with no watch table (has_watch False, no has_gravity key at all any
    more), it must not still claim "measured", which would openly
    contradict the same row."""
    write_fixture(tmp_path)
    a = AirPodsAdapter()
    bundle = a.load(a.discover(tmp_path)[0])
    assert bundle.meta["gravity_source"] == "none"


def test_accel_semantics_still_describes_the_single_motion_stream(tmp_path):
    """Fix-round-2 item 2: unlike gravity_source, accel_semantics is not
    blanked - docs/DESIGN.md's watch/-only scoping is a disambiguation rule
    for recordings with several accel streams, and AirPods has exactly one."""
    write_fixture(tmp_path)
    a = AirPodsAdapter()
    bundle = a.load(a.discover(tmp_path)[0])
    assert bundle.meta["accel_semantics"] == "user"
    assert bundle.meta["accel_calibration"] == "fused"


def test_measured_head_rate_is_reported(tmp_path):
    """Fix-round-1 item 1: the adapter no longer declares head_hz_measured in
    meta at all - manifest.build_manifest recomputes it from the headimu
    table for every source, so an adapter-level value would only ever be
    either redundant or (if it drifted) a silently-overridden lie. This test
    moved from asserting bundle.meta to asserting the manifest, matching the
    new ownership boundary."""
    write_fixture(tmp_path, fs=25.0)
    a = AirPodsAdapter()
    bundle = a.load(a.discover(tmp_path)[0])
    assert "head_hz_measured" not in bundle.meta
    manifest = build_manifest([bundle])
    assert manifest.iloc[0]["head_hz_measured"] == pytest.approx(25.0, rel=0.05)


def test_loaded_headimu_passes_the_validator(tmp_path):
    write_fixture(tmp_path)
    a = AirPodsAdapter()
    head = a.load(a.discover(tmp_path)[0]).tables["headimu"]
    assert [f for f in validate_motion_table(head, "AIRPODS-P1", "headimu", 25.0)
            if not f.passed] == []


# --- Coverage-matrix consistency ------------------------------------------------
#
# The manifest flags this adapter emits (has_headimu, has_attention,
# has_head_gravity, has_head_quaternion, ...) are what feeds check_coverage -
# a flag that says "true" without the matching table/columns actually present
# would let the coverage matrix silently wave a missing physical check
# through. This test runs the real bundle through the real gate end to end,
# not just the adapter's own opinion of itself.

def test_coverage_matrix_agrees_with_the_real_bundle(tmp_path):
    """Hollow test #4 (fix round): this used to hand-build the manifest from
    `bundle.meta` (see git history), the exact pattern the Ege and
    SensorLogger equivalents were already fixed away from - AirPods was the
    only adapter whose actual `build_manifest()` derivation path was never
    exercised by a coverage test. build_manifest(), not a hand-picked dict of
    flags: a hand-built manifest that forgets to name a flag (or, as here,
    that never calls the real derivation at all) makes check_coverage
    silently skip a requirement rather than fail it.
    """
    write_fixture(tmp_path)
    a = AirPodsAdapter()
    ref = a.discover(tmp_path)[0]
    bundle = a.load(ref)
    manifest = build_manifest([bundle])

    findings = validate_motion_table(bundle.tables["headimu"], ref.recording_id, "headimu",
                                     bundle.meta["head_hz_nominal"])
    findings += validate_attention_table(bundle.tables["attention"], ref.recording_id)
    findings += validate_recording(ref.recording_id, bundle.tables, bundle.meta)
    assert check_coverage(manifest, findings) == []


def test_coverage_matrix_flags_a_stale_has_attention():
    """A manifest that claims has_attention while no attention (or any other)
    table ever produced a time_magnitude finding must be rejected - this is
    the check that catches a flag drifting from the data it describes."""
    manifest = pd.DataFrame([{
        "recording_id": "AIRPODS-P1",
        "has_watch": False, "has_watch_rawaccel": False, "has_headimu": False,
        "has_pen": False, "has_markers": False,
        "has_attention": True,                     # stale: nothing below backs it
        "has_head_gravity": False, "has_head_quaternion": False,
    }])
    problems = check_coverage(manifest, [])
    assert any("time_magnitude" in p for p in problems)


def test_coverage_matrix_catches_a_stale_attention_flag_beside_a_real_headimu(tmp_path):
    """Regression for the (recording_id, modality) fix to check_coverage:
    before it, headimu's own time_magnitude finding masked a dropped
    attention table that still claimed has_attention - the exact case a
    real AirPods bundle produces (headimu is always present alongside
    attention). Scoped per-modality, headimu's finding must no longer cover
    for a missing attention finding."""
    write_fixture(tmp_path)
    a = AirPodsAdapter()
    ref = a.discover(tmp_path)[0]
    bundle = a.load(ref)

    manifest = pd.DataFrame([{
        "recording_id": ref.recording_id,
        "has_watch": False, "has_watch_rawaccel": False, "has_headimu": True,
        "has_pen": False, "has_markers": False,
        "has_attention": True,                      # stale: attention table dropped below
        "has_head_gravity": bundle.meta["has_head_gravity"],
        "has_head_quaternion": bundle.meta["has_head_quaternion"],
        "has_head_gyro": bundle.meta["has_head_gyro"],
    }])
    findings = validate_motion_table(bundle.tables["headimu"], ref.recording_id, "headimu",
                                     bundle.meta["head_hz_nominal"])
    findings += validate_recording(ref.recording_id, {"headimu": bundle.tables["headimu"]}, bundle.meta)
    problems = check_coverage(manifest, findings)
    assert any("time_magnitude" in p and "attention" in p for p in problems)

