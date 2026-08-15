import json

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.transform import Rotation

from focuswatch_dataset import schema as S
from focuswatch_dataset.adapters.sensorlogger import SensorLoggerAdapter
from focuswatch_dataset.manifest import build_channels, build_manifest
from focuswatch_dataset.validate import (
    check_coverage, validate_motion_table, validate_pen_table, validate_recording,
)

T0_NS = 1780853585220_000_000


def write_fixture(root, sid="E2_session6", n=400, standardisation=False, with_pen=True,
                  with_rawaccel=True):
    d = root / sid
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2)
    rots = Rotation.random(n, random_state=2)
    grav = rots.inv().apply([0.0, 0.0, -1.0])
    q = rots.as_quat()
    t = T0_NS + np.arange(n, dtype=np.int64) * 10_000_000
    scale = S.G_TO_MS2 if standardisation else 1.0

    pd.DataFrame({
        "time": t, "seconds_elapsed": np.arange(n) / 100,
        "rotationRateX": rng.normal(0, 0.15, n), "rotationRateY": rng.normal(0, 0.15, n),
        "rotationRateZ": rng.normal(0, 0.15, n),
        "gravityX": grav[:, 0] * scale, "gravityY": grav[:, 1] * scale, "gravityZ": grav[:, 2] * scale,
        "accelerationX": rng.normal(0, 0.04, n) * scale,
        "accelerationY": rng.normal(0, 0.04, n) * scale,
        "accelerationZ": rng.normal(0, 0.04, n) * scale,
        "quaternionW": q[:, 3], "quaternionX": q[:, 0], "quaternionY": q[:, 1], "quaternionZ": q[:, 2],
    }).to_csv(d / "WristMotion.csv", index=False)

    # Headphone gravity is always m/s2, independent of the standardisation flag.
    pd.DataFrame({
        "time": t[:200], "seconds_elapsed": np.arange(200) / 100,
        "rotationRateX": 0.01, "rotationRateY": 0.01, "rotationRateZ": 0.01,
        "gravityX": grav[:200, 0] * S.G_TO_MS2, "gravityY": grav[:200, 1] * S.G_TO_MS2,
        "gravityZ": grav[:200, 2] * S.G_TO_MS2,
        "accelerationX": rng.normal(0, 0.02, 200) * scale,
        "accelerationY": rng.normal(0, 0.02, 200) * scale,
        "accelerationZ": rng.normal(0, 0.02, 200) * scale,
        "quaternionW": q[:200, 3], "quaternionX": q[:200, 0],
        "quaternionY": q[:200, 1], "quaternionZ": q[:200, 2],
        "roll": 0.0, "pitch": 0.0, "yaw": 0.0, "devicelocation": "unknown",
    }).to_csv(d / "Headphone.csv", index=False)

    if with_rawaccel:
        pd.DataFrame({"time": t, "seconds_elapsed": np.arange(n) / 100,
                      "x": grav[:, 0] * scale, "y": grav[:, 1] * scale,
                      "z": grav[:, 2] * scale}).to_csv(
            d / "WatchAccelerometerUncalibrated.csv", index=False)

    pd.DataFrame([{
        "version": 3, "device name": "iPhone 15 Pro", "recording epoch time": T0_NS // 1_000_000,
        "recording time": "2026-06-07_17-33-05", "recording timezone": "Europe/Zurich",
        "platform": "ios", "appVersion": "1.59", "device id": "abc",
        # Wrist Motion sits at position 1 with a rate no other stream shares, so a
        # parser that took a fixed index instead of the named one reads the wrong value.
        "sensors": "Headphone|Wrist Motion|Annotation", "sampleRateMs": "40|10|",
        "standardisation": str(standardisation).lower(), "platform version": "26.5",
    }]).to_csv(d / "Metadata.csv", index=False)

    (d / "Annotation.csv").write_text("")

    events = [{"t_ms": int(t[0] // 1_000_000), "t_session_ms": 0.0, "event": "session_start",
               "session_id": "x", "payload": {}},
              {"t_ms": int(t[10] // 1_000_000), "t_session_ms": 100.0, "event": "pen_session_sync",
               "session_id": "x",
               "payload": {"pen_connected_t_ms": 1.0, "session_start_t_ms": 2.0,
                           "pen_minus_session_ms": 1.0}}]
    if with_pen:
        for i, ev in enumerate(["pen_down", "pen_move", "pen_move", "pen_up"]):
            events.append({"t_ms": int(t[20 + i] // 1_000_000), "t_session_ms": 200.0 + i,
                           "event": ev, "session_id": "x",
                           "payload": {"x": 11.68 + i, "y": 56.19, "force": 408,
                                       "tilt": {"x": 93, "y": 139, "twist": 9},
                                       "timestamp": 1716121921598}})
    # Why: without a trailing session_end, markers' span truncates to its last
    # early event (~100 ms in) while pen strokes and the motion streams run to
    # the recording's actual end - validate_recording's streams_overlap then
    # sees markers "end" before pen "starts" and fails on a fixture artefact,
    # not a real gap. Mirrors the ege fixture's events.csv, which already
    # brackets the session with session_start/session_end.
    events.append({"t_ms": int(t[-1] // 1_000_000), "t_session_ms": float((n - 1) * 10),
                   "event": "session_end", "session_id": "x", "payload": {}})
    (d / f"{sid}.json").write_text(json.dumps({"events": events}))
    return root


def test_units_are_harmonised_to_g_when_standardisation_is_on(tmp_path):
    write_fixture(tmp_path, standardisation=True)
    a = SensorLoggerAdapter()
    watch = a.load(a.discover(tmp_path)[0]).tables["watch"]
    norm = np.linalg.norm(watch[list(S.COLUMNS[S.Quantity.GRAVITY])].to_numpy(), axis=1)
    assert np.allclose(norm, 1.0, atol=1e-6)


def test_standardisation_on_harmonises_acceleration_and_rawaccel_to_g(tmp_path):
    """H1: the only prior standardisation=True test (above) asserts the
    GRAVITY norm, which `_harmonise_gravity_to_g` derives from the measured
    magnitude and is independent of `accel_factor` (finding I1) - replacing
    sensorlogger.py's `accel_factor = 1.0 / S.G_TO_MS2 if si else 1.0` with a
    hardcoded `1.0` left 213/213 green under the old suite. This asserts the
    branch that flag actually controls: watch `accel_user_*` and
    `watch_rawaccel`'s `accel_total_*`.

    Correction round 1, Finding 1: also pins the RECORDED
    unit_conversion_factor via build_channels, not just the harmonised value -
    the brief's literal C1 mutation (hardcoding the factor returned by
    `_motion`/`_rawaccel` to 1.0 instead of the true `accel_factor`) leaves the
    harmonised-norm assertions below unaffected (a prior test run with
    standardisation=False already showed factor 1.0 is correct there), so only
    a standardisation=True recorded-factor assertion can catch it.

    Correction round 1, Finding 7: making the watch norm assertion two-sided is
    not enough - ACCEL_USER_BAND starts at 0.0, so its lower bound is vacuous
    for a vector norm and an over-division on the watch path alone still passed
    17/17. The guard that bites pins the published value to the RECORDED factor:
    the two are artefacts of one decision and nothing else tied them together,
    so a scaling applied to the data but not reflected in the factor - or the
    reverse - went unnoticed.
    """
    write_fixture(tmp_path, standardisation=True)
    a = SensorLoggerAdapter()
    bundle = a.load(a.discover(tmp_path)[0])

    watch = bundle.tables["watch"]
    accel_norm = np.linalg.norm(watch[list(S.COLUMNS[S.Quantity.ACCEL_USER])].to_numpy(), axis=1)
    assert S.ACCEL_USER_BAND[0] <= np.median(accel_norm) <= S.ACCEL_USER_BAND[1]

    rawaccel = bundle.tables["watch_rawaccel"]
    raw_norm = np.linalg.norm(rawaccel[list(S.COLUMNS[S.Quantity.ACCEL_TOTAL])].to_numpy(), axis=1)
    assert S.ACCEL_TOTAL_BAND[0] <= np.median(raw_norm) <= S.ACCEL_TOTAL_BAND[1]

    ch = build_channels([bundle]).set_index(["modality", "column"])["unit_conversion_factor"]
    assert ch.loc[("watch", "accel_user_x")] == pytest.approx(1.0 / S.G_TO_MS2)
    assert ch.loc[("watch_rawaccel", "accel_total_x")] == pytest.approx(1.0 / S.G_TO_MS2)

    src = pd.read_csv(a.discover(tmp_path)[0].path / "WristMotion.csv").sort_values(
        "time", kind="stable")
    assert watch["accel_user_x"].to_numpy() == pytest.approx(
        src["accelerationX"].to_numpy() * ch.loc[("watch", "accel_user_x")])


def test_unit_conversion_factor_is_recorded_per_column_not_restated(tmp_path):
    """C1: manifest.py used to hardcode unit_conversion_factor=1.0 for every
    channel regardless of what the adapter actually divided by - false
    provenance for all 7 SensorLogger recordings (headphone gravity is
    always divided by G_TO_MS2 per the app's documented behaviour;
    standardisation gates whether the wrist stream also is). With
    standardisation=False the wrist stream's own gravity is untouched
    (factor 1.0) while the SAME recording's headphone gravity is still
    divided (factor 1/G_TO_MS2) - one recording, two true factors for the
    same channel name, which a single restated constant cannot express.
    """
    write_fixture(tmp_path, standardisation=False)
    a = SensorLoggerAdapter()
    bundle = a.load(a.discover(tmp_path)[0])
    ch = build_channels([bundle]).set_index(["modality", "column"])["unit_conversion_factor"]

    assert ch.loc[("watch", "accel_user_x")] == pytest.approx(1.0)
    assert ch.loc[("watch", "gravity_x")] == pytest.approx(1.0)
    assert ch.loc[("headimu", "gravity_x")] == pytest.approx(1.0 / S.G_TO_MS2)


def test_headphone_gravity_is_divided_even_when_standardisation_is_off(tmp_path):
    write_fixture(tmp_path, standardisation=False)
    a = SensorLoggerAdapter()
    head = a.load(a.discover(tmp_path)[0]).tables["headimu"]
    norm = np.linalg.norm(head[list(S.COLUMNS[S.Quantity.GRAVITY])].to_numpy(), axis=1)
    assert np.allclose(norm, 1.0, atol=1e-6)


def test_head_gravity_already_in_g_is_not_divided_again(tmp_path):
    """Guards magnitude-based harmonisation against a table-identity shortcut.

    A future SensorLogger export could ship headphone gravity already in g (the
    app's documented behaviour could change, or a differently-configured
    exporter could be added). The conversion must be gated on the measured
    norm, not on "this came from the headimu table" - otherwise a column
    already in g gets divided a second time and silently corrupted.
    """
    write_fixture(tmp_path, standardisation=False)
    d = tmp_path / "E2_session6"
    head_raw = pd.read_csv(d / "Headphone.csv")
    for c in ("gravityX", "gravityY", "gravityZ"):
        head_raw[c] = head_raw[c] / S.G_TO_MS2
    head_raw.to_csv(d / "Headphone.csv", index=False)

    a = SensorLoggerAdapter()
    head = a.load(a.discover(tmp_path)[0]).tables["headimu"]
    norm = np.linalg.norm(head[list(S.COLUMNS[S.Quantity.GRAVITY])].to_numpy(), axis=1)
    assert np.allclose(norm, 1.0, atol=1e-6)


def test_raw_accel_goes_to_its_own_table(tmp_path):
    write_fixture(tmp_path)
    a = SensorLoggerAdapter()
    bundle = a.load(a.discover(tmp_path)[0])
    assert "accel_total_x" in bundle.tables["watch_rawaccel"].columns
    assert "accel_total_x" not in bundle.tables["watch"].columns
    assert bundle.meta["has_watch_rawaccel"] is True


def test_head_capability_flags_are_scoped_to_the_headphone_table(tmp_path):
    """has_head_gravity/has_head_quaternion (fix-round-1 item 2) describe the
    headimu stream specifically, distinct from has_gravity/has_quaternion,
    which describe watch/. Both tables carry both columns in this fixture, so
    both flag pairs must independently come back true."""
    write_fixture(tmp_path)
    a = SensorLoggerAdapter()
    bundle = a.load(a.discover(tmp_path)[0])
    assert bundle.meta["has_gravity"] is True
    assert bundle.meta["has_quaternion"] is True
    assert bundle.meta["has_head_gravity"] is True
    assert bundle.meta["has_head_quaternion"] is True


def test_head_capability_flags_are_false_without_a_headphone_table(tmp_path):
    write_fixture(tmp_path, sid="E3_session2")
    d = tmp_path / "E3_session2"
    (d / "Headphone.csv").unlink()
    a = SensorLoggerAdapter()
    ref = next(r for r in a.discover(tmp_path) if r.recording_id.endswith("E3_session2"))
    bundle = a.load(ref)
    assert "headimu" not in bundle.tables
    assert bundle.meta["has_head_gravity"] is False
    assert bundle.meta["has_head_quaternion"] is False


def test_missing_raw_accel_is_reported_not_faked(tmp_path):
    write_fixture(tmp_path, sid="E3_session1", with_rawaccel=False)
    a = SensorLoggerAdapter()
    ref = next(r for r in a.discover(tmp_path) if r.recording_id.endswith("E3_session1"))
    bundle = a.load(ref)
    assert "watch_rawaccel" not in bundle.tables
    assert bundle.meta["has_watch_rawaccel"] is False


def test_pen_comes_from_the_session_json(tmp_path):
    write_fixture(tmp_path)
    a = SensorLoggerAdapter()
    pen = a.load(a.discover(tmp_path)[0]).tables["pen"]
    assert set(pen["dot_type"]) == {"PEN_DOWN", "PEN_MOVE", "PEN_UP"}
    assert "src_timestamp" in pen.columns          # the pen device clock is kept as metadata


def test_recording_without_strokes_has_no_pen_table(tmp_path):
    write_fixture(tmp_path, sid="focuswatch_T8_s1", with_pen=False)
    a = SensorLoggerAdapter()
    ref = next(r for r in a.discover(tmp_path) if "T8" in r.recording_id)
    bundle = a.load(ref)
    assert "pen" not in bundle.tables
    assert bundle.meta["has_pen"] is False
    assert "markers" in bundle.tables              # phase events still exist


def test_pen_session_sync_is_a_marker_not_a_stroke(tmp_path):
    write_fixture(tmp_path)
    a = SensorLoggerAdapter()
    bundle = a.load(a.discover(tmp_path)[0])
    assert "pen_session_sync" in set(bundle.tables["markers"]["event"])
    assert "pen_session_sync" not in set(bundle.tables["pen"]["dot_type"])


def test_loaded_watch_passes_the_validator(tmp_path):
    write_fixture(tmp_path)
    a = SensorLoggerAdapter()
    watch = a.load(a.discover(tmp_path)[0]).tables["watch"]
    assert [f for f in validate_motion_table(watch, "ETH-SL-E2", "watch", 100.0)
            if not f.passed] == []


def test_loaded_headimu_gravity_passes_the_validator(tmp_path):
    """The magnitude-gated headphone gravity conversion is the riskiest part
    of this adapter; push it through the physical gate, not just the
    adapter's own np.allclose(..., 1.0) - the check should come from the
    validator agreeing, not from the adapter agreeing with itself."""
    write_fixture(tmp_path)
    a = SensorLoggerAdapter()
    head = a.load(a.discover(tmp_path)[0]).tables["headimu"]
    findings = validate_motion_table(head, "ETH-SL-E2", "headimu", 100.0)
    gravity_finding = next(f for f in findings if f.check == "gravity_norm")
    assert gravity_finding.passed
    assert S.GRAVITY_NORM_BAND[0] <= gravity_finding.observed <= S.GRAVITY_NORM_BAND[1]


def test_watch_hz_nominal_is_parsed_from_metadata_sample_rate(tmp_path):
    write_fixture(tmp_path)
    d = tmp_path / "E2_session6"
    meta = pd.read_csv(d / "Metadata.csv")
    meta["sampleRateMs"] = "40|20|"          # Wrist Motion at 20 ms -> 50 Hz
    meta.to_csv(d / "Metadata.csv", index=False)

    a = SensorLoggerAdapter()
    bundle = a.load(a.discover(tmp_path)[0])
    assert bundle.meta["watch_hz_nominal"] == 50.0


def test_watch_hz_nominal_falls_back_to_measured_when_sample_rate_is_absent(tmp_path):
    write_fixture(tmp_path)
    d = tmp_path / "E2_session6"
    meta = pd.read_csv(d / "Metadata.csv")
    meta = meta.drop(columns=["sampleRateMs"])
    meta.to_csv(d / "Metadata.csv", index=False)

    a = SensorLoggerAdapter()
    bundle = a.load(a.discover(tmp_path)[0])
    assert bundle.meta["watch_hz_nominal"] is None
    # The measured rate is still recorded - just downstream, by the validator.
    findings = validate_motion_table(bundle.tables["watch"], "ETH-SL-E2", "watch", None)
    rate_finding = next(f for f in findings if f.check == "sample_rate")
    assert rate_finding.passed
    assert rate_finding.observed == pytest.approx(100.0, rel=0.05)


def test_annotation_csv_is_never_read(tmp_path):
    """M14: Annotation.csv is genuinely zero-byte in the real export (e.g.
    E3_session1) - phase markers and pen strokes live in the session JSON
    instead (see module docstring). `pd.read_csv` raises `EmptyDataError` on
    a zero-byte file, so if anything upstream ever opened it, load() would
    raise too. write_fixture already writes it empty; this test just asserts
    that stays true rather than being an accident nothing exercises.
    """
    write_fixture(tmp_path)
    d = tmp_path / "E2_session6"
    assert (d / "Annotation.csv").stat().st_size == 0
    a = SensorLoggerAdapter()
    bundle = a.load(a.discover(tmp_path)[0])
    assert "watch" in bundle.tables


def test_coverage_matrix_agrees_with_the_real_bundle(tmp_path):
    """End-to-end check_coverage regression (fix-round-3 item 3): the only
    adapter with such a test (AirPods) was the only adapter whose coverage
    was actually exercised, which is why the watch_rawaccel/gyro_range and
    headimu/gyro_range gaps went uncaught. Builds a real bundle from the
    existing fixture (watch + headimu + watch_rawaccel + pen + markers, all
    present) and runs it through the real gate.

    Why build_manifest(), not a hand-picked dict of flags: a hand-built
    manifest that forgets to name a flag makes check_coverage silently skip
    that flag's requirement rather than fail it - this exact class of bug
    hid a real gap in tests/test_adapter_ege.py's equivalent test
    (has_head_quaternion, fixed in the same review round as this change).
    build_manifest derives every flag the same way the real build does, so
    this test can never again omit one by hand.
    """
    write_fixture(tmp_path)
    a = SensorLoggerAdapter()
    ref = a.discover(tmp_path)[0]
    bundle = a.load(ref)
    manifest = build_manifest([bundle])

    findings = validate_motion_table(bundle.tables["watch"], ref.recording_id, "watch",
                                     bundle.meta["watch_hz_nominal"])
    findings += validate_motion_table(bundle.tables["headimu"], ref.recording_id, "headimu",
                                      bundle.meta["watch_hz_nominal"])
    findings += validate_motion_table(bundle.tables["watch_rawaccel"], ref.recording_id,
                                      "watch_rawaccel", bundle.meta["watch_hz_nominal"])
    findings += validate_pen_table(bundle.tables["pen"], ref.recording_id)
    findings += validate_recording(ref.recording_id, bundle.tables, bundle.meta)

    assert check_coverage(manifest, findings) == []
