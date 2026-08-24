import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.transform import Rotation

from focuswatch_dataset import schema as S
from focuswatch_dataset.adapters.sensorlogger import SensorLoggerAdapter
from focuswatch_dataset.manifest import build_channels, build_manifest
from focuswatch_dataset.validate import (
    check_coverage, detect_dropouts, validate_motion_table, validate_pen_table, validate_recording,
)

T0_NS = 1780853585220_000_000


def write_fixture(root, sid="E2_session6", n=400, standardisation=False, with_pen=True,
                  with_rawaccel=True, generation="B", participant_id="E2", handedness="right",
                  head_rows=200):
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
    # Why (C4): head_rows is independently settable (default 200, matching the
    # historical fixture) so a dropout scenario (a headphone stream covering
    # only a sliver of the recording, e.g. E3_session1's 58 samples/2.1s
    # inside 8713.6s) can be built without perturbing every other test.
    pd.DataFrame({
        "time": t[:head_rows], "seconds_elapsed": np.arange(head_rows) / 100,
        "rotationRateX": 0.01, "rotationRateY": 0.01, "rotationRateZ": 0.01,
        "gravityX": grav[:head_rows, 0] * S.G_TO_MS2, "gravityY": grav[:head_rows, 1] * S.G_TO_MS2,
        "gravityZ": grav[:head_rows, 2] * S.G_TO_MS2,
        "accelerationX": rng.normal(0, 0.02, head_rows) * scale,
        "accelerationY": rng.normal(0, 0.02, head_rows) * scale,
        "accelerationZ": rng.normal(0, 0.02, head_rows) * scale,
        "quaternionW": q[:head_rows, 3], "quaternionX": q[:head_rows, 0],
        "quaternionY": q[:head_rows, 1], "quaternionZ": q[:head_rows, 2],
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

    # Why (I9/C5): the session's own token and the (unread until now) handedness
    # covariate both live in session_start's payload - both are on the
    # allow-list (C5), so redaction never strips them.
    start_payload = {"participant_id": participant_id}
    if handedness is not None:
        start_payload["handedness"] = handedness
    events = [{"t_ms": int(t[0] // 1_000_000), "t_session_ms": 0.0, "event": "session_start",
               "session_id": "x", "payload": start_payload},
              {"t_ms": int(t[10] // 1_000_000), "t_session_ms": 100.0, "event": "pen_session_sync",
               "session_id": "x",
               "payload": {"pen_connected_t_ms": 1.0, "session_start_t_ms": 2.0,
                           "pen_minus_session_ms": 1.0}}]
    # Why (C3): generation B (E1/E2/E3) embeds strokes in `events` with a
    # nested payload; generation A (Ege's own export, plus SL's S3/T8/T9/T10)
    # stores them under a separate, flat `pen_events` key instead - `events`
    # never carries stroke data for a genA session.
    pen_events = []
    if with_pen and generation == "B":
        for i, ev in enumerate(["pen_down", "pen_move", "pen_move", "pen_up"]):
            events.append({"t_ms": int(t[20 + i] // 1_000_000), "t_session_ms": 200.0 + i,
                           "event": ev, "session_id": "x",
                           "payload": {"x": 11.68 + i, "y": 56.19, "force": 408,
                                       "tilt": {"x": 93, "y": 139, "twist": 9},
                                       "timestamp": 1716121921598}})
    elif with_pen and generation == "A":
        for i, ev in enumerate(["pen_down", "pen_dot", "pen_dot", "pen_up"]):
            pen_events.append({"t_ms": int(t[20 + i] // 1_000_000), "t_session_ms": 200.0 + i,
                               "type": ev, "x": 11.68 + i, "y": 56.19, "force": 408})
        # A generation-A event class carrying no position (C3/I8) - must
        # route to markers/, never to pen/.
        pen_events.append({"t_ms": int(t[24] // 1_000_000), "t_session_ms": 204.0,
                           "type": "pen_paper_info"})
    # Why: without a trailing session_end, markers' span truncates to its last
    # early event (~100 ms in) while pen strokes and the motion streams run to
    # the recording's actual end - validate_recording's streams_overlap then
    # sees markers "end" before pen "starts" and fails on a fixture artefact,
    # not a real gap. Mirrors the ege fixture's events.csv, which already
    # brackets the session with session_start/session_end.
    events.append({"t_ms": int(t[-1] // 1_000_000), "t_session_ms": float((n - 1) * 10),
                   "event": "session_end", "session_id": "x", "payload": {}})
    body = {"events": events}
    if generation == "A":
        body["pen_events"] = pen_events
    (d / f"{sid}.json").write_text(json.dumps(body))
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


# --- I9: participant id from session_start's payload, not the directory ---

def test_participant_id_comes_from_session_start_payload_not_directory_name(tmp_path):
    """Real SL directories carry session suffixes (E1_session3,
    focuswatch_T10_s1_d7498f51) that are not DESIGN §2.2's `ETH-T8` token
    form - the participant id must come from the session's own stated token,
    not a directory-name parse."""
    write_fixture(tmp_path, sid="E1_session3_d7498f51", participant_id="E1")
    ref = SensorLoggerAdapter().discover(tmp_path)[0]
    assert ref.participant_id == "ETH-E1"
    assert ref.recording_id == "ETH-SL-E1_session3_d7498f51"   # recording_id is unaffected


def test_discover_fails_loudly_without_a_participant_id(tmp_path):
    write_fixture(tmp_path)
    d = tmp_path / "E2_session6"
    payload = json.loads(next(d.glob("*.json")).read_text())
    payload["events"][0]["payload"].pop("participant_id")
    next(d.glob("*.json")).write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="participant_id"):
        SensorLoggerAdapter().discover(tmp_path)


# --- C5: allow-listed payload, promoted handedness -------------------------

def test_handedness_reaches_the_manifest(tmp_path):
    write_fixture(tmp_path, handedness="right")
    a = SensorLoggerAdapter()
    bundle = a.load(a.discover(tmp_path)[0])
    assert bundle.meta["handedness"] == "right"
    assert build_manifest([bundle]).iloc[0]["handedness"] == "right"


def test_handedness_defaults_to_unknown_when_absent(tmp_path):
    write_fixture(tmp_path, handedness=None)
    a = SensorLoggerAdapter()
    bundle = a.load(a.discover(tmp_path)[0])
    assert bundle.meta["handedness"] == "unknown"


def test_unrecognised_handedness_value_fails_loudly(tmp_path):
    write_fixture(tmp_path, handedness="ambidextrous")
    a = SensorLoggerAdapter()
    with pytest.raises(ValueError, match="handedness"):
        a.load(a.discover(tmp_path)[0])


def test_device_fingerprint_and_free_text_never_reach_the_published_payload(tmp_path):
    """C5: user_agent (browser build) and screen (display geometry) are a
    device fingerprint; notes is free text an experimenter can type anything
    into (measured: "S3_Sensor_logger_dl-studying"). None of the three may
    survive into src_payload - the allow-list, not the adapter's own
    discretion, is what guarantees that."""
    write_fixture(tmp_path)
    d = tmp_path / "E2_session6"
    jf = next(d.glob("*.json"))
    payload = json.loads(jf.read_text())
    payload["events"][0]["payload"].update({
        "user_agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Chrome/150.0.0.0",
        "screen": {"w": 1800, "h": 1169, "dpr": 2},
        "notes": "S3_Sensor_logger_dl-studying",
    })
    jf.write_text(json.dumps(payload))

    a = SensorLoggerAdapter()
    bundle = a.load(a.discover(tmp_path)[0])
    markers = bundle.tables["markers"]
    combined = " ".join(markers["src_payload"])
    assert "user_agent" not in combined
    assert "screen" not in combined
    assert "notes" not in combined
    assert "Chrome" not in combined
    # Why: participant_id is on the allow-list (C5) - redaction must not
    # collaterally strip fields the pipeline elsewhere depends on.
    assert "participant_id" in combined


def test_dropped_payload_key_count_is_reported(tmp_path):
    write_fixture(tmp_path)
    d = tmp_path / "E2_session6"
    jf = next(d.glob("*.json"))
    payload = json.loads(jf.read_text())
    payload["events"][0]["payload"].update({"user_agent": "x", "screen": {"w": 1}})
    jf.write_text(json.dumps(payload))

    a = SensorLoggerAdapter()
    bundle = a.load(a.discover(tmp_path)[0])
    assert bundle.meta["src_payload_dropped_key_count"] == 2

    finding = next(f for f in validate_recording(bundle.ref.recording_id, bundle.tables, bundle.meta)
                   if f.check == "payload_keys_redacted")
    assert finding.observed == 2
    assert finding.passed is True


def test_generation_a_pen_events_key_is_read_not_dropped(tmp_path):
    """C3: S3/T8/T9/T10 store strokes under `pen_events`, a key the adapter
    used to ignore entirely (it only ever read `events`) - all four
    recordings published has_pen=false while carrying real stroke ground
    truth. write_fixture's generation="A" pen_events entries are 3 strokes
    (pen_down/pen_dot/pen_dot/pen_up = 4, but the 2 pen_dot rows both map to
    PEN_MOVE) plus one pen_paper_info.
    """
    write_fixture(tmp_path, sid="focuswatch_T8_s1", generation="A")
    a = SensorLoggerAdapter()
    ref = next(r for r in a.discover(tmp_path) if "T8" in r.recording_id)
    bundle = a.load(ref)
    assert bundle.meta["has_pen"] is True
    pen = bundle.tables["pen"]
    assert len(pen) == 4
    assert set(pen["dot_type"]) == {"PEN_DOWN", "PEN_MOVE", "PEN_UP"}
    assert (pen["dot_type"] == "PEN_MOVE").sum() == 2


def test_generation_a_pen_events_carry_no_tilt_or_pen_device_clock(tmp_path):
    """C3: generation A carries no tilt and no pen-device clock (open
    question 4) - NaN here is the honest value, not a missing mapping."""
    write_fixture(tmp_path, sid="focuswatch_T9_s1", generation="A")
    a = SensorLoggerAdapter()
    ref = next(r for r in a.discover(tmp_path) if "T9" in r.recording_id)
    pen = a.load(ref).tables["pen"]
    assert pen["tilt_x"].isna().all()
    assert pen["tilt_y"].isna().all()
    assert pen["src_timestamp"].isna().all()


def test_generation_a_paper_info_routes_to_markers_not_dropped(tmp_path):
    write_fixture(tmp_path, sid="focuswatch_T10_s1", generation="A")
    a = SensorLoggerAdapter()
    ref = next(r for r in a.discover(tmp_path) if "T10" in r.recording_id)
    bundle = a.load(ref)
    assert "pen_paper_info" in set(bundle.tables["markers"]["event"])
    assert "pen_paper_info" not in set(bundle.tables["pen"]["dot_type"])


def test_generation_a_pen_xy_unit_matches_ege_not_the_e_series(tmp_path):
    """C3: pen_xy_unit/pen_pressure_scale follow the export GENERATION, not
    the SensorLogger pipeline directory - genA recordings must publish the
    same label Ege's own genA export does, distinct from genB (E1/E2/E3)."""
    write_fixture(tmp_path, sid="focuswatch_S3_s1", generation="A")
    write_fixture(tmp_path, sid="E2_session6", generation="B")
    a = SensorLoggerAdapter()
    gen_a = a.load(next(r for r in a.discover(tmp_path) if "S3" in r.recording_id))
    gen_b = a.load(next(r for r in a.discover(tmp_path) if "E2" in r.recording_id))
    assert gen_a.meta["pen_xy_unit"] == S.PEN_XY_UNIT_GEN_A == "webapp_raw"
    assert gen_a.meta["pen_pressure_scale"] == S.PEN_PRESSURE_SCALE_GEN_A == "webapp_force"
    assert gen_b.meta["pen_xy_unit"] == S.PEN_XY_UNIT_GEN_B
    assert gen_b.meta["pen_pressure_scale"] == S.PEN_PRESSURE_SCALE_GEN_B
    assert gen_a.meta["pen_xy_unit"] != gen_b.meta["pen_xy_unit"]


def test_generation_a_pen_events_entry_missing_a_required_key_raises(tmp_path):
    """Fix round C item 6: this raise (adapters/sensorlogger.py:274-280) had no
    test at all - `missing = []` (never triggering it) left 293/293 green.
    The flat generation-A pen_events shape is inferred, not verified against a
    real export, so a malformed entry must fail loudly and specifically
    rather than surface as a bare, unhelpful KeyError deep in the mapping
    below.
    """
    write_fixture(tmp_path, sid="focuswatch_S3_s1", generation="A")
    json_path = tmp_path / "focuswatch_S3_s1" / "focuswatch_S3_s1.json"
    body = json.loads(json_path.read_text())
    del body["pen_events"][0]["type"]
    json_path.write_text(json.dumps(body))

    a = SensorLoggerAdapter()
    ref = next(r for r in a.discover(tmp_path) if "S3" in r.recording_id)
    with pytest.raises(ValueError, match="missing required key"):
        a.load(ref)


def test_generation_a_loaded_pen_passes_the_validator(tmp_path):
    write_fixture(tmp_path, sid="focuswatch_T8_s1", generation="A")
    a = SensorLoggerAdapter()
    ref = next(r for r in a.discover(tmp_path) if "T8" in r.recording_id)
    pen = a.load(ref).tables["pen"]
    assert [f for f in validate_pen_table(pen, ref.recording_id) if not f.passed] == []


def test_pen_comes_from_the_session_json(tmp_path):
    write_fixture(tmp_path)
    a = SensorLoggerAdapter()
    pen = a.load(a.discover(tmp_path)[0]).tables["pen"]
    assert set(pen["dot_type"]) == {"PEN_DOWN", "PEN_MOVE", "PEN_UP"}
    assert "src_timestamp" in pen.columns          # the pen device clock is kept as metadata


def test_pen_provenance_columns_declare_their_own_clock_not_backend_wall_clock(tmp_path):
    """Item 1 (fix round C): pen.src_timestamp is the pen hardware's own
    free-running clock (~749 days off backend_wall_clock, measured on
    ETH-SL-E2 - whole-branch-review-2.md finding 1), and src_t_session_ms is
    a session-relative offset, not a wall-clock reading. The modality
    default must still apply to every other column, including pen/'s own
    t_ns axis.
    """
    write_fixture(tmp_path)
    a = SensorLoggerAdapter()
    bundle = a.load(a.discover(tmp_path)[0])
    ch = build_channels([bundle]).set_index(["modality", "column"])["time_domain"]

    assert ch.loc[("pen", "src_timestamp")] == "pen_device_clock"
    assert ch.loc[("pen", "src_t_session_ms")] == "session_relative_offset_ms"
    assert ch.loc[("markers", "src_t_session_ms")] == "session_relative_offset_ms"
    assert ch.loc[("pen", "t_ns")] == "backend_wall_clock"
    assert ch.loc[("watch", "t_ns")] == "backend_wall_clock"


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


def test_load_reads_metadata_csv_once_while_preserving_its_derived_values(tmp_path, monkeypatch):
    """One parsed metadata frame supplies all fields load publishes from it."""
    import focuswatch_dataset.adapters.sensorlogger as sensorlogger_mod

    write_fixture(tmp_path, standardisation=True)
    metadata_path = tmp_path / "E2_session6" / "Metadata.csv"
    real_read_csv = sensorlogger_mod.pd.read_csv
    metadata_reads = 0

    def read_csv(path, *args, **kwargs):
        nonlocal metadata_reads
        if Path(path) == metadata_path:
            metadata_reads += 1
        return real_read_csv(path, *args, **kwargs)

    monkeypatch.setattr(sensorlogger_mod.pd, "read_csv", read_csv)
    adapter = SensorLoggerAdapter()
    bundle = adapter.load(adapter.discover(tmp_path)[0])

    assert bundle.meta["src_standardisation"] is True
    assert bundle.meta["watch_hz_nominal"] == 100.0
    assert bundle.meta["session_start_ns"] == T0_NS
    assert metadata_reads == 1


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


# --- C4: a dropped-out modality (ETH-SL-E3_session1's headimu) -------------
#
# Real measured case: Headphone.csv holds 58 rows spanning 2.1 s inside an
# 8713.6 s recording (coverage ratio ~0.00024) - the head unit disconnected
# 2 s in and never returned. The only two failures of the whole 983-check
# real-corpus validity run were this recording's streams_overlap (headimu
# has no overlap with watch) and quat_norm "never ran" (58 < the 100-sample
# floor). This fixture reproduces the same SHAPE (a headimu stream far too
# short relative to watch to mean anything physically) without claiming the
# exact real ratio.

def _dropout_fixture(tmp_path, sid="E3_session1"):
    # n=3000 @ 10 ms -> 30 s watch span; head_rows=5 @ 10 ms -> 50 ms head
    # span. Ratio ~0.0017 (0.17%), comfortably below DROPOUT_COVERAGE_RATIO_MIN
    # (0.01) and comfortably below the smallest legitimate fixture coverage
    # (Ege's headimu fixture, ~0.0497 - see schema.py's threshold comment).
    write_fixture(tmp_path, sid=sid, n=3000, head_rows=5, with_rawaccel=False)
    a = SensorLoggerAdapter()
    ref = next(r for r in a.discover(tmp_path) if sid in r.recording_id)
    return a, ref, a.load(ref)


def test_headimu_dropout_ratio_is_measured_correctly(tmp_path):
    _, _, bundle = _dropout_fixture(tmp_path)
    dropouts = detect_dropouts(bundle.tables)
    assert set(dropouts) == {"headimu"}
    assert dropouts["headimu"].n_samples == 5
    assert dropouts["headimu"].coverage_ratio < S.DROPOUT_COVERAGE_RATIO_MIN


def test_headimu_dropout_is_exempt_from_overlap_and_never_ran_gaps(tmp_path):
    """The real-corpus failure this item fixes: without the exemption,
    validate_recording's streams_overlap fails outright (headimu genuinely
    does not overlap watch) and check_coverage flags quat_norm/gravity_norm
    as never-ran (58 samples cannot clear the 100-sample floor)."""
    a, ref, bundle = _dropout_fixture(tmp_path)
    manifest = build_manifest([bundle])
    dropouts = {ref.recording_id: detect_dropouts(bundle.tables)}

    findings = validate_motion_table(bundle.tables["watch"], ref.recording_id, "watch",
                                     bundle.meta["watch_hz_nominal"])
    findings += validate_motion_table(bundle.tables["headimu"], ref.recording_id, "headimu",
                                      bundle.meta["watch_hz_nominal"])
    findings += validate_pen_table(bundle.tables["pen"], ref.recording_id)
    findings += validate_recording(ref.recording_id, bundle.tables, bundle.meta)

    assert [f for f in findings if f.check == "streams_overlap"] == [] or \
        all(f.passed for f in findings if f.check == "streams_overlap")
    assert any(f.check == "modality_dropout" and f.modality == "headimu" and f.passed
              for f in findings)
    assert check_coverage(manifest, findings, dropouts) == []


def test_headimu_dropout_without_the_dropouts_param_still_flags_the_gap(tmp_path):
    """Backward-compatibility check: check_coverage's dropouts param is
    optional - a caller that never passes it (every pre-C4 call site) must
    keep seeing the quat_norm/gravity_norm gap, not silently start passing."""
    a, ref, bundle = _dropout_fixture(tmp_path)
    manifest = build_manifest([bundle])

    findings = validate_motion_table(bundle.tables["watch"], ref.recording_id, "watch",
                                     bundle.meta["watch_hz_nominal"])
    findings += validate_motion_table(bundle.tables["headimu"], ref.recording_id, "headimu",
                                      bundle.meta["watch_hz_nominal"])
    findings += validate_pen_table(bundle.tables["pen"], ref.recording_id)
    findings += validate_recording(ref.recording_id, bundle.tables, bundle.meta)

    problems = check_coverage(manifest, findings)   # no dropouts param
    assert any("headimu" in p and ("quat_norm" in p or "gravity_norm" in p) for p in problems)


def test_manifest_issue_codes_reports_the_dropout(tmp_path):
    _, _, bundle = _dropout_fixture(tmp_path)
    row = build_manifest([bundle]).iloc[0]
    issues = json.loads(row["issue_codes"])
    assert issues == [{"code": "modality_dropout", "modality": "headimu",
                       "n_samples": 5, "coverage_ratio": pytest.approx(0.001334, abs=1e-4)}]
