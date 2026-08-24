import importlib
import json
import sys

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.transform import Rotation

from focuswatch_dataset import schema as S
from focuswatch_dataset.adapters import base
from focuswatch_dataset.adapters.ege import EgeAdapter
from focuswatch_dataset.manifest import build_channels, build_manifest
from focuswatch_dataset.validate import (
    check_coverage, validate_motion_table, validate_pen_table, validate_recording,
)

T0 = 1780577357025


@pytest.fixture(autouse=True)
def _isolated_registry():
    # Why: same pattern as tests/test_adapter_base.py:12-17 - register() mutates
    # a module-level dict, so a test that pops/re-adds entries must not leak
    # into later tests in the same session.
    snapshot = dict(base._REGISTRY)
    yield
    base._REGISTRY.clear()
    base._REGISTRY.update(snapshot)


def write_fixture(root, sid="T6", n=3000):
    rng = np.random.default_rng(1)
    d = root / sid
    d.mkdir(parents=True, exist_ok=True)
    rots = Rotation.random(n, random_state=1)
    grav = rots.inv().apply([0.0, 0.0, -1.0])
    total = grav + rng.normal(0, 0.02, (n, 3))       # gravity is included
    # Why: this pipeline stores the gyroscope under g*, colliding with our gravity names.
    # A leading quiet stretch (still_mask's 2 s / 200-sample window needs contiguous
    # coverage) lets the validator's gravity-column-absent quaternion fallback run,
    # matching the pattern in tests/test_validate.py::make_ege_like.
    gyro = rng.normal(0, 0.15, (n, 3))
    n_still = int(n * 0.2)
    gyro[:n_still] = rng.normal(0, 0.001, (n_still, 3))
    t = T0 + np.arange(n) * 10
    pd.DataFrame({
        "id": np.arange(n), "session_id": sid, "t_ms": t,
        "ax": total[:, 0], "ay": total[:, 1], "az": total[:, 2],
        "gx": gyro[:, 0], "gy": gyro[:, 1], "gz": gyro[:, 2],
        "qw": rots.as_quat()[:, 3], "qx": rots.as_quat()[:, 0],
        "qy": rots.as_quat()[:, 1], "qz": rots.as_quat()[:, 2],
        "created_at": "2026-06-04",
    }).to_csv(d / "imu_samples_rows.csv", index=False)

    # Why: at least 100 rows - validate._validate_quaternion silently skips
    # the quat_norm check below that count (guards against ML4SCS's
    # partly-empty forward-only capture), which would otherwise leave
    # has_head_quaternion=True with no finding to back it, an
    # indistinguishable-from-untested coverage gap on every build.
    n_head = 150
    pd.DataFrame({
        "id": np.arange(n_head), "session_id": sid, "t_ms": t[:n_head],
        "qw": 1.0, "qx": 0.0, "qy": 0.0, "qz": 0.0,
        "ax": 0.0, "ay": 0.0, "az": -1.0, "created_at": "2026-06-04",
    }).to_csv(d / "head_motion_samples_rows.csv", index=False)

    pd.DataFrame({
        "id": np.arange(6), "session_id": sid, "t_ms": t[:6], "t_session_ms": np.arange(6) * 10.0,
        "type": ["pen_down", "pen_dot", "pen_dot", "pen_up", "pen_paper_info", "pen_dot"],
        "x": [1.0, 2.0, 3.0, 4.0, np.nan, 5.0], "y": [1.0, 2.0, 3.0, 4.0, np.nan, 5.0],
        "force": [400, 410, 405, 0, np.nan, 402], "created_at": "2026-06-04",
    }).to_csv(d / "pen_events.csv", index=False)

    # Why (C5/I9): the real session_start payload, verbatim in shape - this
    # export carries the same browser fingerprint fields as the SensorLogger
    # one, and a fixture of "{}" could not tell a redacting adapter from a
    # passthrough one.
    start_payload = json.dumps({
        "mode": "free", "notes": "", "screen": {"h": 956, "w": 1470, "dpr": 2},
        "pen_name": "LAMY_safari", "handedness": "right",
        "user_agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Chrome/148.0.0.0",
        "phase_count": 1, "operator_mode": False, "participant_id": sid,
        "session_number": 1, "time_origin_ms": T0, "pen_connected_at_start": True,
    })
    pd.DataFrame({
        "id": [0, 1], "session_id": sid, "t_ms": [t[0], t[-1]], "t_session_ms": [0.0, 6000.0],
        "event_type": ["session_start", "session_end"], "payload": [start_payload, "{}"],
        "created_at": "2026-06-04",
    }).to_csv(d / "events.csv", index=False)

    pd.DataFrame([{"session_id": sid, "participant_id": sid, "device": "watch",
                   "started_at_ms": T0, "ended_at_ms": int(t[-1]), "notes": "",
                   "created_at": "2026-06-04", "mode": "study"}]).to_csv(
        d / "sensor_session.csv", index=False)
    return root


def test_gyro_is_renamed_and_gravity_is_absent(tmp_path):
    write_fixture(tmp_path)
    a = EgeAdapter()
    watch = a.load(a.discover(tmp_path)[0]).tables["watch"]
    for c in S.COLUMNS[S.Quantity.GYRO]:
        assert c in watch.columns
    # The source g* columns are the gyroscope, so no gravity channel exists here.
    assert "gravity_x" not in watch.columns


def test_acceleration_is_declared_total(tmp_path):
    write_fixture(tmp_path)
    a = EgeAdapter()
    bundle = a.load(a.discover(tmp_path)[0])
    assert "accel_total_x" in bundle.tables["watch"].columns
    assert "accel_user_x" not in bundle.tables["watch"].columns
    assert bundle.meta["accel_semantics"] == "total"
    assert bundle.meta["accel_calibration"] == "raw_uncalibrated"


def test_quaternion_columns_pass_through_already_scalar_last(tmp_path):
    write_fixture(tmp_path)
    a = EgeAdapter()
    watch = a.load(a.discover(tmp_path)[0]).tables["watch"]
    raw = pd.read_csv(tmp_path / "T6" / "imu_samples_rows.csv")
    assert np.allclose(watch["quat_w"], raw["qw"])
    assert np.allclose(watch["quat_x"], raw["qx"])


def test_adapter_refuses_a_gravity_like_g_column(tmp_path):
    write_fixture(tmp_path)
    p = tmp_path / "T6" / "imu_samples_rows.csv"
    raw = pd.read_csv(p)
    rots = Rotation.random(len(raw), random_state=2)
    raw[["gx", "gy", "gz"]] = rots.inv().apply([0.0, 0.0, -1.0])   # unit norm: not a gyro
    raw.to_csv(p, index=False)
    a = EgeAdapter()
    with pytest.raises(ValueError, match="not gyroscope-like"):
        a.load(a.discover(tmp_path)[0])


def test_time_columns_are_split_correctly(tmp_path):
    write_fixture(tmp_path)
    a = EgeAdapter()
    bundle = a.load(a.discover(tmp_path)[0])
    assert bundle.tables["watch"]["t_ns"].iloc[0] == T0 * 1_000_000
    assert "src_t_session_ms" in bundle.tables["pen"].columns
    assert bundle.meta["time_alignment"] == "shared_clock"


def test_src_t_session_ms_is_declared_as_an_offset_not_a_wall_clock(tmp_path):
    """Item 1 (fix round C): src_t_session_ms is a session-relative
    millisecond offset, not a wall-
    clock reading - no clock name would be honest for it. Both pen/ and
    markers/ carry the column on this fixture (Ege's own imu_samples_rows.csv
    has no t_session_ms, so watch/ never emits it here); the modality
    default (backend_wall_clock) must still apply to every other column.
    """
    write_fixture(tmp_path)
    a = EgeAdapter()
    bundle = a.load(a.discover(tmp_path)[0])
    ch = build_channels([bundle]).set_index(["modality", "column"])["time_domain"]

    assert ch.loc[("pen", "src_t_session_ms")] == "session_relative_offset_ms"
    assert ch.loc[("markers", "src_t_session_ms")] == "session_relative_offset_ms"
    assert ch.loc[("pen", "src_timestamp")] == "pen_device_clock"
    assert ch.loc[("watch", "t_ns")] == "backend_wall_clock"
    assert ch.loc[("pen", "t_ns")] == "backend_wall_clock"


def test_pen_vocabulary_maps_to_canonical(tmp_path):
    write_fixture(tmp_path)
    a = EgeAdapter()
    bundle = a.load(a.discover(tmp_path)[0])
    pen = bundle.tables["pen"]
    assert set(pen["dot_type"]) <= set(S.DOT_TYPES)
    assert (pen["dot_type"] == "PEN_MOVE").sum() == 3
    framing = pen[pen["dot_type"] == "PEN_HOVER"]
    assert framing.empty  # pen_paper_info carries no position, never becomes a pen/ row


def test_pen_event_with_an_unmapped_type_raises(tmp_path):
    """Fix round C item 6 audit: this raise (adapters/ege.py:181-182) guards
    `strokes["type"].map(S.PEN_EVENTS_GEN_A)` against a `type` value outside
    the known vocabulary - it had no test, so the map silently producing NaN
    for an unrecognised source type could have published a stroke with
    `dot_type=NaN` instead of failing loudly.
    """
    write_fixture(tmp_path)
    pen_path = tmp_path / "T6" / "pen_events.csv"
    df = pd.read_csv(pen_path)
    df.loc[0, "type"] = "pen_wiggle"
    df.to_csv(pen_path, index=False)
    a = EgeAdapter()
    with pytest.raises(ValueError, match="unmapped pen event types"):
        a.load(a.discover(tmp_path)[0])


def test_pen_paper_info_routes_to_markers_not_dropped(tmp_path):
    """C3/I8: pen_paper_info used to be silently discarded (I8) - it carries
    no position, so it still never becomes a pen/ row, but it must not
    vanish entirely. Routed to markers/, matching how this adapter (and
    SensorLogger) already treat pen_session_sync."""
    write_fixture(tmp_path)
    a = EgeAdapter()
    markers = a.load(a.discover(tmp_path)[0]).tables["markers"]
    assert "pen_paper_info" in set(markers["event"])


def test_loaded_watch_passes_the_validator(tmp_path):
    write_fixture(tmp_path)
    a = EgeAdapter()
    watch = a.load(a.discover(tmp_path)[0]).tables["watch"]
    assert [f for f in validate_motion_table(watch, "ETH-EGE-T6", "watch", 100.0)
            if not f.passed] == []


def test_coverage_matrix_agrees_with_the_real_bundle(tmp_path):
    """End-to-end check_coverage regression (fix-round-3 item 3): the only
    adapter with such a test (AirPods) was the only adapter whose coverage
    was actually exercised, which is why the headimu/gyro_range gap (Ege's
    head table structurally carries no gyroscope) went uncaught. Builds a
    real bundle from the existing fixture (watch + headimu + pen + markers,
    all present) and runs it through the real gate.

    Why build_manifest(), not a hand-picked dict of flags: a hand-built
    manifest that forgets to name a flag (e.g. has_head_quaternion) makes
    check_coverage silently skip that flag's requirement rather than fail
    it - the exact masking bug this test exists to catch, previously true
    of has_head_gyro and, until this fix, still true of has_head_quaternion/
    has_head_gravity here. build_manifest derives every flag the same way
    the real build does, so this test can never again omit one by hand.
    """
    write_fixture(tmp_path)
    a = EgeAdapter()
    ref = a.discover(tmp_path)[0]
    bundle = a.load(ref)
    manifest = build_manifest([bundle])

    nominal = bundle.meta["watch_hz_nominal"]   # None - this source states no nominal rate
    findings = validate_motion_table(bundle.tables["watch"], ref.recording_id, "watch", nominal)
    findings += validate_motion_table(bundle.tables["headimu"], ref.recording_id, "headimu", nominal)
    findings += validate_pen_table(bundle.tables["pen"], ref.recording_id)
    findings += validate_recording(ref.recording_id, bundle.tables, bundle.meta)

    assert check_coverage(manifest, findings) == []


def test_importing_the_adapters_package_registers_ege():
    """`adapters/__init__.py`'s `from . import ege` line must do the registering.

    Every other test in this module imports the `ege` submodule directly, which
    fires its module-level register() regardless of what __init__.py does - that
    would make this test pass even if the package-level wiring were deleted. To
    actually exercise __init__.py's own import statement, the entry it already
    produced (at collection time, via this file's own direct import above) is
    scrubbed from both the registry and the import cache first, then only the
    package - never the ege submodule - is re-imported.
    """
    base._REGISTRY.pop("ege", None)
    saved = {name: sys.modules.pop(name, None)
             for name in ("focuswatch_dataset.adapters", "focuswatch_dataset.adapters.ege")}
    try:
        importlib.import_module("focuswatch_dataset.adapters")
        assert base.get_adapter("ege").name == "ege"
    finally:
        for name, mod in saved.items():
            if mod is not None:
                sys.modules[name] = mod
            else:
                sys.modules.pop(name, None)


def test_marker_payloads_are_allow_listed_here_too(tmp_path):
    """C5 was applied to the SensorLogger adapter alone, and this export carries
    the same session_start payload - so `user_agent` and `screen` stayed in the
    published Ege markers. Both ETH pipelines now share one allow-list.
    """
    write_fixture(tmp_path)
    a = EgeAdapter()
    bundle = a.load(a.discover(tmp_path)[0])
    blob = " ".join(bundle.tables["markers"]["src_payload"].astype(str))

    for banned in ("user_agent", "Mozilla", "screen", "notes"):
        assert banned not in blob, f"{banned} reached the published markers"
    assert "handedness" in blob and "participant_id" in blob
    # user_agent, screen, notes - the three excluded keys, counted so a future
    # export's new field is visible rather than silently discarded.
    assert bundle.meta["src_payload_dropped_key_count"] == 3


def test_handedness_is_read_from_the_session_payload(tmp_path):
    """The covariate is stated in events.csv's payload and was published as
    "unknown" for both Ege recordings while SensorLogger's carried it."""
    write_fixture(tmp_path)
    a = EgeAdapter()
    assert a.load(a.discover(tmp_path)[0]).meta["handedness"] == "right"


def test_markers_carry_one_column_set_across_both_source_files(tmp_path):
    """I15: session events come from events.csv and pen_paper_info from
    pen_events.csv, which has no payload column. Concatenating the two used to
    leave `src_payload` NaN on half the rows of one table."""
    write_fixture(tmp_path)
    a = EgeAdapter()
    mk = a.load(a.discover(tmp_path)[0]).tables["markers"]
    assert "src_payload" in mk.columns
    assert mk["src_payload"].notna().all()
