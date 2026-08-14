import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from focuswatch_dataset import schema as S
from focuswatch_dataset.adapters.ml4scs import Ml4scsAdapter
from focuswatch_dataset.validate import validate_motion_table


def write_fixture(root, sid="S096", n=500, fs=100.0, gravity=True):
    rng = np.random.default_rng(0)
    rots = Rotation.random(n, random_state=0)
    g = rots.inv().apply([0.0, 0.0, -1.0])
    ts = 1780577357025 + (np.arange(n) * (1000 / fs)).astype(np.int64)
    watch = pd.DataFrame({
        "local_ts_ms": ts + 40, "session_id": sid, "sequence": np.arange(n),
        "sample_rate_hz": fs, "server_received_ms": ts + 50, "source": "watch",
        "ts": ts,
        "ax": rng.normal(0, 0.04, n), "ay": rng.normal(0, 0.04, n), "az": rng.normal(0, 0.04, n),
        "rx": rng.normal(0, 0.15, n), "ry": rng.normal(0, 0.15, n), "rz": rng.normal(0, 0.15, n),
    })
    if gravity:
        watch[["gx", "gy", "gz"]] = g
        watch[["qx", "qy", "qz", "qw"]] = rots.as_quat()
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

    pd.DataFrame([{
        "session_id": sid, "person_id": "P76", "study_mode": "study", "protocol_id": "v2",
        "subject_index": 3, "watch_profile": "100hz_grav" if gravity else "50hz",
        "start_time": pd.Timestamp(int(ts[0]), unit="ms", tz="UTC").isoformat(),
    }]).to_csv(root / "sessions.csv", index=False)
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


def test_pen_framing_sentinel_survives(tmp_path):
    write_fixture(tmp_path)
    a = Ml4scsAdapter()
    pen = a.load(a.discover(tmp_path)[0]).tables["pen"]
    assert (pen["x"] == -1.0).sum() == 1
