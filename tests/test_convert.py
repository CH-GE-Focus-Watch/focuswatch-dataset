import numpy as np
import pandas as pd
import pytest
from scipy.spatial.transform import Rotation

from focuswatch_dataset import schema as S
from focuswatch_dataset.convert import decimation_factor, downsample, to_user_acceleration


def total_frame(n=300, seed=5):
    rng = np.random.default_rng(seed)
    rots = Rotation.random(n, random_state=seed)
    user = rng.normal(0, 0.04, (n, 3))
    total = rots.inv().apply([0.0, 0.0, -1.0]) + user
    df = pd.DataFrame({"t_ns": np.arange(n, dtype=np.int64)})
    df[list(S.COLUMNS[S.Quantity.ACCEL_TOTAL])] = total
    df[list(S.COLUMNS[S.Quantity.QUAT])] = rots.as_quat()
    return df, user


def test_recovers_the_user_component():
    df, user = total_frame()
    out = to_user_acceleration(df)
    assert np.allclose(out[list(S.COLUMNS[S.Quantity.ACCEL_USER])].to_numpy(), user, atol=1e-9)


def test_total_columns_are_replaced_not_duplicated():
    df, _ = total_frame()
    out = to_user_acceleration(df)
    assert "accel_total_x" not in out.columns
    assert "accel_user_x" in out.columns


def test_already_user_frames_pass_through():
    df = pd.DataFrame({"t_ns": [0, 1], "accel_user_x": [0.1, 0.2],
                       "accel_user_y": [0.0, 0.0], "accel_user_z": [0.0, 0.0]})
    pd.testing.assert_frame_equal(to_user_acceleration(df), df)


def test_without_quaternion_it_refuses():
    df, _ = total_frame()
    df = df.drop(columns=list(S.COLUMNS[S.Quantity.QUAT]))
    with pytest.raises(ValueError, match="needs a quaternion"):
        to_user_acceleration(df)


# --- downsample ---------------------------------------------------------------

def _motion_table(n=1000, hz=100.0, freq=5.0):
    t = np.arange(n) / hz
    wave = np.sin(2 * np.pi * freq * t)
    return pd.DataFrame({
        "t_ns": (t * 1e9).astype(np.int64),
        "accel_user_x": wave, "accel_user_y": wave, "accel_user_z": wave,
        "quat_x": np.zeros(n), "quat_y": np.zeros(n),
        "quat_z": np.sin(t), "quat_w": np.cos(t),
        "src_sequence": np.arange(n, dtype=np.int64),
    })


def test_downsample_halves_rows_and_keeps_real_timestamps():
    df = _motion_table()
    out = downsample(df, 100.0, 50.0)
    assert len(out) == 500
    assert out["t_ns"].tolist() == df["t_ns"].iloc[::2].tolist()
    assert out["src_sequence"].tolist() == list(range(0, 1000, 2))


def test_downsample_keeps_a_slow_signal_and_removes_aliasing_content():
    slow = downsample(_motion_table(freq=5.0), 100.0, 50.0)["accel_user_x"]
    fast = downsample(_motion_table(freq=40.0), 100.0, 50.0)["accel_user_x"]
    # 5 Hz is far below the 25 Hz Nyquist limit of 50 Hz; 40 Hz is above it
    # and would alias to 10 Hz without the anti-aliasing filter.
    assert slow.iloc[50:-50].abs().max() > 0.95
    assert fast.iloc[50:-50].abs().max() < 0.05


def test_downsample_picks_quaternions_instead_of_filtering_them():
    df = _motion_table()
    out = downsample(df, 100.0, 50.0)
    quat = out[["quat_x", "quat_y", "quat_z", "quat_w"]].to_numpy()
    assert np.allclose(quat, df[["quat_x", "quat_y", "quat_z", "quat_w"]].to_numpy()[::2])
    assert np.allclose(np.linalg.norm(quat, axis=1), 1.0)


def test_downsample_accepts_measured_rates_near_nominal():
    assert decimation_factor(99.82, 50.0) == 2
    assert len(downsample(_motion_table(), 99.82, 50.0)) == 500


def test_downsample_at_target_rate_returns_an_unchanged_copy():
    df = _motion_table(hz=50.0)
    out = downsample(df, 50.0, 50.0)
    pd.testing.assert_frame_equal(out, df)
    assert out is not df


@pytest.mark.parametrize("source, target", [(100.0, 30.0), (50.0, 100.0)])
def test_downsample_refuses_non_integer_ratios_and_upsampling(source, target):
    with pytest.raises(ValueError, match="cannot downsample"):
        downsample(_motion_table(), source, target)


def test_downsample_refuses_missing_values():
    df = _motion_table()
    df.loc[10, "accel_user_x"] = np.nan
    with pytest.raises(ValueError, match=r"missing values: \['accel_user_x'\]"):
        downsample(df, 100.0, 50.0)
