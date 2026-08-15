import numpy as np
import pandas as pd
import pytest
from scipy.spatial.transform import Rotation

from focuswatch_dataset import schema as S
from focuswatch_dataset.convert import to_user_acceleration


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
