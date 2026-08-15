"""Read-side conversions the published tables deliberately do not perform.

Removing gravity from a total-acceleration channel requires a per-sample
orientation estimate. That is a model, not a measurement, so it stays out of the
dataset and lives here for reusers who need one common representation.
"""
from __future__ import annotations

import pandas as pd

from . import schema as S
from .physics import gravity_from_quaternion

_TOTAL = list(S.COLUMNS[S.Quantity.ACCEL_TOTAL])
_USER = list(S.COLUMNS[S.Quantity.ACCEL_USER])
_QUAT = list(S.COLUMNS[S.Quantity.QUAT])


def to_user_acceleration(df: pd.DataFrame) -> pd.DataFrame:
    """Return `df` with `accel_total_*` replaced by derived `accel_user_*`.

    Already-`user` frames pass through unchanged. A `total`-only frame
    needs `quat_*` to derive gravity's direction in the device frame; without
    it there is no way to subtract gravity from the total, and this raises
    rather than guessing.
    """
    if all(c in df.columns for c in _USER):
        return df
    if not all(c in df.columns for c in _TOTAL):
        raise ValueError("frame carries neither user nor total acceleration")
    if not all(c in df.columns for c in _QUAT):
        raise ValueError("removing gravity needs a quaternion channel")
    gravity = gravity_from_quaternion(df[_QUAT].to_numpy(dtype=float))
    out = df.drop(columns=_TOTAL)
    out[_USER] = df[_TOTAL].to_numpy(dtype=float) - gravity
    return out
