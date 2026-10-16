"""Read-side conversions the published tables deliberately do not perform.

Removing gravity from a total-acceleration channel requires a per-sample
orientation estimate, and a common sample rate requires a filter. Both are
models, not measurements, so they stay out of the dataset and live here for
reusers who need one common representation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import signal

from . import schema as S
from .physics import gravity_from_quaternion

_TOTAL = list(S.COLUMNS[S.Quantity.ACCEL_TOTAL])
_USER = list(S.COLUMNS[S.Quantity.ACCEL_USER])
_QUAT = list(S.COLUMNS[S.Quantity.QUAT])

# Why: quaternion components are not a band-limited signal - filtering them
# breaks the unit norm and smears sign flips - so they are picked, not filtered.
_FILTERED = [c for q, cols in S.COLUMNS.items() if q != S.Quantity.QUAT for c in cols]

# Why: measured rates sit slightly off nominal (99.82 Hz for a 100 Hz watch);
# a ratio this close to an integer is that integer.
DECIMATION_RATIO_TOLERANCE = 0.05


def decimation_factor(source_hz: float, target_hz: float) -> int:
    """The integer factor that takes `source_hz` to `target_hz`, or raise."""
    if target_hz <= 0:
        raise ValueError(f"target_hz must be positive, got {target_hz}")
    ratio = source_hz / target_hz
    factor = int(round(ratio))
    if factor < 1 or abs(ratio - factor) > DECIMATION_RATIO_TOLERANCE * factor:
        raise ValueError(
            f"cannot downsample {source_hz:g} Hz to {target_hz:g} Hz: the ratio "
            f"{ratio:.3f} is not an integer (upsampling is not supported)"
        )
    return factor


def downsample(df: pd.DataFrame, source_hz: float, target_hz: float) -> pd.DataFrame:
    """Return a motion table decimated from `source_hz` to `target_hz`.

    Sensor channels go through `scipy.signal.decimate` (8th-order Chebyshev
    IIR anti-aliasing, zero phase so the samples do not shift against pen or
    marker timestamps); the same method the ML4SCS pipeline uses. `t_ns`,
    quaternions and provenance columns keep every `factor`-th original value,
    so timestamps stay real readings instead of interpolated ones. A table
    already at `target_hz` is returned unchanged.
    """
    factor = decimation_factor(source_hz, target_hz)
    if factor == 1:
        return df.copy()
    filtered = [c for c in _FILTERED if c in df.columns]
    with_nan = [c for c in filtered if df[c].isna().any()]
    if with_nan:
        # Why: filtfilt spreads one NaN across the whole channel.
        raise ValueError(f"cannot filter columns with missing values: {with_nan}")
    out = df.iloc[::factor].reset_index(drop=True)
    for column in filtered:
        decimated = signal.decimate(df[column].to_numpy(dtype=float), factor,
                                    ftype="iir", zero_phase=True)
        out[column] = np.asarray(decimated[: len(out)])
    return out


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
