"""Time-unit detection and conversion to the canonical int64 Unix-nanosecond axis."""
from __future__ import annotations

import numpy as np
import pandas as pd

_NS_PER = {"s": 1_000_000_000, "ms": 1_000_000, "us": 1_000, "ns": 1}

# Magnitude brackets for a contemporary wall clock. Anything below 1e8 cannot be one.
_BRACKETS = (("s", 1e9, 1e10), ("ms", 1e12, 1e13), ("us", 1e15, 1e16), ("ns", 1e18, 1e19))


def classify_time_unit(values: np.ndarray) -> str:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("no finite timestamps")
    ref = float(np.median(np.abs(finite)))
    for unit, lo, hi in _BRACKETS:
        if lo <= ref < hi:
            return unit
    if ref < 1e8:
        return "session_relative"
    raise ValueError(f"timestamp magnitude {ref:g} matches no known unit")


def to_unix_ns(values: np.ndarray, unit: str) -> np.ndarray:
    """Convert to int64 Unix nanoseconds without going through a float product.

    A wall clock in nanoseconds sits near 1.78e18, where float64 resolves only to
    256 ns. Scaling in floating point therefore shifts every timestamp by tens of
    nanoseconds and corrupts sources that are already nanosecond integers.
    """
    if unit == "session_relative":
        raise ValueError("session-relative timestamps need an epoch offset first")
    factor = _NS_PER[unit]
    v = np.asarray(values)
    if np.issubdtype(v.dtype, np.integer):
        return v.astype(np.int64) * factor
    v = v.astype(float)
    whole = np.floor(v)
    frac = v - whole
    return whole.astype(np.int64) * factor + np.rint(frac * factor).astype(np.int64)


def sort_stable_by_time(df: pd.DataFrame, column: str = "t_ns") -> pd.DataFrame:
    # Why: batched captures share timestamps; an unstable sort reorders tied samples.
    return df.sort_values(column, kind="stable").reset_index(drop=True)


def median_rate_hz(t_ns: np.ndarray) -> float:
    diffs = np.diff(np.sort(np.asarray(t_ns, dtype=np.int64)))
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return float("nan")
    return 1e9 / float(np.median(diffs))
