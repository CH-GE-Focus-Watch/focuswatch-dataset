"""Time-unit detection and conversion to the canonical int64 Unix-nanosecond axis."""
from __future__ import annotations

import numpy as np
import pandas as pd

from focuswatch_dataset.schema import TIME_COLUMN

_NS_PER = {"s": 1_000_000_000, "ms": 1_000_000, "us": 1_000, "ns": 1}

# Magnitude brackets for a contemporary wall clock. Anything below 1e8 cannot be one.
_BRACKETS = (("s", 1e9, 1e10), ("ms", 1e12, 1e13), ("us", 1e15, 1e16), ("ns", 1e18, 1e19))


def classify_time_unit(values: np.ndarray) -> str:
    """Classify by order of magnitude only.

    A session-relative clock in fine-grained units over a long enough session can
    reach the same magnitude as an absolute epoch and be misclassified as
    absolute; callers who know their unit should pass it rather than infer it.
    """
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


def parse_iso_to_unix_ns(values, iso_format: str = "ISO8601") -> np.ndarray:
    """Parse ISO-8601 timestamp strings to int64 Unix nanoseconds.

    pandas' default datetime64 resolution from `to_datetime` is not pinned
    across versions - pandas 3 defaults to microseconds, not nanoseconds, so
    a plain `.astype("int64")` after it silently scales by the wrong factor
    (1000x too small) instead of raising. `as_unit` fixes a known resolution
    before the int64 view; the result is then routed through `to_unix_ns`'s
    integer path so the final scaling never depends on the ambient default.
    """
    parsed = pd.to_datetime(pd.Series(values), format=iso_format, utc=True).dt.as_unit("us")
    return to_unix_ns(parsed.astype("int64").to_numpy(), "us")


def sort_stable_by_time(df: pd.DataFrame, column: str = TIME_COLUMN) -> pd.DataFrame:
    # Why: batched captures share timestamps; an unstable sort reorders tied samples.
    return df.sort_values(column, kind="stable").reset_index(drop=True)


def median_rate_hz(t_ns: np.ndarray) -> float:
    diffs = np.diff(np.sort(np.asarray(t_ns, dtype=np.int64)))
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return float("nan")
    return 1e9 / float(np.median(diffs))
