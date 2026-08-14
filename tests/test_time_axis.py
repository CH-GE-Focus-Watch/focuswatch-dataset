import numpy as np
import pandas as pd
import pytest
from hypothesis import given, strategies as st

from focuswatch_dataset.time_axis import (
    classify_time_unit, median_rate_hz, sort_stable_by_time, to_unix_ns,
)

# Reference epochs taken from the real corpora.
EGE_MS = 1780577357025
SL_NS = 1780853585220_000_000


@pytest.mark.parametrize("values,expected", [
    (np.array([1780577357.025]), "s"),
    (np.array([EGE_MS], dtype=float), "ms"),
    (np.array([EGE_MS * 1000], dtype=float), "us"),
    (np.array([SL_NS], dtype=float), "ns"),
    (np.array([0.0, 800106.0]), "session_relative"),
])
def test_classify_time_unit(values, expected):
    assert classify_time_unit(values) == expected


def test_ms_to_ns_is_exact():
    out = to_unix_ns(np.array([EGE_MS], dtype=float), "ms")
    assert out.dtype == np.int64
    assert out[0] == EGE_MS * 1_000_000


def test_integral_input_never_goes_through_float():
    # float64 has a 256 ns ULP at 1.78e18, so a naive `value * 1e6` is off by ~64 ns.
    for ms in (1780577357025, 1780853816507, 1780585671996):
        assert to_unix_ns(np.array([ms], dtype=float), "ms")[0] == ms * 1_000_000


def test_nanosecond_source_passes_through_unchanged():
    ns = 1780853585220123456
    assert to_unix_ns(np.array([ns], dtype=np.int64), "ns")[0] == ns


def test_fractional_seconds_keep_millisecond_resolution():
    out = to_unix_ns(np.array([1780577357.025]), "s")[0]
    assert abs(out - 1780577357_025_000_000) < 1_000


def test_stable_sort_preserves_order_within_ties():
    df = pd.DataFrame({"t_ns": [2, 1, 1, 1], "seq": [0, 1, 2, 3]})
    out = sort_stable_by_time(df)
    assert out["seq"].tolist() == [1, 2, 3, 0]


def test_median_rate_from_10ms_spacing():
    t = np.arange(0, 1_000_000_000, 10_000_000, dtype=np.int64)
    assert median_rate_hz(t) == pytest.approx(100.0)


@given(st.lists(st.integers(min_value=0, max_value=10_000), min_size=2, max_size=200))
def test_sort_is_monotonic_and_preserves_rows(values):
    df = pd.DataFrame({"t_ns": values, "i": range(len(values))})
    out = sort_stable_by_time(df)
    assert len(out) == len(df)
    assert out["t_ns"].is_monotonic_increasing
