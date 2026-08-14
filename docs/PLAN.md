# FocusWatch Dataset Package — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A pip-installable Python package that reads four heterogeneous sensor-capture formats through adapters, enforces physical invariants, and writes one uniformly described Parquet bundle for publication as an Elsevier Data in Brief data descriptor.

**Architecture:** Adapters normalise each source into canonical tables (`watch`, `watch_rawaccel`, `headimu`, `pen`, `markers`, `attention`) whose column names carry the signal semantics. A physics validator gates every write. All capability flags live in a single manifest; Parquet key-value metadata is generated from it. Raw data never enters the git tree.

**Tech Stack:** Python ≥ 3.11, pandas, pyarrow, numpy, scipy, pytest, hypothesis.

**Spec:** `docs/specs/2026-08-14-focuswatch-dataset-design.md` — read it before starting. The plan argues from the spec; where they disagree, the spec wins.

## Global Constraints

- **Target repo:** new, standalone, `CH-GE-Focus-Watch/focuswatch-dataset`. Not this repo. Tasks 1–15 build it from scratch.
- **No real data in the git tree, ever.** Tests use synthetic fixtures only. The build reads from an external `--source-root`; output goes outside the repo.
- **float64 throughout.** float32 does not round-trip bit-identically for CoreMotion values and is not acceptable for a published dataset.
- **Canonical units:** acceleration and gravity in **g**, angular velocity in **rad/s**, quaternions dimensionless in **xyzw** order, time as **int64 Unix nanoseconds**.
- **`G_TO_MS2 = 9.80665`** — the only unit constant; every conversion records its factor in `channels`.
- **Semantics live in column names**, not in flags: `accel_user_*` versus `accel_total_*`. Manifest fields duplicate this as a searchable facet, never as its replacement.
- **Parquet write options (fixed):** `compression="zstd"`, `BYTE_STREAM_SPLIT` for float columns, `DELTA_BINARY_PACKED` for `t_ns`, `use_dictionary=False` where an explicit `column_encoding` is set (pyarrow rejects the combination).
- **Sorting is always `kind="stable"`.** Unstable sorts on tied timestamps scrambled sample order once already in this project's history.
- **Code comments in English, short, only where the constraint is not obvious.** Prefer `# Why:` lines over restating the code.
- **Every physical threshold is a named constant in `schema.py`**, imported by both the CI tests and the build gate, so the two cannot drift.
- **Tests are normative.** Where a test and the implementation shown in a task disagree, the test is right and the implementation must change. Never relax an assertion — in particular never turn an exactness assertion into `pytest.approx` — to make a task pass.
- **Task 16 runs on a machine that holds the data and is not delegable.** Every other task must pass on synthetic fixtures alone.

---

### Task 1: Repository scaffold and the data-policy guard

The guard comes first because everything after it handles participant data. On 2026-08-08 participant recordings reached both remotes of the sibling repo and had to be removed with `git filter-repo`; `origin` there is public.

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `README.md`
- Create: `src/focuswatch_dataset/__init__.py`
- Create: `tests/__init__.py` (empty — Task 15 imports fixtures across test modules, which needs package context)
- Create: `scripts/check_no_data.py`
- Create: `.pre-commit-config.yaml`
- Create: `.github/workflows/ci.yml`
- Test: `tests/test_data_policy.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `scripts/check_no_data.py::check_paths(paths: list[Path], max_bytes: int = 1_048_576) -> list[str]` returning human-readable violation strings, empty when clean.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_data_policy.py
from pathlib import Path

from scripts.check_no_data import check_paths


def test_rejects_large_file(tmp_path: Path):
    big = tmp_path / "watch.parquet"
    big.write_bytes(b"\0" * (1_048_576 + 1))
    assert check_paths([big]) != []


def test_rejects_known_data_extensions_regardless_of_size(tmp_path: Path):
    for name in ("S008_watch.csv", "P1_airpod_motion_labeled.csv", "sweep_data.zip"):
        f = tmp_path / name
        f.write_text("x")
        assert check_paths([f]) != [], name


def test_allows_small_source_and_fixtures(tmp_path: Path):
    ok = tmp_path / "adapters" / "ege.py"
    ok.parent.mkdir()
    ok.write_text("def load(): ...")
    assert check_paths([ok]) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_data_policy.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.check_no_data'`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/check_no_data.py
"""Pre-commit guard: keeps participant recordings out of the git tree."""
from __future__ import annotations

import re
import sys
from pathlib import Path

MAX_BYTES = 1_048_576

# Why: names seen in the source corpora. Fixtures must not look like real captures.
DATA_NAME_PATTERNS = (
    re.compile(r"_watch\.csv$"),
    re.compile(r"_pen\.csv$"),
    re.compile(r"_markers\.csv$"),
    re.compile(r"airpod_motion.*\.csv$"),
    re.compile(r"^(WristMotion|Headphone|WatchAccelerometerUncalibrated)\.csv$"),
    re.compile(r"\.zip$"),
    re.compile(r"\.parquet$"),
)


def check_paths(paths: list[Path], max_bytes: int = MAX_BYTES) -> list[str]:
    violations = []
    for p in paths:
        if not p.is_file():
            continue
        if any(pat.search(p.name) for pat in DATA_NAME_PATTERNS):
            violations.append(f"{p}: filename matches a capture-data pattern")
        elif p.stat().st_size > max_bytes:
            violations.append(f"{p}: {p.stat().st_size} bytes exceeds {max_bytes}")
    return violations


def main(argv: list[str]) -> int:
    violations = check_paths([Path(a) for a in argv])
    for v in violations:
        print(f"blocked: {v}", file=sys.stderr)
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_data_policy.py -v`
Expected: 3 passed

- [ ] **Step 5: Add packaging and CI**

```toml
# pyproject.toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "focuswatch-dataset"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "pandas>=2.0",
    # Why: the Parquet footer records `created_by: parquet-cpp-arrow <version>`,
    # so byte-level reproducibility of a release holds only within a major version.
    "pyarrow>=24.0,<25.0",
    "numpy>=1.26",
    "scipy>=1.11",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "hypothesis>=6.100"]

[project.scripts]
fw = "focuswatch_dataset.cli:main"

[tool.hatch.build.targets.wheel]
packages = ["src/focuswatch_dataset"]

[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
```

```yaml
# .github/workflows/ci.yml
name: ci
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: "3.11"}
      - run: pip install -e ".[dev]"
      - run: pytest tests/ -v
      - name: data-policy guard over tracked files
        run: git ls-files | xargs python scripts/check_no_data.py
```

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: no-participant-data
        name: block participant data
        entry: python scripts/check_no_data.py
        language: system
        stages: [pre-commit]
```

`.gitignore` must contain at minimum: `build/`, `dist/`, `*.egg-info/`, `__pycache__/`, `.pytest_cache/`, `out/`, `data/`, `*.parquet`, `*.zip`.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml .gitignore .pre-commit-config.yaml .github README.md src scripts tests
git commit -m "chore: scaffold package and add the data-policy guard"
```

---

### Task 2: Canonical schema and physical thresholds

**Files:**
- Create: `src/focuswatch_dataset/schema.py`
- Test: `tests/test_schema.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `SCHEMA_VERSION: str`
  - `G_TO_MS2: float`
  - `Quantity` (StrEnum): `ACCEL_USER`, `ACCEL_TOTAL`, `GYRO`, `GRAVITY`, `QUAT`
  - `COLUMNS: dict[Quantity, tuple[str, ...]]`
  - `UNITS: dict[Quantity, str]`
  - `ACCEL_USER_BAND`, `ACCEL_TOTAL_BAND`, `ACCEL_FORBIDDEN_BANDS`, `GRAVITY_NORM_BAND`, `GYRO_MEDIAN_BAND`, `GYRO_P95_MAX`, `QUAT_NORM_BAND`, `QUAT_GRAVITY_ANGLE_MAX_DEG`, `QUAT_STILL_ANGLE_MAX_DEG`, `RATE_TOLERANCE`, `SPILL_GUARD_S`
  - `MODALITIES: tuple[str, ...]`
  - `DOT_TYPES: tuple[str, ...]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_schema.py
from focuswatch_dataset import schema as S


def test_every_quantity_has_columns_and_a_unit():
    for q in S.Quantity:
        assert S.COLUMNS[q], q
        assert S.UNITS[q], q


def test_accel_bands_do_not_overlap_and_leave_a_forbidden_gap():
    lo_hi_user, lo_hi_total = S.ACCEL_USER_BAND, S.ACCEL_TOTAL_BAND
    assert lo_hi_user[1] < lo_hi_total[0]
    # Why: the gap is the detector. A user/total mix-up averages into it.
    assert (lo_hi_user[1], lo_hi_total[0]) in S.ACCEL_FORBIDDEN_BANDS


def test_ms2_band_is_forbidden():
    # A forgotten division by 9.80665 lands near 9.8.
    assert any(lo <= 9.80665 <= hi for lo, hi in S.ACCEL_FORBIDDEN_BANDS)


def test_quaternion_columns_are_scalar_last():
    assert S.COLUMNS[S.Quantity.QUAT] == ("quat_x", "quat_y", "quat_z", "quat_w")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_schema.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/focuswatch_dataset/schema.py
"""Canonical column names, units and the physical thresholds the validator enforces.

Both the CI tests and the build gate import from here so the two cannot drift.
"""
from __future__ import annotations

from enum import StrEnum

SCHEMA_VERSION = "1.0"
G_TO_MS2 = 9.80665


class Quantity(StrEnum):
    ACCEL_USER = "accel_user"
    ACCEL_TOTAL = "accel_total"
    GYRO = "gyro"
    GRAVITY = "gravity"
    QUAT = "quat"


COLUMNS: dict[Quantity, tuple[str, ...]] = {
    Quantity.ACCEL_USER: ("accel_user_x", "accel_user_y", "accel_user_z"),
    Quantity.ACCEL_TOTAL: ("accel_total_x", "accel_total_y", "accel_total_z"),
    Quantity.GYRO: ("gyro_x", "gyro_y", "gyro_z"),
    Quantity.GRAVITY: ("gravity_x", "gravity_y", "gravity_z"),
    # Why: scalar-last, matching scipy.spatial.transform.Rotation.from_quat.
    Quantity.QUAT: ("quat_x", "quat_y", "quat_z", "quat_w"),
}

UNITS: dict[Quantity, str] = {
    Quantity.ACCEL_USER: "g",
    Quantity.ACCEL_TOTAL: "g",
    Quantity.GYRO: "rad/s",
    Quantity.GRAVITY: "g",
    Quantity.QUAT: "1",
}

TIME_COLUMN = "t_ns"

ACCEL_USER_BAND = (0.0, 0.2)
ACCEL_TOTAL_BAND = (0.9, 1.1)
# Why: a user/total confusion averages into the gap; a missed /9.80665 lands near 9.8.
ACCEL_FORBIDDEN_BANDS = ((0.2, 0.9), (5.0, 15.0))

GRAVITY_NORM_BAND = (0.99, 1.01)
GRAVITY_NORM_MAX_IQR = 0.01
GYRO_MEDIAN_BAND = (0.005, 2.0)
GYRO_P95_MAX = 20.0          # deg/s data would sit far above this
QUAT_NORM_BAND = (0.999, 1.001)
QUAT_GRAVITY_ANGLE_MAX_DEG = 2.0
QUAT_STILL_ANGLE_MAX_DEG = 5.0
RATE_TOLERANCE = 0.20
SPILL_GUARD_S = 60.0

MODALITIES = ("watch", "watch_rawaccel", "headimu", "pen", "markers", "attention")
DOT_TYPES = ("PEN_DOWN", "PEN_MOVE", "PEN_UP", "PEN_HOVER")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_schema.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/focuswatch_dataset/schema.py tests/test_schema.py
git commit -m "feat: canonical schema and physical thresholds"
```

---

### Task 3: Time-axis normalisation

Every adapter depends on this. Three of the four sources use a different time unit, and one carries both an absolute and a session-relative column.

**Files:**
- Create: `src/focuswatch_dataset/time_axis.py`
- Test: `tests/test_time_axis.py`

**Interfaces:**
- Consumes: `schema.TIME_COLUMN`
- Produces:
  - `classify_time_unit(values: np.ndarray) -> str` returning one of `"s"`, `"ms"`, `"us"`, `"ns"`, `"session_relative"`
  - `to_unix_ns(values: np.ndarray, unit: str) -> np.ndarray` (int64)
  - `sort_stable_by_time(df: pd.DataFrame, column: str = "t_ns") -> pd.DataFrame`
  - `median_rate_hz(t_ns: np.ndarray) -> float`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_time_axis.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_time_axis.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/focuswatch_dataset/time_axis.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_time_axis.py -v`
Expected: all passed

- [ ] **Step 5: Commit**

```bash
git add src/focuswatch_dataset/time_axis.py tests/test_time_axis.py
git commit -m "feat: canonical time axis with unit detection"
```

---

### Task 4: Physical primitives

These are the measurements that expose semantic errors invisible at the column-name level. Numbers in the tests come from the real corpora and are documented in the spec, §5.1–5.2.

**Files:**
- Create: `src/focuswatch_dataset/physics.py`
- Test: `tests/test_physics.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `NormStats` dataclass with fields `median, p05, p95, iqr, mean, std, n`
  - `norm_stats(vectors: np.ndarray) -> NormStats`
  - `gravity_from_quaternion(q_xyzw: np.ndarray) -> np.ndarray` shape `(N, 3)`
  - `angle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray`
  - `still_mask(gyro: np.ndarray, fs_hz: float, window_s: float = 2.0, threshold: float = 0.05) -> np.ndarray`
  - `detect_quaternion_order(q_first_last: np.ndarray, gravity: np.ndarray) -> str` returning `"xyzw"` or `"wxyz"`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_physics.py
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from focuswatch_dataset.physics import (
    angle_deg, detect_quaternion_order, gravity_from_quaternion, norm_stats, still_mask,
)


def _random_rotations(n=500, seed=0):
    return Rotation.random(n, random_state=seed)


def test_gravity_from_quaternion_matches_the_rotated_down_vector():
    rots = _random_rotations()
    q = rots.as_quat()                       # scipy emits xyzw
    expected = rots.inv().apply([0.0, 0.0, -1.0])
    assert np.allclose(gravity_from_quaternion(q), expected, atol=1e-12)


def test_gravity_from_quaternion_is_unit_norm():
    g = gravity_from_quaternion(_random_rotations().as_quat())
    assert np.allclose(np.linalg.norm(g, axis=1), 1.0, atol=1e-12)


def test_detect_quaternion_order_finds_xyzw():
    rots = _random_rotations()
    q_xyzw = rots.as_quat()
    gravity = rots.inv().apply([0.0, 0.0, -1.0])
    assert detect_quaternion_order(q_xyzw, gravity) == "xyzw"


def test_detect_quaternion_order_finds_wxyz():
    rots = _random_rotations()
    q_xyzw = rots.as_quat()
    gravity = rots.inv().apply([0.0, 0.0, -1.0])
    q_wxyz = np.roll(q_xyzw, 1, axis=1)      # stored scalar-first
    assert detect_quaternion_order(q_wxyz, gravity) == "wxyz"


def test_angle_deg_endpoints():
    a = np.array([[0.0, 0.0, -1.0]])
    assert angle_deg(a, a)[0] == pytest.approx(0.0, abs=1e-9)
    assert angle_deg(a, -a)[0] == pytest.approx(180.0, abs=1e-9)


def test_norm_stats_on_a_unit_vector_field():
    v = np.tile([0.0, 0.0, 1.0], (100, 1))
    s = norm_stats(v)
    assert s.median == pytest.approx(1.0)
    assert s.iqr == pytest.approx(0.0)
    assert s.n == 100


def test_still_mask_selects_the_quiet_stretch():
    rng = np.random.default_rng(0)
    gyro = np.vstack([rng.normal(0, 0.5, (300, 3)), rng.normal(0, 0.001, (300, 3))])
    mask = still_mask(gyro, fs_hz=100.0, window_s=1.0, threshold=0.05)
    assert mask[350:550].mean() > 0.9
    assert mask[:250].mean() < 0.1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_physics.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/focuswatch_dataset/physics.py
"""Vector-level measurements that expose semantic errors column names cannot.

Gravity is a unit vector, angular velocity is not, and an attitude quaternion
reconstructs the gravity direction exactly. Those three facts identify the
quantity a column actually holds, independent of what it is called.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

DOWN = np.array([0.0, 0.0, -1.0])


@dataclass(frozen=True)
class NormStats:
    median: float
    p05: float
    p95: float
    iqr: float
    mean: float
    std: float
    n: int


def norm_stats(vectors: np.ndarray) -> NormStats:
    v = np.asarray(vectors, dtype=float)
    v = v[np.isfinite(v).all(axis=1)]
    n = np.linalg.norm(v, axis=1)
    q25, q75 = np.percentile(n, [25, 75])
    return NormStats(
        median=float(np.median(n)), p05=float(np.percentile(n, 5)),
        p95=float(np.percentile(n, 95)), iqr=float(q75 - q25),
        mean=float(n.mean()), std=float(n.std()), n=int(n.size),
    )


def gravity_from_quaternion(q_xyzw: np.ndarray) -> np.ndarray:
    """Rotate the world-frame down vector into the device frame."""
    return Rotation.from_quat(np.asarray(q_xyzw, dtype=float)).inv().apply(DOWN)


def angle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, float) / np.linalg.norm(a, axis=1, keepdims=True)
    b = np.asarray(b, float) / np.linalg.norm(b, axis=1, keepdims=True)
    return np.degrees(np.arccos(np.clip((a * b).sum(axis=1), -1.0, 1.0)))


def still_mask(gyro: np.ndarray, fs_hz: float, window_s: float = 2.0,
               threshold: float = 0.05) -> np.ndarray:
    """Samples whose surrounding window stays below `threshold` rad/s."""
    win = max(int(round(window_s * fs_hz)), 1)
    n = np.linalg.norm(np.asarray(gyro, dtype=float), axis=1)
    rolling_max = pd.Series(n).rolling(win, center=True, min_periods=1).max().to_numpy()
    return rolling_max < threshold


def detect_quaternion_order(q_first_last: np.ndarray, gravity: np.ndarray) -> str:
    """Decide whether stored components are scalar-last or scalar-first.

    Norm checks cannot tell these apart; the reconstructed gravity direction can.
    """
    q = np.asarray(q_first_last, dtype=float)
    as_xyzw = float(np.median(angle_deg(gravity_from_quaternion(q), gravity)))
    as_wxyz = float(np.median(angle_deg(gravity_from_quaternion(np.roll(q, -1, axis=1)), gravity)))
    return "xyzw" if as_xyzw <= as_wxyz else "wxyz"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_physics.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/focuswatch_dataset/physics.py tests/test_physics.py
git commit -m "feat: physical primitives for semantic validation"
```

---

### Task 5: Validator

The negative tests are the point of this task. A validator whose failure path is never exercised is decoration.

**Files:**
- Create: `src/focuswatch_dataset/validate.py`
- Test: `tests/test_validate.py`

**Interfaces:**
- Consumes: `schema`, `physics` (incl. `still_mask`), `time_axis.median_rate_hz`
- Produces:
  - `Finding` dataclass: `check: str, recording_id: str, modality: str, column: str, observed: float | str, expected: str, passed: bool`
  - `validate_motion_table(df, recording_id, modality, nominal_hz: float | None) -> list[Finding]`
  - `validate_pen_table(df, recording_id) -> list[Finding]`
  - `validate_recording(recording_id, tables: dict[str, pd.DataFrame], meta: dict) -> list[Finding]`
  - `check_coverage(manifest: pd.DataFrame, findings: list[Finding]) -> list[str]`
  - `ValidationReport` with `findings: list[Finding]`, `.failed -> list[Finding]`, `.to_json() -> str`

Three design points a reviewer should hold on to. **The quaternion check must run
for sources without a gravity column** — that is exactly the Ege pipeline, the
only source storing components scalar-first, so leaving it unchecked would leave
the highest-ranked risk unguarded; the fallback compares against the acceleration
direction during still windows. **Non-finite rows are excluded rather than
poisoning a median**, because forward-only quaternion capture leaves older ML4SCS
sessions partly empty. And **`check_coverage` exists because a skipped check and
a passed check look identical in the report**: an adapter that drops a column
would otherwise produce a vacuously green build.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_validate.py
import numpy as np
import pandas as pd
import pytest
from scipy.spatial.transform import Rotation

from focuswatch_dataset import schema as S
from focuswatch_dataset.validate import ValidationReport, validate_motion_table, validate_pen_table


def make_watch(n=2000, fs=100.0, accel_scale=0.04, gravity_scale=1.0, seed=0):
    rng = np.random.default_rng(seed)
    rots = Rotation.random(n, random_state=seed)
    q = rots.as_quat()
    grav = rots.inv().apply([0.0, 0.0, -1.0]) * gravity_scale
    return pd.DataFrame({
        "t_ns": np.arange(n, dtype=np.int64) * int(1e9 / fs),
        **dict(zip(S.COLUMNS[S.Quantity.ACCEL_USER], rng.normal(0, accel_scale, (n, 3)).T)),
        **dict(zip(S.COLUMNS[S.Quantity.GYRO], rng.normal(0, 0.15, (n, 3)).T)),
        **dict(zip(S.COLUMNS[S.Quantity.GRAVITY], grav.T)),
        **dict(zip(S.COLUMNS[S.Quantity.QUAT], q.T)),
    })


def failed_checks(findings):
    return {f.check for f in findings if not f.passed}


def test_clean_table_passes():
    assert failed_checks(validate_motion_table(make_watch(), "R1", "watch", 100.0)) == set()


def test_gravity_left_in_ms2_fails():
    df = make_watch(gravity_scale=S.G_TO_MS2)
    assert "gravity_norm" in failed_checks(validate_motion_table(df, "R1", "watch", 100.0))


def test_accel_in_the_forbidden_gap_fails_and_names_the_cause():
    # A constant 0.3 per axis gives a norm of 0.52, inside the (0.2, 0.9) gap.
    df = make_watch(accel_scale=0.0)
    df[list(S.COLUMNS[S.Quantity.ACCEL_USER])] = 0.3
    findings = validate_motion_table(df, "R1", "watch", 100.0)
    bad = next(f for f in findings if f.check == "accel_semantic_band")
    assert not bad.passed
    assert "user/total" in bad.expected


def test_accel_in_si_units_is_diagnosed_as_such():
    df = make_watch(accel_scale=0.0)
    df[list(S.COLUMNS[S.Quantity.ACCEL_USER])] = S.G_TO_MS2 / np.sqrt(3)
    bad = next(f for f in validate_motion_table(df, "R1", "watch", 100.0)
               if f.check == "accel_semantic_band")
    assert not bad.passed
    assert "m/s2" in bad.expected


def test_gyro_in_degrees_per_second_fails():
    df = make_watch()
    df[list(S.COLUMNS[S.Quantity.GYRO])] *= 180.0 / np.pi * 50
    assert "gyro_range" in failed_checks(validate_motion_table(df, "R1", "watch", 100.0))


def test_scalar_first_quaternion_fails_the_gravity_crosscheck():
    df = make_watch()
    q = df[list(S.COLUMNS[S.Quantity.QUAT])].to_numpy()
    df[list(S.COLUMNS[S.Quantity.QUAT])] = np.roll(q, 1, axis=1)
    assert "quat_gravity_agreement" in failed_checks(validate_motion_table(df, "R1", "watch", 100.0))


def test_rate_mismatch_fails():
    assert "sample_rate" in failed_checks(validate_motion_table(make_watch(fs=100.0), "R1", "watch", 50.0))


def test_non_monotonic_time_fails():
    df = make_watch()
    df.loc[10, "t_ns"] = 0
    assert "time_monotonic" in failed_checks(validate_motion_table(df, "R1", "watch", 100.0))


def test_pen_vocabulary_is_checked():
    df = pd.DataFrame({"t_ns": [1, 2], "dot_type": ["PEN_DOWN", "SCRIBBLE"],
                       "x": [1.0, 2.0], "y": [1.0, 2.0]})
    assert "dot_type_vocabulary" in failed_checks(validate_pen_table(df, "R1"))


def test_pen_framing_sentinel_is_allowed():
    df = pd.DataFrame({"t_ns": [1, 2], "dot_type": ["PEN_DOWN", "PEN_UP"],
                       "x": [-1.0, 2.0], "y": [-1.0, 2.0]})
    assert failed_checks(validate_pen_table(df, "R1")) == set()


def test_report_serialises_and_reports_failure():
    r = ValidationReport(validate_motion_table(make_watch(gravity_scale=S.G_TO_MS2), "R1", "watch", 100.0))
    assert r.failed
    assert '"check"' in r.to_json()


# --- The Ege case: quaternion present, gravity column absent -------------------

def make_ege_like(n=3000, seed=7, still_fraction=0.2, roll_quaternion=False):
    """Total acceleration plus quaternion, no gravity column, with quiet stretches."""
    rng = np.random.default_rng(seed)
    rots = Rotation.random(n, random_state=seed)
    q = rots.as_quat()
    grav = rots.inv().apply([0.0, 0.0, -1.0])
    gyro = rng.normal(0, 0.15, (n, 3))
    n_still = int(n * still_fraction)
    gyro[:n_still] = rng.normal(0, 0.001, (n_still, 3))
    total = grav + rng.normal(0, 0.005, (n, 3))
    stored_q = np.roll(q, 1, axis=1) if roll_quaternion else q
    return pd.DataFrame({
        "t_ns": np.arange(n, dtype=np.int64) * 10_000_000,
        **dict(zip(S.COLUMNS[S.Quantity.ACCEL_TOTAL], total.T)),
        **dict(zip(S.COLUMNS[S.Quantity.GYRO], gyro.T)),
        **dict(zip(S.COLUMNS[S.Quantity.QUAT], stored_q.T)),
    })


def test_still_window_quaternion_check_runs_without_a_gravity_column():
    findings = validate_motion_table(make_ege_like(), "ETH-EGE-T6", "watch", 100.0)
    checks = {f.check for f in findings}
    assert "quat_still_agreement" in checks
    assert failed_checks(findings) == set()


def test_still_window_check_catches_a_scalar_first_quaternion():
    findings = validate_motion_table(make_ege_like(roll_quaternion=True), "ETH-EGE-T6", "watch", 100.0)
    assert "quat_still_agreement" in failed_checks(findings)


def test_nan_quaternion_rows_are_excluded_not_poisoning():
    # ML4SCS quaternion capture is forward-only; older sessions carry empty columns.
    df = make_watch()
    df.loc[:200, list(S.COLUMNS[S.Quantity.QUAT])] = np.nan
    findings = validate_motion_table(df, "R1", "watch", 100.0)
    agreement = next(f for f in findings if f.check == "quat_gravity_agreement")
    assert agreement.passed
    assert not np.isnan(float(agreement.observed))


def test_fully_missing_quaternion_reports_skipped_not_passed():
    df = make_watch()
    df[list(S.COLUMNS[S.Quantity.QUAT])] = np.nan
    checks = {f.check for f in validate_motion_table(df, "R1", "watch", 100.0)}
    assert "quat_gravity_agreement" not in checks


# --- Recording level -----------------------------------------------------------

def test_streams_that_do_not_overlap_fail():
    watch = make_watch(n=500)
    pen = pd.DataFrame({"t_ns": watch["t_ns"].max() + 10 ** 12 + np.arange(5, dtype=np.int64),
                        "dot_type": ["PEN_DOWN"] * 5, "x": 1.0, "y": 1.0})
    findings = validate_recording("R1", {"watch": watch, "pen": pen}, {})
    assert "streams_overlap" in failed_checks(findings)


def test_samples_far_before_the_session_start_fail():
    watch = make_watch(n=500)
    declared_start = int(watch["t_ns"].min())
    watch.loc[0, "t_ns"] = declared_start - 300 * 10 ** 9
    findings = validate_recording("R1", {"watch": watch}, {"session_start_ns": declared_start})
    assert "spill_guard" in failed_checks(findings)


def test_spill_guard_is_skipped_when_the_source_declares_no_start():
    # Not every source records a session start; the coverage matrix must not
    # demand a check the data cannot support.
    checks = {f.check for f in validate_recording("R1", {"watch": make_watch(n=500)}, {})}
    assert "spill_guard" not in checks


def test_declared_time_unit_must_match_the_magnitude():
    watch = make_watch(n=100)
    watch["t_ns"] = np.arange(100, dtype=np.int64)          # session-relative, not an epoch
    findings = validate_recording("R1", {"watch": watch}, {"time_domain": "backend_wall_clock"})
    assert "time_magnitude" in failed_checks(findings)


# --- Coverage matrix -----------------------------------------------------------

def test_coverage_accepts_a_complete_report():
    manifest = pd.DataFrame([{"recording_id": "R1", "has_watch": True, "has_quaternion": True,
                              "has_gravity": True, "has_headimu": False, "has_pen": False,
                              "has_watch_rawaccel": False, "has_markers": False,
                              "has_attention": False}])
    findings = validate_motion_table(make_watch(), "R1", "watch", 100.0)
    findings += validate_recording("R1", {"watch": make_watch()}, {})
    assert check_coverage(manifest, findings) == []


def test_coverage_rejects_a_silently_skipped_quaternion_check():
    manifest = pd.DataFrame([{"recording_id": "R1", "has_watch": True, "has_quaternion": True,
                              "has_gravity": True, "has_headimu": False, "has_pen": False,
                              "has_watch_rawaccel": False, "has_markers": False,
                              "has_attention": False}])
    # An adapter that dropped the quaternion columns produces no quat findings at all.
    df = make_watch().drop(columns=list(S.COLUMNS[S.Quantity.QUAT]))
    findings = validate_motion_table(df, "R1", "watch", 100.0)
    findings += validate_recording("R1", {"watch": df}, {})
    problems = check_coverage(manifest, findings)
    assert any("quat_norm" in p for p in problems)
```

Import line for the new names: `from focuswatch_dataset.validate import (ValidationReport, check_coverage, validate_motion_table, validate_pen_table, validate_recording)`.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_validate.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/focuswatch_dataset/validate.py
"""Physical and structural gate applied to every table before it is written."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from . import schema as S
from .physics import angle_deg, gravity_from_quaternion, norm_stats, still_mask
from .time_axis import median_rate_hz


@dataclass(frozen=True)
class Finding:
    check: str
    recording_id: str
    modality: str
    column: str
    observed: float | str
    expected: str
    passed: bool


@dataclass
class ValidationReport:
    findings: list[Finding] = field(default_factory=list)

    @property
    def failed(self) -> list[Finding]:
        return [f for f in self.findings if not f.passed]

    def to_json(self) -> str:
        return json.dumps([asdict(f) for f in self.findings], indent=2)


def _in_band(value: float, band: tuple[float, float]) -> bool:
    return band[0] <= value <= band[1]


def _has(df: pd.DataFrame, q: S.Quantity) -> bool:
    return all(c in df.columns for c in S.COLUMNS[q])


def _vec(df: pd.DataFrame, q: S.Quantity) -> np.ndarray:
    return df[list(S.COLUMNS[q])].to_numpy(dtype=float)


def _diagnose_accel(median: float, band: tuple[float, float]) -> str:
    """Name the likely cause when an acceleration norm lands outside its band."""
    if S.ACCEL_FORBIDDEN_BANDS[0][0] <= median <= S.ACCEL_FORBIDDEN_BANDS[0][1]:
        return f"median in {band}; observed value suggests a user/total mix"
    if S.ACCEL_FORBIDDEN_BANDS[1][0] <= median <= S.ACCEL_FORBIDDEN_BANDS[1][1]:
        return f"median in {band}; observed value suggests m/s2 instead of g"
    return f"median in {band}"


def validate_motion_table(df: pd.DataFrame, recording_id: str, modality: str,
                          nominal_hz: float | None) -> list[Finding]:
    out: list[Finding] = []

    def add(check, column, observed, expected, passed):
        out.append(Finding(check, recording_id, modality, column, observed, expected, passed))

    t = df[S.TIME_COLUMN].to_numpy(dtype=np.int64)
    add("time_monotonic", S.TIME_COLUMN, int(np.sum(np.diff(t) < 0)),
        "no backward steps", bool(np.all(np.diff(t) >= 0)))

    if nominal_hz:
        measured = median_rate_hz(t)
        add("sample_rate", S.TIME_COLUMN, round(measured, 3), f"{nominal_hz} Hz +-20%",
            abs(measured - nominal_hz) / nominal_hz < S.RATE_TOLERANCE)

    for q, band in ((S.Quantity.ACCEL_USER, S.ACCEL_USER_BAND),
                    (S.Quantity.ACCEL_TOTAL, S.ACCEL_TOTAL_BAND)):
        if not _has(df, q):
            continue
        st = norm_stats(_vec(df, q))
        add("accel_semantic_band", q.value, round(st.median, 5),
            _diagnose_accel(st.median, band), _in_band(st.median, band))

    if _has(df, S.Quantity.GRAVITY):
        st = norm_stats(_vec(df, S.Quantity.GRAVITY))
        add("gravity_norm", S.Quantity.GRAVITY.value, round(st.median, 5),
            f"median in {S.GRAVITY_NORM_BAND}, IQR < {S.GRAVITY_NORM_MAX_IQR}",
            _in_band(st.median, S.GRAVITY_NORM_BAND) and st.iqr < S.GRAVITY_NORM_MAX_IQR)
        flat = np.abs(_vec(df, S.Quantity.GRAVITY)[:, :2]).max(axis=1) < 0.1
        if flat.sum() > 10:
            gz = float(np.median(_vec(df, S.Quantity.GRAVITY)[flat, 2]))
            add("gravity_sign", S.Quantity.GRAVITY.value, round(gz, 4),
                "gz near -1 when the device lies flat", gz < 0)

    if _has(df, S.Quantity.GYRO):
        st = norm_stats(_vec(df, S.Quantity.GYRO))
        add("gyro_range", S.Quantity.GYRO.value, round(st.median, 5),
            f"median in {S.GYRO_MEDIAN_BAND}, p95 < {S.GYRO_P95_MAX}",
            _in_band(st.median, S.GYRO_MEDIAN_BAND) and st.p95 < S.GYRO_P95_MAX)

    out += _validate_quaternion(df, recording_id, modality)
    return out


def _validate_quaternion(df: pd.DataFrame, recording_id: str, modality: str) -> list[Finding]:
    """Quaternion checks, including the fallback for sources without a gravity column.

    A scalar-first storage order leaves every norm at 1.0 and raises no exception;
    only the reconstructed gravity direction exposes it. Sources that ship no
    gravity channel are compared against the acceleration direction while the
    device is still, which is the gravity direction to within the sensor bias.
    """
    out: list[Finding] = []

    def add(check, observed, expected, passed):
        out.append(Finding(check, recording_id, modality, S.Quantity.QUAT.value,
                           observed, expected, passed))

    if not _has(df, S.Quantity.QUAT):
        return out
    q_arr = _vec(df, S.Quantity.QUAT)
    finite_q = np.isfinite(q_arr).all(axis=1)
    # Why: ML4SCS quaternion capture is forward-only, so older sessions carry
    # partly empty columns. Rotation.from_quat raises "Found zero norm
    # quaternions" on those rows, so an unfiltered array aborts the build.
    if finite_q.sum() < 100:
        return out

    st = norm_stats(q_arr[finite_q])
    add("quat_norm", round(st.median, 6), f"median in {S.QUAT_NORM_BAND}",
        _in_band(st.median, S.QUAT_NORM_BAND))

    reconstructed = gravity_from_quaternion(q_arr[finite_q])

    if _has(df, S.Quantity.GRAVITY):
        grav = _vec(df, S.Quantity.GRAVITY)[finite_q]
        rows = np.isfinite(grav).all(axis=1)
        if rows.sum() >= 100:
            ang = float(np.median(angle_deg(reconstructed[rows], grav[rows])))
            add("quat_gravity_agreement", round(ang, 4),
                f"median angle < {S.QUAT_GRAVITY_ANGLE_MAX_DEG} deg",
                ang < S.QUAT_GRAVITY_ANGLE_MAX_DEG)
        return out

    accel_q = S.Quantity.ACCEL_TOTAL if _has(df, S.Quantity.ACCEL_TOTAL) else None
    if accel_q is None or not _has(df, S.Quantity.GYRO):
        return out
    gyro = _vec(df, S.Quantity.GYRO)[finite_q]
    accel = _vec(df, accel_q)[finite_q]
    fs = median_rate_hz(df[S.TIME_COLUMN].to_numpy(dtype=np.int64)[finite_q])
    still = still_mask(gyro, fs_hz=fs if np.isfinite(fs) else 100.0)
    if still.sum() < 100:
        add("quat_still_agreement", f"{int(still.sum())} still samples",
            "at least 100 still samples for the fallback check", False)
        return out
    ang = float(np.median(angle_deg(accel[still], reconstructed[still])))
    add("quat_still_agreement", round(ang, 4),
        f"median angle < {S.QUAT_STILL_ANGLE_MAX_DEG} deg over still windows",
        ang < S.QUAT_STILL_ANGLE_MAX_DEG)
    return out


def validate_pen_table(df: pd.DataFrame, recording_id: str) -> list[Finding]:
    unknown = sorted(set(df["dot_type"]) - set(S.DOT_TYPES))
    return [Finding("dot_type_vocabulary", recording_id, "pen", "dot_type",
                    ", ".join(unknown) or "-", f"subset of {S.DOT_TYPES}", not unknown)]


_INTERVAL_START, _INTERVAL_END = "t_start_ns", "t_end_ns"
# A contemporary wall clock in nanoseconds; anything far below is session-relative.
_EPOCH_NS_MIN, _EPOCH_NS_MAX = 1e18, 2e18


def _table_span(df: pd.DataFrame) -> tuple[int, int] | None:
    if _INTERVAL_START in df.columns and _INTERVAL_END in df.columns and len(df):
        return int(df[_INTERVAL_START].min()), int(df[_INTERVAL_END].max())
    if S.TIME_COLUMN in df.columns and len(df):
        return int(df[S.TIME_COLUMN].min()), int(df[S.TIME_COLUMN].max())
    return None


def validate_recording(recording_id: str, tables: dict[str, pd.DataFrame],
                       meta: dict) -> list[Finding]:
    """Checks that only make sense across the tables of one recording."""
    out: list[Finding] = []

    def add(check, modality, observed, expected, passed):
        out.append(Finding(check, recording_id, modality, "-", observed, expected, passed))

    spans = {m: s for m, s in ((m, _table_span(df)) for m, df in tables.items()) if s}

    for modality, (lo, _hi) in spans.items():
        add("time_magnitude", modality, lo,
            "timestamps on the declared wall clock", _EPOCH_NS_MIN <= lo <= _EPOCH_NS_MAX)

    if len(spans) > 1:
        latest_start = max(lo for lo, _ in spans.values())
        earliest_end = min(hi for _, hi in spans.values())
        overlap_s = (earliest_end - latest_start) / 1e9
        add("streams_overlap", "+".join(sorted(spans)), round(overlap_s, 3),
            "modality time ranges overlap", overlap_s > 0)

    # Why: the declared session start, taken from the source's own session
    # metadata - not the computed span, which would make the check tautological.
    start = meta.get("session_start_ns")
    if start:
        for modality, (lo, _hi) in spans.items():
            lag_s = (int(start) - lo) / 1e9
            add("spill_guard", modality, round(lag_s, 3),
                f"no sample more than {S.SPILL_GUARD_S} s before session start",
                lag_s <= S.SPILL_GUARD_S)
    return out


# Which checks a recording must have produced, derived from its manifest flags.
# A skipped check and a passed check are otherwise indistinguishable.
_REQUIRED_MOTION_CHECKS = ("time_monotonic", "sample_rate", "accel_semantic_band")


def check_coverage(manifest: pd.DataFrame, findings: list[Finding]) -> list[str]:
    problems: list[str] = []
    by_recording: dict[str, set[str]] = {}
    for f in findings:
        by_recording.setdefault(f.recording_id, set()).add(f.check)

    for _, row in manifest.iterrows():
        rid = row["recording_id"]
        seen = by_recording.get(rid, set())

        def require(check: str, reason: str) -> None:
            if check not in seen:
                problems.append(f"{rid}: {check} never ran ({reason})")

        if row.get("has_watch") or row.get("has_headimu"):
            for check in _REQUIRED_MOTION_CHECKS:
                require(check, "motion table present")
            require("time_magnitude", "motion table present")
        if row.get("has_gravity"):
            require("gravity_norm", "has_gravity is true")
        if row.get("has_quaternion"):
            require("quat_norm", "has_quaternion is true")
            if not ({"quat_gravity_agreement", "quat_still_agreement"} & seen):
                problems.append(
                    f"{rid}: neither quat_gravity_agreement nor quat_still_agreement ran "
                    "(has_quaternion is true)")
        if row.get("has_pen"):
            require("dot_type_vocabulary", "has_pen is true")
    return problems
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_validate.py -v`
Expected: 20 passed

- [ ] **Step 5: Commit**

```bash
git add src/focuswatch_dataset/validate.py tests/test_validate.py
git commit -m "feat: physical validator with negative-path coverage"
```

---

### Task 6: Adapter protocol and registry

**Files:**
- Create: `src/focuswatch_dataset/adapters/__init__.py`
- Create: `src/focuswatch_dataset/adapters/base.py`
- Test: `tests/test_adapter_base.py`

**Interfaces:**
- Consumes: `schema.MODALITIES`
- Produces:
  - `RecordingRef` dataclass: `recording_id, participant_id, cohort, pipeline, path`
  - `RecordingBundle` dataclass: `ref, tables: dict[str, pd.DataFrame], meta: dict[str, object]`
  - `Adapter` Protocol with `name: str`, `discover(root: Path) -> list[RecordingRef]`, `load(ref: RecordingRef) -> RecordingBundle`
  - `register(adapter)` / `get_adapter(name)` / `all_adapters()`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_adapter_base.py
from pathlib import Path

import pandas as pd
import pytest

from focuswatch_dataset.adapters.base import (
    Adapter, RecordingBundle, RecordingRef, all_adapters, get_adapter, register,
)


class _Dummy:
    name = "dummy"

    def discover(self, root: Path) -> list[RecordingRef]:
        return [RecordingRef("D-1", "D-P1", "dummy", "dummy", root)]

    def load(self, ref: RecordingRef) -> RecordingBundle:
        return RecordingBundle(ref, {"watch": pd.DataFrame({"t_ns": [1]})}, {})


def test_dummy_satisfies_the_protocol():
    assert isinstance(_Dummy(), Adapter)


def test_register_and_retrieve(tmp_path):
    register(_Dummy())
    assert get_adapter("dummy").discover(tmp_path)[0].recording_id == "D-1"
    assert "dummy" in {a.name for a in all_adapters()}


def test_bundle_rejects_unknown_modality():
    ref = RecordingRef("D-1", "D-P1", "dummy", "dummy", Path("."))
    with pytest.raises(ValueError, match="unknown modality"):
        RecordingBundle(ref, {"telepathy": pd.DataFrame()}, {})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_adapter_base.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/focuswatch_dataset/adapters/base.py
"""Adapter contract. Each source pipeline implements discover() and load()."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

import pandas as pd

from .. import schema as S


@dataclass(frozen=True)
class RecordingRef:
    recording_id: str
    participant_id: str
    cohort: str
    pipeline: str
    path: Path


@dataclass
class RecordingBundle:
    ref: RecordingRef
    tables: dict[str, pd.DataFrame]
    meta: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        unknown = set(self.tables) - set(S.MODALITIES)
        if unknown:
            raise ValueError(f"unknown modality: {sorted(unknown)}")


@runtime_checkable
class Adapter(Protocol):
    name: str

    def discover(self, root: Path) -> list[RecordingRef]: ...
    def load(self, ref: RecordingRef) -> RecordingBundle: ...


_REGISTRY: dict[str, Adapter] = {}


def register(adapter: Adapter) -> None:
    _REGISTRY[adapter.name] = adapter


def get_adapter(name: str) -> Adapter:
    return _REGISTRY[name]


def all_adapters() -> list[Adapter]:
    return list(_REGISTRY.values())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_adapter_base.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/focuswatch_dataset/adapters tests/test_adapter_base.py
git commit -m "feat: adapter protocol and registry"
```

---

### Task 7: ML4SCS adapter

Source layout: `{root}/watch/{SID}_watch.csv`, `{root}/pen/{SID}_pen.csv`, `{root}/markers/{SID}_markers.csv`, plus `{root}/sessions.csv`. 33 recordings, one per participant.

**Files:**
- Create: `src/focuswatch_dataset/adapters/ml4scs.py`
- Test: `tests/test_adapter_ml4scs.py`
- Test fixture: `tests/fixtures/ml4scs/` (synthetic, written by the test)

**Interfaces:**
- Consumes: `base.RecordingRef`, `base.RecordingBundle`, `time_axis`, `schema`
- Produces: `Ml4scsAdapter` with `name = "ml4scs"`, registered on import.

Column mapping. `ax/ay/az` is `userAcceleration` in g (measured median norm 0.039), `gx/gy/gz` is the CoreMotion gravity unit vector, `rx/ry/rz` is the gyroscope in rad/s. The join axis is the per-sample capture clock `ts`, not `local_ts_ms`.

| Source | Canonical |
|---|---|
| `ts` (ms) | `t_ns` |
| `ax, ay, az` | `accel_user_x/y/z` |
| `rx, ry, rz` | `gyro_x/y/z` |
| `gx, gy, gz` | `gravity_x/y/z` |
| `qx, qy, qz, qw` | `quat_x/y/z/w` |
| `local_ts_ms`, `sequence`, `server_received_ms` | `src_*` |

- [ ] **Step 1: Write the failing test**

```python
# tests/test_adapter_ml4scs.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_adapter_ml4scs.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/focuswatch_dataset/adapters/ml4scs.py
"""Adapter for the ML4SCS capture pipeline (watch + Moleskine pen + study markers)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .. import schema as S
from ..time_axis import sort_stable_by_time, to_unix_ns
from .base import RecordingBundle, RecordingRef, register

COHORT = "ML4SCS"

_WATCH_MAP = {
    "ax": "accel_user_x", "ay": "accel_user_y", "az": "accel_user_z",
    "rx": "gyro_x", "ry": "gyro_y", "rz": "gyro_z",
    "gx": "gravity_x", "gy": "gravity_y", "gz": "gravity_z",
    "qx": "quat_x", "qy": "quat_y", "qz": "quat_z", "qw": "quat_w",
}
_KEEP_AS_SOURCE = ("local_ts_ms", "sequence", "server_received_ms", "phone_received_at")


class Ml4scsAdapter:
    name = "ml4scs"

    def discover(self, root: Path) -> list[RecordingRef]:
        sessions = pd.read_csv(root / "sessions.csv").set_index("session_id")
        refs = []
        for f in sorted((root / "watch").glob("*_watch.csv")):
            sid = f.name.removesuffix("_watch.csv")
            if sid not in sessions.index:
                raise ValueError(f"{sid} has no row in sessions.csv")
            person = str(sessions.loc[sid, "person_id"]).strip()
            # Why: a placeholder here would collapse several recordings onto one
            # participant, and the article's participant count would silently lie.
            if not person or person.lower() in {"nan", "none"}:
                raise ValueError(f"{sid} has no person_id in sessions.csv")
            refs.append(RecordingRef(f"{COHORT}-{sid}", f"{COHORT}-{person}", COHORT, self.name, root))
        return refs

    def load(self, ref: RecordingRef) -> RecordingBundle:
        sid = ref.recording_id.removeprefix(f"{COHORT}-")
        root = ref.path
        sessions = pd.read_csv(root / "sessions.csv").set_index("session_id")
        row = sessions.loc[sid]

        watch = self._watch(root / "watch" / f"{sid}_watch.csv")
        tables = {"watch": watch}

        pen_path = root / "pen" / f"{sid}_pen.csv"
        if pen_path.exists():
            tables["pen"] = self._pen(pen_path)
        marker_path = root / "markers" / f"{sid}_markers.csv"
        if marker_path.exists():
            tables["markers"] = self._markers(marker_path)

        meta = {
            "watch_hz_nominal": 50.0 if str(row.get("watch_profile")) == "50hz" else 100.0,
            "has_gravity": "gravity_x" in watch.columns,
            "has_quaternion": "quat_x" in watch.columns,
            "accel_semantics": "user",
            "accel_calibration": "fused",
            "gravity_source": "measured" if "gravity_x" in watch.columns else "none",
            "time_domain": "watch_capture_clock",
            "time_alignment": "estimated_delta",
            "protocol_id": f"ml4scs_{row.get('protocol_id')}",
            "study_mode": row.get("study_mode"),
            "subject_index": row.get("subject_index"),
            "pen_xy_unit": "ncode_grid",
            "pen_pressure_scale": "moleskine_raw",
            "session_start_ns": self._session_start_ns(row.get("start_time")),
        }
        return RecordingBundle(ref, tables, meta)

    @staticmethod
    def _session_start_ns(start_time: object) -> int | None:
        """Session start from the server clock, used only for the spill guard.

        It sits within NTP distance of the watch capture clock, well inside the
        60 s tolerance, so no conversion between the two is needed here.
        """
        if start_time is None or pd.isna(start_time):
            return None
        # utc=True accepts both the offset-carrying and the naive form found in
        # sessions.csv and returns nanoseconds since the epoch either way.
        return int(pd.to_datetime(start_time, utc=True).value)

    def _watch(self, path: Path) -> pd.DataFrame:
        raw = pd.read_csv(path)
        # Why: `ts` is the per-sample capture clock; `local_ts_ms` is batch arrival
        # and lags by minutes when the watch drains a spill buffer.
        out = pd.DataFrame({"t_ns": to_unix_ns(raw["ts"].to_numpy(), "ms")})
        for src, dst in _WATCH_MAP.items():
            if src in raw.columns and raw[src].notna().any():
                out[dst] = raw[src].astype(float)
        for c in _KEEP_AS_SOURCE:
            if c in raw.columns:
                out[f"src_{c}"] = raw[c]
        return sort_stable_by_time(out)

    def _pen(self, path: Path) -> pd.DataFrame:
        raw = pd.read_csv(path)
        out = pd.DataFrame({
            "t_ns": to_unix_ns(raw["local_ts_ms"].to_numpy(), "ms"),
            "dot_type": raw["dot_type"].astype(str),
            "x": raw["x"].astype(float), "y": raw["y"].astype(float),
            "pressure": raw["pressure"].astype(float),
            "tilt_x": raw["tilt_x"].astype(float), "tilt_y": raw["tilt_y"].astype(float),
            "src_timestamp": raw["timestamp"],
        })
        return sort_stable_by_time(out)

    def _markers(self, path: Path) -> pd.DataFrame:
        raw = pd.read_csv(path)
        out = raw.rename(columns={"timestamp_ms": "_ms"})
        out["t_ns"] = to_unix_ns(out.pop("_ms").to_numpy(), "ms")
        cols = ["t_ns", "event", "task_id", "task_name", "task_index", "task_category", "protocol_id"]
        return sort_stable_by_time(out[cols])


register(Ml4scsAdapter())
```

Add `from . import ml4scs  # noqa: F401` to `src/focuswatch_dataset/adapters/__init__.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_adapter_ml4scs.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/focuswatch_dataset/adapters/ml4scs.py tests/test_adapter_ml4scs.py
git commit -m "feat: ML4SCS adapter"
```

---

### Task 8: ETH Ege adapter

This source carries the two documented traps: `ax/ay/az` is **total** acceleration (measured median norm 0.9946, so gravity is included), and the columns named `gx/gy/gz` hold the **gyroscope**, not gravity (median norm 0.177, p95 1.58 rad/s). The rename is mandatory and must be gated on physics, never on the column name.

Source layout: `{root}/{T6,T7}/{imu_samples_rows,head_motion_samples_rows,pen_events,events,sensor_session,web_session}.csv`. `t_ms` is Unix milliseconds; `t_session_ms` is session-relative and is preserved as `src_t_session_ms`.

**Files:**
- Create: `src/focuswatch_dataset/adapters/ege.py`
- Test: `tests/test_adapter_ege.py`

**Interfaces:**
- Consumes: `base`, `time_axis`, `physics.norm_stats`, `schema`
- Produces: `EgeAdapter` with `name = "ege"`, registered on import.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_adapter_ege.py
import numpy as np
import pandas as pd
import pytest
from scipy.spatial.transform import Rotation

from focuswatch_dataset import schema as S
from focuswatch_dataset.adapters.ege import EgeAdapter
from focuswatch_dataset.validate import validate_motion_table

T0 = 1780577357025


def write_fixture(root, sid="T6", n=600):
    rng = np.random.default_rng(1)
    d = root / sid
    d.mkdir(parents=True, exist_ok=True)
    rots = Rotation.random(n, random_state=1)
    grav = rots.inv().apply([0.0, 0.0, -1.0])
    total = grav + rng.normal(0, 0.02, (n, 3))       # gravity is included
    t = T0 + np.arange(n) * 10
    pd.DataFrame({
        "id": np.arange(n), "session_id": sid, "t_ms": t,
        "ax": total[:, 0], "ay": total[:, 1], "az": total[:, 2],
        # Why: this pipeline stores the gyroscope under g*, colliding with our gravity names.
        "gx": rng.normal(0, 0.15, n), "gy": rng.normal(0, 0.15, n), "gz": rng.normal(0, 0.15, n),
        "qw": rots.as_quat()[:, 3], "qx": rots.as_quat()[:, 0],
        "qy": rots.as_quat()[:, 1], "qz": rots.as_quat()[:, 2],
        "created_at": "2026-06-04",
    }).to_csv(d / "imu_samples_rows.csv", index=False)

    pd.DataFrame({
        "id": np.arange(50), "session_id": sid, "t_ms": t[:50],
        "qw": 1.0, "qx": 0.0, "qy": 0.0, "qz": 0.0,
        "ax": 0.0, "ay": 0.0, "az": -1.0, "created_at": "2026-06-04",
    }).to_csv(d / "head_motion_samples_rows.csv", index=False)

    pd.DataFrame({
        "id": np.arange(6), "session_id": sid, "t_ms": t[:6], "t_session_ms": np.arange(6) * 10.0,
        "type": ["pen_down", "pen_dot", "pen_dot", "pen_up", "pen_paper_info", "pen_dot"],
        "x": [1.0, 2.0, 3.0, 4.0, np.nan, 5.0], "y": [1.0, 2.0, 3.0, 4.0, np.nan, 5.0],
        "force": [400, 410, 405, 0, np.nan, 402], "created_at": "2026-06-04",
    }).to_csv(d / "pen_events.csv", index=False)

    pd.DataFrame({
        "id": [0, 1], "session_id": sid, "t_ms": [t[0], t[-1]], "t_session_ms": [0.0, 6000.0],
        "event_type": ["session_start", "session_end"], "payload": ["{}", "{}"],
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


def test_quaternion_is_reordered_to_scalar_last(tmp_path):
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


def test_pen_vocabulary_maps_to_canonical(tmp_path):
    write_fixture(tmp_path)
    a = EgeAdapter()
    pen = a.load(a.discover(tmp_path)[0]).tables["pen"]
    assert set(pen["dot_type"]) <= set(S.DOT_TYPES)
    assert (pen["dot_type"] == "PEN_MOVE").sum() == 3
    framing = pen[pen["dot_type"] == "PEN_HOVER"]
    assert framing.empty  # pen_paper_info carries no position and is dropped from pen/


def test_loaded_watch_passes_the_validator(tmp_path):
    write_fixture(tmp_path)
    a = EgeAdapter()
    watch = a.load(a.discover(tmp_path)[0]).tables["watch"]
    assert [f for f in validate_motion_table(watch, "ETH-EGE-T6", "watch", 100.0)
            if not f.passed] == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_adapter_ege.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/focuswatch_dataset/adapters/ege.py
"""Adapter for the custom watchOS pipeline that streams to a Supabase backend.

Two source quirks drive this module: acceleration includes gravity, and the
columns named g* hold the gyroscope. Both are asserted, not assumed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .. import schema as S
from ..physics import norm_stats, still_mask
from ..time_axis import median_rate_hz, sort_stable_by_time, to_unix_ns
from .base import RecordingBundle, RecordingRef, register

COHORT = "ETH-EGE"

_DOT_TYPES = {"pen_down": "PEN_DOWN", "pen_dot": "PEN_MOVE", "pen_up": "PEN_UP"}
_NON_STROKE_EVENTS = ("pen_paper_info", "pen_session_sync")


def _assert_gyroscope_like(values: np.ndarray) -> None:
    st = norm_stats(values)
    # Why: gravity would sit at 1.000 with near-zero spread; a gyro does not.
    if 0.99 <= st.median <= 1.01 and st.iqr < 0.01:
        raise ValueError(
            f"g* columns are not gyroscope-like (norm median {st.median:.4f}, "
            f"IQR {st.iqr:.4f}) - they look like a gravity unit vector"
        )


class EgeAdapter:
    name = "ege"

    def discover(self, root: Path) -> list[RecordingRef]:
        refs = []
        for d in sorted(p for p in root.iterdir() if (p / "imu_samples_rows.csv").exists()):
            refs.append(RecordingRef(f"{COHORT}-{d.name}", f"ETH-{d.name}", "ETH", self.name, d))
        return refs

    def load(self, ref: RecordingRef) -> RecordingBundle:
        d = ref.path
        tables = {"watch": self._motion(d / "imu_samples_rows.csv")}
        head = d / "head_motion_samples_rows.csv"
        if head.exists():
            tables["headimu"] = self._motion(head)
        pen = d / "pen_events.csv"
        if pen.exists():
            tables["pen"] = self._pen(pen)
        events = d / "events.csv"
        if events.exists():
            tables["markers"] = self._markers(events)

        meta = {
            "watch_hz_nominal": 100.0,
            "has_gravity": False,
            "has_quaternion": True,
            "accel_semantics": "total",
            # Established by a still-window test on the real corpus: the norm sits at
            # 0.9954 (T6) / 0.9932 (T7), a persistent per-device bias. A recombination
            # of userAcceleration and gravity would sit at exactly 1.000.
            "accel_calibration": "raw_uncalibrated",
            "accel_still_bias": self._still_bias(tables["watch"]),
            "gravity_source": "none",
            "time_domain": "backend_wall_clock",
            "time_alignment": "shared_clock",
            "protocol_id": "eth_ege_web",
            "study_mode": "study",
            "pen_xy_unit": "webapp_raw",
            "pen_pressure_scale": "webapp_force",
            "session_start_ns": self._session_start_ns(d / "sensor_session.csv"),
        }
        return RecordingBundle(ref, tables, meta)

    @staticmethod
    def _session_start_ns(path: Path) -> int | None:
        if not path.exists():
            return None
        started = pd.read_csv(path)["started_at_ms"].iloc[0]
        return int(started) * 1_000_000 if pd.notna(started) else None

    @staticmethod
    def _still_bias(watch: pd.DataFrame) -> float:
        """Mean deviation of the acceleration norm from 1 g while the wrist is still.

        Reported, never corrected. It is the evidence for accel_calibration and it
        differs per session, so folding it in would bake a guess into the data.
        """
        cols = list(S.COLUMNS[S.Quantity.GYRO])
        if not set(cols) <= set(watch.columns):
            return float("nan")
        fs = median_rate_hz(watch["t_ns"].to_numpy(dtype=np.int64))
        mask = still_mask(watch[cols].to_numpy(dtype=float), fs_hz=fs if np.isfinite(fs) else 100.0)
        if mask.sum() < 100:
            return float("nan")
        norms = np.linalg.norm(
            watch.loc[mask, list(S.COLUMNS[S.Quantity.ACCEL_TOTAL])].to_numpy(dtype=float), axis=1)
        return round(float(norms.mean() - 1.0), 6)

    def _motion(self, path: Path) -> pd.DataFrame:
        raw = pd.read_csv(path)
        out = pd.DataFrame({"t_ns": to_unix_ns(raw["t_ms"].to_numpy(), "ms")})
        out[list(S.COLUMNS[S.Quantity.ACCEL_TOTAL])] = raw[["ax", "ay", "az"]].astype(float).to_numpy()
        if {"gx", "gy", "gz"} <= set(raw.columns):
            gyro = raw[["gx", "gy", "gz"]].astype(float).to_numpy()
            _assert_gyroscope_like(gyro)
            out[list(S.COLUMNS[S.Quantity.GYRO])] = gyro
        if {"qx", "qy", "qz", "qw"} <= set(raw.columns):
            out[list(S.COLUMNS[S.Quantity.QUAT])] = raw[["qx", "qy", "qz", "qw"]].astype(float).to_numpy()
        if "t_session_ms" in raw.columns:
            out["src_t_session_ms"] = raw["t_session_ms"]
        return sort_stable_by_time(out)

    def _pen(self, path: Path) -> pd.DataFrame:
        raw = pd.read_csv(path)
        raw = raw[~raw["type"].isin(_NON_STROKE_EVENTS)]
        out = pd.DataFrame({
            "t_ns": to_unix_ns(raw["t_ms"].to_numpy(), "ms"),
            "dot_type": raw["type"].map(_DOT_TYPES).to_numpy(),
            "x": raw["x"].astype(float).to_numpy(), "y": raw["y"].astype(float).to_numpy(),
            "pressure": raw["force"].astype(float).to_numpy(),
            "src_t_session_ms": raw["t_session_ms"].to_numpy(),
        })
        if out["dot_type"].isna().any():
            raise ValueError(f"unmapped pen event types in {path}")
        return sort_stable_by_time(out)

    def _markers(self, path: Path) -> pd.DataFrame:
        raw = pd.read_csv(path)
        out = pd.DataFrame({
            "t_ns": to_unix_ns(raw["t_ms"].to_numpy(), "ms"),
            "event": raw["event_type"].astype(str),
            "task_id": "", "task_name": "", "task_index": -1,
            "task_category": "", "protocol_id": "eth_ege_web",
            "src_payload": raw.get("payload", ""),
            "src_t_session_ms": raw.get("t_session_ms"),
        })
        return sort_stable_by_time(out)


register(EgeAdapter())
```

Add `from . import ege  # noqa: F401` to `adapters/__init__.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_adapter_ege.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/focuswatch_dataset/adapters/ege.py tests/test_adapter_ege.py
git commit -m "feat: ETH Ege adapter with a physics-gated gyro rename"
```

---

### Task 9: ETH SensorLogger adapter

Three quirks. Units depend on the per-recording `standardisation` flag in `Metadata.csv`, and headphone gravity is written in m/s² regardless of it. Pen strokes and markers live in the session JSON, not in the empty `Annotation.csv`. `WatchAccelerometerUncalibrated.csv` is a second, raw acceleration stream with its own rate and goes to its own table.

**Files:**
- Create: `src/focuswatch_dataset/adapters/sensorlogger.py`
- Test: `tests/test_adapter_sensorlogger.py`

**Interfaces:**
- Consumes: `base`, `time_axis`, `schema`
- Produces: `SensorLoggerAdapter` with `name = "sensorlogger"`, registered on import.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_adapter_sensorlogger.py
import json

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.transform import Rotation

from focuswatch_dataset import schema as S
from focuswatch_dataset.adapters.sensorlogger import SensorLoggerAdapter
from focuswatch_dataset.validate import validate_motion_table

T0_NS = 1780853585220_000_000


def write_fixture(root, sid="E2_session6", n=400, standardisation=False, with_pen=True,
                  with_rawaccel=True):
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
    pd.DataFrame({
        "time": t[:200], "seconds_elapsed": np.arange(200) / 100,
        "rotationRateX": 0.01, "rotationRateY": 0.01, "rotationRateZ": 0.01,
        "gravityX": grav[:200, 0] * S.G_TO_MS2, "gravityY": grav[:200, 1] * S.G_TO_MS2,
        "gravityZ": grav[:200, 2] * S.G_TO_MS2,
        "accelerationX": rng.normal(0, 0.02, 200) * scale,
        "accelerationY": rng.normal(0, 0.02, 200) * scale,
        "accelerationZ": rng.normal(0, 0.02, 200) * scale,
        "quaternionW": q[:200, 3], "quaternionX": q[:200, 0],
        "quaternionY": q[:200, 1], "quaternionZ": q[:200, 2],
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
        "sensors": "Wrist Motion|Headphone|Annotation", "sampleRateMs": "10|10|",
        "standardisation": str(standardisation).lower(), "platform version": "26.5",
    }]).to_csv(d / "Metadata.csv", index=False)

    (d / "Annotation.csv").write_text("")

    events = [{"t_ms": int(t[0] // 1_000_000), "t_session_ms": 0.0, "event": "session_start",
               "session_id": "x", "payload": {}},
              {"t_ms": int(t[10] // 1_000_000), "t_session_ms": 100.0, "event": "pen_session_sync",
               "session_id": "x",
               "payload": {"pen_connected_t_ms": 1.0, "session_start_t_ms": 2.0,
                           "pen_minus_session_ms": 1.0}}]
    if with_pen:
        for i, ev in enumerate(["pen_down", "pen_move", "pen_move", "pen_up"]):
            events.append({"t_ms": int(t[20 + i] // 1_000_000), "t_session_ms": 200.0 + i,
                           "event": ev, "session_id": "x",
                           "payload": {"x": 11.68 + i, "y": 56.19, "force": 408,
                                       "tilt": {"x": 93, "y": 139, "twist": 9},
                                       "timestamp": 1716121921598}})
    (d / f"{sid}.json").write_text(json.dumps({"events": events}))
    return root


def test_units_are_harmonised_to_g_when_standardisation_is_on(tmp_path):
    write_fixture(tmp_path, standardisation=True)
    a = SensorLoggerAdapter()
    watch = a.load(a.discover(tmp_path)[0]).tables["watch"]
    norm = np.linalg.norm(watch[list(S.COLUMNS[S.Quantity.GRAVITY])].to_numpy(), axis=1)
    assert np.allclose(norm, 1.0, atol=1e-6)


def test_headphone_gravity_is_divided_even_when_standardisation_is_off(tmp_path):
    write_fixture(tmp_path, standardisation=False)
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


def test_missing_raw_accel_is_reported_not_faked(tmp_path):
    write_fixture(tmp_path, sid="E3_session1", with_rawaccel=False)
    a = SensorLoggerAdapter()
    ref = next(r for r in a.discover(tmp_path) if r.recording_id.endswith("E3_session1"))
    bundle = a.load(ref)
    assert "watch_rawaccel" not in bundle.tables
    assert bundle.meta["has_watch_rawaccel"] is False


def test_pen_comes_from_the_session_json(tmp_path):
    write_fixture(tmp_path)
    a = SensorLoggerAdapter()
    pen = a.load(a.discover(tmp_path)[0]).tables["pen"]
    assert set(pen["dot_type"]) == {"PEN_DOWN", "PEN_MOVE", "PEN_UP"}
    assert "src_timestamp" in pen.columns          # the pen device clock is kept as metadata


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_adapter_sensorlogger.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/focuswatch_dataset/adapters/sensorlogger.py
"""Adapter for the Sensor Logger iOS app export.

Unit handling follows the app's documented behaviour: motion channels are in g
when `standardisation` is off and in SI when it is on, but headphone gravity is
always written in m/s2. See awesome-sensor-logger/UNITS.md.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .. import schema as S
from ..time_axis import sort_stable_by_time, to_unix_ns
from .base import RecordingBundle, RecordingRef, register

COHORT = "ETH-SL"

_DOT_TYPES = {"pen_down": "PEN_DOWN", "pen_move": "PEN_MOVE", "pen_up": "PEN_UP"}


def _standardisation(meta_path: Path) -> bool:
    meta = pd.read_csv(meta_path)
    return str(meta["standardisation"].iloc[0]).strip().lower() == "true"


class SensorLoggerAdapter:
    name = "sensorlogger"

    def discover(self, root: Path) -> list[RecordingRef]:
        refs = []
        for d in sorted(p for p in root.iterdir() if (p / "WristMotion.csv").exists()):
            refs.append(RecordingRef(f"{COHORT}-{d.name}", f"ETH-{d.name}", "ETH", self.name, d))
        return refs

    def load(self, ref: RecordingRef) -> RecordingBundle:
        d = ref.path
        si = _standardisation(d / "Metadata.csv")
        accel_factor = 1.0 / S.G_TO_MS2 if si else 1.0

        tables = {"watch": self._motion(d / "WristMotion.csv", accel_factor, accel_factor)}

        head = d / "Headphone.csv"
        if head.exists():
            # Why: headphone gravity is m/s2 regardless of the standardisation flag.
            tables["headimu"] = self._motion(head, accel_factor, 1.0 / S.G_TO_MS2)

        rawaccel = d / "WatchAccelerometerUncalibrated.csv"
        if rawaccel.exists():
            tables["watch_rawaccel"] = self._rawaccel(rawaccel, accel_factor)

        pen, markers = self._from_session_json(d)
        if pen is not None:
            tables["pen"] = pen
        if markers is not None:
            tables["markers"] = markers

        meta = {
            "watch_hz_nominal": 100.0,
            "has_gravity": True, "has_quaternion": True,
            "has_watch_rawaccel": "watch_rawaccel" in tables,
            "has_pen": "pen" in tables,
            "accel_semantics": "user", "accel_calibration": "fused",
            "gravity_source": "measured",
            "time_domain": "backend_wall_clock", "time_alignment": "shared_clock",
            "protocol_id": "eth_ege_web", "study_mode": "study",
            "pen_xy_unit": "webapp_raw", "pen_pressure_scale": "webapp_force",
            "src_standardisation": si,
            "session_start_ns": self._session_start_ns(d / "Metadata.csv"),
        }
        return RecordingBundle(ref, tables, meta)

    @staticmethod
    def _session_start_ns(meta_path: Path) -> int | None:
        epoch_ms = pd.read_csv(meta_path)["recording epoch time"].iloc[0]
        return int(epoch_ms) * 1_000_000 if pd.notna(epoch_ms) else None

    def _motion(self, path: Path, accel_factor: float, gravity_factor: float) -> pd.DataFrame:
        raw = pd.read_csv(path)
        out = pd.DataFrame({"t_ns": to_unix_ns(raw["time"].to_numpy(), "ns")})
        out[list(S.COLUMNS[S.Quantity.ACCEL_USER])] = (
            raw[["accelerationX", "accelerationY", "accelerationZ"]].astype(float).to_numpy() * accel_factor)
        out[list(S.COLUMNS[S.Quantity.GYRO])] = (
            raw[["rotationRateX", "rotationRateY", "rotationRateZ"]].astype(float).to_numpy())
        out[list(S.COLUMNS[S.Quantity.GRAVITY])] = (
            raw[["gravityX", "gravityY", "gravityZ"]].astype(float).to_numpy() * gravity_factor)
        out[list(S.COLUMNS[S.Quantity.QUAT])] = (
            raw[["quaternionX", "quaternionY", "quaternionZ", "quaternionW"]].astype(float).to_numpy())
        return sort_stable_by_time(out)

    def _rawaccel(self, path: Path, factor: float) -> pd.DataFrame:
        raw = pd.read_csv(path)
        out = pd.DataFrame({"t_ns": to_unix_ns(raw["time"].to_numpy(), "ns")})
        out[list(S.COLUMNS[S.Quantity.ACCEL_TOTAL])] = raw[["x", "y", "z"]].astype(float).to_numpy() * factor
        return sort_stable_by_time(out)

    def _from_session_json(self, d: Path) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
        candidates = list(d.glob("*.json"))
        if len(candidates) != 1:
            raise ValueError(f"expected exactly one session JSON in {d}, found {candidates}")
        events = json.loads(candidates[0].read_text())["events"]

        strokes = [e for e in events if e["event"] in _DOT_TYPES]
        pen = None
        if strokes:
            pen = sort_stable_by_time(pd.DataFrame({
                # Why: no float cast. float64 resolves only to 256 ns at wall-clock
                # magnitude, which would shift every timestamp and can collapse
                # neighbouring samples onto the same value.
                "t_ns": to_unix_ns(np.array([e["t_ms"] for e in strokes], dtype=np.int64), "ms"),
                "dot_type": [_DOT_TYPES[e["event"]] for e in strokes],
                "x": [e["payload"].get("x", np.nan) for e in strokes],
                "y": [e["payload"].get("y", np.nan) for e in strokes],
                "pressure": [e["payload"].get("force", np.nan) for e in strokes],
                "tilt_x": [e["payload"].get("tilt", {}).get("x", np.nan) for e in strokes],
                "tilt_y": [e["payload"].get("tilt", {}).get("y", np.nan) for e in strokes],
                # Why: the pen's own clock, roughly 749 days behind wall clock. Metadata only.
                "src_timestamp": [e["payload"].get("timestamp", np.nan) for e in strokes],
            }))

        others = [e for e in events if e["event"] not in _DOT_TYPES]
        markers = None
        if others:
            markers = sort_stable_by_time(pd.DataFrame({
                "t_ns": to_unix_ns(np.array([e["t_ms"] for e in others], dtype=np.int64), "ms"),
                "event": [e["event"] for e in others],
                "task_id": "", "task_name": "", "task_index": -1,
                "task_category": "", "protocol_id": "eth_ege_web",
                "src_payload": [json.dumps(e.get("payload", {})) for e in others],
                "src_t_session_ms": [e.get("t_session_ms", np.nan) for e in others],
            }))
        return pen, markers


register(SensorLoggerAdapter())
```

Add `from . import sensorlogger  # noqa: F401` to `adapters/__init__.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_adapter_sensorlogger.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add src/focuswatch_dataset/adapters/sensorlogger.py tests/test_adapter_sensorlogger.py
git commit -m "feat: SensorLogger adapter with flag-driven unit handling"
```

---

### Task 10: AirPods attention adapter

25 recordings of head IMU at roughly 25 Hz. The per-sample `label` column is an expansion of the interval file and is dropped; the intervals are the raw annotation.

Source layout: `{root}/P{n}_{dur}_airpod_motion_{date}_{time}_labeled.csv` plus `{root}/1_Data_Protocols+GroundTruth/P{n}_ground_truth_{date}_{time}.txt` in `M:SS label` form.

**Files:**
- Create: `src/focuswatch_dataset/adapters/airpods.py`
- Test: `tests/test_adapter_airpods.py`

**Interfaces:**
- Consumes: `base`, `time_axis`, `schema`
- Produces: `AirPodsAdapter` with `name = "airpods"`, registered on import; `parse_ground_truth(text: str) -> list[tuple[float, str]]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_adapter_airpods.py
import numpy as np
import pandas as pd
import pytest
from scipy.spatial.transform import Rotation

from focuswatch_dataset import schema as S
from focuswatch_dataset.adapters.airpods import AirPodsAdapter, parse_ground_truth
from focuswatch_dataset.validate import validate_motion_table

GT = "0:00 focused\n4:00 distracted\n9:30 focused\n"


def write_fixture(root, pid="P1", n=1500, fs=25.0):
    rng = np.random.default_rng(3)
    rots = Rotation.random(n, random_state=3)
    q, grav = rots.as_quat(), rots.inv().apply([0.0, 0.0, -1.0])
    t0 = pd.Timestamp("2026-04-28T12:29:29.515Z")
    iso = (t0 + pd.to_timedelta(np.arange(n) / fs, unit="s")).strftime("%Y-%m-%dT%H:%M:%S.%f").str[:-3] + "Z"
    labels = np.where(np.arange(n) / fs < 240, "focused", "distracted")
    pd.DataFrame({
        "timestamp_iso": iso, "sensor_timestamp_s": 6679.42 + np.arange(n) / fs,
        "attitude_roll_rad": 0.0, "attitude_pitch_rad": 0.0, "attitude_yaw_rad": 0.0,
        "quaternion_x": q[:, 0], "quaternion_y": q[:, 1], "quaternion_z": q[:, 2], "quaternion_w": q[:, 3],
        "rotation_rate_x_rad_s": rng.normal(0, 0.04, n),
        "rotation_rate_y_rad_s": rng.normal(0, 0.04, n),
        "rotation_rate_z_rad_s": rng.normal(0, 0.04, n),
        "gravity_x_g": grav[:, 0], "gravity_y_g": grav[:, 1], "gravity_z_g": grav[:, 2],
        "user_acceleration_x_g": rng.normal(0, 0.015, n),
        "user_acceleration_y_g": rng.normal(0, 0.015, n),
        "user_acceleration_z_g": rng.normal(0, 0.015, n),
        "label": labels,
    }).to_csv(root / f"{pid}_17m30s_airpod_motion_2026-04-28_14-47-30_labeled.csv", index=False)
    gt = root / "1_Data_Protocols+GroundTruth"
    gt.mkdir(exist_ok=True)
    (gt / f"{pid}_ground_truth_2026-04-28_13-40-36.txt").write_text(GT)
    return root


def test_parse_ground_truth():
    assert parse_ground_truth(GT) == [(0.0, "focused"), (240.0, "distracted"), (570.0, "focused")]


def test_discover_prefixes_the_participant_id(tmp_path):
    write_fixture(tmp_path, pid="P17")
    ref = AirPodsAdapter().discover(tmp_path)[0]
    # Why: AirPods P17 and ML4SCS P17 are different people.
    assert ref.recording_id == "AIRPODS-P17"
    assert ref.participant_id == "AIRPODS-P17"


def test_attention_table_holds_intervals_not_samples(tmp_path):
    write_fixture(tmp_path)
    a = AirPodsAdapter()
    att = a.load(a.discover(tmp_path)[0]).tables["attention"]
    assert list(att.columns) == ["t_start_ns", "t_end_ns", "label"]
    assert len(att) == 3
    assert att["t_end_ns"].iloc[0] - att["t_start_ns"].iloc[0] == 240 * 1_000_000_000


def test_per_sample_label_is_dropped(tmp_path):
    write_fixture(tmp_path)
    a = AirPodsAdapter()
    head = a.load(a.discover(tmp_path)[0]).tables["headimu"]
    assert "label" not in head.columns


def test_expansion_crosscheck_rejects_a_wrong_clock_anchor(tmp_path):
    # The interval offsets are relative to an unstated clock. Before the
    # per-sample column is dropped it is used to prove the anchor.
    write_fixture(tmp_path)
    gt = next((tmp_path / "1_Data_Protocols+GroundTruth").glob("*.txt"))
    gt.write_text("0:00 distracted\n4:00 focused\n9:30 distracted\n")   # inverted
    a = AirPodsAdapter()
    with pytest.raises(ValueError, match="anchor is wrong"):
        a.load(a.discover(tmp_path)[0])


def test_no_watch_table_and_flags_say_so(tmp_path):
    write_fixture(tmp_path)
    a = AirPodsAdapter()
    bundle = a.load(a.discover(tmp_path)[0])
    assert "watch" not in bundle.tables
    assert bundle.meta["has_watch"] is False
    assert bundle.meta["has_attention"] is True


def test_measured_head_rate_is_reported(tmp_path):
    write_fixture(tmp_path, fs=25.0)
    a = AirPodsAdapter()
    bundle = a.load(a.discover(tmp_path)[0])
    assert bundle.meta["head_hz_measured"] == pytest.approx(25.0, rel=0.05)


def test_loaded_headimu_passes_the_validator(tmp_path):
    write_fixture(tmp_path)
    a = AirPodsAdapter()
    head = a.load(a.discover(tmp_path)[0]).tables["headimu"]
    assert [f for f in validate_motion_table(head, "AIRPODS-P1", "headimu", 25.0)
            if not f.passed] == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_adapter_airpods.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/focuswatch_dataset/adapters/airpods.py
"""Adapter for the AirPods attention study (head IMU with observer annotation).

The per-sample `label` column in the source is an expansion of the interval
protocol; the intervals are the raw annotation and the expansion is dropped.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .. import schema as S
from ..time_axis import median_rate_hz, sort_stable_by_time
from .base import RecordingBundle, RecordingRef, register

COHORT = "AIRPODS"
_RECORDING = re.compile(r"^(P\d+)_.*_labeled\.csv$")
_GT_LINE = re.compile(r"^\s*(\d+):(\d{2})\s+(\S+)\s*$")

_MOTION_MAP = {
    "user_acceleration_x_g": "accel_user_x", "user_acceleration_y_g": "accel_user_y",
    "user_acceleration_z_g": "accel_user_z",
    "rotation_rate_x_rad_s": "gyro_x", "rotation_rate_y_rad_s": "gyro_y",
    "rotation_rate_z_rad_s": "gyro_z",
    "gravity_x_g": "gravity_x", "gravity_y_g": "gravity_y", "gravity_z_g": "gravity_z",
    "quaternion_x": "quat_x", "quaternion_y": "quat_y",
    "quaternion_z": "quat_z", "quaternion_w": "quat_w",
}


def parse_ground_truth(text: str) -> list[tuple[float, str]]:
    """Parse `M:SS label` lines into (offset_seconds, label) pairs."""
    out = []
    for line in text.splitlines():
        m = _GT_LINE.match(line)
        if m:
            out.append((int(m.group(1)) * 60 + int(m.group(2)), m.group(3)))
    return [(float(s), lab) for s, lab in out]


class AirPodsAdapter:
    name = "airpods"

    def discover(self, root: Path) -> list[RecordingRef]:
        refs = []
        for f in sorted(root.glob("*_labeled.csv")):
            m = _RECORDING.match(f.name)
            if m:
                rid = f"{COHORT}-{m.group(1)}"
                refs.append(RecordingRef(rid, rid, COHORT, self.name, f))
        return refs

    def load(self, ref: RecordingRef) -> RecordingBundle:
        raw = pd.read_csv(ref.path)
        t_ns = pd.to_datetime(raw["timestamp_iso"], format="ISO8601", utc=True).astype("int64").to_numpy()

        head = pd.DataFrame({"t_ns": t_ns})
        for src, dst in _MOTION_MAP.items():
            head[dst] = raw[src].astype(float)
        head["src_sensor_timestamp_s"] = raw["sensor_timestamp_s"]
        head = sort_stable_by_time(head)

        tables = {"headimu": head}
        attention = self._attention(ref, t_ns, raw["label"])
        if attention is not None:
            tables["attention"] = attention

        return RecordingBundle(ref, tables, {
            "has_watch": False, "has_pen": False, "has_markers": False,
            "has_attention": attention is not None,
            "has_head_gravity": True, "has_head_quaternion": True,
            "head_hz_nominal": 25.0,
            "head_hz_measured": round(median_rate_hz(head["t_ns"].to_numpy()), 3),
            "accel_semantics": "user", "accel_calibration": "fused",
            "gravity_source": "measured",
            "time_domain": "device_wall_clock", "time_alignment": "shared_clock",
            "protocol_id": "airpods_attention", "study_mode": "study",
        })

    def _attention(self, ref: RecordingRef, t_ns: np.ndarray,
                   source_labels: pd.Series) -> pd.DataFrame | None:
        pid = ref.recording_id.removeprefix(f"{COHORT}-")
        gt_dir = ref.path.parent / "1_Data_Protocols+GroundTruth"
        matches = sorted(gt_dir.glob(f"{pid}_ground_truth_*.txt")) if gt_dir.exists() else []
        if not matches:
            return None
        intervals = parse_ground_truth(matches[0].read_text())
        if not intervals:
            return None
        t0, t_end = int(t_ns.min()), int(t_ns.max())
        starts = [t0 + int(round(s * 1e9)) for s, _ in intervals]
        ends = starts[1:] + [t_end]
        table = pd.DataFrame({"t_start_ns": np.array(starts, dtype=np.int64),
                              "t_end_ns": np.array(ends, dtype=np.int64),
                              "label": [lab for _, lab in intervals]})
        self._verify_expansion(table, t_ns, source_labels)
        return table

    @staticmethod
    def _verify_expansion(table: pd.DataFrame, t_ns: np.ndarray,
                          source_labels: pd.Series, min_agreement: float = 0.99) -> None:
        """Check the interval anchor against the source's own per-sample labels.

        The intervals are offsets from an unstated clock; mapping them onto the
        first sample is an assumption. The per-sample column we are about to drop
        is that same assumption already applied, so re-expanding and comparing
        turns the discarded derivative into a test of the anchor.
        """
        expanded = np.empty(len(t_ns), dtype=object)
        for _, row in table.iterrows():
            expanded[(t_ns >= row.t_start_ns) & (t_ns <= row.t_end_ns)] = row.label
        agreement = float((expanded == source_labels.to_numpy()).mean())
        if agreement < min_agreement:
            raise ValueError(
                f"interval expansion matches only {agreement:.3f} of the source labels; "
                "the ground-truth clock anchor is wrong"
            )


register(AirPodsAdapter())
```

Add `from . import airpods  # noqa: F401` to `adapters/__init__.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_adapter_airpods.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add src/focuswatch_dataset/adapters/airpods.py tests/test_adapter_airpods.py
git commit -m "feat: AirPods attention adapter with interval ground truth"
```

---

### Task 11: Deterministic Parquet writer

**Files:**
- Create: `src/focuswatch_dataset/write.py`
- Test: `tests/test_write.py`

**Interfaces:**
- Consumes: `schema`
- Produces:
  - `write_table(df: pd.DataFrame, path: Path, recording_id: str) -> None`
  - `read_table(path: Path) -> pd.DataFrame`
  - `table_metadata(path: Path) -> dict[str, str]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_write.py
import hashlib

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from focuswatch_dataset import schema as S
from focuswatch_dataset.write import read_table, table_metadata, write_table


def sample(n=1000):
    rng = np.random.default_rng(4)
    return pd.DataFrame({
        "t_ns": np.arange(n, dtype=np.int64) * 10_000_000 + 1780577357025_000_000,
        "accel_user_x": rng.normal(0, 0.04, n),
        "accel_user_y": rng.normal(0, 0.04, n),
        "accel_user_z": rng.normal(0, 0.04, n),
        "dot_type": ["PEN_MOVE"] * n,
    })


def test_roundtrip_is_bit_identical(tmp_path):
    df = sample()
    p = tmp_path / "r.parquet"
    write_table(df, p, "R1")
    back = read_table(p)
    pd.testing.assert_frame_equal(df, back)
    assert (df["accel_user_x"].to_numpy() == back["accel_user_x"].to_numpy()).all()


def test_floats_stay_float64(tmp_path):
    p = tmp_path / "r.parquet"
    write_table(sample(), p, "R1")
    assert read_table(p)["accel_user_x"].dtype == np.float64


def test_metadata_carries_schema_version_and_id(tmp_path):
    p = tmp_path / "r.parquet"
    write_table(sample(), p, "ML4SCS-S096")
    md = table_metadata(p)
    assert md["recording_id"] == "ML4SCS-S096"
    assert md["schema_version"] == S.SCHEMA_VERSION


def test_writes_are_byte_identical_across_runs(tmp_path):
    df = sample()
    a, b = tmp_path / "a.parquet", tmp_path / "b.parquet"
    write_table(df, a, "R1")
    write_table(df, b, "R1")
    assert hashlib.sha256(a.read_bytes()).hexdigest() == hashlib.sha256(b.read_bytes()).hexdigest()


def test_encodings_are_applied(tmp_path):
    p = tmp_path / "r.parquet"
    write_table(sample(), p, "R1")
    meta = pq.ParquetFile(p).metadata.row_group(0)
    by_name = {meta.column(i).path_in_schema: meta.column(i) for i in range(meta.num_columns)}
    assert "BYTE_STREAM_SPLIT" in str(by_name["accel_user_x"].encodings)
    assert "DELTA_BINARY_PACKED" in str(by_name["t_ns"].encodings)


def test_compression_beats_uncompressed_csv(tmp_path):
    df = sample(20_000)
    p, c = tmp_path / "r.parquet", tmp_path / "r.csv"
    write_table(df, p, "R1")
    df.to_csv(c, index=False)
    assert p.stat().st_size < c.stat().st_size / 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_write.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/focuswatch_dataset/write.py
"""Deterministic Parquet writer.

Encoding choices are fixed here rather than exposed: byte-stream split exploits
the shared exponent bytes of neighbouring IMU samples, delta packing exploits
the constant timestamp spacing.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import schema as S


def _column_encoding(table: pa.Table) -> dict[str, str]:
    enc: dict[str, str] = {}
    for field in table.schema:
        if pa.types.is_floating(field.type):
            enc[field.name] = "BYTE_STREAM_SPLIT"
        elif field.name == S.TIME_COLUMN:
            enc[field.name] = "DELTA_BINARY_PACKED"
    return enc


def write_table(df: pd.DataFrame, path: Path, recording_id: str) -> None:
    table = pa.Table.from_pandas(df, preserve_index=False)
    table = table.replace_schema_metadata({
        "schema_version": S.SCHEMA_VERSION,
        "recording_id": recording_id,
    })
    encoding = _column_encoding(table)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        table, path,
        compression="zstd",
        column_encoding=encoding,
        # Why: pyarrow rejects dictionary encoding together with an explicit
        # column_encoding, and dictionaries buy nothing on continuous signals.
        use_dictionary=[c for c in table.schema.names if c not in encoding],
        write_statistics=True,
        # Why: store_schema=False would drop the key-value metadata set above.
        # replace_schema_metadata has already removed the non-deterministic
        # pandas block, so writes stay byte-identical across runs.
    )


def read_table(path: Path) -> pd.DataFrame:
    return pq.read_table(path).to_pandas()


def table_metadata(path: Path) -> dict[str, str]:
    raw = pq.ParquetFile(path).schema_arrow.metadata or {}
    return {k.decode(): v.decode() for k, v in raw.items()}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_write.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/focuswatch_dataset/write.py tests/test_write.py
git commit -m "feat: deterministic Parquet writer"
```

---

### Task 12: Manifest and channel description

**Files:**
- Create: `src/focuswatch_dataset/manifest.py`
- Test: `tests/test_manifest.py`

**Interfaces:**
- Consumes: `base.RecordingBundle`, `schema`, `time_axis.median_rate_hz`, `physics.norm_stats`
- Produces:
  - `MANIFEST_COLUMNS: tuple[str, ...]`
  - `build_manifest(bundles: list[RecordingBundle]) -> pd.DataFrame`
  - `build_channels(bundles: list[RecordingBundle]) -> pd.DataFrame`
  - `check_manifest_consistency(manifest: pd.DataFrame, root: Path) -> list[str]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_manifest.py
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from focuswatch_dataset import schema as S
from focuswatch_dataset.adapters.base import RecordingBundle, RecordingRef
from focuswatch_dataset.manifest import (
    MANIFEST_COLUMNS, build_channels, build_manifest, check_manifest_consistency,
)


def bundle(rid="ML4SCS-S096", with_pen=True, **meta):
    n = 200
    watch = pd.DataFrame({
        "t_ns": np.arange(n, dtype=np.int64) * 10_000_000,
        **{c: np.zeros(n) for c in S.COLUMNS[S.Quantity.ACCEL_USER]},
        **{c: np.zeros(n) for c in S.COLUMNS[S.Quantity.GYRO]},
    })
    tables = {"watch": watch}
    if with_pen:
        tables["pen"] = pd.DataFrame({"t_ns": [0, 1], "dot_type": ["PEN_DOWN", "PEN_UP"],
                                      "x": [1.0, 2.0], "y": [1.0, 2.0]})
    base = {"watch_hz_nominal": 100.0, "accel_semantics": "user",
            "accel_calibration": "fused", "gravity_source": "none",
            "time_domain": "watch_capture_clock", "time_alignment": "estimated_delta",
            "protocol_id": "ml4scs_v2"}
    return RecordingBundle(RecordingRef(rid, "ML4SCS-P76", "ML4SCS", "ml4scs", Path(".")),
                           tables, base | meta)


def test_manifest_has_all_declared_columns():
    m = build_manifest([bundle()])
    assert set(MANIFEST_COLUMNS) <= set(m.columns)


def test_modality_flags_follow_the_tables():
    m = build_manifest([bundle(with_pen=False)]).iloc[0]
    assert m["has_watch"] is np.True_ or m["has_watch"] is True
    assert not m["has_pen"]
    assert not m["has_attention"]


def test_measured_rate_is_derived_not_copied():
    m = build_manifest([bundle()]).iloc[0]
    assert m["watch_hz_measured"] == pytest.approx(100.0)


def test_channels_declare_a_unit_for_every_signal_column():
    ch = build_channels([bundle()])
    signal = ch[ch["column"] != "t_ns"]
    assert (signal["unit"] != "").all()
    assert (signal["semantics"] != "").all()
    assert set(ch[ch["column"] == "accel_user_x"]["unit"]) == {"g"}


def test_consistency_flags_a_missing_file(tmp_path):
    m = build_manifest([bundle()])
    problems = check_manifest_consistency(m, tmp_path)
    assert any("watch/ML4SCS-S096.parquet" in p for p in problems)


def test_consistency_is_clean_when_files_exist(tmp_path):
    m = build_manifest([bundle(with_pen=False)])
    (tmp_path / "watch").mkdir()
    (tmp_path / "watch" / "ML4SCS-S096.parquet").write_bytes(b"x")
    assert check_manifest_consistency(m, tmp_path) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_manifest.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/focuswatch_dataset/manifest.py
"""The manifest is the single source of truth for capability flags.

Parquet key-value metadata and the Frictionless descriptor are generated from it;
nothing else is maintained independently.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import schema as S
from .adapters.base import RecordingBundle
from .time_axis import median_rate_hz

MANIFEST_COLUMNS = (
    "recording_id", "participant_id", "cohort", "pipeline",
    "has_watch", "has_watch_rawaccel", "has_headimu", "has_pen", "has_markers", "has_attention",
    "watch_hz_nominal", "watch_hz_measured", "has_gravity", "has_quaternion",
    "accel_semantics", "accel_calibration", "accel_still_bias", "gravity_source",
    "head_hz_nominal", "head_hz_measured", "has_head_gravity", "has_head_quaternion",
    "time_domain", "time_alignment", "t_start_ns", "t_end_ns", "duration_s",
    "protocol_id", "study_mode", "subject_index",
    "watch_wrist_side",
    "pen_xy_unit", "pen_pressure_scale", "pen_delta_s", "pen_delta_sigma",
    "delta_applied", "alignment_note",
    "n_samples_watch", "n_samples_pen", "n_samples_head", "issue_codes",
    "source_pipeline", "schema_version", "redaction_policy", "build_git_sha",
)

_DEFAULTS: dict[str, object] = {
    "watch_wrist_side": "unknown", "delta_applied": False,
    "pen_delta_s": np.nan, "pen_delta_sigma": np.nan, "alignment_note": "",
    "accel_still_bias": np.nan, "issue_codes": "", "redaction_policy": "none",
    "build_git_sha": "", "subject_index": -1, "study_mode": "",
    "head_hz_nominal": np.nan, "head_hz_measured": np.nan,
    "has_head_gravity": False, "has_head_quaternion": False,
    "pen_xy_unit": "", "pen_pressure_scale": "",
}


def _span(bundle: RecordingBundle) -> tuple[int, int]:
    starts, ends = [], []
    for df in bundle.tables.values():
        if not len(df):
            continue
        # Why: interval tables carry start and end separately. Reading the end
        # from t_start_ns would drop the final interval from the duration.
        if {"t_start_ns", "t_end_ns"} <= set(df.columns):
            starts.append(int(df["t_start_ns"].min()))
            ends.append(int(df["t_end_ns"].max()))
        elif "t_ns" in df.columns:
            starts.append(int(df["t_ns"].min()))
            ends.append(int(df["t_ns"].max()))
    return (min(starts), max(ends)) if starts else (0, 0)


def build_manifest(bundles: list[RecordingBundle]) -> pd.DataFrame:
    rows = []
    for b in bundles:
        t0, t1 = _span(b)
        row: dict[str, object] = dict(_DEFAULTS)
        row.update({
            "recording_id": b.ref.recording_id, "participant_id": b.ref.participant_id,
            "cohort": b.ref.cohort, "pipeline": b.ref.pipeline,
            "source_pipeline": b.ref.pipeline, "schema_version": S.SCHEMA_VERSION,
            "t_start_ns": t0, "t_end_ns": t1, "duration_s": round((t1 - t0) / 1e9, 3),
            "n_samples_watch": len(b.tables.get("watch", [])),
            "n_samples_pen": len(b.tables.get("pen", [])),
            "n_samples_head": len(b.tables.get("headimu", [])),
        })
        for m in S.MODALITIES:
            row[f"has_{m}"] = m in b.tables
        if "watch" in b.tables:
            w = b.tables["watch"]
            row["watch_hz_measured"] = round(median_rate_hz(w["t_ns"].to_numpy()), 3)
            row["has_gravity"] = "gravity_x" in w.columns
            row["has_quaternion"] = "quat_x" in w.columns
        if "headimu" in b.tables:
            h = b.tables["headimu"]
            row["head_hz_measured"] = round(median_rate_hz(h["t_ns"].to_numpy()), 3)
            row["has_head_gravity"] = "gravity_x" in h.columns
            row["has_head_quaternion"] = "quat_x" in h.columns
        row.update(b.meta)
        rows.append(row)
    df = pd.DataFrame(rows)
    for c in MANIFEST_COLUMNS:
        if c not in df.columns:
            df[c] = _DEFAULTS.get(c, np.nan)
    return df[list(MANIFEST_COLUMNS)]


def build_channels(bundles: list[RecordingBundle]) -> pd.DataFrame:
    col_to_quantity = {c: q for q, cols in S.COLUMNS.items() for c in cols}
    rows = []
    for b in bundles:
        for modality, df in b.tables.items():
            hz = median_rate_hz(df["t_ns"].to_numpy()) if "t_ns" in df.columns and len(df) else np.nan
            for column in df.columns:
                q = col_to_quantity.get(column)
                rows.append({
                    "recording_id": b.ref.recording_id, "modality": modality, "column": column,
                    "quantity": q.value if q else ("time" if column.startswith("t_") else "other"),
                    "unit": S.UNITS[q] if q else ("ns" if column.startswith("t_") else ""),
                    "semantics": q.value if q else ("timestamp" if column.startswith("t_") else "source_metadata"),
                    "frame": "device" if q else "",
                    "sample_rate_hz": round(hz, 3) if np.isfinite(hz) else np.nan,
                    "unit_conversion_factor": 1.0,
                })
    return pd.DataFrame(rows)


def check_manifest_consistency(manifest: pd.DataFrame, root: Path) -> list[str]:
    problems = []
    for _, row in manifest.iterrows():
        for m in S.MODALITIES:
            path = root / m / f"{row['recording_id']}.parquet"
            declared, exists = bool(row[f"has_{m}"]), path.exists()
            if declared and not exists:
                problems.append(f"{m}/{row['recording_id']}.parquet declared but missing")
            if exists and not declared:
                problems.append(f"{m}/{row['recording_id']}.parquet present but not declared")
    return problems
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_manifest.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/focuswatch_dataset/manifest.py tests/test_manifest.py
git commit -m "feat: manifest and channel description"
```

---

### Task 13: Pen redaction policy

Off by default. The switch must be one config value plus a rebuild, and the applied policy must be recorded in the manifest.

**Files:**
- Create: `src/focuswatch_dataset/redact.py`
- Test: `tests/test_redact.py`

**Interfaces:**
- Consumes: `schema`
- Produces:
  - `RedactionPolicy` StrEnum: `NONE`, `FREE_WRITING_XY`, `ALL_XY`
  - `apply_redaction(pen: pd.DataFrame, markers: pd.DataFrame | None, policy: RedactionPolicy) -> pd.DataFrame`
  - `FREE_WRITING_TASKS: frozenset[str]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_redact.py
import numpy as np
import pandas as pd

from focuswatch_dataset.redact import RedactionPolicy, apply_redaction


def pen():
    return pd.DataFrame({"t_ns": [10, 20, 30, 40], "dot_type": ["PEN_DOWN"] * 4,
                         "x": [1.0, 2.0, 3.0, 4.0], "y": [1.0, 2.0, 3.0, 4.0],
                         "pressure": [300.0] * 4})


def markers():
    return pd.DataFrame({
        "t_ns": [5, 25, 26, 45],
        "event": ["task_start", "task_end", "task_start", "task_end"],
        "task_id": ["abschreiben", "abschreiben", "free_writing", "free_writing"],
        "task_index": [0, 0, 1, 1], "task_category": ["writing"] * 4,
        "protocol_id": ["ml4scs_v2"] * 4,
    })


def test_none_is_the_default_and_changes_nothing():
    pd.testing.assert_frame_equal(apply_redaction(pen(), markers(), RedactionPolicy.NONE), pen())


def test_free_writing_xy_blanks_only_the_free_writing_block():
    out = apply_redaction(pen(), markers(), RedactionPolicy.FREE_WRITING_XY)
    assert out.loc[:1, ["x", "y"]].notna().all().all()
    assert out.loc[2:, ["x", "y"]].isna().all().all()
    # Non-positional channels survive: the label only needs dot_type and time.
    assert out["pressure"].notna().all()
    assert out["dot_type"].tolist() == ["PEN_DOWN"] * 4


def test_all_xy_blanks_everything_positional():
    out = apply_redaction(pen(), markers(), RedactionPolicy.ALL_XY)
    assert out[["x", "y"]].isna().all().all()


def test_free_writing_without_markers_falls_back_to_all_xy():
    out = apply_redaction(pen(), None, RedactionPolicy.FREE_WRITING_XY)
    assert out[["x", "y"]].isna().all().all()


def test_markers_without_task_structure_also_fall_back():
    # The ETH sources emit phase events with empty task ids. Matching nothing
    # would silently publish every coordinate under a redaction policy.
    structureless = pd.DataFrame({
        "t_ns": [5, 45], "event": ["session_start", "session_end"],
        "task_id": ["", ""], "task_index": [-1, -1],
        "task_category": ["", ""], "protocol_id": ["eth_ege_web"] * 2,
    })
    out = apply_redaction(pen(), structureless, RedactionPolicy.FREE_WRITING_XY)
    assert out[["x", "y"]].isna().all().all()


def test_row_count_never_changes():
    for p in RedactionPolicy:
        assert len(apply_redaction(pen(), markers(), p)) == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_redact.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/focuswatch_dataset/redact.py
"""Optional removal of pen coordinates.

Stroke geometry reconstructs the written text. Structured tasks have prescribed
content; free-writing blocks do not. The policy runs on the canonical pen table
so it covers every source, and it blanks values rather than dropping rows.
"""
from __future__ import annotations

from enum import StrEnum

import numpy as np
import pandas as pd

XY_COLUMNS = ("x", "y")
FREE_WRITING_TASKS = frozenset({"free_writing", "think_pause_writing"})


class RedactionPolicy(StrEnum):
    NONE = "none"
    FREE_WRITING_XY = "free_writing_xy"
    ALL_XY = "all_xy"


def _free_writing_spans(markers: pd.DataFrame) -> list[tuple[int, int]]:
    spans = []
    for task_index, block in markers[markers["task_id"].isin(FREE_WRITING_TASKS)].groupby("task_index"):
        starts = block.loc[block["event"] == "task_start", "t_ns"]
        ends = block.loc[block["event"] == "task_end", "t_ns"]
        if len(starts) and len(ends):
            spans.append((int(starts.min()), int(ends.max())))
    return spans


def apply_redaction(pen: pd.DataFrame, markers: pd.DataFrame | None,
                    policy: RedactionPolicy) -> pd.DataFrame:
    if policy is RedactionPolicy.NONE:
        return pen
    out = pen.copy()
    present = [c for c in XY_COLUMNS if c in out.columns]
    if policy is RedactionPolicy.ALL_XY:
        out[present] = np.nan
        return out
    spans = _free_writing_spans(markers) if markers is not None and not markers.empty else []
    if not _has_task_structure(markers):
        # Why: an absent marker table and a marker table without task ids are the
        # same situation - we cannot tell free writing apart, so we do not guess.
        # The ETH sources emit phase events without task ids and land here.
        out[present] = np.nan
        return out
    mask = pd.Series(False, index=out.index)
    for lo, hi in spans:
        mask |= out["t_ns"].between(lo, hi)
    out.loc[mask, present] = np.nan
    return out


def _has_task_structure(markers: pd.DataFrame | None) -> bool:
    if markers is None or markers.empty or "task_id" not in markers.columns:
        return False
    return markers["task_id"].astype(str).str.strip().ne("").any()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_redact.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/focuswatch_dataset/redact.py tests/test_redact.py
git commit -m "feat: pen redaction policy, disabled by default"
```

---

### Task 14: Consumer API

This is the surface a reuser touches. `load_recordings` refuses to mix acceleration semantics, because that mix is invisible after per-session normalisation.

**Files:**
- Create: `src/focuswatch_dataset/load.py`
- Create: `src/focuswatch_dataset/select.py`
- Create: `src/focuswatch_dataset/convert.py`
- Test: `tests/test_load.py`
- Test: `tests/test_convert.py`

**Interfaces:**
- Consumes: `write.read_table`, `schema`, `physics.gravity_from_quaternion`
- Produces:
  - `load_manifest(root: Path | str) -> pd.DataFrame`
  - `load_channels(root: Path | str) -> pd.DataFrame`
  - `load_recording(root, recording_id: str, modality: str = "watch") -> pd.DataFrame`
  - `load_recordings(root, recording_ids: list[str], modality: str = "watch") -> pd.DataFrame`
  - `select.by_flags(manifest: pd.DataFrame, **flags) -> pd.DataFrame`
  - `convert.to_user_acceleration(df: pd.DataFrame) -> pd.DataFrame`

`to_user_acceleration` is the escape hatch the spec promises in §5: the dataset
declares semantics rather than converting them, and a reuser who needs one
common representation calls this. It is a derivation, not a measurement — hence
it lives on the read side and never runs during a build.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_load.py
import numpy as np
import pandas as pd
import pytest

from focuswatch_dataset import select
from focuswatch_dataset.load import load_manifest, load_recording, load_recordings
from focuswatch_dataset.write import write_table


def build_bundle(root):
    rows = []
    for rid, sem, hz, grav in [("A", "user", 100.0, True), ("B", "user", 50.0, False),
                               ("C", "total", 100.0, False)]:
        col = f"accel_{sem}_x"
        df = pd.DataFrame({"t_ns": np.arange(50, dtype=np.int64) * 10_000_000,
                           col: np.zeros(50)})
        write_table(df, root / "watch" / f"{rid}.parquet", rid)
        rows.append({"recording_id": rid, "has_watch": True, "has_pen": rid != "C",
                     "watch_hz_nominal": hz, "has_gravity": grav, "accel_semantics": sem})
    m = pd.DataFrame(rows)
    m.to_parquet(root / "sessions.parquet", index=False)
    m.to_csv(root / "sessions.csv", index=False)
    return root


def test_load_manifest(tmp_path):
    build_bundle(tmp_path)
    assert len(load_manifest(tmp_path)) == 3


def test_load_recording(tmp_path):
    build_bundle(tmp_path)
    assert len(load_recording(tmp_path, "A")) == 50


def test_by_flags_filters(tmp_path):
    build_bundle(tmp_path)
    m = load_manifest(tmp_path)
    got = select.by_flags(m, has_watch=True, has_pen=True, watch_hz_nominal=100.0)
    assert got["recording_id"].tolist() == ["A"]


def test_query_string_works_too(tmp_path):
    build_bundle(tmp_path)
    m = load_manifest(tmp_path)
    assert m.query("has_watch and has_gravity")["recording_id"].tolist() == ["A"]


def test_load_recordings_concatenates_and_tags(tmp_path):
    build_bundle(tmp_path)
    out = load_recordings(tmp_path, ["A", "B"])
    assert len(out) == 100
    assert set(out["recording_id"]) == {"A", "B"}


def test_load_recordings_refuses_mixed_accel_semantics(tmp_path):
    build_bundle(tmp_path)
    with pytest.raises(ValueError, match="mixed acceleration semantics"):
        load_recordings(tmp_path, ["A", "C"])
```

```python
# tests/test_convert.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_load.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/focuswatch_dataset/convert.py
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
```

```python
# src/focuswatch_dataset/select.py
"""Flag-based subsetting over the manifest."""
from __future__ import annotations

import pandas as pd


def by_flags(manifest: pd.DataFrame, **flags) -> pd.DataFrame:
    out = manifest
    for column, value in flags.items():
        if column not in out.columns:
            raise KeyError(f"unknown manifest column: {column}")
        out = out[out[column] == value]
    return out.reset_index(drop=True)
```

```python
# src/focuswatch_dataset/load.py
"""Reader API for a published bundle."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .write import read_table


def load_manifest(root: Path | str) -> pd.DataFrame:
    return pd.read_parquet(Path(root) / "sessions.parquet")


def load_channels(root: Path | str) -> pd.DataFrame:
    return pd.read_parquet(Path(root) / "channels.parquet")


def load_recording(root: Path | str, recording_id: str, modality: str = "watch") -> pd.DataFrame:
    return read_table(Path(root) / modality / f"{recording_id}.parquet")


def load_recordings(root: Path | str, recording_ids: list[str],
                    modality: str = "watch") -> pd.DataFrame:
    manifest = load_manifest(root).set_index("recording_id")
    if "accel_semantics" in manifest.columns:
        semantics = set(manifest.loc[recording_ids, "accel_semantics"].dropna())
        if len(semantics) > 1:
            # Why: per-session normalisation hides this mismatch instead of surfacing it.
            raise ValueError(
                f"mixed acceleration semantics {sorted(semantics)}; convert with "
                "to_user_acceleration() or restrict the selection"
            )
    frames = []
    for rid in recording_ids:
        df = load_recording(root, rid, modality)
        df.insert(0, "recording_id", rid)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_load.py tests/test_convert.py -v`
Expected: 10 passed

- [ ] **Step 5: Commit**

```bash
git add src/focuswatch_dataset/load.py src/focuswatch_dataset/select.py \
        src/focuswatch_dataset/convert.py tests/test_load.py tests/test_convert.py
git commit -m "feat: consumer API with a semantics-homogeneity guard"
```

---

### Task 15: Build orchestration, CLI and generated documentation

**Files:**
- Create: `src/focuswatch_dataset/build.py`
- Create: `src/focuswatch_dataset/docs.py`
- Create: `src/focuswatch_dataset/cli.py`
- Test: `tests/test_build.py`

**Interfaces:**
- Consumes: every module above.
- Produces:
  - `build_dataset(source_roots: dict[str, Path], out: Path, policy: RedactionPolicy = RedactionPolicy.NONE, strict: bool = True) -> ValidationReport`
  - `docs.write_datapackage(out: Path, manifest, channels) -> None`
  - `docs.write_data_dictionary(out: Path, channels) -> None`
  - `cli.main(argv: list[str] | None = None) -> int` implementing `fw build`, `fw validate`, `fw report`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_build.py
import hashlib
import json

import pytest

from focuswatch_dataset.build import build_dataset
from focuswatch_dataset.cli import main
from focuswatch_dataset.load import load_manifest
from focuswatch_dataset.redact import RedactionPolicy

from .test_adapter_airpods import write_fixture as write_airpods
from .test_adapter_ege import write_fixture as write_ege
from .test_adapter_ml4scs import write_fixture as write_ml4scs
from .test_adapter_sensorlogger import write_fixture as write_sl


def sources(tmp_path):
    roots = {}
    for name, writer in (("ml4scs", write_ml4scs), ("ege", write_ege),
                         ("sensorlogger", write_sl), ("airpods", write_airpods)):
        d = tmp_path / "src" / name
        d.mkdir(parents=True)
        writer(d)
        roots[name] = d
    return roots


def test_build_produces_all_expected_artefacts(tmp_path):
    out = tmp_path / "out"
    build_dataset(sources(tmp_path), out)
    for f in ("sessions.parquet", "sessions.csv", "channels.parquet",
              "datapackage.json", "data_dictionary.md", "validation_report.json"):
        assert (out / f).exists(), f


def test_manifest_covers_every_cohort(tmp_path):
    out = tmp_path / "out"
    build_dataset(sources(tmp_path), out)
    assert set(load_manifest(out)["cohort"]) == {"ML4SCS", "ETH", "AIRPODS"}


def test_build_is_deterministic(tmp_path):
    src = sources(tmp_path)
    a, b = tmp_path / "a", tmp_path / "b"
    build_dataset(src, a)
    build_dataset(src, b)
    for f in sorted(p.relative_to(a) for p in a.rglob("*.parquet")):
        assert hashlib.sha256((a / f).read_bytes()).hexdigest() == \
               hashlib.sha256((b / f).read_bytes()).hexdigest(), f


def test_validation_report_is_written_and_all_checks_pass(tmp_path):
    out = tmp_path / "out"
    build_dataset(sources(tmp_path), out)
    findings = json.loads((out / "validation_report.json").read_text())
    assert findings
    assert [f for f in findings if not f["passed"]] == []


def test_strict_build_aborts_on_a_physical_failure(tmp_path, monkeypatch):
    from focuswatch_dataset import schema as S
    src = sources(tmp_path)
    # Force every acceleration median into the forbidden gap.
    monkeypatch.setattr(S, "ACCEL_USER_BAND", (0.9, 1.1))
    with pytest.raises(RuntimeError, match="validation failed"):
        build_dataset(src, tmp_path / "out", strict=True)


def test_redaction_policy_is_recorded_in_the_manifest(tmp_path):
    out = tmp_path / "out"
    build_dataset(sources(tmp_path), out, policy=RedactionPolicy.FREE_WRITING_XY)
    assert set(load_manifest(out)["redaction_policy"]) == {"free_writing_xy"}


def test_cli_build_and_validate(tmp_path):
    src = sources(tmp_path)
    out = tmp_path / "out"
    args = ["build", "--out", str(out)]
    for name, path in src.items():
        args += ["--source", f"{name}={path}"]
    assert main(args) == 0
    assert main(["validate", "--dataset", str(out)]) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_build.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/focuswatch_dataset/build.py
"""Orchestration: discover, load, validate, write, describe."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd

from . import docs, schema as S
from .adapters import base
from .adapters.base import RecordingBundle
from .manifest import build_channels, build_manifest, check_manifest_consistency
from .redact import RedactionPolicy, apply_redaction
from .validate import (
    ValidationReport, check_coverage, validate_motion_table, validate_pen_table,
    validate_recording,
)
from .write import write_table

_MOTION_MODALITIES = ("watch", "watch_rawaccel", "headimu")


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return ""


def _validate(bundle: RecordingBundle) -> list:
    findings = []
    for modality in _MOTION_MODALITIES:
        if modality in bundle.tables:
            nominal = bundle.meta.get("head_hz_nominal" if modality == "headimu"
                                      else "watch_hz_nominal")
            findings += validate_motion_table(bundle.tables[modality],
                                              bundle.ref.recording_id, modality, nominal)
    if "pen" in bundle.tables:
        findings += validate_pen_table(bundle.tables["pen"], bundle.ref.recording_id)
    findings += validate_recording(bundle.ref.recording_id, bundle.tables, bundle.meta)
    return findings


def build_dataset(source_roots: dict[str, Path], out: Path,
                  policy: RedactionPolicy = RedactionPolicy.NONE,
                  strict: bool = True) -> ValidationReport:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    report = ValidationReport()
    bundles: list[RecordingBundle] = []

    for name, root in sorted(source_roots.items()):
        adapter = base.get_adapter(name)
        for ref in adapter.discover(Path(root)):
            bundle = adapter.load(ref)
            if "pen" in bundle.tables:
                bundle.tables["pen"] = apply_redaction(
                    bundle.tables["pen"], bundle.tables.get("markers"), policy)
            report.findings += _validate(bundle)
            bundle.meta["redaction_policy"] = str(policy)
            bundle.meta["build_git_sha"] = _git_sha()
            bundles.append(bundle)

    manifest_preview = build_manifest(bundles)
    gaps = check_coverage(manifest_preview, report.findings)
    if strict and (report.failed or gaps):
        (out / "validation_report.json").write_text(report.to_json())
        detail = f"{len(report.failed)} failed checks"
        if gaps:
            # Why: an unrun check is indistinguishable from a passed one in the
            # report, so coverage gaps abort just as hard as failures.
            detail += f", {len(gaps)} coverage gaps: " + "; ".join(gaps[:5])
        raise RuntimeError(f"validation failed - {detail}; see validation_report.json")

    for bundle in bundles:
        for modality, df in bundle.tables.items():
            write_table(df, out / modality / f"{bundle.ref.recording_id}.parquet",
                        bundle.ref.recording_id)

    manifest = manifest_preview
    channels = build_channels(bundles)
    manifest.to_parquet(out / "sessions.parquet", index=False)
    manifest.to_csv(out / "sessions.csv", index=False)
    channels.to_parquet(out / "channels.parquet", index=False)

    problems = check_manifest_consistency(manifest, out)
    if problems and strict:
        raise RuntimeError("manifest inconsistent: " + "; ".join(problems))

    docs.write_datapackage(out, manifest, channels)
    docs.write_data_dictionary(out, channels)
    (out / "validation_report.json").write_text(report.to_json())
    return report
```

```python
# src/focuswatch_dataset/docs.py
"""Generated bundle documentation. Both files derive from the manifest."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import schema as S


def write_datapackage(out: Path, manifest: pd.DataFrame, channels: pd.DataFrame) -> None:
    resources = [{"name": "sessions", "path": "sessions.csv", "format": "csv"}]
    for modality in S.MODALITIES:
        d = out / modality
        for f in sorted(d.glob("*.parquet")) if d.exists() else []:
            resources.append({"name": f"{modality}/{f.stem}",
                              "path": f"{modality}/{f.name}", "format": "parquet"})
    (out / "datapackage.json").write_text(json.dumps({
        "profile": "data-package", "name": "focuswatch-dataset",
        "title": "FocusWatch: wrist and head IMU with pen and observer ground truth",
        "licenses": [{"name": "CC-BY-4.0", "path": "https://creativecommons.org/licenses/by/4.0/"}],
        "version": S.SCHEMA_VERSION, "resources": resources,
    }, indent=2))


def write_data_dictionary(out: Path, channels: pd.DataFrame) -> None:
    lines = ["# Data dictionary", "",
             "Units are canonical across the bundle: acceleration and gravity in g,",
             "angular velocity in rad/s, quaternions scalar-last (x, y, z, w),",
             "timestamps as int64 Unix nanoseconds on the clock named by `time_domain`.", "",
             "Pen rows with `x = y = -1` are framing events without a position. They are",
             "retained deliberately; treating them as measurements skews any positional statistic.", ""]
    summary = (channels.groupby(["modality", "column", "quantity", "unit", "semantics"])
               .size().reset_index(name="recordings"))
    lines += ["| modality | column | quantity | unit | semantics | recordings |",
              "|---|---|---|---|---|---|"]
    for _, r in summary.iterrows():
        lines.append(f"| {r.modality} | {r.column} | {r.quantity} | {r.unit} | "
                     f"{r.semantics} | {r.recordings} |")
    (out / "data_dictionary.md").write_text("\n".join(lines) + "\n")
```

```python
# src/focuswatch_dataset/cli.py
"""Command line entry point: fw build | validate | report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .build import build_dataset
from .load import load_manifest
from .manifest import check_manifest_consistency
from .redact import RedactionPolicy


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fw")
    sub = parser.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build")
    b.add_argument("--source", action="append", required=True, metavar="NAME=PATH")
    b.add_argument("--out", required=True)
    b.add_argument("--redact", choices=[p.value for p in RedactionPolicy],
                   default=RedactionPolicy.NONE.value)
    b.add_argument("--no-strict", action="store_true")

    v = sub.add_parser("validate")
    v.add_argument("--dataset", required=True)

    r = sub.add_parser("report")
    r.add_argument("--dataset", required=True)

    args = parser.parse_args(argv)

    if args.command == "build":
        roots = dict(s.split("=", 1) for s in args.source)
        report = build_dataset({k: Path(v) for k, v in roots.items()}, Path(args.out),
                               RedactionPolicy(args.redact), strict=not args.no_strict)
        print(f"{len(report.findings)} checks, {len(report.failed)} failed")
        return 1 if report.failed else 0

    dataset = Path(args.dataset)
    if args.command == "validate":
        problems = check_manifest_consistency(load_manifest(dataset), dataset)
        failed = [f for f in json.loads((dataset / "validation_report.json").read_text())
                  if not f["passed"]]
        for p in problems:
            print(f"manifest: {p}")
        for f in failed:
            print(f"physics: {f['recording_id']} {f['check']} observed={f['observed']}")
        return 1 if (problems or failed) else 0

    manifest = load_manifest(dataset)
    print(f"{len(manifest)} recordings, {manifest['participant_id'].nunique()} participants")
    print(manifest.groupby("cohort").size().to_string())
    return 0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/ -v`
Expected: all tests pass

- [ ] **Step 5: Commit**

```bash
git add src/focuswatch_dataset/build.py src/focuswatch_dataset/docs.py src/focuswatch_dataset/cli.py tests/test_build.py
git commit -m "feat: build orchestration, CLI and generated documentation"
```

---

### Task 16: Build gate against the real corpus

**Not delegable to a subagent.** This task runs outside CI, on the machine that holds the data, and the source paths below must be filled in by whoever runs it. It is the only step that proves the pipeline works on the actual recordings, and its output ships with the dataset.

**Files:**
- Create: `docs/RUNBOOK.md`
- No test file — this task *is* the test.

**Interfaces:**
- Consumes: `cli.main`
- Produces: a `validation_report.json` and a populated bundle, plus the recorded expected counts below.

- [ ] **Step 1: Run the build against the Google Drive bundle**

```bash
fw build \
  --source ml4scs=<extracted>/ml4scs_partner_export \
  --source ege=<bundle>/Data/data_ege_pipeline \
  --source sensorlogger=<bundle>/Data/data_ege_sensorlogger_pipeline \
  --source airpods=<bundle>/Data/labeled_data_AirPods_GS \
  --out ~/focuswatch-dataset-v1.0
```

- [ ] **Step 2: Check the counts against the corpus audit**

Expected, from the spec §2:

| Table | Recordings |
|---|---:|
| `watch/` | 42 |
| `watch_rawaccel/` | 6 |
| `headimu/` | 34 |
| `pen/` | 38 |
| `markers/` | 42 |
| `attention/` | 25 |

`sessions.parquet`: 67 rows, 3 cohorts, and `has_pen == False` for `ETH-SL-focuswatch_T8_s1_1afe92b1`, `…T9…`, `…T10…`, `…S3…`.

- [ ] **Step 3: Confirm every physical check passed**

Run: `fw validate --dataset ~/focuswatch-dataset-v1.0`
Expected: exit 0, no output lines.

If `accel_semantic_band` fails for the ETH Ege recordings, the adapter mapped them as `user` — that is the failure this gate exists for. If `gravity_norm` fails for a headphone table, the m/s² division was skipped.

- [ ] **Step 4: Record the measured spot-checks in the runbook**

Write `docs/RUNBOOK.md` with the build command, the expected table above, and these reference values from the corpus audit, so a future run can be compared rather than re-derived:

| Recording | Check | Expected |
|---|---|---|
| `ML4SCS-S096` | accel_user norm median | ≈ 0.039 |
| `ML4SCS-S096` | gravity norm median | ≈ 1.000 |
| `ETH-EGE-T6` | accel_total norm median | ≈ 0.995 |
| `ETH-EGE-T6` | gyro norm median / p95 | ≈ 0.177 / 1.58 |
| `ETH-SL-E2` (headimu) | gravity norm median after ÷ 9.80665 | ≈ 1.000 |
| `AIRPODS-P1` | head rate measured | ≈ 25 Hz |
| all with quaternion | quat/gravity angle median | < 0.5° |
| `ETH-EGE-T6` / `T7` | `accel_still_bias` in the manifest | ≈ −0.005 / −0.007 |

- [ ] **Step 5: Render one stroke plot per pen source**

The cheapest and sharpest check on the pen coordinate scales, and the only one
that settles the open item in spec §8.2. Plot `x` against `y` for one writing
block from `ML4SCS-S096`, `ETH-EGE-T6` and `ETH-SL-E2`, with equal axis scaling.
Letters must be legible and not mirrored or squashed. Record the observed value
ranges and the aspect ratio in the runbook, and write the resulting `pen_xy_unit`
and `pen_pressure_scale` descriptions into the data dictionary. If a plot is
illegible, the coordinate convention differs from the assumption and the adapter
needs a fix — not the plot.

- [ ] **Step 6: Commit the runbook**

```bash
git add docs/RUNBOOK.md
git commit -m "docs: build runbook with corpus reference values"
```

---

## Self-Review

**Spec coverage.** §2 corpus → Tasks 7–10 and 16. §4 bundle structure → Tasks 11, 12, 15. §5 units and semantics contract → Tasks 2, 3, 8, 9. §5.0 alignment regimes → the `time_alignment` field set in Tasks 7–10. §5.2 verified conventions → Task 4 and the validator checks in Task 5. §6 canonical schema → Task 2, applied in 7–10. §7 manifest → Task 12; §7.1 query surface → Task 14. §8 adapters → Tasks 7–10; §8.1 pen vocabulary → the mapping in Tasks 8 and 9. §9 validator → Tasks 5 and 16. §10 architecture and data policy → Tasks 1 and 15. §11 risk ranking → each risk has a failing test: quaternion order (Task 4 `detect_quaternion_order`, Task 5 `quat_gravity_agreement`), unit and semantics mix (Task 5 forbidden bands, Task 14 homogeneity guard), `standardisation` branching (Task 9), wrist side (manifest default `unknown`, Task 12), pen sentinels (Tasks 7 and 15 data dictionary), time epochs (Task 3).

**Spec §8.2, now closed with mechanisms rather than intentions.** The attention clock is verified in Task 10: the interval table is re-expanded under its anchor assumption and compared against the source's own per-sample labels before those are dropped, so the discarded derivative becomes the test. `accel_still_bias` is computed by the Ege adapter itself in Task 8, because Task 16 has no write path into the manifest. The pen coordinate scales stay a human judgement, but Task 16 Step 5 now makes rendering the stroke plots a required step rather than a suggestion.

**Review findings folded in after the plan was first written.** Twelve corrections, four of them blocking: the integer path in `to_unix_ns` plus the removal of the adapters' own float casts (a wall clock in nanoseconds sits where float64 resolves only to 256 ns); the still-window quaternion check, without which the Ege source — the only one storing components scalar-first — passed the gate unexamined; `check_coverage`, because a skipped check and a passed check are otherwise indistinguishable in the published report; and `tests/__init__.py`, which Task 15's cross-module fixture imports require. The rest: dropping `store_schema=False` (it discarded the key-value metadata the writer had just set), non-finite filtering in the quaternion angle path (forward-only capture leaves older sessions partly empty), the interval-aware `_span`, redaction escalating when markers carry no task structure, an honest diagnosis instead of a dead forbidden-band conjunction, a hard failure instead of an `UNKNOWN` participant, the spill and overlap checks the risk ranking had promised, and a pyarrow major-version pin so byte-level reproducibility is a claim rather than a hope.

**Type consistency.** `RecordingRef` and `RecordingBundle` signatures are identical across Tasks 6–10 and 12. `median_rate_hz` is defined in Task 3 and used in 10 and 12. `norm_stats` is defined in Task 4 and used in 5 and 8. `validate_motion_table(df, recording_id, modality, nominal_hz)` keeps its four-argument form in Tasks 5, 7–10 and 15. `write_table(df, path, recording_id)` is consistent between Tasks 11, 14 and 15. `RedactionPolicy` values match between Tasks 13 and 15.
