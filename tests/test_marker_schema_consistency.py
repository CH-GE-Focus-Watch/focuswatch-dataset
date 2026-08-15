"""I15: one modality, two schemas - `markers/` must carry the same canonical
columns and dtypes regardless of which cohort produced the recording.
`src_`-prefixed columns are the documented exception (docs/DESIGN.md §6):
provenance passthrough is legitimately cohort-specific, so this test scopes
the "same columns" requirement to everything else.
"""
from __future__ import annotations

import numpy as np

from focuswatch_dataset.adapters.ege import EgeAdapter
from focuswatch_dataset.adapters.ml4scs import Ml4scsAdapter
from focuswatch_dataset.adapters.sensorlogger import SensorLoggerAdapter
from tests.test_adapter_ege import write_fixture as ege_fixture
from tests.test_adapter_ml4scs import write_fixture as ml4scs_fixture
from tests.test_adapter_sensorlogger import write_fixture as sl_fixture

_CANONICAL = ("t_ns", "event", "task_id", "task_name", "task_index", "task_category", "protocol_id")


def _markers_tables(tmp_path):
    ege_fixture(tmp_path / "ege")
    ml4scs_fixture(tmp_path / "ml4scs")
    sl_fixture(tmp_path / "sl")
    return {
        "ege": EgeAdapter().load(EgeAdapter().discover(tmp_path / "ege")[0]).tables["markers"],
        "ml4scs": Ml4scsAdapter().load(Ml4scsAdapter().discover(tmp_path / "ml4scs")[0]).tables["markers"],
        "sensorlogger": SensorLoggerAdapter().load(
            SensorLoggerAdapter().discover(tmp_path / "sl")[0]).tables["markers"],
    }


def test_canonical_marker_columns_and_dtypes_agree_across_cohorts(tmp_path):
    tables = _markers_tables(tmp_path)
    for cohort, df in tables.items():
        canonical = [c for c in df.columns if not c.startswith("src_")]
        assert set(canonical) == set(_CANONICAL), (cohort, canonical)

    reference = tables["ml4scs"]
    for cohort, df in tables.items():
        for col in _CANONICAL:
            assert df[col].dtype == reference[col].dtype, (cohort, col, df[col].dtype, reference[col].dtype)


def test_task_index_is_null_not_a_sentinel_int_across_cohorts(tmp_path):
    """The concrete I15 bug: -1 looks like a valid index to a reuser filtering
    `task_index >= 0`, NaN cannot be. Every cohort's "not applicable" marker
    row (the ETH pipelines never populate task_index; ml4scs does not either
    for structureless events) must publish NaN."""
    tables = _markers_tables(tmp_path)
    for cohort in ("ege", "sensorlogger"):
        df = tables[cohort]
        assert df["task_index"].dtype == np.float64
        assert df["task_index"].isna().all(), cohort
