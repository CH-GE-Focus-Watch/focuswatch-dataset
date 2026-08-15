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
