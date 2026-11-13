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
    # Why: from_pandas stamps a "pandas" key with a per-call column-index blob
    # into schema metadata; replacing it (rather than merging) is what keeps
    # writes byte-identical across runs, not just value-identical.
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
        # Why: store_schema=False would drop the key-value metadata set above
        # (an earlier review caught this) - leave the pyarrow default (True).
    )


def read_table(path: Path) -> pd.DataFrame:
    return pq.read_table(path).to_pandas()


def table_metadata(path: Path) -> dict[str, str]:
    raw = pq.ParquetFile(path).schema_arrow.metadata or {}
    return {k.decode(): v.decode() for k, v in raw.items()}
