"""Flag-based subsetting over the manifest."""
from __future__ import annotations

import pandas as pd


def by_flags(manifest: pd.DataFrame, **flags) -> pd.DataFrame:
    """Filter the manifest to rows matching every `column=value` pair.

    A shorthand for the equality-only case; `manifest.query(...)` (plain
    pandas, works directly on `load_manifest`'s return value) covers
    everything this does not - ranges, `and`/`or`, membership. An unknown
    column raises rather than matching nothing silently, so a typo'd flag
    name fails loudly instead of quietly returning an empty selection that
    reads the same as "no recordings meet this criterion".
    """
    out = manifest
    for column, value in flags.items():
        if column not in out.columns:
            raise KeyError(
                f"unknown manifest column {column!r}; see load_manifest(...).columns "
                "for the published fields"
            )
        out = out[out[column] == value]
    return out.reset_index(drop=True)
