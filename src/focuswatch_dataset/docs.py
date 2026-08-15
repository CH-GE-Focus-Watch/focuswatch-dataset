"""Generated bundle documentation. Both files derive from the manifest/channels
tables actually written to `out`, never from a hand-maintained file list - so
the docs cannot drift from what is on disk (see build.py's module docstring
and the plan's property 4).
"""
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
