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


# I12: `write_datapackage` below asserts CC-BY-4.0 in `datapackage.json`'s
# `licenses` field; without an actual LICENSE file beside it that assertion
# is unbacked - a downloader has the claim but not the license text. Kept
# short (a summary + the canonical link), matching how CC licenses are
# normally shipped: the full legal code is long and lives at the
# Creative Commons URL, not duplicated per-archive.
_DATA_LICENSE_TEXT = """\
FocusWatch dataset - data license

The data files in this bundle (everything except this LICENSE file and the
generated documentation) are licensed under the Creative Commons
Attribution 4.0 International License (CC BY 4.0).

You are free to share and adapt this material for any purpose, even
commercially, as long as you give appropriate credit, provide a link to the
license, and indicate if changes were made.

Full license text: https://creativecommons.org/licenses/by/4.0/legalcode

This license covers the DATA only. The focuswatch-dataset software that
produced this bundle is separately licensed (Apache-2.0); see that
project's own repository and LICENSE file.
"""


def write_license(out: Path) -> None:
    (out / "LICENSE").write_text(_DATA_LICENSE_TEXT)


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
    # C2: the old sentence named a single recording-level `time_domain` as
    # covering every timestamp in a recording. False for any cohort whose
    # modalities are not all on one clock (ML4SCS: watch on its own capture
    # clock, pen/markers on the server clock) - the clock is only truthfully
    # named per (recording, modality) in channels.parquet's own `time_domain`
    # column, which the table below now includes.
    lines = ["# Data dictionary", "",
             "Units are canonical across the bundle: acceleration and gravity in g,",
             "angular velocity in rad/s, quaternions scalar-last (x, y, z, w),",
             "timestamps as int64 Unix nanoseconds. The clock each column is stamped on",
             "is named per (recording, modality) in this table's `time_domain` column -",
             "not by `sessions.parquet`'s recording-level `time_domain` alone, which names",
             "only the primary motion stream's clock and can differ from other modalities",
             "of the same recording. For a recording with `time_alignment =",
             "\"estimated_delta\"`, read its `alignment_note` before assuming any two",
             "modalities share a clock: the offset between them is estimated but not",
             "published, and `pen_delta_sigma` is that estimate's confidence, not its value.", "",
             "Pen rows with `x = y = -1` are framing events without a position. They are",
             "retained deliberately; treating them as measurements skews any positional statistic.", ""]
    summary = (channels.groupby(["modality", "column", "quantity", "unit", "semantics", "time_domain"])
               .size().reset_index(name="recordings"))
    lines += ["| modality | column | quantity | unit | semantics | time_domain | recordings |",
              "|---|---|---|---|---|---|---|"]
    for _, r in summary.iterrows():
        lines.append(f"| {r.modality} | {r.column} | {r.quantity} | {r.unit} | "
                     f"{r.semantics} | {r.time_domain} | {r.recordings} |")
    (out / "data_dictionary.md").write_text("\n".join(lines) + "\n")
