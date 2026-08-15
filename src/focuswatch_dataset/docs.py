"""Generated bundle documentation. Both files derive from the manifest/channels
tables actually written to `out`, never from a hand-maintained file list - so
the docs cannot drift from what is on disk (see build.py's module docstring
and the plan's property 4).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from . import schema as S

# I4: a Frictionless resource `name` is lowercase alphanumeric plus `.`, `-`,
# `_` only - no `/`, no upper case. `path` (the actual file location) is left
# untouched; only the descriptor's own `name` field needs slugifying.
_SLUG_INVALID = re.compile(r"[^a-z0-9._-]+")


def _slugify(name: str) -> str:
    return _SLUG_INVALID.sub("-", name.lower())


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
    # I4: without these four, a Frictionless consumer only ever sees the
    # per-recording modality tables and sessions.csv - never channels.parquet
    # (the one place a column's unit/semantics is declared), never the
    # validation evidence, never the dictionary that explains the unit
    # vocabulary.
    resources = [
        {"name": "sessions", "path": "sessions.csv", "format": "csv"},
        {"name": "sessions-parquet", "path": "sessions.parquet", "format": "parquet"},
        {"name": "channels", "path": "channels.parquet", "format": "parquet"},
        {"name": "data-dictionary", "path": "data_dictionary.md", "format": "markdown"},
        {"name": "validation-report", "path": "validation_report.json", "format": "json"},
    ]
    for modality in S.MODALITIES:
        d = out / modality
        for f in sorted(d.glob("*.parquet")) if d.exists() else []:
            resources.append({"name": _slugify(f"{modality}-{f.stem}"),
                              "path": f"{modality}/{f.name}", "format": "parquet"})
    (out / "datapackage.json").write_text(json.dumps({
        "profile": "data-package", "name": "focuswatch-dataset",
        "title": "FocusWatch: wrist and head IMU with pen and observer ground truth",
        "licenses": [{"name": "CC-BY-4.0", "path": "https://creativecommons.org/licenses/by/4.0/"}],
        "version": S.SCHEMA_VERSION, "resources": resources,
    }, indent=2))


def _modality_file_counts(out: Path) -> dict[str, int]:
    counts = {}
    for modality in S.MODALITIES:
        d = out / modality
        counts[modality] = len(list(d.glob("*.parquet"))) if d.exists() else 0
    return counts


def write_readme(out: Path, manifest: pd.DataFrame, channels: pd.DataFrame) -> None:
    """Bundle entry point + recipe chapters (DESIGN §4).

    Every count below is read from `manifest`/`channels`/`out` itself - not
    restated, so it cannot drift from the archive it describes (I4).
    """
    counts = _modality_file_counts(out)
    n_recordings = len(manifest)
    n_participants = int(manifest["participant_id"].nunique())
    cohort_counts = manifest.groupby("cohort").size().sort_index()

    lines = [
        "# FocusWatch dataset", "",
        "Wrist and head IMU, pen and observer-annotation ground truth for a",
        f"handwriting-detection study. {n_recordings} recordings, "
        f"{n_participants} participants, schema version {S.SCHEMA_VERSION}.", "",
        "## Contents", "",
        "| file | rows | what it is |",
        "|---|---|---|",
        f"| `sessions.parquet` / `sessions.csv` | {n_recordings} | the manifest - one row per "
        "recording: every capability flag (`has_watch`, `has_pen`, ...), timing and protocol field |",
        f"| `channels.parquet` | {len(channels)} | one row per (recording, modality, column): unit, "
        "semantics, sample rate, time domain |",
        "| `data_dictionary.md` | - | the canonical unit vocabulary, and what is/isn't harmonised "
        "across sources |",
        "| `validation_report.json` | - | every physical/structural check this build ran, and "
        "whether it passed |",
        "| `datapackage.json` | - | machine-readable Frictionless Data Package descriptor |",
        "| `LICENSE` | - | the data license (CC BY 4.0) |",
    ]
    for modality in S.MODALITIES:
        if counts[modality]:
            lines.append(f"| `{modality}/{{recording_id}}.parquet` | {counts[modality]} files | "
                        f"see `channels.parquet` (`modality == \"{modality}\"`) for its columns |")
    lines += [
        "", "## Quick start", "",
        "```python",
        "from focuswatch_dataset import load_manifest, load_recording, by_flags", "",
        'root = "."  # the directory this README is in',
        "m = load_manifest(root)", "",
        "# every recording with a 100 Hz watch stream carrying gravity",
        "subset = by_flags(m, has_watch=True, has_gravity=True)", "",
        'rid = subset.iloc[0]["recording_id"]',
        'df = load_recording(root, rid, modality="watch")',
        "```", "",
        "## Recipes", "",
        "- **Select recordings by capability**: `m.query(...)` (any pandas expression over "
        "`sessions.parquet`'s columns), or `by_flags(m, has_pen=True, ...)` for the equality-only "
        "shorthand.",
        '- **Load one recording\'s table**: `load_recording(root, recording_id, modality="watch")`.',
        "- **Load several at once**: `load_recordings(root, recording_ids, modality=\"watch\")` - "
        "refuses to silently mix `user` and `total` acceleration semantics.",
        "- **Total acceleration but you need gravity removed**: `to_user_acceleration(df)` derives "
        "it from the quaternion; the bundle itself never guesses an orientation.",
        "- **What did this build verify?**: read `validation_report.json` - every physical check, "
        "per recording, with the observed value and the tolerance it was checked against.",
        "", "## License", "",
        "The data in this bundle is CC BY 4.0 - see `LICENSE`. The `focuswatch-dataset` software "
        "that produced it is Apache-2.0, in its own source repository.",
        "", "## Corpus by cohort", "",
        "| cohort | recordings |",
        "|---|---|",
    ]
    for cohort, n in cohort_counts.items():
        lines.append(f"| {cohort} | {n} |")
    (out / "README.md").write_text("\n".join(lines) + "\n")


# I5: the unit vocabulary glossary. Previously defined only in
# docs/DESIGN.md (German, never shipped in the bundle) and a schema.py code
# comment (_KNOWN_METADATA_COLUMNS) a reuser downloading the archive never
# sees. Physical quantities carry their own SI-ish unit directly (g, rad/s,
# "1" for a unit quaternion, "ns" for a timestamp); every other column's
# `unit` cell is one of these five.
_UNIT_VOCABULARY = (
    ("category", "one of a fixed, enumerated set of string values."),
    ("ordinal", "the position of an item within a sequence, not a magnitude "
               "(e.g. `task_index`)."),
    ("device_native", "a raw value in the source stream's own scale. For pen `x`/`y`/"
                      "`pressure` the actual scale is a genuinely different unit per "
                      "recording, not a fixed constant - look up `pen_xy_unit`/"
                      "`pen_pressure_scale` in the per-recording table below (Moleskine's "
                      "Ncode grid and the ETH web app's own pixel/force scale are unrelated "
                      "scales that happen to share a column name)."),
    ("source_native", "a `src_`-prefixed provenance passthrough column: the source "
                      "device's own value, kept for audit, never reinterpreted or rescaled."),
    ("n/a", "no physical or source unit applies (e.g. a categorical id)."),
)

# I5 (open question 4): the real Ege `pen_events.csv` header, measured
# directly from the source - NOT `t_ms, x, y, force, tilt.x, tilt.y,
# timestamp` as an earlier draft of docs/DESIGN.md assumed. Ege's export
# carries neither tilt nor a pen-device clock at all.
_EGE_PEN_EVENTS_HEADER = "id, session_id, t_ms, t_session_ms, type, x, y, force, created_at"


def write_data_dictionary(out: Path, manifest: pd.DataFrame, channels: pd.DataFrame) -> None:
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
             "retained deliberately; treating them as measurements skews any positional statistic.", "",
             "## Unit vocabulary", "",
             "Physical quantities carry their own unit directly. Every other column's `unit` "
             "cell is one of:", "",
             "| value | meaning |", "|---|---|"]
    for value, meaning in _UNIT_VOCABULARY:
        lines.append(f"| `{value}` | {meaning} |")

    lines += [
        "", "## Pen tilt and the pen-device clock: generation-B ETH only", "",
        "Moleskine (ML4SCS) carries real tilt from its own pen hardware - this section is "
        "about the ETH cohort only. Both ETH pipelines export the same web app's pen events, "
        "but the corpus mixes two export generations (see the adapter split for "
        "`pen_events`/`events` keys). WITHIN the ETH cohort, tilt (`tilt_x`/`tilt_y`) and the "
        "pen-device clock (`src_timestamp`) exist ONLY for the three generation-B ETH "
        "SensorLogger recordings (E1, E2, E3). Ege's own export and "
        "the four generation-A recordings SensorLogger also carries (S3, T8, T9, T10) have "
        "neither in their source - the real Ege `pen_events.csv` header is:", "",
        f"```\n{_EGE_PEN_EVENTS_HEADER}\n```", "",
        "so `tilt_x`, `tilt_y` and `src_timestamp` are NaN there BY NATURE (the source never "
        "measured them), not by loss in this pipeline.", "",
        "## Pen coordinate scale: occasional far-scale samples, not a millimetre unit", "",
        "ETH pen `x`/`y` occasionally jump scale within a single recording. Measured on the "
        "actual corpus: `ETH-SL-E1_session3`'s `x` spans 5.69 … 16416.00 - 18,846 of "
        "18,848 samples sit in [5.69, 63.55] and the remaining 2 sit at exactly 16416.00, "
        "≈258× the low cluster's maximum (`ETH-SL-E2_session6` shows the same "
        "pattern, ≈260×, on 4 of 7,130 samples). `ETH-EGE-T6`'s `y` shows the same "
        "phenomenon as a substantial cluster rather than a handful of points: 149 samples "
        "confined to [2048.01, 3072.83] end within 0.255 s of a `pen_paper_info` marker "
        "event, immediately followed by 2,367 samples in [8704.57, 34816.62] - consistent "
        "with a page or section change. Plotting a recording's raw strokes without "
        "segmenting will flatten the main cluster to a line under the far-scale points. For "
        "generation-A recordings, segment on `pen_paper_info` events (`markers/`) to find "
        "the transitions; generation-B recordings (E1/E2/E3) never carry `pen_paper_info` "
        "at all, so this cue is unavailable there and far-scale samples must be found by "
        "inspection instead (e.g. a percentile filter). Do NOT treat any of this as a "
        "millimetre conversion - no such mapping is established for either pen coordinate "
        "scale.", ""]

    summary = (channels.groupby(["modality", "column", "quantity", "unit", "semantics", "time_domain"])
               .size().reset_index(name="recordings"))
    lines += ["## Channels", "", "| modality | column | quantity | unit | semantics | "
              "time_domain | recordings |",
              "|---|---|---|---|---|---|---|"]
    for _, r in summary.iterrows():
        lines.append(f"| {r.modality} | {r.column} | {r.quantity} | {r.unit} | "
                     f"{r.semantics} | {r.time_domain} | {r.recordings} |")

    # I5: the table above lists `pen | x` once per distinct unit VALUE
    # ("ncode_grid", "webapp_raw", ...), but a unit string alone does not say
    # which recording it belongs to - this ties each value back to a
    # recording_id, straight from the manifest that already carries it.
    pen_rows = manifest.loc[manifest["has_pen"],
                            ["recording_id", "cohort", "pen_xy_unit", "pen_pressure_scale"]]
    if len(pen_rows):
        lines += ["", "## Pen coordinate/pressure units by recording", "",
                  "| recording_id | cohort | pen_xy_unit | pen_pressure_scale |",
                  "|---|---|---|---|"]
        for _, r in pen_rows.sort_values("recording_id").iterrows():
            lines.append(f"| {r.recording_id} | {r.cohort} | {r.pen_xy_unit} | "
                        f"{r.pen_pressure_scale} |")

    (out / "data_dictionary.md").write_text("\n".join(lines) + "\n")
