"""Generated documentation for a dataset bundle."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from . import schema as S

# Frictionless resource names are lowercase alphanumeric plus `.`, `-`, or `_`.
_SLUG_INVALID = re.compile(r"[^a-z0-9._-]+")


def _slugify(name: str) -> str:
    return _SLUG_INVALID.sub("-", name.lower())


# Package metadata and the bundled license file must name the same license.
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
    # Include schema, validation, and unit-vocabulary resources for consumers.
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
    """Write the bundle README and recipe chapters.

    Every count below is read from `manifest`/`channels`/`out` itself - not
    restated, so it cannot drift from the archive it describes.
    """
    counts = _modality_file_counts(out)
    n_recordings = len(manifest)
    n_participants = int(manifest["participant_id"].nunique())
    cohort_counts = manifest.groupby("cohort").size().sort_index()

    lines = [
        "# FocusWatch dataset", "",
        "Wrist and head IMU, pen and observer-annotation ground truth for a",
        f"handwriting-detection study. {n_recordings} recordings, "
        f"{n_participants} participant identifiers, schema version "
        f"{S.SCHEMA_VERSION}.", "",
        # Why: "participants" is the N a citing paper will quote, and this count
        # cannot support it. Identifiers are namespaced per cohort, so the
        # number would be identical even if all three cohorts had recorded the
        # same people - the cohorts were run by different teams and no mapping
        # between them exists.
        "Identifiers are namespaced per cohort (`ML4SCS-`, `ETH-`, `AIRPODS-`), so "
        "this is a count of identifiers, not of established distinct people: no "
        "participant mapping across cohorts exists. Within a cohort, one identifier "
        "is one person.", "",
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


# Standalone glossary for the non-physical unit vocabulary.
_UNIT_VOCABULARY = (
    ("category", "one of a fixed, enumerated set of string values."),
    ("ordinal", "the position of an item within a sequence, not a magnitude "
               "(e.g. `task_index`)."),
    ("device_native", "a raw value in the source stream's own scale. For pen `x`/`y`/"
                      "`pressure`, consult that recording's `pen_xy_unit` and "
                      "`pen_pressure_scale` in `sessions.parquet`."),
    ("source_native", "a `src_`-prefixed provenance passthrough column: the source "
                      "device's own value, kept for audit, never reinterpreted or rescaled."),
    ("n/a", "no physical or source unit applies (e.g. a categorical id)."),
)

# Fixed time-domain overrides for src_-prefixed provenance columns. A
# modality's primary clock remains adapter-defined.
_TIME_DOMAIN_OVERRIDE_VOCABULARY = (
    (S.TIME_DOMAIN_PEN_DEVICE_CLOCK, "the pen hardware's free-running clock. It is not aligned "
                                     "to a bundle wall clock and must not be used for joins."),
    (S.TIME_DOMAIN_SESSION_RELATIVE_OFFSET_MS, "milliseconds since this recording's own session "
                                               "start - not a wall-clock reading at all, so no "
                                               "clock name would be honest."),
    (S.TIME_DOMAIN_PHONE_WALL_CLOCK, "a phone bridge wall clock, distinct from the motion "
                                     "device's capture clock."),
    (S.TIME_DOMAIN_DEVICE_MONOTONIC_CLOCK, "a free-running uptime clock, not a wall-clock epoch."),
)


def write_data_dictionary(out: Path, manifest: pd.DataFrame, channels: pd.DataFrame) -> None:
    # channels.parquet declares time domains per recording, modality, and
    # column because canonical and provenance timestamps can use different clocks.
    lines = ["# Data dictionary", "",
             "Units are canonical across the bundle: acceleration and gravity in g,",
             "angular velocity in rad/s, quaternions scalar-last (x, y, z, w),",
             "timestamps as int64 Unix nanoseconds. `time_domain` is named per",
             "(recording, modality, column) in this table, and it answers one question",
             "per kind of column: a TIME-VALUED column declares the clock ITS OWN VALUES",
             "are expressed in; every other column declares the clock ITS ROW is stamped",
             "on, which is that modality's default. So `pen.src_timestamp` reads",
             "`pen_device_clock` because that is what its numbers are, while",
             "`pen.pressure` carries the modality's own clock because that is when it",
             "was measured. What a column holds is stated by `quantity` and `unit`, not",
             "by this field. The recording-level `time_domain` in `sessions.parquet` names",
             "only the primary motion stream. Consult `channels.parquet` before joining",
             "modalities or provenance timestamps. For `time_alignment = \"estimated_delta\"`,",
             "read `alignment_note`; `pen_delta_sigma` is the estimate's confidence, not an",
             "offset value.", "",
             "Pen rows with `x = y = -1` are framing events without a position. They are",
             "retained deliberately; treating them as measurements skews any positional statistic.", "",
             "## Unit vocabulary", "",
             "Physical quantities carry their own unit directly. Every other column's `unit` "
             "cell is one of:", "",
             "| value | meaning |", "|---|---|"]
    for value, meaning in _UNIT_VOCABULARY:
        lines.append(f"| `{value}` | {meaning} |")

    lines += [
        "", "## Time domain vocabulary", "",
        "A modality's default `time_domain` names its capture pipeline clock. A `src_` "
        "provenance column may use a different clock and therefore has an explicit "
        "per-column override in the Channels table. The following terms describe those "
        "overrides:", "",
        "| value | meaning |", "|---|---|"]
    for value, meaning in _TIME_DOMAIN_OVERRIDE_VOCABULARY:
        lines.append(f"| `{value}` | {meaning} |")

    lines += [
        "", "## Pen coordinate representation", "",
        "Pen `x`, `y`, and `pressure` use device-native scales. They are not comparable "
        "across all recordings and are not a millimetre conversion. Use each recording's "
        "`pen_xy_unit` and `pen_pressure_scale` from `sessions.parquet` when interpreting "
        "them. Some recordings do not contain pen tilt or a pen-device timestamp; missing "
        "values are represented as `NaN`, not inferred. Coordinate scales can also change "
        "within a recording, so inspect or segment the trace before using positional "
        "statistics.", ""]

    summary = (channels.groupby(["modality", "column", "quantity", "unit", "semantics", "time_domain"])
               .size().reset_index(name="recordings"))
    lines += ["## Channels", "", "| modality | column | quantity | unit | semantics | "
              "time_domain | recordings |",
              "|---|---|---|---|---|---|---|"]
    for _, r in summary.iterrows():
        lines.append(f"| {r.modality} | {r.column} | {r.quantity} | {r.unit} | "
                     f"{r.semantics} | {r.time_domain} | {r.recordings} |")

    (out / "data_dictionary.md").write_text("\n".join(lines) + "\n")
