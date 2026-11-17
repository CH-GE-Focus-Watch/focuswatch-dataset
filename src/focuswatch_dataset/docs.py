"""Generated documentation for a dataset bundle."""
from __future__ import annotations

import importlib.metadata
import json
import re
from pathlib import Path

import pandas as pd

from . import schema as S
from .checksums import CHECKSUM_FILE
from .manifest import MANIFEST_COLUMNS
from .redact import RedactionPolicy
from .time_origin import TimeOrigin

# Frictionless resource names are lowercase alphanumeric plus `.`, `-`, or `_`.
_SLUG_INVALID = re.compile(r"[^a-z0-9._-]+")


def _slugify(name: str) -> str:
    return _SLUG_INVALID.sub("-", name.lower())


# Package metadata, README and the bundled license file must name the same license.
_DATA_LICENSE = "CC BY-NC 4.0"
_DATA_LICENSE_SPDX = "CC-BY-NC-4.0"
_DATA_LICENSE_URL = "https://creativecommons.org/licenses/by-nc/4.0/"
_DATA_LICENSE_TEXT = f"""\
FocusWatch dataset - data license

The data files in this bundle (everything except this LICENSE file and the
generated documentation) are licensed under the Creative Commons
Attribution-NonCommercial 4.0 International License ({_DATA_LICENSE}).

You are free to share and adapt this material, as long as you give
appropriate credit, provide a link to the license, and indicate if changes
were made. You may not use the material for commercial purposes.

Full license text: {_DATA_LICENSE_URL}legalcode

This license covers the DATA only. The focuswatch-dataset software that
produced this bundle is separately licensed (Apache-2.0); see that
project's own repository and LICENSE file.
"""

_TITLE = "FocusWatch: wrist and head motion with pen events and protocol attention labels"
_ARTICLE_TITLE = ("A Multimodal Wrist and Head Motion Dataset for Handwriting and Focus "
                  "Detection with Consumer Wearables")
_DATASET_DOI = "10.17632/nt4dvs2sk4.1"
_SOFTWARE_DOI = "10.5281/zenodo.24303697"
_REPOSITORY_URL = "https://github.com/CH-GE-Focus-Watch/focuswatch-dataset"


def _doi_url(doi: str) -> str:
    return f"https://doi.org/{doi}"


def _software_version() -> str:
    return importlib.metadata.version("focuswatch-dataset")


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
        "id": _doi_url(_DATASET_DOI), "title": _TITLE, "homepage": _REPOSITORY_URL,
        "licenses": [{"name": _DATA_LICENSE_SPDX, "path": _DATA_LICENSE_URL}],
        "version": S.SCHEMA_VERSION, "resources": resources,
    }, indent=2))


def _modality_file_counts(out: Path) -> dict[str, int]:
    counts = {}
    for modality in S.MODALITIES:
        d = out / modality
        counts[modality] = len(list(d.glob("*.parquet"))) if d.exists() else 0
    return counts


def _withdrawn_streams(manifest: pd.DataFrame) -> list[tuple[str, str]]:
    streams = []
    for rid, codes in zip(manifest["recording_id"], manifest["issue_codes"]):
        for issue in json.loads(codes) if codes else []:
            if issue["code"] == "modality_withdrawn":
                streams.append((rid, issue["modality"]))
    return sorted(streams)


def _pen_limitation(policy: str) -> str:
    if policy == RedactionPolicy.ALL_XY:
        return ("- Pen `x` and `y` are empty in every row, so handwriting shape cannot be "
                "analysed. Pen pressure is device-native, not a cross-device calibrated scale.")
    return ("- Pen coordinates and pressure are device-native, not millimetres or a "
            "cross-device calibrated scale.")


def write_readme(out: Path, manifest: pd.DataFrame, channels: pd.DataFrame) -> None:
    """Write the bundle README and recipe chapters.

    Every count below is read from `manifest`/`channels`/`out` itself - not
    restated, so it cannot drift from the archive it describes.
    """
    counts = _modality_file_counts(out)
    n_recordings = len(manifest)
    n_participants = int(manifest["participant_id"].nunique())
    identifier_noun = "identifier" if n_participants == 1 else "identifiers"
    redaction_policy = str(manifest["redaction_policy"].iloc[0])
    time_origin = str(manifest["time_origin"].iloc[0])
    has_overlap_only = bool(
        (manifest["time_alignment"] == S.TIME_ALIGNMENT_OVERLAP_ONLY).any()
    )
    cohort_counts = manifest.groupby("cohort").size().sort_index()
    version = _software_version()

    lines = [
        "# FocusWatch dataset", "",
        "Wrist and head motion, pen events and protocol attention labels, recorded with",
        "consumer wearables during desk-based writing, non-writing and attention tasks. "
        f"{n_recordings} recordings, {n_participants} participant {identifier_noun}, "
        f"schema version {S.SCHEMA_VERSION}.", "",
        # Why: "participants" is the N a citing paper will quote, and this count
        # cannot support it. Identifiers are namespaced per cohort, so the
        # number would be identical even if every cohort had recorded the
        # same people - the cohorts were run by different teams and no mapping
        # between them exists.
        "Identifiers are source-scoped, so this is not a count of established distinct "
        "people: no complete mapping across cohorts or sources exists.", "",
        "## Citation", "",
        f"- Dataset: Mendeley Data, DOI [{_DATASET_DOI}]({_doi_url(_DATASET_DOI)}).",
        f"- Article: *{_ARTICLE_TITLE}*, Data in Brief.",
        f"- Software: `focuswatch-dataset` {version}, DOI "
        f"[{_SOFTWARE_DOI}]({_doi_url(_SOFTWARE_DOI)}), source code at {_REPOSITORY_URL}.",
        "",
        "## Contents", "",
        "| file | rows | what it is |",
        "|---|---|---|",
        f"| `sessions.parquet` / `sessions.csv` | {n_recordings} | the manifest - one row per "
        "recording: every capability flag (`has_watch`, `has_pen`, ...), timing and protocol field |",
        f"| `channels.parquet` | {len(channels)} | one row per (recording, modality, column): unit, "
        "semantics, sample rate, time domain |",
        "| `data_dictionary.md` | - | every manifest column, clock, unit and category value |",
        "| `validation_report.json` | - | every physical/structural check this build ran, and "
        "whether it passed |",
        "| `datapackage.json` | - | machine-readable Frictionless Data Package descriptor |",
        f"| `LICENSE` | - | the data license ({_DATA_LICENSE}) |",
        f"| `{CHECKSUM_FILE}` | - | SHA-256 checksum of every other file |",
    ]
    for modality in S.MODALITIES:
        if counts[modality]:
            lines.append(f"| `{modality}/{{recording_id}}.parquet` | {counts[modality]} files | "
                        f"see `channels.parquet` (`modality == \"{modality}\"`) for its columns |")
    lines += [
        "", "## Verify the download", "",
        "```bash",
        f"sha256sum -c {CHECKSUM_FILE}          # Linux",
        f"shasum -a 256 -c {CHECKSUM_FILE}      # macOS",
        "```", "",
        "## Installation", "",
        "Every table is an Apache Parquet file, so pandas, pyarrow or R `arrow` read the "
        "bundle without further software. The helper functions below come from the "
        "`focuswatch-dataset` package, which is installed from its repository:", "",
        "```bash",
        f'pip install "focuswatch-dataset @ git+{_REPOSITORY_URL}@v{version}"',
        "```", "",
        "## Quick start", "",
        "```python",
        "from focuswatch_dataset import load_manifest, load_recording, by_flags", "",
        'root = "."  # the directory this README is in',
        "m = load_manifest(root)", "",
        "# every recording with a watch stream",
        "subset = by_flags(m, has_watch=True)", "",
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
        "- **Total acceleration but you need gravity removed**: when `has_quaternion` is true, "
        "`to_user_acceleration(df)` derives it from orientation. Without a quaternion this "
        "conversion is not available, and the bundle does not guess.",
        "- **What did this build verify?**: read `validation_report.json` - every physical check, "
        "per recording, with the observed value and the tolerance it was checked against. "
        "`fw validate --dataset .` reruns the checks on this copy.",
        "", "## Limitations", "",
        f"- This bundle's pen-coordinate redaction policy is `{redaction_policy}`; see "
        "`sessions.parquet` for the per-recording declaration.",
        f"- This bundle's time origin is `{time_origin}`. Under `recording_start` every "
        "timestamp counts from the start of its own recording, so the dates and times of "
        "day of the sessions are not published; see the data dictionary.",
        "- Attention labels state the condition the session protocol instructed, not a "
        "measured attentional state.",
        "- Acceleration semantics vary by source. Filter on `accel_semantics` before "
        "combining recordings.",
        _pen_limitation(redaction_policy),
        "- Automated validation covers structure, timing, physical plausibility, redaction, "
        "and archive consistency. It does not establish participant consent, redistribution "
        "rights, anonymisation sufficiency, or scientific validity.",
    ]
    if withdrawn := _withdrawn_streams(manifest):
        lines.append(
            "- These streams are left out because they cover only part of their session: "
            + ", ".join(f"`{rid}` ({modality})" for rid, modality in withdrawn)
            + ". Each row lists `modality_withdrawn` in `issue_codes`; the other modalities "
            "of these recordings are published."
        )
    if has_overlap_only:
        lines += [
            "- Recordings with `time_alignment = \"overlap_only\"` pair independent wall "
            "clocks by overlapping ranges; no offset is applied, so sample-level joins are "
            "approximate. Read `alignment_note` before combining modalities."
        ]
    lines += [
        "", "## License", "",
        f"The data in this bundle is {_DATA_LICENSE} - see `LICENSE`. The `focuswatch-dataset` software "
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

# Every clock a `time_domain` cell can name. The first group are the clocks a
# modality's `t_ns` is stamped on; the second are clocks only `src_` provenance
# columns use (see schema.TIME_DOMAIN_*).
_CLOCK_VOCABULARY = (
    ("watch_capture_clock", "ML4SCS: the Apple Watch's timestamp of each sample, as the "
                            "watch app captured it."),
    ("server_wall_clock", "ML4SCS: the study server's wall clock, which stamped pen dots "
                          "and markers on arrival."),
    ("backend_wall_clock", "ZHAW: the study web app's backend clock, which stamped every "
                           "modality of a recording."),
    ("device_wall_clock", "AIRPODS: the wall clock of the iPhone that recorded the "
                          "headphone motion; the attention intervals are anchored to it."),
    ("watch_device_wall_clock", "MOLESKINE: the Apple Watch's own wall clock, as Sensor "
                                "Logger logged it."),
    ("moleskine_app_wall_clock", "MOLESKINE: the wall clock of the phone running the "
                                 "Moleskine app, which stored the pen dots."),
    (S.TIME_DOMAIN_PEN_DEVICE_CLOCK, "the pen hardware's free-running clock. It is not aligned "
                                     "to a bundle wall clock and must not be used for joins."),
    (S.TIME_DOMAIN_SESSION_RELATIVE_OFFSET_MS, "milliseconds since this recording's own session "
                                               "start - not a wall-clock reading at all, so no "
                                               "clock name would be honest."),
    (S.TIME_DOMAIN_PHONE_WALL_CLOCK, "a phone bridge wall clock, distinct from the motion "
                                     "device's capture clock."),
    (S.TIME_DOMAIN_DEVICE_MONOTONIC_CLOCK, "a free-running uptime clock, not a wall-clock epoch."),
)

# (type, meaning) per manifest column. Columns typed `category` also get the
# values this bundle holds listed beside them.
_MANIFEST_COLUMN_DOCS: dict[str, tuple[str, str]] = {
    "recording_id": ("string", "Unique recording id. It is the file name of the recording "
                               "in every modality directory."),
    "participant_id": ("string", "Participant identifier, scoped to its cohort. The same "
                                 "number in two cohorts says nothing about the person."),
    "cohort": ("category", "Study the recording belongs to."),
    "pipeline": ("category", "Capture pipeline that produced the source files."),
    "has_watch": ("bool", "`watch/` holds a table for this recording."),
    "has_watch_rawaccel": ("bool", "`watch_rawaccel/` holds a table for this recording."),
    "has_headimu": ("bool", "`headimu/` holds a table for this recording."),
    "has_pen": ("bool", "`pen/` holds a table for this recording."),
    "has_markers": ("bool", "`markers/` holds a table for this recording."),
    "has_attention": ("bool", "`attention/` holds a table for this recording."),
    "watch_model": ("category", "Apple Watch model; empty without a watch table."),
    "watch_hz_nominal": ("number", "Watch sampling rate the app was set to, in Hz; empty "
                                   "where the source does not state one."),
    "watch_hz_measured": ("number", "1 / median interval between watch samples, in Hz."),
    "has_gravity": ("bool", "The watch table has `gravity_x/y/z`."),
    "has_quaternion": ("bool", "The watch table has `quat_x/y/z/w`."),
    "has_watch_gyro": ("bool", "The watch table has `gyro_x/y/z`."),
    "accel_semantics": ("category", "`user`: acceleration without gravity (`accel_user_*`). "
                                    "`total`: acceleration including gravity "
                                    "(`accel_total_*`)."),
    "accel_calibration": ("category", "`fused`: Core Motion's sensor-fused output. `raw`: the "
                                      "accelerometer reading. `raw_uncalibrated`: the "
                                      "accelerometer reading with a measurable per-device "
                                      "bias, see `accel_still_bias`."),
    "accel_still_bias": ("number", "Mean deviation of the acceleration norm from 1 g while "
                                   "the wrist is still, in g. Reported, never corrected; "
                                   "empty where not measured."),
    "gravity_source": ("category", "`measured`: the source logged a gravity vector. `none`: "
                                   "it did not."),
    "head_model": ("category", "Headphone model; empty without a head-IMU table."),
    "head_hz_nominal": ("number", "Head-IMU sampling rate the app was set to, in Hz; empty "
                                  "where the source does not state one."),
    "head_hz_measured": ("number", "1 / median interval between head-IMU samples, in Hz."),
    "has_head_gravity": ("bool", "The head-IMU table has `gravity_x/y/z`."),
    "has_head_quaternion": ("bool", "The head-IMU table has `quat_x/y/z/w`."),
    "has_head_gyro": ("bool", "The head-IMU table has `gyro_x/y/z`."),
    "time_domain": ("category", "Clock of the primary motion stream: the watch, or the head "
                                "IMU without a watch. See Clocks."),
    "time_alignment": ("category", "How the clocks of the modalities relate. See Time "
                                   "alignment."),
    "time_origin": ("category", "Where the timestamps count from. See Time origin."),
    "t_start_ns": ("integer", "Earliest timestamp of any table of the recording, in ns."),
    "t_end_ns": ("integer", "Latest timestamp of any table of the recording, in ns."),
    "duration_s": ("number", "`(t_end_ns - t_start_ns) / 1e9`, in s."),
    "protocol_id": ("category", "Task protocol of the session; empty without one."),
    "study_mode": ("category", "`study` for a regular study session; empty where the "
                               "source has no such field."),
    "subject_index": ("integer", "ML4SCS: row of the Williams counterbalancing design that "
                                 "set the task order. `-1` in every other cohort."),
    "n_writing_tasks": ("integer", "Number of `task_start` markers with `task_category = "
                                   "writing`; empty without a task protocol."),
    "n_idle_tasks": ("integer", "Number of `task_start` markers with `task_category = "
                                "idle`; empty without a task protocol."),
    "watch_wrist_side": ("category", "Wrist the watch was worn on; `unknown` where not "
                                     "recorded."),
    "handedness": ("category", "Writing hand as the ZHAW web app logged it at session "
                               "start; `unknown` where not recorded."),
    "pen_model": ("category", "Smart pen model; empty without a pen table, `unknown` where "
                              "the source did not log it."),
    "pen_hz_measured": ("number", "Pen dots per second within strokes, in Hz. See Pen "
                                  "sampling rate."),
    "pen_hz_clock": ("category", "Clock `pen_hz_measured` was measured on. See Pen sampling "
                                 "rate."),
    "pen_xy_unit": ("category", "Device-native scale of pen `x`/`y`; empty without a pen "
                                "table."),
    "pen_pressure_scale": ("category", "Device-native scale of pen `pressure`; empty without "
                                       "a pen table."),
    "pen_delta_s": ("number", "Offset between pen and watch clocks, in s. Always empty: an "
                              "estimated offset is not published."),
    "pen_delta_sigma": ("number", "ML4SCS: the confidence score the study server stored for "
                                  "its pen-watch offset estimate. It is not an offset. The "
                                  "source does not document its scale."),
    "delta_applied": ("bool", "An offset was applied to the pen timestamps."),
    "alignment_note": ("string", "Which modalities run on another clock than the primary "
                                 "motion stream, and what that means for joins; empty where "
                                 "all share one clock."),
    "n_samples_watch": ("integer", "Rows in the watch table; 0 without one."),
    "n_samples_pen": ("integer", "Rows in the pen table; 0 without one."),
    "n_samples_head": ("integer", "Rows in the head-IMU table; 0 without one."),
    "issue_codes": ("JSON", "List of known issues; empty if none. `modality_dropout` marks a "
                            "stream that covers less than 1 % of the recording, with its "
                            "`n_samples` and `coverage_ratio`. `modality_withdrawn` marks a "
                            "stream left out of the bundle because it covers only part of "
                            "the session."),
    "source_pipeline": ("category", "Same value as `pipeline`."),
    "schema_version": ("category", "Version of the bundle schema."),
    "redaction_policy": ("category", "Which pen coordinates were removed. `none`: none. "
                                     "`free_writing_xy`: `x`/`y` within free-writing tasks. "
                                     "`all_xy`: every `x`/`y`."),
    "build_git_sha": ("string", "Commit of the software that built this bundle."),
}

# Category columns of the modality tables whose values are listed.
_TABLE_CATEGORY_COLUMNS = (
    ("pen", "dot_type"),
    ("markers", "event"), ("markers", "protocol_id"), ("markers", "task_category"),
    ("markers", "task_id"), ("markers", "task_name"),
    ("attention", "label"), ("attention", "activity"),
)


def _format_values(values) -> str:
    distinct = set(map(str, values))
    cells = [f"`{v}`" for v in sorted(distinct - {""})] + (["empty"] if "" in distinct else [])
    return ", ".join(cells) if cells else "-"


def _manifest_column_lines(manifest: pd.DataFrame) -> list[str]:
    lines = ["## Manifest columns", "",
             "One row per recording in `sessions.parquet` and `sessions.csv`. An empty cell "
             "means not applicable. Values are listed for category columns as they occur "
             "in this bundle.", "",
             "| column | type | meaning | values |", "|---|---|---|---|"]
    for column in MANIFEST_COLUMNS:
        kind, meaning = _MANIFEST_COLUMN_DOCS[column]
        values = _format_values(manifest[column].dropna()) if kind == "category" else "-"
        lines.append(f"| `{column}` | {kind} | {meaning} | {values} |")
    return [*lines, ""]


def _table_values(out: Path, modality: str, column: str) -> set[str]:
    values: set[str] = set()
    for path in sorted((out / modality).glob("*.parquet")):
        frame = pd.read_parquet(path)
        if column in frame.columns:
            values.update(map(str, frame[column].dropna()))
    return values


def _table_category_lines(out: Path) -> list[str]:
    lines = ["## Category values in the tables", "",
             "The ML4SCS task names and the AIRPODS activities are English translations "
             "of the German values in the source files.", "",
             "| modality | column | values |", "|---|---|---|"]
    for modality, column in _TABLE_CATEGORY_COLUMNS:
        values = _table_values(out, modality, column)
        if values:
            lines.append(f"| {modality} | `{column}` | {_format_values(values)} |")
    return [*lines, ""]


def _pen_coordinate_lines(policy: str) -> list[str]:
    lines = ["## Pen coordinate representation", ""]
    if policy == RedactionPolicy.ALL_XY:
        lines.append("This bundle's redaction policy is `all_xy`: pen `x` and `y` are empty "
                     "(`NaN`) in every row, and marker payloads carry no coordinates.")
    else:
        lines.append("Pen `x` and `y` use device-native scales. They are not comparable "
                     "across all recordings and are not a millimetre conversion. Use each "
                     "recording's `pen_xy_unit` from `sessions.parquet` when interpreting "
                     "them. Coordinate scales can also change within a recording, so inspect "
                     "or segment the trace before using positional statistics. Rows with "
                     "`x = y = -1` are framing events without a position.")
        if policy == RedactionPolicy.FREE_WRITING_XY:
            lines.append("Under `free_writing_xy`, `x` and `y` are empty within free-writing "
                         "tasks and in recordings without task markers.")
    lines += ["", "Pen `pressure` is device-native too: use `pen_pressure_scale`, not a "
              "physical unit. Some recordings do not contain pen tilt or a pen-device "
              "timestamp; missing values are `NaN`, not inferred.", ""]
    return lines


def _pen_rate_summary(manifest: pd.DataFrame) -> list[str]:
    pen = manifest[manifest["has_pen"]]
    if pen.empty:
        return []
    lines = ["Measured rates per pen model in this bundle:", "",
             "| pen_model | pen_hz_clock | recordings | median Hz | range Hz |",
             "|---|---|---:|---:|---|"]
    for (model, clock), group in pen.groupby(["pen_model", "pen_hz_clock"]):
        hz = group["pen_hz_measured"].dropna()
        median, span = ((f"{hz.median():.1f}", f"{hz.min():.1f}-{hz.max():.1f}") if len(hz)
                        else ("-", "-"))
        lines.append(f"| {model} | {f'`{clock}`' if clock else 'empty'} | {len(group)} "
                     f"| {median} | {span} |")
    return [*lines, ""]


def write_data_dictionary(out: Path, manifest: pd.DataFrame, channels: pd.DataFrame) -> None:
    # channels.parquet declares time domains per recording, modality, and
    # column because canonical and provenance timestamps can use different clocks.
    if str(manifest["time_origin"].iloc[0]) == TimeOrigin.RECORDING_START:
        time_unit = ("timestamps as int64 nanoseconds since the start of the recording "
                     "(`time_origin = \"recording_start\"`)")
    else:
        time_unit = "timestamps as int64 Unix nanoseconds"
    lines = ["# Data dictionary", "",
             "Units are canonical across the bundle: acceleration and gravity in g,",
             "angular velocity in rad/s, quaternions scalar-last (x, y, z, w),",
             f"{time_unit}. `time_domain` is named per",
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
             "offset value. For `time_alignment = \"overlap_only\"`, the independently",
             "maintained device clocks only establish coarse recording membership, not exact",
             "sample alignment.", "",
             *_manifest_column_lines(manifest),
             "## Unit vocabulary", "",
             "Physical quantities carry their own unit directly. Every other column's `unit` "
             "cell is one of:", "",
             "| value | meaning |", "|---|---|"]
    for value, meaning in _UNIT_VOCABULARY:
        lines.append(f"| `{value}` | {meaning} |")

    lines += [
        "", "## Clocks", "",
        "Every value a `time_domain` cell can hold. The first six are the clocks a "
        "modality's `t_ns` is stamped on. The last four occur only on `src_` provenance "
        "columns, which have an explicit per-column entry in the Channels table.", "",
        "| value | meaning |", "|---|---|"]
    for value, meaning in _CLOCK_VOCABULARY:
        lines.append(f"| `{value}` | {meaning} |")

    lines += [
        "", "## Time origin", "",
        "`time_origin` in `sessions.parquet` states where the timestamps count from.",
        "",
        "| value | meaning |", "|---|---|",
        "| `unix_epoch` | Every clock value is as the source logged it, in Unix time. |",
        "| `recording_start` | The earliest instant of each recording was subtracted from every "
        "Unix clock value of that recording: `t_ns`, `t_start_ns`, `t_end_ns`, the Unix "
        "millisecond `src_` columns and the Unix millisecond marker payload fields. "
        "Differences between samples, modalities and clocks are unchanged, and a "
        "`src_` clock that ran off by days still shows that offset. |",
        "", "## Time alignment", "",
        "| value | meaning |", "|---|---|",
        "| `shared_clock` | Modalities use the same capture or backend clock. |",
        "| `estimated_delta` | An inter-clock offset was estimated but is not applied; read "
        "`alignment_note`. |",
        "| `overlap_only` | Independent wall-clock ranges overlap, but no offset was estimated "
        "or applied; cross-modal sample joins are approximate. |",
        "",
    ]

    lines += [
        *_pen_coordinate_lines(str(manifest["redaction_policy"].iloc[0])),
        "## Pen sampling rate", "",
        "A pen reports dots only while it touches the paper, so it has no fixed sample "
        "rate. `pen_hz_measured` in `sessions.parquet`, and `sample_rate_hz` on the pen "
        "rows of `channels.parquet`, give the dots per second within strokes. "
        "`pen_hz_clock` names the clock it was measured on:", "",
        "| `pen_hz_clock` | Meaning |",
        "|---|---|",
        "| `src_timestamp` | The pen's own clock. Preferred wherever the source logs it. |",
        "| `t_ns` | No pen clock is logged, and at most 20 % of the intervals within strokes "
        "are zero, so `t_ns` stamps dots one by one. On recordings that log both clocks, "
        "`t_ns` at this level gave a rate 2-13 % above the pen clock. |",
        "| empty | Not measurable: `pen_hz_measured` is `NaN`. `t_ns` is the backend's "
        "arrival time in Bluetooth batches that share one timestamp and would inflate "
        "the rate. |", "",
        "Dots are also dropped during fast strokes, so the rate is an average, not a "
        "fixed sensor frequency. There is no nominal pen rate: the manufacturer "
        "specifications we found do not state one.", "",
        *_pen_rate_summary(manifest),
        *_table_category_lines(out)]

    summary = (channels.groupby(["modality", "column", "quantity", "unit", "semantics", "time_domain"])
               .size().reset_index(name="recordings"))
    lines += ["## Channels", "", "| modality | column | quantity | unit | semantics | "
              "time_domain | recordings |",
              "|---|---|---|---|---|---|---|"]
    for _, r in summary.iterrows():
        lines.append(f"| {r.modality} | {r.column} | {r.quantity} | {r.unit} | "
                     f"{r.semantics} | {r.time_domain} | {r.recordings} |")

    (out / "data_dictionary.md").write_text("\n".join(lines) + "\n")
