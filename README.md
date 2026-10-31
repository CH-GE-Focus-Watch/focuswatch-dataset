# focuswatch-dataset

`focuswatch-dataset` converts heterogeneous wearable and annotation exports
into one validated Parquet dataset for the FocusWatch / ML4SCS handwriting
detection project. It standardises schemas and units without pretending that
different sensors, clocks, or acceleration semantics are interchangeable.

## Dataset scope

The current release candidate contains 88 recordings across four cohorts.

| Cohort | Recordings | Primary modalities |
| --- | ---: | --- |
| ML4SCS | 33 | wrist IMU, Moleskine pen, task markers |
| AIRPODS | 38 | head IMU, observer attention intervals |
| ZHAW | 9 | wrist/head IMU, selected pen and marker streams |
| MOLESKINE | 8 | watch accelerometer and Moleskine pen |

`sessions.parquet` is the one-row-per-recording manifest.
`channels.parquet` documents every published column's unit, semantics, sample
rate, and time domain. Modality files use one uniform layout:
`{modality}/{recording_id}.parquet`.

Participant identifiers are source-scoped, not a defensible count of distinct
people across cohorts. The eight Moleskine sessions are eight different people,
as confirmed by the data collector. They are numbered in chronological order:
recording `MOLESKINE-S01` belongs to participant `MOLESKINE-P01`, up to `S08`
and `P08`. Neither the raw filenames' informal labels nor the recording dates
are published.

## Install

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11 or later.

```bash
git clone https://github.com/CH-GE-Focus-Watch/focuswatch-dataset.git
cd focuswatch-dataset
uv sync --all-extras
```

Use `uv sync --extra dev` for tests only or `uv sync --extra tui` for the
terminal explorer.

## Build and validate a bundle

Raw participant data is intentionally excluded from Git. Place each source in
a local ignored directory, extract the ML4SCS partner export, and pass every
source explicitly:

```bash
uv run fw build \
  --source ml4scs=PATH/TO/ml4scs_partner_export \
  --source ege=PATH/TO/data_ege_pipeline \
  --source sensorlogger=PATH/TO/data_ege_sensorlogger_pipeline \
  --source airpods=PATH/TO/Headphone_6Tasks_38People \
  --source moleskine=PATH/TO/raw_moleskine \
  --out out/focuswatch-dataset \
  --redact all_xy \
  --time-origin recording_start

uv run fw validate --dataset out/focuswatch-dataset
uv run fw report --dataset out/focuswatch-dataset
```

`--redact` and `--time-origin` are required so a release cannot silently
inherit a coordinate or timestamp policy. `--time-origin recording_start`
subtracts each recording's earliest instant from every Unix clock value, so
the bundle no longer says on which day and at what time someone was recorded.
Differences between samples, modalities and clocks stay unchanged.
`unix_epoch` keeps the source clocks for internal builds. Pen coordinates
can reconstruct handwriting; `all_xy` is the safe
public-release choice unless consent and disclosure review explicitly approve
another policy. `--no-strict` is for diagnostics, not publication.

Builds are strict and atomic by default: any source, physics, coverage, or
manifest failure prevents replacement of the previous output. A successful
bundle contains:

```text
sessions.parquet        canonical manifest
sessions.csv            accessible manifest copy
channels.parquet        per-recording channel metadata
{modality}/{recording_id}.parquet
validation_report.json
data_dictionary.md
datapackage.json
README.md
LICENSE
```

The names are intentionally uniform. Source filenames never become output
filenames; `recording_id` is the stable join key across the manifest, channels,
and modality directories.

## Use a published bundle

Start with metadata, then load only selected recordings:

```python
from focuswatch_dataset import load_manifest, load_recordings

root = "PATH/TO/DATASET"
manifest = load_manifest(root)
selected = manifest.query(
    "has_watch and has_pen and accel_semantics == 'user' and watch_hz_nominal == 100"
)
watch = load_recordings(
    root,
    selected["recording_id"].head(10).tolist(),
    modality="watch",
)
```

`load_recordings` refuses to mix incompatible watch acceleration semantics.
When total acceleration has a quaternion, `to_user_acceleration` can derive
gravity-removed acceleration; without orientation, the conversion is not
possible and the package does not guess.

It also refuses motion tables whose sample rates differ by more than 10 %
(native rates are never resampled), and pen tables whose `pen_xy_unit` or
`pen_pressure_scale` differ. For per-recording processing, pass
`allow_mixed_rates=True` or `allow_mixed_pen_units=True`; the loaded frame then
carries a `sample_rate_hz` column or the two unit columns on every row.

To combine 100 Hz and 50 Hz recordings, bring them to one rate while loading:

```python
watch = load_recordings(root, ids, modality="watch", target_hz=50)
```

Each recording is decimated with an anti-aliasing filter
(`scipy.signal.decimate`, IIR, zero phase) before concatenation;
`convert.downsample` does the same for a single table. Timestamps,
quaternions and provenance columns keep real readings, and the published
files are not changed.

For interactive metadata filtering and subset export:

```bash
uv run fw explore PATH/TO/DATASET
```

The explorer is a three-step assistant: pick a goal ("Writing detection" for
watch plus pen, "Attention" for head IMU plus observer annotation, or
"Everything") and refine devices and cohorts, narrow by signals, rates and
timing, then export. Each filter shows how many recordings it would leave, and
the recording preview stays visible in every step. The export step lists the
files, their size and the watch rates in the selection, and can bring the
watch to 50 Hz (see `target_hz` above) in the same export instead of copying it
at its native rate. Press `1`, `2`, `3` to switch steps and the arrow keys and
space to move between and toggle filters.

Filters on the same manifest field are alternatives (two cohorts select
either), filters on different fields narrow each other. "Shared clock across
modalities" keeps only recordings whose pen and motion timestamps can be
joined directly. Exporting pen or observer labels also exports the watch or
head IMU they label.

Press `e` to export the selected recordings, `c` to copy the equivalent pandas
query, `p` for provenance, and `q` to quit. Use `--export NEW/DIRECTORY` to
choose a destination; existing directories are never overwritten.

## Derive training labels

The bundle publishes ground truth as recorded: pen contact events and observer
intervals. `fw preprocess` turns them into per-sample labels in a new
directory next to the bundle. Every choice is a parameter and is recorded in
`preprocess.json`.

```bash
uv run fw preprocess --dataset PATH/TO/DATASET --task writing \
  --gap-closing-s 3 --out labels-writing
uv run fw preprocess --dataset PATH/TO/DATASET --task attention \
  --out labels-attention
```

- `writing` adds `label_writing` to each watch table. A sample counts as
  writing when the nearest pen event within 40 ms is a `PEN_DOWN` or
  `PEN_MOVE`. Pauses up to `--gap-closing-s` between contacts then count as
  writing. A pause at the very start or end of a recording is left alone.
  `--gap-closing-s` has no default, because it defines what "writing" means.
- `attention` adds `label_attention` (`focused` / `distracted`) to each head IMU
  table from the observer interval that contains the sample.
- Pen timestamps are joined as published (`--pen-shift-s 0`). Recordings with
  `time_alignment = "overlap_only"` (Moleskine) are skipped for `writing`
  unless `--include-overlap-only` is given, because their watch and pen clocks
  are independent.
- `writing` refuses a selection that mixes watch acceleration with gravity
  (`accel_total_*`) and without it (`accel_user_*`), unless
  `--allow-mixed-accel` is given. `preprocess.json` records each recording's
  `accel_semantics` either way. In the current release, ZHAW-EGE-T6/T7 and
  the Moleskine recordings are `total`; all other watch recordings are `user`.
- `--target-hz 50` labels the downsampled timeline; `--cohort` and
  `--recording` restrict the selection.
- `preprocess.json` names the source bundle by its directory name only, so a
  shared label set does not reveal a local path.

The same functions are available in Python:

```python
from focuswatch_dataset.preprocess import PreprocessConfig, preprocess_recordings

config = PreprocessConfig(task="writing", gap_closing_s=3.0, target_hz=50.0)
labelled = preprocess_recordings(root, ids, config)
```

The two label columns are never merged into one: writing and attention are
different questions with different ground truth.

## Pen sampling rate

A pen reports dots only while it touches the paper, so it has no fixed sample
rate, and none of the manufacturers we checked states one. The manifest gives
the dots per second within strokes as `pen_hz_measured` and names the clock
it was measured on in `pen_hz_clock`. The pen's own clock (`src_timestamp`)
is used where the source logs it. Otherwise `t_ns` is used, but only if it
stamps dots one by one, not in Bluetooth batches. Measured on the current
release candidate:

| Pen model | Clock | Recordings | Median | Range |
| --- | --- | ---: | ---: | --- |
| Moleskine Smart Pen NWP-F130 | `src_timestamp` | 33 | 79.2 Hz | 68.2–83.5 Hz |
| Moleskine Smart Pen NWP-F130 | `t_ns` | 8 | 39.0 Hz | 31.1–39.5 Hz |
| Neo smartpen M1 NWP-F50 | `src_timestamp` | 1 | 70.0 Hz | – |
| LAMY safari ncode NWP-F80 | `src_timestamp` | 1 | 45.1 Hz | – |
| LAMY safari ncode NWP-F80 | `t_ns` | 2 | 48.7 Hz | 46.9–50.5 Hz |
| unknown | `src_timestamp` | 1 | 69.1 Hz | – |

Four NWP-F50 recordings stay `NaN`: their `t_ns` arrives in Bluetooth
batches. On recordings that log both clocks, `t_ns` gave a rate 2–13 % above
the pen clock. The eight Moleskine-cohort recordings show about half the
NWP-F130 rate of the ML4SCS cohort; the cause is not known. Each bundle's
`data_dictionary.md` repeats this table for its own recordings.

## Important limitations

- Moleskine watch and pen streams use independent wall clocks. They are paired
  by overlapping recording ranges only; no clock offset is estimated or
  applied, so sample-level joins are approximate.
- Acceleration may include gravity (`total`) or exclude it (`user`). Filter on
  `accel_semantics` before combining recordings.
- Pen coordinates and pressure remain device-native, not millimetres or a
  cross-device calibrated scale. See each row's `pen_xy_unit` and
  `pen_pressure_scale`.
- Automated validation covers structure, timing, physical plausibility,
  redaction, and archive consistency. It does not establish participant
  consent, redistribution rights, anonymisation sufficiency, or scientific
  validity.

## Publication checklist

Before creating a public archive:

1. Build from a clean reviewed commit so `build_git_sha` is not suffixed
   `-dirty`.
2. Use an explicitly approved redaction policy and `--time-origin
   recording_start`, and independently review pen, marker, and participant
   metadata for disclosure risk.
3. Confirm consent and redistribution rights for every cohort and confirm that
   CC BY-NC 4.0 is authorised for the dataset.
4. Run `uv run pytest` and `uv run fw validate --dataset ...`; require zero
   failed checks.
5. Inspect `sessions.parquet`, `validation_report.json`, and the generated
   bundle documentation; record a checksum for the final archive.
6. Replace the author and DOI placeholders in `CITATION.cff`, and add the final
   dataset citation/DOI to the release record.
7. Confirm no raw data is tracked:

   ```bash
   git ls-files | xargs uv run python scripts/check_no_data.py
   ```

## License and citation

The software is Apache-2.0; see [`LICENSE`](LICENSE). A generated dataset bundle
declares CC BY-NC 4.0 separately. `CITATION.cff` describes the software and still
contains explicit placeholders that must be completed before release.
