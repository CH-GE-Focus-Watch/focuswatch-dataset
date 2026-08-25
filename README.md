# focuswatch-dataset

`focuswatch-dataset` converts heterogeneous wearable and annotation exports
into one validated Parquet dataset for the FocusWatch / ML4SCS handwriting
detection project. It standardises schemas and units without pretending that
different sensors, clocks, or acceleration semantics are interchangeable.

## Dataset scope

The current release candidate contains 75 recordings across four cohorts.

| Cohort | Recordings | Primary modalities |
| --- | ---: | --- |
| ML4SCS | 33 | wrist IMU, Moleskine pen, task markers |
| AIRPODS | 25 | head IMU, observer attention intervals |
| ETH | 9 | wrist/head IMU, selected pen and marker streams |
| MOLESKINE | 8 | watch accelerometer and Moleskine pen |

`sessions.parquet` is the one-row-per-recording manifest.
`channels.parquet` documents every published column's unit, semantics, sample
rate, and time domain. Modality files use one uniform layout:
`{modality}/{recording_id}.parquet`.

Participant identifiers are source-scoped, not a defensible count of distinct
people across cohorts. The Moleskine pilot has no complete publishable
participant attribution and therefore uses `MOLESKINE-UNKNOWN`; the raw
filenames' informal labels are not published.

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
  --source airpods=PATH/TO/labeled_data_AirPods_GS \
  --source moleskine=PATH/TO/raw_moleskine \
  --out out/focuswatch-dataset \
  --redact all_xy

uv run fw validate --dataset out/focuswatch-dataset
uv run fw report --dataset out/focuswatch-dataset
```

`--redact` is required so a release cannot silently inherit a coordinate
policy. Pen coordinates can reconstruct handwriting; `all_xy` is the safe
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
selected = manifest.query("has_watch and has_pen and accel_semantics == 'user'")
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

For interactive metadata filtering and subset export:

```bash
uv run fw explore PATH/TO/DATASET
```

Press `e` to export the selected recordings, `c` to copy the equivalent pandas
query, `p` for provenance, and `q` to quit. Use `--export NEW/DIRECTORY` to
choose a destination; existing directories are never overwritten.

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
2. Use an explicitly approved redaction policy and independently review pen,
   marker, and participant metadata for disclosure risk.
3. Confirm consent and redistribution rights for every cohort and confirm that
   CC BY 4.0 is authorised for the dataset.
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
declares CC BY 4.0 separately. `CITATION.cff` describes the software and still
contains explicit placeholders that must be completed before release.
