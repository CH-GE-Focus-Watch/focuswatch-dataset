# focuswatch-dataset

`focuswatch-dataset` converts heterogeneous wearable-sensor and annotation
pipelines into a documented Parquet dataset for the FocusWatch / ML4SCS
handwriting-detection project. It gives downstream researchers one manifest
schema and a metadata-first way to select recordings, without having to learn
the original capture formats.

> **Project status:** This repository is ready for project review and
> reproducibility testing. The published Parquet archive and its DOI will be
> released separately; neither raw recordings nor private source paths are
> stored in this repository.

## What the dataset contains

The current release-candidate archive contains 67 recordings across three
cohorts:

| Cohort | Recordings | Primary modalities |
| --- | ---: | --- |
| ML4SCS | 33 | wrist IMU, pen, task markers |
| AIRPODS | 25 | head IMU, observer attention annotations |
| ETH | 9 | wrist and head IMU; selected pen/marker streams |

The archive uses Parquet tables for wrist and head IMU, pen traces, task
markers, and observer attention intervals. `sessions.parquet` is the manifest:
one row per recording with capability flags such as `has_watch`, `has_headimu`,
`has_gravity`, and `has_pen`. `channels.parquet` documents each published
column's unit, semantics, sampling rate, and time domain.

Different cohorts expose different sensors. Always filter the manifest before
loading samples; the capability flags make those differences explicit.

## Install

Requires Python 3.11 or later.

```bash
git clone https://github.com/CH-GE-Focus-Watch/focuswatch-dataset.git
cd focuswatch-dataset
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

For the optional interactive terminal explorer, install the `tui` extra:

```bash
pip install -e ".[tui]"
```

## Work with a dataset bundle

Obtain the separately distributed dataset bundle and set `DATASET_ROOT` to its
directory. A bundle contains `sessions.parquet`, `channels.parquet`,
`validation_report.json`, and the modality directories described above.

Start by inspecting the manifest:

```bash
fw report --dataset DATASET_ROOT
fw validate --dataset DATASET_ROOT
```

Select recordings by metadata, then load only the samples you requested:

```bash
python examples/select_and_load.py DATASET_ROOT --require-pen --modality watch \
  --accel-semantics user
```

The example reads only `sessions.parquet` during selection and loads sensor rows
only after the recording IDs are known. The equivalent Python workflow is:

```python
from focuswatch_dataset import load_manifest, load_recordings

manifest = load_manifest("DATASET_ROOT")
selected = manifest.query(
    "has_watch and has_pen and accel_semantics == 'user'"
)
samples = load_recordings(
    "DATASET_ROOT", selected["recording_id"].tolist(), modality="watch"
)
```

`load_recordings` refuses to silently mix incompatible watch acceleration
semantics. Use the manifest and `channels.parquet` to make a deliberate choice.

## Explore the manifest in the terminal

The optional terminal UI filters only manifest metadata; it does not load or
export sensor samples. Plain-language facets (modalities, signals, acquisition
settings) update the recording count live, show the equivalent pandas query,
and can export the selected manifest rows together with that query.

```bash
fw explore DATASET_ROOT --export focuswatch-selection.json
```

Press `e` to export, `c` to copy the query, `r` to reset all filters, `p` for
provenance details (cohort, protocol, source pipeline, build SHA), and `q` to
quit.

## Reproducibility and validation

The build process normalizes source-specific schemas, records provenance, and
runs structural and physical validation. The current release-candidate archive
was built with the strict gate and contains 1,146 passing validation checks.
`fw validate` independently reopens the published archive and verifies its
manifest, tables, channel descriptions, redaction declaration, and derived
summaries.

Before a journal release, rebuild the archive from a clean Git checkout. This
ensures its recorded `build_git_sha` identifies a clean, reproducible source
revision rather than a `-dirty` working tree.

## Develop and test

Run the full test suite:

```bash
pytest tests/
```

The repository intentionally contains no participant recordings. Verify that
no data artefacts were added before committing:

```bash
git ls-files | python scripts/check_no_data.py
```

## License and citation

The software in this repository is licensed under Apache-2.0; see
[`LICENSE`](LICENSE). The dataset archive is licensed separately as CC BY 4.0.

Citation metadata for the software is in [`CITATION.cff`](CITATION.cff). The
dataset DOI and final author metadata remain placeholders until the public
archive is released.

## Further documentation

- [`docs/DESIGN.md`](docs/DESIGN.md) explains the canonical schema, validation,
  and release decisions.
- [`examples/select_and_load.py`](examples/select_and_load.py) is the
  executable metadata-first loading example.
