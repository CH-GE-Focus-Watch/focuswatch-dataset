# focuswatch-dataset

Converts four heterogeneous sensor-capture pipelines into one published
Parquet dataset for a journal data descriptor. Adapters read each source
format, a physical validator checks the result, and the package writes a
uniformly described Parquet bundle that downstream users assemble via
manifest flags instead of learning three source schemas.

## Install

```bash
pip install -e ".[dev]"
```

After installation, select from the published manifest and load samples
explicitly with the reproducible example:

```bash
python examples/select_and_load.py DATASET_ROOT --require-pen --modality watch \
  --accel-semantics user
```

The example reads only `sessions.parquet` while applying its filters. It calls
`load_recordings` only after the selected recording IDs are known; selecting a
manifest row never loads sensor samples implicitly.

The same metadata-first selection is available directly in Python:

```python
from focuswatch_dataset import load_manifest, load_recordings

manifest = load_manifest("DATASET_ROOT")
selected = manifest.query(
    "has_watch and has_pen and watch_hz_nominal == 100 and accel_semantics == 'user'"
)
recording_ids = selected["recording_id"].tolist()
samples = load_recordings("DATASET_ROOT", recording_ids, modality="watch")
```

The optional terminal explorer reads only the bundle manifest while filtering:

```bash
pip install "focuswatch-dataset[tui]"
fw explore DATASET_ROOT --export focuswatch-selection.json
```

Use the German-labelled facets to update the Aufnahme count immediately. Press
`e` to export selected manifest rows plus their reproducible pandas query, `c`
to copy that query, `p` for source/provenance details, or `q` to quit. The
default export is `focuswatch-selection.json`; no sensor samples are exported.

The strict external-data build and validation gate is still to be run on the
machine containing the source bundle. Local tests and code-only checks do not
claim that external-data gate has run.

## Tests

```bash
pytest tests/
```

See [docs/DESIGN.md](docs/DESIGN.md) and [docs/PLAN.md](docs/PLAN.md) for
the full design and task plan.
