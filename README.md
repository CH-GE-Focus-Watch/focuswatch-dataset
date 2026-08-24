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

The optional terminal explorer reads only the bundle manifest while filtering:

```bash
pip install "focuswatch-dataset[tui]"
fw explore DATASET_ROOT
```

Use the German-labelled facets to update the Aufnahme count immediately. Press
`e` to export selected manifest rows plus their reproducible pandas query, `c`
to copy that query, `p` for source/provenance details, or `q` to quit. The
default export is `focuswatch-selection.json`; no sensor samples are exported.

## Tests

```bash
pytest tests/
```

See [docs/DESIGN.md](docs/DESIGN.md) and [docs/PLAN.md](docs/PLAN.md) for
the full design and task plan.
