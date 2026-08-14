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

## Tests

```bash
pytest tests/
```

See [docs/DESIGN.md](docs/DESIGN.md) and [docs/PLAN.md](docs/PLAN.md) for
the full design and task plan.
