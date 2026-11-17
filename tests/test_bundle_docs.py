"""The generated bundle documentation covers every published column, clock and value."""
import hashlib
import importlib.metadata
import json

import pandas as pd
import pytest

from focuswatch_dataset import docs
from focuswatch_dataset.build import build_dataset
from focuswatch_dataset.checksums import CHECKSUM_FILE
from focuswatch_dataset.load import load_manifest
from focuswatch_dataset.manifest import MANIFEST_COLUMNS
from focuswatch_dataset.redact import RedactionPolicy
from focuswatch_dataset.time_origin import TimeOrigin
from focuswatch_dataset.validate import validate_dataset
from focuswatch_dataset.write import read_table

from .test_build import sources


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("bundle")
    out = tmp / "out"
    build_dataset(sources(tmp), out, policy=RedactionPolicy.NONE,
                  time_origin=TimeOrigin.UNIX_EPOCH)
    return out


@pytest.fixture(scope="module")
def release_bundle(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("release")
    out = tmp / "out"
    build_dataset(sources(tmp), out, policy=RedactionPolicy.ALL_XY,
                  time_origin=TimeOrigin.RECORDING_START)
    return out


def _dictionary(out):
    return (out / "data_dictionary.md").read_text()


def test_every_manifest_column_has_a_documented_meaning():
    assert set(docs._MANIFEST_COLUMN_DOCS) == set(MANIFEST_COLUMNS)


def test_data_dictionary_explains_every_manifest_column(bundle):
    text = _dictionary(bundle)
    for column in MANIFEST_COLUMNS:
        assert f"| `{column}` |" in text, column


def test_data_dictionary_lists_the_values_of_every_category_column(bundle):
    manifest = load_manifest(bundle)
    text = _dictionary(bundle)
    for column, (kind, _) in docs._MANIFEST_COLUMN_DOCS.items():
        if kind != "category":
            continue
        row = next(line for line in text.splitlines() if line.startswith(f"| `{column}` |"))
        for value in manifest[column].dropna().astype(str):
            assert ("empty" if value == "" else f"`{value}`") in row, (column, value)


def test_data_dictionary_defines_every_clock_the_bundle_names(bundle):
    manifest = load_manifest(bundle)
    channels = read_table(bundle / "channels.parquet")
    clocks_section = _dictionary(bundle).split("## Clocks", 1)[1].split("\n## ", 1)[0]
    used = set(channels["time_domain"]) | set(manifest["time_domain"])
    for clock in used:
        assert f"| `{clock}` |" in clocks_section, clock


def test_data_dictionary_lists_category_values_of_the_tables(bundle):
    text = _dictionary(bundle)
    section = text.split("## Category values in the tables", 1)[1].split("\n## ", 1)[0]
    for modality, column in docs._TABLE_CATEGORY_COLUMNS:
        values = set()
        for path in (bundle / modality).glob("*.parquet"):
            frame = read_table(path)
            if column in frame.columns:
                values.update(frame[column].dropna().astype(str))
        if not values:
            continue
        row = next(line for line in section.splitlines()
                   if line.startswith(f"| {modality} | `{column}` |"))
        for value in values:
            assert ("empty" if value == "" else f"`{value}`") in row, (modality, column, value)


def test_all_xy_documentation_states_that_coordinates_are_empty(release_bundle):
    text = _dictionary(release_bundle)
    section = text.split("## Pen coordinate representation", 1)[1].split("\n## ", 1)[0]
    assert "empty (`NaN`) in every row" in section
    assert "x = y = -1" not in text
    assert "Pen coordinates and pressure are device-native" not in (
        release_bundle / "README.md").read_text()


def test_unredacted_documentation_keeps_the_coordinate_scales(bundle):
    section = _dictionary(bundle).split("## Pen coordinate representation", 1)[1]
    assert "`pen_xy_unit`" in section
    assert "x = y = -1" in section


def test_bundle_never_calls_the_attention_labels_observer_annotations(bundle):
    for name in ("README.md", "data_dictionary.md", "datapackage.json"):
        assert "observer" not in (bundle / name).read_text().lower(), name
    channels = read_table(bundle / "channels.parquet")
    assert not channels["semantics"].str.contains("observer").any()


def test_readme_names_dois_repository_and_installation(bundle):
    text = (bundle / "README.md").read_text()
    version = importlib.metadata.version("focuswatch-dataset")
    assert f"https://doi.org/{docs._DATASET_DOI}" in text
    assert f"https://doi.org/{docs._SOFTWARE_DOI}" in text
    assert docs._REPOSITORY_URL in text
    assert f'pip install "focuswatch-dataset @ git+{docs._REPOSITORY_URL}@v{version}"' in text


def test_datapackage_carries_the_dataset_doi_and_homepage(bundle):
    package = json.loads((bundle / "datapackage.json").read_text())
    assert package["id"] == f"https://doi.org/{docs._DATASET_DOI}"
    assert package["homepage"] == docs._REPOSITORY_URL


def test_checksum_list_covers_every_other_file(bundle):
    listed = {}
    for line in (bundle / CHECKSUM_FILE).read_text().splitlines():
        digest, rel = line.split("  ", 1)
        listed[rel] = digest
    files = {p.relative_to(bundle).as_posix(): p for p in bundle.rglob("*") if p.is_file()}
    assert set(listed) == set(files) - {CHECKSUM_FILE}
    for rel, digest in listed.items():
        assert hashlib.sha256(files[rel].read_bytes()).hexdigest() == digest, rel


def _copy(bundle, tmp_path):
    import shutil
    copy = tmp_path / "copy"
    shutil.copytree(bundle, copy)
    return copy


def _checksum_failures(root):
    return [f.observed for f in validate_dataset(root).failed if f.check == "checksum"]


def test_validate_passes_an_untouched_bundle(bundle):
    assert _checksum_failures(bundle) == []


def test_validate_flags_a_modified_file(bundle, tmp_path):
    copy = _copy(bundle, tmp_path)
    (copy / "LICENSE").write_text("changed")
    assert _checksum_failures(copy) == ["LICENSE does not match its checksum"]


def test_validate_flags_an_unlisted_file(bundle, tmp_path):
    copy = _copy(bundle, tmp_path)
    (copy / "notes.txt").write_text("extra")
    assert _checksum_failures(copy) == ["notes.txt is not listed"]


def test_validate_flags_a_missing_checksum_list(bundle, tmp_path):
    copy = _copy(bundle, tmp_path)
    (copy / CHECKSUM_FILE).unlink()
    assert _checksum_failures(copy) == [f"{CHECKSUM_FILE} is missing"]


def test_withdrawn_streams_are_read_from_issue_codes():
    manifest = pd.DataFrame({
        "recording_id": ["R2", "R1"],
        "issue_codes": ['[{"code": "modality_withdrawn", "modality": "headimu"}]', ""],
    })
    assert docs._withdrawn_streams(manifest) == [("R2", "headimu")]
