import json
import sys

import pandas as pd
import pytest

import focuswatch_dataset as fw
from focuswatch_dataset import cli
from focuswatch_dataset.explore import (
    FACETS,
    ExplorerSelection,
    export_selected_recordings,
    export_selection,
    filter_manifest,
    selection_query,
)


def test_public_api_exports_the_pure_explorer_model():
    assert fw.ExplorerSelection is ExplorerSelection
    assert fw.filter_manifest is filter_manifest
    assert fw.selection_query is selection_query


@pytest.fixture
def manifest():
    return pd.DataFrame([
        {
            "recording_id": "A",
            "participant_id": "P1",
            "cohort": "internal-a",
            "protocol_id": "protocol-a",
            "has_watch": True,
            "has_headimu": False,
            "has_pen": True,
            "has_attention": False,
            "has_gravity": True,
            "has_quaternion": False,
            "has_head_gravity": False,
            "has_head_quaternion": False,
            "has_head_gyro": False,
            "has_watch_rawaccel": False,
            "watch_hz_nominal": 100.0,
            "accel_semantics": "user",
            "study_mode": "study",
        },
        {
            "recording_id": "B",
            "participant_id": "P2",
            "cohort": "internal-b",
            "protocol_id": "protocol-b",
            "has_watch": True,
            "has_headimu": True,
            "has_pen": True,
            "has_attention": True,
            "has_gravity": False,
            "has_quaternion": True,
            "has_head_gravity": True,
            "has_head_quaternion": True,
            "has_head_gyro": True,
            "has_watch_rawaccel": True,
            "watch_hz_nominal": 50.0,
            "accel_semantics": "total",
            "study_mode": "study",
        },
    ])


def test_filter_manifest_intersects_plain_language_facets(manifest):
    selected = filter_manifest(manifest, {"smartwatch": True, "watch_gravity": True})

    assert selected["recording_id"].tolist() == ["A"]


def test_selection_query_is_reproducible_and_uses_manifest_columns():
    assert selection_query({"watch_gravity": True, "smartwatch": True}) == (
        "has_watch and has_gravity"
    )
    assert selection_query({"smartwatch": True, "watch_gravity": False}) == "has_watch"


def test_empty_selection_query_reproduces_all_manifest_rows(manifest):
    query = selection_query({})

    assert manifest.query(query)["recording_id"].tolist() == ["A", "B"]


def test_explorer_selection_filters_and_counts_without_mutating_manifest(manifest):
    selection = ExplorerSelection.from_mapping({"head_imu": True, "head_gyro": True})

    selected = selection.filter(manifest)

    assert selection.query == "has_headimu and has_head_gyro"
    assert selection.count(manifest) == 1
    assert selected["recording_id"].tolist() == ["B"]
    assert manifest["recording_id"].tolist() == ["A", "B"]


def test_explorer_selection_normalizes_mutable_input_to_frozenset():
    active = {"smartwatch"}

    selection = ExplorerSelection(active)
    active.add("watch_gravity")

    assert isinstance(selection.active_facets, frozenset)
    assert selection.active_facets == frozenset({"smartwatch"})
    assert selection.query == "has_watch"


def test_facets_have_understandable_german_labels_and_manifest_conditions():
    labels = {facet.label for facet in FACETS.values()}

    assert {"Smartwatch", "Head IMU", "Smart Pen", "Observer annotation"} <= labels
    assert "Watch gravity" in labels
    assert FACETS["watch_gyro"].condition == ("has_watch_gyro", True)
    assert FACETS["watch_50_hz"].condition == ("watch_hz_nominal", 50.0)
    assert FACETS["acceleration_user"].condition == ("accel_semantics", "user")


def test_unknown_facet_fails_loudly(manifest):
    with pytest.raises(KeyError, match="unknown explorer facet"):
        filter_manifest(manifest, {"source_code_x": True})


def test_missing_manifest_column_fails_loudly():
    with pytest.raises(KeyError, match="has_gravity"):
        filter_manifest(pd.DataFrame({"recording_id": ["A"]}), {"watch_gravity": True})


def test_export_selection_json_contains_only_manifest_rows_and_query(tmp_path, manifest):
    destination = tmp_path / "selection.json"

    export_selection(
        manifest,
        {"smartwatch": True, "watch_gravity": True},
        destination,
    )

    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert set(payload) == {"query", "recording_ids", "rows"}
    assert payload["query"] == "has_watch and has_gravity"
    assert payload["recording_ids"] == ["A"]
    assert [row["recording_id"] for row in payload["rows"]] == ["A"]
    assert "samples" not in json.dumps(payload).lower()


def test_export_selection_csv_writes_rows_and_reproducible_query(tmp_path, manifest):
    destination = tmp_path / "selection.csv"

    written = export_selection(manifest, {"head_imu": True}, destination)

    assert pd.read_csv(destination)["recording_id"].tolist() == ["B"]
    assert destination.with_suffix(".query.txt").read_text(encoding="utf-8") == "has_headimu\n"
    assert written == (destination, destination.with_suffix(".query.txt"))


def test_export_selection_rejects_non_metadata_format(tmp_path, manifest):
    with pytest.raises(ValueError, match=r"\.csv or \.json"):
        export_selection(manifest, {}, tmp_path / "selection.parquet")


def test_export_selected_recordings_creates_a_portable_subset(tmp_path, manifest):
    root = tmp_path / "bundle"
    root.mkdir()
    source_manifest = manifest.assign(
        has_watch_rawaccel=False,
        has_markers=[True, False],
        has_pen=[True, False],
    )
    source_manifest.to_parquet(root / "sessions.parquet", index=False)
    pd.DataFrame([
        {"recording_id": "A", "modality": "watch", "column": "accel_user_x"},
        {"recording_id": "A", "modality": "pen", "column": "x"},
        {"recording_id": "A", "modality": "markers", "column": "event"},
        {"recording_id": "B", "modality": "watch", "column": "accel_total_x"},
        {"recording_id": "B", "modality": "headimu", "column": "accel_x"},
    ]).to_parquet(root / "channels.parquet", index=False)
    for modality, recording_id, content in (
        ("watch", "A", b"watch-a"),
        ("pen", "A", b"pen-a"),
        ("markers", "A", b"markers-a"),
        ("watch", "B", b"watch-b"),
        ("headimu", "B", b"head-b"),
    ):
        table = root / modality / f"{recording_id}.parquet"
        table.parent.mkdir(exist_ok=True)
        table.write_bytes(content)

    destination = tmp_path / "Downloads" / "focuswatch-selection"
    written = export_selected_recordings(
        root,
        source_manifest,
        {"smartwatch": True, "digital_pen": True},
        destination,
    )

    assert written == destination
    assert (destination / "selection.json").is_file()
    assert (destination / "sessions.parquet").is_file()
    assert (destination / "channels.parquet").is_file()
    assert not (destination / ".incomplete").exists()
    assert (destination / "watch" / "A.parquet").read_bytes() == b"watch-a"
    assert (destination / "pen" / "A.parquet").read_bytes() == b"pen-a"
    assert (destination / "markers" / "A.parquet").read_bytes() == b"markers-a"
    assert not (destination / "headimu").exists()
    assert not (destination / "watch" / "B.parquet").exists()

    payload = json.loads((destination / "selection.json").read_text(encoding="utf-8"))
    assert payload["recording_ids"] == ["A"]
    assert payload["modalities"] == ["watch", "pen", "markers"]
    assert pd.read_parquet(destination / "channels.parquet")["modality"].tolist() == [
        "watch", "pen", "markers"
    ]


def test_export_selected_recordings_rejects_an_empty_selection(tmp_path, manifest):
    with pytest.raises(ValueError, match="matched no recordings"):
        export_selected_recordings(
            tmp_path,
            manifest,
            {"head_imu": True, "watch_gravity": True},
            tmp_path / "Downloads" / "selection",
        )


def test_export_selected_recordings_rejects_an_unsafe_recording_id(tmp_path, manifest):
    unsafe = manifest.assign(recording_id="../outside")

    with pytest.raises(ValueError, match="unsafe recording_id"):
        export_selected_recordings(
            tmp_path,
            unsafe,
            {"smartwatch": True},
            tmp_path / "Downloads" / "selection",
        )


def test_export_selected_recordings_refuses_a_dangling_destination_symlink(
    tmp_path, manifest
):
    destination = tmp_path / "Downloads" / "selection"
    destination.parent.mkdir()
    destination.symlink_to(tmp_path / "missing-target", target_is_directory=True)

    with pytest.raises(FileExistsError, match="already exists"):
        export_selected_recordings(
            tmp_path,
            manifest,
            {"smartwatch": True},
            destination,
        )

    assert destination.is_symlink()


def test_export_selected_recordings_refuses_an_existing_destination(tmp_path, manifest):
    destination = tmp_path / "Downloads" / "selection"
    destination.mkdir(parents=True)

    with pytest.raises(FileExistsError, match="already exists"):
        export_selected_recordings(
            tmp_path,
            manifest,
            {"smartwatch": True},
            destination,
        )


@pytest.mark.parametrize("recording_id", ["CON", "AUX", "A:B", "session.", "session "])
def test_export_selected_recordings_rejects_windows_unsafe_recording_ids(
    tmp_path, manifest, recording_id
):
    unsafe = manifest.assign(recording_id=recording_id)

    with pytest.raises(ValueError, match="unsafe recording_id"):
        export_selected_recordings(
            tmp_path,
            unsafe,
            {"smartwatch": True},
            tmp_path / "Downloads" / "selection",
        )


def test_cli_missing_textual_prints_exact_install_command(tmp_path, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "focuswatch_dataset.tui", None)

    assert cli.main(["explore", str(tmp_path)]) == 1
    assert capsys.readouterr().out.strip() == (
        'Explorer unavailable: install it with pip install "focuswatch-dataset[tui]"'
    )
