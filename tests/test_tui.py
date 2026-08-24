import asyncio
import json

import pandas as pd
import pytest

pytest.importorskip("textual")

from textual.widgets import Checkbox, Static

from focuswatch_dataset.tui import DatasetExplorerApp, ProvenanceScreen


def build_manifest(root):
    pd.DataFrame([
        {
            "recording_id": "A",
            "participant_id": "P1",
            "cohort": "source-a",
            "protocol_id": "protocol-a",
            "source_pipeline": "pipeline-a",
            "has_watch": True,
            "has_headimu": False,
            "has_pen": True,
            "has_attention": False,
            "has_gravity": True,
        },
        {
            "recording_id": "B",
            "participant_id": "P2",
            "cohort": "source-b",
            "protocol_id": "protocol-b",
            "source_pipeline": "pipeline-b",
            "has_watch": True,
            "has_headimu": True,
            "has_pen": False,
            "has_attention": True,
            "has_gravity": False,
        },
        {
            "recording_id": "C",
            "participant_id": "P3",
            "cohort": "source-c",
            "protocol_id": "protocol-c",
            "source_pipeline": "pipeline-c",
            "has_watch": False,
            "has_headimu": False,
            "has_pen": True,
            "has_attention": False,
            "has_gravity": False,
        },
    ]).to_parquet(root / "sessions.parquet", index=False)
    return root


def count_text(app):
    return str(app.query_one("#recording-count", Static).render())


def table_text(table):
    return " ".join(
        str(cell)
        for row_index in range(table.row_count)
        for cell in table.get_row_at(row_index)
    )


def test_toggling_smartwatch_and_gravity_updates_visible_count(tmp_path):
    root = build_manifest(tmp_path)

    async def scenario():
        app = DatasetExplorerApp(root, export_path=tmp_path / "selection.json")
        async with app.run_test() as pilot:
            assert "3 of 3 recordings" in count_text(app)
            labels = {str(box.label) for box in app.query(Checkbox)}
            assert "Smartwatch" in labels
            assert "Watch gravity" in labels

            await pilot.click("#facet-smartwatch")
            await pilot.pause()
            assert "2 of 3 recordings" in count_text(app)

            await pilot.click("#facet-watch_gravity")
            await pilot.pause()
            assert "1 of 3 recording " in count_text(app)

    asyncio.run(scenario())


def test_export_reads_only_sessions_manifest_and_never_sensor_data(tmp_path, monkeypatch):
    root = build_manifest(tmp_path)
    destination = tmp_path / "selection.json"
    sensor = root / "watch" / "A.parquet"
    sensor.parent.mkdir()
    sentinel = b"SENSOR_SECRET_SENTINEL"
    sensor.write_bytes(sentinel)
    parquet_reads = []
    real_read_parquet = pd.read_parquet

    def tracked_read_parquet(path, *args, **kwargs):
        parquet_reads.append(path)
        return real_read_parquet(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", tracked_read_parquet)

    async def scenario():
        app = DatasetExplorerApp(root, export_path=destination)
        async with app.run_test() as pilot:
            await pilot.click("#facet-smartwatch")
            await pilot.click("#facet-watch_gravity")
            await pilot.press("e")
            await pilot.pause()

    asyncio.run(scenario())
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert set(payload) == {"query", "rows"}
    assert payload["query"] == "has_watch and has_gravity"
    assert [row["recording_id"] for row in payload["rows"]] == ["A"]
    assert [str(path) for path in parquet_reads] == [str(root / "sessions.parquet")]
    assert sensor.read_bytes() == sentinel
    assert sentinel not in destination.read_bytes()


def test_copy_action_and_provenance_details_are_separate_from_overview(tmp_path, monkeypatch):
    root = build_manifest(tmp_path)
    copied = []

    async def scenario():
        app = DatasetExplorerApp(root, export_path=tmp_path / "selection.json")
        monkeypatch.setattr(app, "copy_to_clipboard", copied.append)
        async with app.run_test() as pilot:
            await pilot.click("#facet-smartwatch")
            await pilot.press("c")
            await pilot.pause()
            assert copied == ["has_watch"]

            overview = " ".join(
                str(widget.render()) for widget in app.screen.query(Static)
            ) + table_text(app.query_one("#preview"))
            assert "source-a" not in overview
            assert "protocol-a" not in overview

            await pilot.press("p")
            await pilot.pause()
            assert isinstance(app.screen, ProvenanceScreen)
            details = table_text(app.screen.query_one("#provenance-table"))
            assert "source-a" in details
            assert "protocol-a" in details

    asyncio.run(scenario())


def test_reset_clears_facets_and_preview_is_naturally_sorted(tmp_path):
    root = build_manifest(tmp_path)

    async def scenario():
        app = DatasetExplorerApp(root, export_path=tmp_path / "selection.json")
        async with app.run_test() as pilot:
            await pilot.click("#facet-smartwatch")
            await pilot.pause()
            assert "2 of 3 recordings" in count_text(app)
            assert "has_watch" in str(app.query_one("#query", Static).render())

            await pilot.press("r")
            await pilot.pause()
            assert "3 of 3 recordings" in count_text(app)
            assert all(not box.value for box in app.query(Checkbox))

    asyncio.run(scenario())


def test_natural_sort_key_orders_numbered_ids_numerically():
    from focuswatch_dataset.tui import _natural_key

    ids = ["AIRPODS-P10", "AIRPODS-P2", "AIRPODS-P1", "ML4SCS-S008"]
    assert sorted(ids, key=_natural_key) == [
        "AIRPODS-P1", "AIRPODS-P2", "AIRPODS-P10", "ML4SCS-S008"
    ]
