import asyncio
import json
import sys

import pandas as pd
from textual.widgets import Checkbox, Static

from focuswatch_dataset import cli
from focuswatch_dataset.tui import DatasetExplorerApp


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


def test_toggling_smartwatch_and_gravity_updates_visible_count(tmp_path):
    root = build_manifest(tmp_path)

    async def scenario():
        app = DatasetExplorerApp(root, export_path=tmp_path / "selection.json")
        async with app.run_test() as pilot:
            assert "3 Aufnahmen" in count_text(app)
            labels = {str(box.label) for box in app.query(Checkbox)}
            assert "Smartwatch" in labels
            assert "Gravity vorhanden" in labels

            await pilot.click("#facet-smartwatch")
            await pilot.pause()
            assert "2 Aufnahmen" in count_text(app)

            await pilot.click("#facet-watch_gravity")
            await pilot.pause()
            assert "1 Aufnahme" in count_text(app)

    asyncio.run(scenario())


def test_export_key_writes_only_selected_manifest_rows_and_query(tmp_path):
    root = build_manifest(tmp_path)
    destination = tmp_path / "selection.json"

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
    assert not list(tmp_path.glob("watch/*.parquet"))


def test_cli_missing_textual_prints_exact_install_command(tmp_path, monkeypatch, capsys):
    build_manifest(tmp_path)
    monkeypatch.setitem(sys.modules, "focuswatch_dataset.tui", None)

    assert cli.main(["explore", str(tmp_path)]) == 1
    assert capsys.readouterr().out.strip() == (
        'Explorer unavailable: install it with pip install "focuswatch-dataset[tui]"'
    )
