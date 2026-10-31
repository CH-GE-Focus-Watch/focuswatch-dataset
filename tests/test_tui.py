import asyncio
import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("textual")

from textual.widgets import Checkbox, DataTable, Static, TabbedContent

from focuswatch_dataset.tui import (
    DatasetExplorerApp,
    ProvenanceScreen,
    _default_export_directory,
)


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
            "watch_model": "Apple Watch Series 6",
            "pen_model": "Moleskine Smart Pen NWP-F130",
            "head_model": "",
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
            "watch_model": "Apple Watch SE (2nd generation)",
            "pen_model": "",
            "head_model": "AirPods Pro 3",
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
            "watch_model": "",
            "pen_model": "LAMY safari all black ncode NWP-F80",
            "head_model": "",
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
            assert "Study recording" not in labels

            await pilot.click("#facet-smartwatch")
            await pilot.pause()
            assert "2 of 3 recordings" in count_text(app)

            # Signal facets live in step 2 of the assistant.
            await pilot.press("2")
            await pilot.click("#facet-watch_gravity")
            await pilot.pause()
            assert "1 of 3 recording " in count_text(app)

    asyncio.run(scenario())


def test_export_writes_selected_sensor_files_to_the_requested_directory(tmp_path):
    root = build_manifest(tmp_path)
    destination = tmp_path / "Downloads" / "selection"
    sensor = root / "watch" / "A.parquet"
    sensor.parent.mkdir()
    sentinel = b"SENSOR_SECRET_SENTINEL"
    sensor.write_bytes(sentinel)
    pd.DataFrame([
        {"recording_id": "A", "modality": "watch", "column": "accel_user_x"},
        {"recording_id": "B", "modality": "watch", "column": "accel_total_x"},
    ]).to_parquet(root / "channels.parquet", index=False)

    async def scenario():
        app = DatasetExplorerApp(root, export_path=destination)
        async with app.run_test() as pilot:
            await pilot.click("#facet-smartwatch")
            await pilot.press("2")
            await pilot.click("#facet-watch_gravity")
            await pilot.press("e")
            await pilot.pause()

    asyncio.run(scenario())
    payload = json.loads((destination / "selection.json").read_text(encoding="utf-8"))
    assert set(payload) == {"query", "rows", "recording_ids", "modalities"}
    assert payload["query"] == "has_watch and has_gravity"
    assert [row["recording_id"] for row in payload["rows"]] == ["A"]
    assert payload["modalities"] == ["watch"]
    assert (destination / "watch" / "A.parquet").read_bytes() == sentinel
    assert not (destination / "watch" / "B.parquet").exists()


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


def test_preview_names_each_device_by_its_model(tmp_path):
    root = build_manifest(tmp_path)

    async def scenario():
        app = DatasetExplorerApp(root, export_path=tmp_path / "selection.json")
        async with app.run_test() as pilot:
            await pilot.pause()
            preview = table_text(app.query_one("#preview", DataTable))
            assert "Watch S6 · Pen F130" in preview
            assert "Watch SE2 · AirPods Pro 3" in preview
            assert "Observer" not in preview
            assert "Pen F80" in preview

    asyncio.run(scenario())


@pytest.mark.parametrize("row, expected", [
    ({"has_pen": True, "pen_model": "unknown"}, "Pen unknown"),
    ({"has_pen": True, "pen_model": float("nan")}, "Pen unknown"),
    ({"has_watch": True, "watch_model": "Apple Watch Ultra"}, "Apple Watch Ultra"),
    ({"has_watch": False, "watch_model": "Apple Watch Series 6"}, "–"),
])
def test_device_names_never_hide_an_unknown_or_unlisted_model(row, expected):
    assert DatasetExplorerApp._devices(pd.Series(row)) == expected


def test_natural_sort_key_orders_numbered_ids_numerically():
    from focuswatch_dataset.tui import _natural_key

    ids = ["AIRPODS-P10", "AIRPODS-P2", "AIRPODS-P1", "ML4SCS-S008"]
    assert sorted(ids, key=_natural_key) == [
        "AIRPODS-P1", "AIRPODS-P2", "AIRPODS-P10", "ML4SCS-S008"
    ]


def test_default_export_directory_uses_downloads_or_current_directory(tmp_path):
    home = tmp_path / "home"
    downloads = home / "Downloads"
    downloads.mkdir(parents=True)

    assert _default_export_directory(home=home, cwd=tmp_path).parent == downloads
    assert _default_export_directory(home=tmp_path / "no-downloads", cwd=tmp_path).parent == tmp_path


def facet_count(app, facet_id):
    return str(app.query_one(f"#count-{facet_id}", Static).render())


def test_facets_show_how_many_recordings_each_would_leave(tmp_path):
    root = build_manifest(tmp_path)

    async def scenario():
        app = DatasetExplorerApp(root, export_path=tmp_path / "selection")
        async with app.run_test() as pilot:
            assert facet_count(app, "smartwatch") == "2"
            assert facet_count(app, "digital_pen") == "2"
            await pilot.click("#facet-smartwatch")
            await pilot.pause()
            assert facet_count(app, "digital_pen") == "1"

    asyncio.run(scenario())


def test_watch_model_facets_count_each_model(tmp_path):
    root = build_manifest(tmp_path)

    async def scenario():
        app = DatasetExplorerApp(root, export_path=tmp_path / "selection")
        async with app.run_test():
            assert facet_count(app, "watch_series_6") == "1"
            assert facet_count(app, "watch_se_2") == "1"

    asyncio.run(scenario())


def test_pen_and_head_model_facets_count_each_model(tmp_path):
    root = build_manifest(tmp_path)

    async def scenario():
        app = DatasetExplorerApp(root, export_path=tmp_path / "selection")
        async with app.run_test():
            assert facet_count(app, "pen_nwp_f130") == "1"
            assert facet_count(app, "pen_nwp_f80") == "1"
            assert facet_count(app, "pen_nwp_f50") == "0"
            assert facet_count(app, "head_airpods_pro_3") == "1"

    asyncio.run(scenario())


def test_facets_whose_column_is_missing_are_disabled_not_fatal(tmp_path):
    # build_manifest has no has_watch_gyro column, like a bundle built before it existed.
    root = build_manifest(tmp_path)

    async def scenario():
        app = DatasetExplorerApp(root, export_path=tmp_path / "selection")
        async with app.run_test():
            assert app.query_one("#facet-watch_gyro", Checkbox).disabled
            assert facet_count(app, "watch_gyro") == "–"
            assert not app.query_one("#facet-smartwatch", Checkbox).disabled

    asyncio.run(scenario())


def test_next_and_back_buttons_walk_the_three_steps(tmp_path):
    root = build_manifest(tmp_path)

    async def scenario():
        app = DatasetExplorerApp(root, export_path=tmp_path / "selection")
        async with app.run_test() as pilot:
            steps = app.query_one("#steps", TabbedContent)
            assert steps.active == "data"
            # Each pause lets the newly shown pane lay out before the next click.
            await pilot.click("#next-data")
            await pilot.pause()
            assert steps.active == "signals"
            await pilot.click("#next-signals")
            await pilot.pause()
            assert steps.active == "export"
            await pilot.click("#back-export")
            await pilot.pause()
            assert steps.active == "signals"

    asyncio.run(scenario())


def test_export_step_can_bring_the_watch_to_50_hz(tmp_path):
    from focuswatch_dataset.write import write_table

    root = build_manifest(tmp_path)
    # A real manifest declares the measured rate the export step reads.
    manifest = pd.read_parquet(root / "sessions.parquet")
    manifest["watch_hz_measured"] = [100.0, 50.0, None]
    manifest.to_parquet(root / "sessions.parquet", index=False)
    t = np.arange(200) / 100.0
    write_table(pd.DataFrame({"t_ns": (t * 1e9).astype(np.int64),
                              "accel_user_x": np.sin(2 * np.pi * t)}),
                root / "watch" / "A.parquet", "A")
    pd.DataFrame([
        {"recording_id": "A", "modality": "watch", "column": "accel_user_x", "sample_rate_hz": 100.0},
    ]).to_parquet(root / "channels.parquet", index=False)
    destination = tmp_path / "at-50"

    async def scenario():
        app = DatasetExplorerApp(root, export_path=destination)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.click("#facet-smartwatch")
            await pilot.press("2")
            await pilot.click("#facet-watch_gravity")
            await pilot.press("3")
            await pilot.pause()
            assert "watch" in str(app.query_one("#export-files", Static).render())
            await pilot.click("#rate-target")
            await pilot.click("#export-button")
            await pilot.pause()

    asyncio.run(scenario())
    payload = json.loads((destination / "selection.json").read_text(encoding="utf-8"))
    assert payload["watch_resampled_to_hz"] == 50.0
    assert len(pd.read_parquet(destination / "watch" / "A.parquet")) == 100


def test_export_step_explains_rates_and_disables_a_pointless_downsample(tmp_path):
    root = build_manifest(tmp_path)
    manifest = pd.read_parquet(root / "sessions.parquet")
    manifest["watch_hz_measured"] = [100.0, 50.0, None]
    manifest.to_parquet(root / "sessions.parquet", index=False)

    async def scenario():
        app = DatasetExplorerApp(root, export_path=tmp_path / "selection")
        async with app.run_test() as pilot:
            def info():
                return str(app.query_one("#rate-info", Static).render())

            assert "1 × 50 Hz · 1 × 100 Hz" in info()
            assert "1 downsampled, 1 copied" in info()
            assert not app.query_one("#rate-target").disabled

            app.query_one("#facet-observer_annotation", Checkbox).value = True
            await pilot.pause()
            assert "nothing to downsample" in info()
            assert app.query_one("#rate-target").disabled

    asyncio.run(scenario())


def test_goal_presets_set_the_matching_facets(tmp_path):
    root = build_manifest(tmp_path)

    async def scenario():
        app = DatasetExplorerApp(root, export_path=tmp_path / "selection")
        async with app.run_test() as pilot:
            await pilot.click("#preset-writing_detection")
            await pilot.pause()
            assert app.selection.query == "has_watch and has_pen"
            assert "1 of 3 recording " in count_text(app)

            await pilot.click("#preset-attention")
            await pilot.pause()
            assert app.selection.query == "has_headimu and has_attention"
            assert not app.query_one("#facet-smartwatch", Checkbox).value

            await pilot.click("#preset-everything")
            await pilot.pause()
            assert "3 of 3 recordings" in count_text(app)

    asyncio.run(scenario())


def test_arrow_keys_move_between_filters(tmp_path):
    root = build_manifest(tmp_path)

    async def scenario():
        app = DatasetExplorerApp(root, export_path=tmp_path / "selection")
        async with app.run_test() as pilot:
            app.query_one("#facet-smartwatch", Checkbox).focus()
            await pilot.press("down")
            assert app.focused.id == "facet-head_imu"
            await pilot.press("space")
            assert app.query_one("#facet-head_imu", Checkbox).value
            await pilot.press("up")
            assert app.focused.id == "facet-smartwatch"

    asyncio.run(scenario())
