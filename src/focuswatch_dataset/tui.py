"""Optional Textual interface for exploring manifest metadata."""
from __future__ import annotations

from pathlib import Path
import re

import pandas as pd
from textual.app import App, ComposeResult
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.screen import ModalScreen
from textual.widgets import Checkbox, DataTable, Footer, Header, Static

from .explore import FACETS, ExplorerSelection, export_selection
from .load import load_manifest


_PROVENANCE_COLUMNS = (
    "recording_id",
    "cohort",
    "protocol_id",
    "source_pipeline",
    "schema_version",
    "build_git_sha",
)

# Plain-language device labels shared by the breakdown line and the preview.
_DEVICE_LABELS = (
    ("has_watch", "Smartwatch"),
    ("has_headimu", "Head IMU"),
    ("has_pen", "Pen"),
    ("has_attention", "Annotation"),
)


def _natural_key(value: object) -> tuple[object, ...]:
    """Sort ``P1, P2, P10`` numerically instead of lexicographically."""

    parts = re.split(r"(\d+)", str(value))
    return tuple(int(part) if part.isdigit() else part.lower() for part in parts)


def _format_duration(seconds: object) -> str:
    if seconds is None or pd.isna(seconds):
        return "–"
    total = int(round(float(seconds)))
    minutes, secs = divmod(total, 60)
    return f"{minutes:d}:{secs:02d} min"


class ProvenanceScreen(ModalScreen[None]):
    """Show source identifiers separately from the plain-language overview."""

    BINDINGS = [("escape", "dismiss", "Back")]

    def __init__(self, manifest: pd.DataFrame) -> None:
        super().__init__()
        self.manifest = manifest

    def compose(self) -> ComposeResult:
        yield Static("Provenance and details", id="details-title")
        yield DataTable(id="provenance-table")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#provenance-table", DataTable)
        columns = [column for column in _PROVENANCE_COLUMNS if column in self.manifest]
        table.add_columns(*columns)
        for row in self.manifest.loc[:, columns].itertuples(index=False, name=None):
            table.add_row(*(str(value) for value in row))

    def action_dismiss(self) -> None:
        self.dismiss()


class DatasetExplorerApp(App[None]):
    """Select recordings using user-facing facets over sessions.parquet."""

    TITLE = "FocusWatch Dataset Explorer"
    SUB_TITLE = "Metadata selection over sessions.parquet"
    BINDINGS = [
        ("e", "export", "Export"),
        ("c", "copy_query", "Copy query"),
        ("r", "reset", "Reset"),
        ("p", "provenance", "Provenance"),
        ("q", "quit", "Quit"),
    ]
    CSS = """
    Screen { layout: vertical; }
    #workspace { height: 1fr; }
    #facets { width: 40; padding: 0 1; border-right: solid $primary; }
    #results { width: 1fr; padding: 0 1; }
    .group-title { margin-top: 1; text-style: bold; color: $accent; }
    #recording-count { text-style: bold; margin-top: 1; }
    #breakdown { height: auto; }
    #query { height: auto; color: $text-muted; margin-bottom: 1; }
    #preview { height: 1fr; }
    ProvenanceScreen { align: center middle; }
    ProvenanceScreen > #details-title { width: 90%; height: 3; padding: 1; background: $panel; text-style: bold; }
    ProvenanceScreen > #provenance-table { width: 90%; height: 75%; background: $surface; }
    """

    def __init__(self, root: Path | str, *, export_path: Path | str | None = None) -> None:
        super().__init__()
        self.root = Path(root)
        self.manifest = load_manifest(self.root)
        self.export_path = Path(export_path or "focuswatch-selection.json")
        self.selection = ExplorerSelection()
        self.selected = self.manifest.copy().reset_index(drop=True)

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="workspace"):
            with ScrollableContainer(id="facets"):
                current_group = None
                for facet_id, facet in FACETS.items():
                    if facet.group != current_group:
                        current_group = facet.group
                        yield Static(current_group, classes="group-title")
                    yield Checkbox(facet.label, id=f"facet-{facet_id}")
            with Vertical(id="results"):
                yield Static(id="recording-count")
                yield Static(id="breakdown")
                yield Static(id="query")
                yield DataTable(id="preview", cursor_type="row", zebra_stripes=True)
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#preview", DataTable).add_columns(
            "Recording ID", "Participant", "Duration", "Modalities"
        )
        self._refresh_selection()

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        self._refresh_selection()

    def _active_facets(self) -> dict[str, bool]:
        return {
            facet_id: self.query_one(f"#facet-{facet_id}", Checkbox).value
            for facet_id in FACETS
        }

    def _refresh_selection(self) -> None:
        self.selection = ExplorerSelection.from_mapping(self._active_facets())
        self.selected = self.selection.filter(self.manifest)
        count, total = len(self.selected), len(self.manifest)
        noun = "recording" if count == 1 else "recordings"
        self.query_one("#recording-count", Static).update(
            f"{count} of {total} {noun} selected"
        )
        self.query_one("#breakdown", Static).update(self._breakdown_text())
        self.query_one("#query", Static).update(f"query: {self.selection.query}")
        self._redraw_preview()

    def _breakdown_text(self) -> str:
        parts = [
            f"{label}: {int(self.selected[column].fillna(False).astype(bool).sum())}"
            for column, label in _DEVICE_LABELS
            if column in self.selected
        ]
        return "  ·  ".join(parts) if parts else "No device flags in manifest"

    @staticmethod
    def _devices(row: pd.Series) -> str:
        return ", ".join(
            label
            for column, label in _DEVICE_LABELS
            if column in row and bool(row[column])
        ) or "–"

    def _redraw_preview(self) -> None:
        table = self.query_one("#preview", DataTable)
        table.clear()
        rows = self.selected
        if "recording_id" in rows:
            rows = rows.sort_values("recording_id", key=lambda s: s.map(_natural_key))
        for _, row in rows.iterrows():
            table.add_row(
                str(row.get("recording_id", "")),
                str(row.get("participant_id", "")),
                _format_duration(row.get("duration_s")),
                self._devices(row),
            )

    def action_export(self) -> None:
        written = export_selection(self.manifest, self.selection, self.export_path)
        self.notify("Exported: " + ", ".join(str(path) for path in written))

    def action_copy_query(self) -> None:
        self.copy_to_clipboard(self.selection.query)
        self.notify("pandas query copied to clipboard")

    def action_reset(self) -> None:
        for checkbox in self.query(Checkbox):
            checkbox.value = False
        self._refresh_selection()

    def action_provenance(self) -> None:
        self.push_screen(ProvenanceScreen(self.selected))
