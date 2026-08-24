"""Optional Textual interface for exploring manifest metadata."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from textual.app import App, ComposeResult
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.screen import ModalScreen
from textual.widgets import Checkbox, DataTable, Footer, Static

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


class ProvenanceScreen(ModalScreen[None]):
    """Show source identifiers separately from the plain-language overview."""

    BINDINGS = [("escape", "dismiss", "Zur\u00fcck")]

    def __init__(self, manifest: pd.DataFrame) -> None:
        super().__init__()
        self.manifest = manifest

    def compose(self) -> ComposeResult:
        yield Static("Provenienz und Details", id="details-title")
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
    SUB_TITLE = "Auswahl aus sessions.parquet"
    BINDINGS = [
        ("e", "export", "Exportieren"),
        ("c", "copy_query", "Query kopieren"),
        ("p", "provenance", "Provenienz/Details"),
        ("q", "quit", "Beenden"),
    ]
    CSS = """
    Screen { layout: vertical; }
    #workspace { height: 1fr; }
    #facets { width: 38; padding: 0 1; border-right: solid $primary; }
    #results { width: 1fr; padding: 0 1; }
    .group-title { margin-top: 1; text-style: bold; }
    #recording-count { text-style: bold; margin-bottom: 1; }
    #breakdown { height: auto; margin-bottom: 1; }
    #preview { height: 1fr; }
    ProvenanceScreen { align: center middle; }
    ProvenanceScreen > #details-title { width: 90%; height: 3; padding: 1; background: $panel; }
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
                yield DataTable(id="preview")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#preview", DataTable).add_columns(
            "Aufnahme-ID", "Teilnehmer-Kennung", "Verf\u00fcgbare Ger\u00e4te"
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
        count = len(self.selected)
        noun = "Aufnahme" if count == 1 else "Aufnahmen"
        self.query_one("#recording-count", Static).update(f"{count} {noun} ausgew\u00e4hlt")
        self.query_one("#breakdown", Static).update(self._breakdown_text())
        self._redraw_preview()

    def _breakdown_text(self) -> str:
        labels = (
            ("has_watch", "Smartwatch"),
            ("has_headimu", "Kopf-IMU/Kopfh\u00f6rer"),
            ("has_pen", "Digitaler Stift"),
            ("has_attention", "Beobachter-Annotation"),
        )
        parts = [
            f"{label}: {int(self.selected[column].fillna(False).astype(bool).sum())}"
            for column, label in labels
            if column in self.selected
        ]
        return "  \u00b7  ".join(parts) if parts else "Keine Ger\u00e4teangaben im Manifest"

    @staticmethod
    def _devices(row: pd.Series) -> str:
        labels = (
            ("has_watch", "Smartwatch"),
            ("has_headimu", "Kopf-IMU/Kopfh\u00f6rer"),
            ("has_pen", "Stift"),
            ("has_attention", "Annotation"),
        )
        return ", ".join(
            label for column, label in labels if column in row and bool(row[column])
        ) or "\u2013"

    def _redraw_preview(self) -> None:
        table = self.query_one("#preview", DataTable)
        table.clear()
        for _, row in self.selected.head(100).iterrows():
            table.add_row(
                str(row.get("recording_id", "")),
                str(row.get("participant_id", "")),
                self._devices(row),
            )

    def action_export(self) -> None:
        written = export_selection(self.manifest, self.selection, self.export_path)
        self.notify("Exportiert: " + ", ".join(str(path) for path in written))

    def action_copy_query(self) -> None:
        self.copy_to_clipboard(self.selection.query)
        self.notify("Pandas-Query kopiert")

    def action_provenance(self) -> None:
        self.push_screen(ProvenanceScreen(self.selected))
