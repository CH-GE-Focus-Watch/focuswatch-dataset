"""Optional Textual interface for exploring manifest metadata.

The explorer is a three-step assistant: choose data, narrow by signals and
timing, then export. The recording preview on the right stays visible in
every step.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re

import pandas as pd
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    Checkbox,
    DataTable,
    Footer,
    Header,
    Input,
    RadioButton,
    RadioSet,
    Static,
    TabbedContent,
    TabPane,
)

from . import schema as S
from .explore import (
    FACETS,
    PRESETS,
    ExplorerSelection,
    export_preview,
    export_selected_recordings,
    watch_rate_summary,
)
from .load import load_manifest


_PROVENANCE_COLUMNS = (
    "recording_id",
    "cohort",
    "protocol_id",
    "source_pipeline",
    "schema_version",
    "build_git_sha",
)

# Plain-language modality labels for the breakdown line above the preview.
_DEVICE_LABELS = (
    ("has_watch", "Smartwatch"),
    ("has_headimu", "Head IMU"),
    ("has_pen", "Pen"),
    ("has_attention", "Annotation"),
)

# The preview names each device by a short model name. A model missing here
# is shown by its full manifest name rather than hidden.
_MODEL_SHORT_NAMES = {
    S.WATCH_MODEL_SERIES_6: "Watch S6",
    S.WATCH_MODEL_SE_2: "Watch SE2",
    S.PEN_MODEL_NWP_F130: "Pen F130",
    S.PEN_MODEL_NWP_F50: "Pen F50",
    S.PEN_MODEL_NWP_F80: "Pen F80",
    S.HEAD_MODEL_AIRPODS_PRO_3: "AirPods Pro 3",
}
# Modality flag, its model column, and the name shown when the model is unknown.
_DEVICE_MODEL_COLUMNS = (
    ("has_watch", "watch_model", "Watch unknown"),
    ("has_pen", "pen_model", "Pen unknown"),
    ("has_headimu", "head_model", "Head IMU unknown"),
)

# Which facet groups each assistant step shows, in registry order.
_STEPS = (
    ("data", "1 Choose data", ("Modalities", "Cohort")),
    ("signals", "2 Signals & timing", ("Signals", "Acquisition", "Timing")),
)

_EXPORT_RATE_HZ = 50.0

# One colour per cohort so mixed selections read at a glance in the preview.
_COHORT_STYLES = {
    "ML4SCS": "bold cyan",
    "ZHAW": "bold magenta",
    "AIRPODS": "bold green",
    "MOLESKINE": "bold yellow",
}


class FacetList(VerticalScroll):
    """Facet column where the arrow keys move between filters."""

    # Why: VerticalScroll binds up/down to scrolling, which would leave the
    # focused checkbox behind; moving focus scrolls it into view instead.
    BINDINGS = [
        Binding("down", "app.focus_next", "Next filter", show=False),
        Binding("up", "app.focus_previous", "Previous filter", show=False),
    ]


def _natural_key(value: object) -> tuple[object, ...]:
    """Sort ``P1, P2, P10`` numerically instead of lexicographically."""

    parts = re.split(r"(\d+)", str(value))
    return tuple(int(part) if part.isdigit() else part.lower() for part in parts)


def _format_duration(seconds: object) -> str:
    if seconds is None or pd.isna(seconds):
        return "–"
    hours, rest = divmod(int(round(float(seconds))), 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours:d}:{minutes:02d}:{secs:02d}"


def _format_bytes(size: int) -> str:
    for unit in ("B", "KB", "MB"):
        if size < 1024:
            return f"{size:.0f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


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
    SUB_TITLE = "Choose data · narrow · export"
    BINDINGS = [
        ("1", "step('data')", "Data"),
        ("2", "step('signals')", "Signals"),
        ("3", "step('export')", "Export step"),
        ("e", "export", "Export"),
        ("c", "copy_query", "Copy query"),
        ("r", "reset", "Reset"),
        ("p", "provenance", "Provenance"),
        ("q", "quit", "Quit"),
    ]
    CSS = """
    Screen { layout: vertical; }
    #summary { height: auto; padding: 0 2; background: $boost; border-bottom: hkey $primary; }
    #recording-count { text-style: bold; color: $accent; }
    #query { color: $text-muted; }
    #workspace { height: 1fr; }
    #steps { width: 48; border: round $primary; border-title-color: $accent; }
    #results { width: 1fr; padding: 0 1; border: round $primary-darken-2; border-title-color: $accent; }
    #presets { height: 1; }
    /* One-line buttons keep the facets below them on a 24-row terminal. */
    #presets Button { height: 1; min-width: 10; border: none; margin-right: 1; }
    #preset-hint { color: $text-muted; height: auto; }
    #rate-info { color: $text-muted; height: auto; margin-bottom: 1; }
    .group-title { margin-top: 1; text-style: bold; color: $accent; }
    .facet-row { height: 1; }
    .facet-row Checkbox { border: none; padding: 0; height: 1; width: 1fr; background: transparent; }
    .facet-row Checkbox:focus { border: none; text-style: bold; }
    .facet-count { width: 6; text-align: right; color: $text-muted; }
    /* Docked so Back/Next/Export stay reachable on short terminals. */
    .step-nav { dock: bottom; height: auto; padding-top: 1; }
    .step-nav Button { margin-right: 1; min-width: 12; }
    #breakdown { height: auto; color: $text-muted; }
    #preview { height: 1fr; }
    #export-files { height: auto; margin: 1 0; }
    #watch-rate { height: auto; margin-bottom: 1; }
    #destination { margin-bottom: 1; }
    ProvenanceScreen { align: center middle; }
    ProvenanceScreen > #details-title { width: 90%; height: 3; padding: 1; background: $panel; text-style: bold; }
    ProvenanceScreen > #provenance-table { width: 90%; height: 75%; background: $surface; }
    """

    def __init__(self, root: Path | str, *, export_path: Path | str | None = None) -> None:
        super().__init__()
        self.root = Path(root)
        self.manifest = load_manifest(self.root)
        self.export_path = Path(export_path) if export_path is not None else None
        self.selection = ExplorerSelection()
        self.selected = self.manifest.copy().reset_index(drop=True)
        # Why: a bundle built before a facet's manifest column existed must
        # still open; such facets are shown disabled instead of crashing.
        self.available = {
            facet_id for facet_id, facet in FACETS.items()
            if facet.condition[0] in self.manifest.columns
        }

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="summary"):
            yield Static(id="recording-count")
            yield Static(id="query")
        with Horizontal(id="workspace"):
            with TabbedContent(id="steps"):
                for step_id, title, groups in _STEPS:
                    with TabPane(title, id=step_id):
                        with Horizontal(classes="step-nav"):
                            if step_id != "data":
                                yield Button("← Back", id=f"back-{step_id}")
                            yield Button("Next →", id=f"next-{step_id}", variant="primary")
                        with FacetList():
                            if step_id == "data":
                                yield Static("Goal", classes="group-title")
                                with Horizontal(id="presets"):
                                    for preset_id, preset in PRESETS.items():
                                        yield Button(preset.label, id=f"preset-{preset_id}")
                                yield Static("Pick a goal, then refine below.", id="preset-hint")
                            yield from self._facet_rows(groups)
                with TabPane("3 Export", id="export"):
                    with Horizontal(classes="step-nav"):
                        yield Button("← Back", id="back-export")
                        yield Button("Export", id="export-button", variant="success")
                    with VerticalScroll():
                        yield Static(id="export-files")
                        yield Static(id="rate-info")
                        with RadioSet(id="watch-rate"):
                            yield RadioButton("Keep native rates", id="rate-native", value=True)
                            yield RadioButton(
                                f"Bring watch to {_EXPORT_RATE_HZ:g} Hz", id="rate-target"
                            )
                        yield Static("Destination")
                        yield Input(id="destination")
            with Vertical(id="results"):
                yield Static(id="breakdown")
                yield DataTable(id="preview", cursor_type="row", zebra_stripes=True)
        yield Footer()

    def _facet_rows(self, groups: tuple[str, ...]) -> ComposeResult:
        for group in groups:
            yield Static(group, classes="group-title")
            for facet_id, facet in FACETS.items():
                if facet.group != group:
                    continue
                with Horizontal(classes="facet-row"):
                    yield Checkbox(
                        facet.label, id=f"facet-{facet_id}",
                        disabled=facet_id not in self.available,
                    )
                    yield Static(id=f"count-{facet_id}", classes="facet-count")

    def on_mount(self) -> None:
        self.theme = "tokyo-night"
        self.query_one("#steps").border_title = "Select"
        self.query_one("#results").border_title = "Preview"
        # Short columns first: long ZHAW recording ids must not push them off screen.
        self.query_one("#preview", DataTable).add_columns(
            "Recording ID", "Duration", "Hz", "Devices", "Participant"
        )
        destination = self.export_path or _default_export_directory()
        self.query_one("#destination", Input).value = str(destination)
        self._refresh_selection()

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        self._refresh_selection()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button = event.button.id or ""
        order = [step_id for step_id, _, _ in _STEPS] + ["export"]
        if button == "export-button":
            self.action_export()
        elif button.startswith("preset-"):
            self.apply_preset(button.removeprefix("preset-"))
        elif button.startswith(("next-", "back-")):
            current = order.index(button.split("-", 1)[1])
            self.action_step(order[current + (1 if button.startswith("next-") else -1)])

    def apply_preset(self, preset_id: str) -> None:
        preset = PRESETS[preset_id]
        with self.prevent(Checkbox.Changed):
            for facet_id in FACETS:
                checkbox = self.query_one(f"#facet-{facet_id}", Checkbox)
                checkbox.value = facet_id in preset.facets and facet_id in self.available
        self.query_one("#preset-hint", Static).update(f"{preset.label}: {preset.description}")
        self._refresh_selection()

    def action_step(self, step_id: str) -> None:
        self.query_one("#steps", TabbedContent).active = step_id
        # Why: TabbedContent re-activates the pane holding the focused widget,
        # so focus left on the old step's button would switch straight back.
        primary = "#export-button" if step_id == "export" else f"#next-{step_id}"
        self.query_one(primary, Button).focus()

    def _active_facets(self) -> dict[str, bool]:
        return {
            facet_id: self.query_one(f"#facet-{facet_id}", Checkbox).value
            for facet_id in FACETS
        }

    def _refresh_selection(self) -> None:
        active = self._active_facets()
        self.selection = ExplorerSelection.from_mapping(active)
        self.selected = self.selection.filter(self.manifest)
        self.query_one("#recording-count", Static).update(self._summary_text())
        self.query_one("#query", Static).update(f"query: {self.selection.query}")
        self.query_one("#breakdown", Static).update(self._breakdown_text())
        for facet_id in FACETS:
            count = "–"
            if facet_id in self.available:
                with_facet = ExplorerSelection.from_mapping({**active, facet_id: True})
                count = str(with_facet.count(self.manifest))
            self.query_one(f"#count-{facet_id}", Static).update(count)
        self._redraw_preview()
        self._redraw_export_files()

    def _summary_text(self) -> str:
        count, total = len(self.selected), len(self.manifest)
        noun = "recording" if count == 1 else "recordings"
        parts = [f"{count} of {total} {noun} selected"]
        if "duration_s" in self.selected:
            parts.append(f"{self.selected['duration_s'].fillna(0).sum() / 3600:.1f} h")
        if "cohort" in self.selected:
            cohorts = self.selected["cohort"].nunique()
            parts.append(f"{cohorts} cohort" + ("" if cohorts == 1 else "s"))
        return " · ".join(parts)

    def _breakdown_text(self) -> str:
        parts = [
            f"{label}: {int(self.selected[column].fillna(False).astype(bool).sum())}"
            for column, label in _DEVICE_LABELS
            if column in self.selected
        ]
        return "  ·  ".join(parts) if parts else "No device flags in manifest"

    @staticmethod
    def _devices(row: pd.Series) -> str:
        parts = []
        for flag, column, unknown in _DEVICE_MODEL_COLUMNS:
            if not (flag in row and bool(row[flag])):
                continue
            model = row.get(column)
            model = "" if model is None or pd.isna(model) else str(model)
            if model in ("", S.DEVICE_MODEL_UNKNOWN):
                parts.append(unknown)
            else:
                parts.append(_MODEL_SHORT_NAMES.get(model, model))
        # Why: observer annotation is no device, but it is the attention ground
        # truth, so the preview keeps showing that a recording has it.
        if "has_attention" in row and bool(row["has_attention"]):
            parts.append("Observer")
        return " · ".join(parts) or "–"

    @staticmethod
    def _rate(row: pd.Series) -> str:
        for column in ("watch_hz_measured", "head_hz_measured"):
            value = row.get(column)
            if value is not None and pd.notna(value):
                return f"{float(value):.0f}"
        return "–"

    def _redraw_preview(self) -> None:
        table = self.query_one("#preview", DataTable)
        table.clear()
        rows = self.selected
        if "recording_id" in rows:
            rows = rows.sort_values("recording_id", key=lambda s: s.map(_natural_key))
        for _, row in rows.iterrows():
            # Why: the id already names its cohort; colour shows it without
            # surfacing the raw cohort field, which stays in Provenance.
            style = _COHORT_STYLES.get(str(row.get("cohort", "")), "")
            table.add_row(
                Text(str(row.get("recording_id", "")), style=style),
                _format_duration(row.get("duration_s")),
                self._rate(row),
                self._devices(row),
                str(row.get("participant_id", "")),
            )

    def _redraw_rate_info(self) -> None:
        counts, above = watch_rate_summary(self.selected, _EXPORT_RATE_HZ)
        info = self.query_one("#rate-info", Static)
        option = self.query_one("#rate-target", RadioButton)
        option.disabled = above == 0
        if above == 0 and option.value:
            self.query_one("#rate-native", RadioButton).value = True
        if not counts:
            info.update("No watch data in this selection.")
            return
        rates = " · ".join(f"{n} × {hz} Hz" for hz, n in counts.items())
        if above == 0:
            info.update(f"Watch rates: {rates}\nAlready at {_EXPORT_RATE_HZ:g} Hz or below; nothing to downsample.")
        else:
            copied = sum(counts.values()) - above
            info.update(
                f"Watch rates: {rates}\nAt {_EXPORT_RATE_HZ:g} Hz: {above} downsampled, {copied} copied as they are."
            )

    def _redraw_export_files(self) -> None:
        self._redraw_rate_info()
        target = self.query_one("#export-files", Static)
        if self.selected.empty:
            target.update("Nothing selected yet.")
            return
        try:
            preview = export_preview(self.root, self.manifest, self.selection)
        except FileNotFoundError as exc:
            target.update(f"Cannot export: {exc}")
            return
        lines = [f"{modality:<10} {files:>3} files  {_format_bytes(size):>9}"
                 for modality, (files, size) in preview.items()]
        total = sum(size for _, size in preview.values())
        lines.append(f"{'total':<10} {'':>9}  {_format_bytes(total):>9}")
        lines.append("Always included: sessions.parquet · channels.parquet · selection.json")
        target.update("\n".join(lines))

    def action_export(self) -> None:
        destination = Path(self.query_one("#destination", Input).value).expanduser()
        target_hz = (
            _EXPORT_RATE_HZ if self.query_one("#rate-target", RadioButton).value else None
        )
        try:
            written = export_selected_recordings(
                self.root, self.manifest, self.selection, destination,
                watch_target_hz=target_hz,
            )
        except Exception as exc:
            self.notify(f"Export failed: {exc}", severity="error")
            return
        self.notify(f"Selected files exported to {written.resolve()}")

    def action_copy_query(self) -> None:
        self.copy_to_clipboard(self.selection.query)
        self.notify("pandas query copied to clipboard")

    def action_reset(self) -> None:
        with self.prevent(Checkbox.Changed):
            for checkbox in self.query(Checkbox):
                checkbox.value = False
        self._refresh_selection()

    def action_provenance(self) -> None:
        self.push_screen(ProvenanceScreen(self.selected))


def _default_export_directory(
    *, home: Path | None = None, cwd: Path | None = None
) -> Path:
    downloads = (home or Path.home()) / "Downloads"
    parent = downloads if downloads.is_dir() else (cwd or Path.cwd())
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return parent / f"focuswatch-selection-{timestamp}"
