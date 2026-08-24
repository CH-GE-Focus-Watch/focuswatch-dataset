"""Pure manifest-selection model for the optional terminal explorer."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import shutil
import tempfile
from typing import Mapping

import pandas as pd

from . import schema as S


@dataclass(frozen=True)
class Facet:
    """A user-facing filter backed by one published manifest condition."""

    label: str
    group: str
    condition: tuple[str, object]


# Registry order is the canonical order for both the UI and generated queries.
FACETS: dict[str, Facet] = {
    "smartwatch": Facet("Smartwatch", "Modalities", ("has_watch", True)),
    "head_imu": Facet("Head IMU", "Modalities", ("has_headimu", True)),
    "digital_pen": Facet("Smart Pen", "Modalities", ("has_pen", True)),
    "observer_annotation": Facet(
        "Observer annotation", "Modalities", ("has_attention", True)
    ),
    "watch_gravity": Facet("Watch gravity", "Signals", ("has_gravity", True)),
    "watch_quaternion": Facet(
        "Watch quaternion", "Signals", ("has_quaternion", True)
    ),
    "head_gravity": Facet(
        "Head gravity", "Signals", ("has_head_gravity", True)
    ),
    "head_quaternion": Facet(
        "Head quaternion", "Signals", ("has_head_quaternion", True)
    ),
    "head_gyro": Facet("Head gyroscope", "Signals", ("has_head_gyro", True)),
    "watch_raw_acceleration": Facet(
        "Watch raw acceleration", "Signals", ("has_watch_rawaccel", True)
    ),
    "watch_50_hz": Facet(
        "Watch nominal rate 50 Hz", "Acquisition", ("watch_hz_nominal", 50.0)
    ),
    "watch_100_hz": Facet(
        "Watch nominal rate 100 Hz", "Acquisition", ("watch_hz_nominal", 100.0)
    ),
    "acceleration_user": Facet(
        "Acceleration without gravity", "Acquisition", ("accel_semantics", "user")
    ),
    "acceleration_total": Facet(
        "Acceleration including gravity", "Acquisition", ("accel_semantics", "total")
    ),
}


def _active_ids(selection: ExplorerSelection | Mapping[str, bool]) -> frozenset[str]:
    if isinstance(selection, ExplorerSelection):
        return selection.active_facets
    unknown = set(selection) - set(FACETS)
    if unknown:
        raise KeyError(f"unknown explorer facet(s): {', '.join(sorted(unknown))}")
    return frozenset(facet_id for facet_id, active in selection.items() if active)


def selection_query(selection: ExplorerSelection | Mapping[str, bool]) -> str:
    """Return a stable pandas-compatible expression for active facets."""

    active = _active_ids(selection)
    expressions: list[str] = []
    for facet_id, facet in FACETS.items():
        if facet_id not in active:
            continue
        column, value = facet.condition
        if value is True:
            expressions.append(column)
        elif value is False:
            expressions.append(f"not {column}")
        else:
            expressions.append(f"{column} == {value!r}")
    # A scalar ``True`` is not a valid DataFrame.query row mask.
    return " and ".join(expressions) or "index == index"


def filter_manifest(
    manifest: pd.DataFrame,
    selection: ExplorerSelection | Mapping[str, bool],
) -> pd.DataFrame:
    """Intersect active facets using manifest columns only."""

    active = _active_ids(selection)
    selected = manifest
    for facet_id, facet in FACETS.items():
        if facet_id not in active:
            continue
        column, value = facet.condition
        if column not in manifest.columns:
            raise KeyError(f"manifest has no column {column!r} required by facet {facet_id!r}")
        selected = selected.loc[selected[column].eq(value)]
    return selected.copy().reset_index(drop=True)


@dataclass(frozen=True)
class ExplorerSelection:
    """Immutable set of active plain-language explorer facets."""

    active_facets: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        normalized = frozenset(self.active_facets)
        object.__setattr__(self, "active_facets", normalized)
        unknown = normalized - set(FACETS)
        if unknown:
            raise KeyError(f"unknown explorer facet(s): {', '.join(sorted(unknown))}")

    @classmethod
    def from_mapping(cls, facets: Mapping[str, bool]) -> ExplorerSelection:
        return cls(_active_ids(facets))

    @property
    def query(self) -> str:
        return selection_query(self)

    def filter(self, manifest: pd.DataFrame) -> pd.DataFrame:
        return filter_manifest(manifest, self)

    def count(self, manifest: pd.DataFrame) -> int:
        return len(self.filter(manifest))


def export_selection(
    manifest: pd.DataFrame,
    selection: ExplorerSelection | Mapping[str, bool],
    destination: Path | str,
) -> tuple[Path, ...]:
    """Export selected manifest metadata and its reproducible query."""

    output = Path(destination)
    selected = filter_manifest(manifest, selection)
    query = selection_query(selection)
    output.parent.mkdir(parents=True, exist_ok=True)

    if output.suffix.lower() == ".json":
        _write_selection_json(output, selected, query)
        return (output,)
    if output.suffix.lower() == ".csv":
        selected.to_csv(output, index=False)
        query_path = output.with_suffix(".query.txt")
        query_path.write_text(query + "\n", encoding="utf-8")
        return output, query_path
    raise ValueError("selection export destination must end in .csv or .json")


_FACET_MODALITIES: dict[str, tuple[str, ...]] = {
    "smartwatch": ("watch",),
    "watch_gravity": ("watch",),
    "watch_quaternion": ("watch",),
    "watch_50_hz": ("watch",),
    "watch_100_hz": ("watch",),
    "acceleration_user": ("watch",),
    "acceleration_total": ("watch",),
    "watch_raw_acceleration": ("watch_rawaccel",),
    "head_imu": ("headimu",),
    "head_gravity": ("headimu",),
    "head_quaternion": ("headimu",),
    "head_gyro": ("headimu",),
    "digital_pen": ("pen", "markers"),
    "observer_annotation": ("attention",),
}

_WINDOWS_RESERVED_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{number}" for number in range(1, 10)}
    | {f"LPT{number}" for number in range(1, 10)}
)


def _write_selection_json(
    destination: Path,
    selected: pd.DataFrame,
    query: str,
    *,
    modalities: tuple[str, ...] | None = None,
) -> None:
    rows = json.loads(selected.to_json(orient="records"))
    payload: dict[str, object] = {
        "query": query,
        "recording_ids": selected["recording_id"].tolist(),
        "rows": rows,
    }
    if modalities is not None:
        payload["modalities"] = list(modalities)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _selected_modalities(
    manifest: pd.DataFrame,
    selection: ExplorerSelection | Mapping[str, bool],
) -> tuple[str, ...]:
    active = _active_ids(selection)
    requested = {
        modality
        for facet_id in active
        for modality in _FACET_MODALITIES.get(facet_id, ())
    }
    if not requested:
        requested = {
            modality
            for modality in S.MODALITIES
            if f"has_{modality}" in manifest
            and manifest[f"has_{modality}"].fillna(False).astype(bool).any()
        }
    return tuple(modality for modality in S.MODALITIES if modality in requested)


def _copy_plan(
    root: Path, selected: pd.DataFrame, modalities: tuple[str, ...]
) -> list[tuple[Path, Path]]:
    plan = []
    for row in selected.itertuples(index=False):
        recording_id = _safe_recording_id(row.recording_id)
        for modality in modalities:
            flag = f"has_{modality}"
            if not bool(getattr(row, flag, False)):
                continue
            source = root / modality / f"{recording_id}.parquet"
            if not source.is_file():
                raise FileNotFoundError(
                    f"selected recording {recording_id!r} declares {flag}, but {source} is missing"
                )
            plan.append((source, Path(modality) / source.name))
    return plan


def _safe_recording_id(value: object) -> str:
    recording_id = str(value)
    posix_path = PurePosixPath(recording_id)
    windows_path = PureWindowsPath(recording_id)
    windows_stem = windows_path.name.split(".", 1)[0].upper()
    if (
        not recording_id
        or recording_id in {".", ".."}
        or posix_path.name != recording_id
        or windows_path.name != recording_id
        or windows_path.drive
        or windows_stem in _WINDOWS_RESERVED_NAMES
        or recording_id.endswith((".", " "))
        or any(character in '<>:"/\\|?*' or ord(character) < 32 for character in recording_id)
    ):
        raise ValueError(f"unsafe recording_id for selection export: {recording_id!r}")
    return recording_id


def _subset_manifest(
    selected: pd.DataFrame, modalities: tuple[str, ...]
) -> pd.DataFrame:
    subset = selected.copy()
    for flag, modality in S.CAPABILITY_FLAG_MODALITY.items():
        if flag in subset and modality not in modalities:
            subset[flag] = False
    return subset


def export_selected_recordings(
    root: Path | str,
    manifest: pd.DataFrame,
    selection: ExplorerSelection | Mapping[str, bool],
    destination: Path | str,
) -> Path:
    """Create a portable folder containing only the selected recordings."""

    root = Path(root)
    output = Path(destination)
    selected = filter_manifest(manifest, selection)
    if selected.empty:
        raise ValueError("selection matched no recordings; nothing was exported")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"export destination already exists: {output}")

    modalities = _selected_modalities(selected, selection)
    plan = _copy_plan(root, selected, modalities)
    channels_path = root / "channels.parquet"
    if not channels_path.is_file():
        raise FileNotFoundError(f"selection export requires {channels_path}")
    channels = pd.read_parquet(channels_path)
    if not {"recording_id", "modality"} <= set(channels.columns):
        raise ValueError("channels.parquet must contain recording_id and modality")
    selected_ids = set(selected["recording_id"])
    subset_channels = channels.loc[
        channels["recording_id"].isin(selected_ids)
        & channels["modality"].isin(modalities)
    ].copy()
    subset_manifest = _subset_manifest(selected, modalities)

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    claimed_output = False
    try:
        subset_manifest.to_parquet(staging / "sessions.parquet", index=False)
        subset_channels.to_parquet(staging / "channels.parquet", index=False)
        _write_selection_json(
            staging / "selection.json",
            subset_manifest,
            selection_query(selection),
            modalities=modalities,
        )
        for source, relative_destination in plan:
            target = staging / relative_destination
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        try:
            output.mkdir()
        except FileExistsError as exc:
            raise FileExistsError(f"export destination already exists: {output}") from exc
        claimed_output = True
        incomplete = output / ".incomplete"
        incomplete.touch()
        for child in staging.iterdir():
            child.replace(output / child.name)
        incomplete.unlink()
        staging.rmdir()
    except Exception:
        if claimed_output:
            shutil.rmtree(output, ignore_errors=True)
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output
