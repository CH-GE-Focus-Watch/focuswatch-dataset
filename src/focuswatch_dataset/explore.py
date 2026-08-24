"""Pure manifest-selection model for the optional terminal explorer."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping

import pandas as pd


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
    "study_recording": Facet(
        "Study recording", "Study", ("study_mode", "study")
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
        # pandas normalizes NumPy scalars and missing values to JSON safely.
        rows = json.loads(selected.to_json(orient="records"))
        output.write_text(
            json.dumps({"query": query, "rows": rows}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return (output,)
    if output.suffix.lower() == ".csv":
        selected.to_csv(output, index=False)
        query_path = output.with_suffix(".query.txt")
        query_path.write_text(query + "\n", encoding="utf-8")
        return output, query_path
    raise ValueError("selection export destination must end in .csv or .json")
