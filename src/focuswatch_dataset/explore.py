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
from .convert import DECIMATION_RATIO_TOLERANCE, decimation_factor, downsample
from .write import read_table, write_table


@dataclass(frozen=True)
class Facet:
    """A user-facing filter backed by one published manifest condition."""

    label: str
    group: str
    condition: tuple[str, object]


# Registry order is the canonical order for both the UI and generated queries.
# Facets on the same manifest column are alternatives (OR), facets on different
# columns narrow each other (AND): "ML4SCS" plus "ZHAW" means either cohort.
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
    "watch_gyro": Facet("Watch gyroscope", "Signals", ("has_watch_gyro", True)),
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
    "watch_series_6": Facet(
        "Apple Watch Series 6", "Acquisition", ("watch_model", S.WATCH_MODEL_SERIES_6)
    ),
    "watch_se_2": Facet(
        "Apple Watch SE (2nd gen.)", "Acquisition", ("watch_model", S.WATCH_MODEL_SE_2)
    ),
    "acceleration_user": Facet(
        "Acceleration without gravity", "Acquisition", ("accel_semantics", "user")
    ),
    "acceleration_total": Facet(
        "Acceleration including gravity", "Acquisition", ("accel_semantics", "total")
    ),
    "cohort_ml4scs": Facet("ML4SCS", "Cohort", ("cohort", "ML4SCS")),
    "cohort_zhaw": Facet("ZHAW", "Cohort", ("cohort", "ZHAW")),
    "cohort_airpods": Facet("AirPods", "Cohort", ("cohort", "AIRPODS")),
    "cohort_moleskine": Facet("Moleskine", "Cohort", ("cohort", "MOLESKINE")),
    # Why: only a shared clock lets pen events label motion samples by timestamp
    # alone; estimated_delta and overlap_only recordings need an offset first.
    "shared_clock": Facet(
        "Shared clock across modalities", "Timing", ("time_alignment", "shared_clock")
    ),
}


@dataclass(frozen=True)
class Preset:
    """A research goal that selects the facets its ground truth needs."""

    label: str
    description: str
    facets: frozenset[str]


# Ground truth decides the pairing: pen contact labels the watch, observer
# intervals label the head IMU (see _FACET_MODALITIES below).
PRESETS: dict[str, Preset] = {
    "writing_detection": Preset(
        "Writing detection", "Smartwatch motion labelled by pen contact",
        frozenset({"smartwatch", "digital_pen"}),
    ),
    "attention": Preset(
        "Attention", "Head IMU motion labelled by observer annotation",
        frozenset({"head_imu", "observer_annotation"}),
    ),
    "everything": Preset("Everything", "All recordings, no filter", frozenset()),
}


def watch_rate_summary(selected: pd.DataFrame, target_hz: float) -> tuple[dict[int, int], int]:
    """Recordings per rounded measured watch rate, and how many exceed `target_hz`."""

    if "watch_hz_measured" not in selected:
        return {}, 0
    rates = selected["watch_hz_measured"].dropna()
    counts = {int(rate): int(n) for rate, n in rates.round().value_counts().sort_index().items()}
    # Why: the export's tolerance, not exact equality, decides whether a
    # 99.8 Hz watch counts as above a 50 Hz target.
    above = int((rates > target_hz * (1 + DECIMATION_RATIO_TOLERANCE)).sum())
    return counts, above


def _active_ids(selection: ExplorerSelection | Mapping[str, bool]) -> frozenset[str]:
    if isinstance(selection, ExplorerSelection):
        return selection.active_facets
    unknown = set(selection) - set(FACETS)
    if unknown:
        raise KeyError(f"unknown explorer facet(s): {', '.join(sorted(unknown))}")
    return frozenset(facet_id for facet_id, active in selection.items() if active)


def _active_conditions(
    selection: ExplorerSelection | Mapping[str, bool],
) -> dict[str, tuple[object, ...]]:
    """Active facet values grouped by manifest column, in registry order."""

    active = _active_ids(selection)
    grouped: dict[str, list[object]] = {}
    for facet_id, facet in FACETS.items():
        if facet_id in active:
            column, value = facet.condition
            grouped.setdefault(column, []).append(value)
    return {column: tuple(values) for column, values in grouped.items()}


def selection_query(selection: ExplorerSelection | Mapping[str, bool]) -> str:
    """Return a stable pandas-compatible expression for active facets."""

    expressions: list[str] = []
    for column, values in _active_conditions(selection).items():
        if len(values) > 1:
            expressions.append(f"{column} in {list(values)!r}")
        elif values[0] is True:
            expressions.append(column)
        elif values[0] is False:
            expressions.append(f"not {column}")
        else:
            expressions.append(f"{column} == {values[0]!r}")
    # A scalar ``True`` is not a valid DataFrame.query row mask.
    return " and ".join(expressions) or "index == index"


def filter_manifest(
    manifest: pd.DataFrame,
    selection: ExplorerSelection | Mapping[str, bool],
) -> pd.DataFrame:
    """Intersect active facets using manifest columns only."""

    selected = manifest
    for column, values in _active_conditions(selection).items():
        if column not in manifest.columns:
            facet_ids = [facet_id for facet_id in sorted(_active_ids(selection))
                         if FACETS[facet_id].condition[0] == column]
            raise KeyError(
                f"manifest has no column {column!r} required by facet(s) {facet_ids}"
            )
        selected = selected.loc[selected[column].isin(values)]
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


# Why: a label stream is exported with the motion stream it labels. Pen
# contact is the writing ground truth for the watch, observer intervals the
# attention ground truth for the head IMU; either alone holds labels without
# the sensor data they describe.
_FACET_MODALITIES: dict[str, tuple[str, ...]] = {
    "smartwatch": ("watch",),
    "watch_gravity": ("watch",),
    "watch_quaternion": ("watch",),
    "watch_gyro": ("watch",),
    "watch_50_hz": ("watch",),
    "watch_100_hz": ("watch",),
    "watch_series_6": ("watch",),
    "watch_se_2": ("watch",),
    "acceleration_user": ("watch",),
    "acceleration_total": ("watch",),
    "watch_raw_acceleration": ("watch_rawaccel",),
    "head_imu": ("headimu",),
    "head_gravity": ("headimu",),
    "head_quaternion": ("headimu",),
    "head_gyro": ("headimu",),
    "digital_pen": ("pen", "markers", "watch"),
    "observer_annotation": ("attention", "headimu"),
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
    watch_resampled_to_hz: float | None = None,
) -> None:
    rows = json.loads(selected.to_json(orient="records"))
    payload: dict[str, object] = {
        "query": query,
        "recording_ids": selected["recording_id"].tolist(),
        "rows": rows,
    }
    if modalities is not None:
        payload["modalities"] = list(modalities)
    if watch_resampled_to_hz is not None:
        payload["watch_resampled_to_hz"] = watch_resampled_to_hz
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
    # Why: the published manifest never names a model for a recording without
    # a watch table, so a subset that drops the watch must not either.
    if "watch_model" in subset and "watch" not in modalities:
        subset["watch_model"] = ""
    return subset


def export_preview(
    root: Path | str,
    manifest: pd.DataFrame,
    selection: ExplorerSelection | Mapping[str, bool],
) -> dict[str, tuple[int, int]]:
    """Files and bytes per modality that `export_selected_recordings` would copy."""

    selected = filter_manifest(manifest, selection)
    preview: dict[str, tuple[int, int]] = {}
    for source, relative in _copy_plan(Path(root), selected, _selected_modalities(selected, selection)):
        modality = relative.parts[0]
        files, size = preview.get(modality, (0, 0))
        preview[modality] = (files + 1, size + source.stat().st_size)
    return preview


def _watch_factors(
    channels: pd.DataFrame, recording_ids: set[str], target_hz: float
) -> dict[str, int]:
    """Decimation factor per selected watch recording, checked before any write."""

    if "sample_rate_hz" not in channels.columns:
        raise ValueError("resampling the watch needs sample_rate_hz in channels.parquet")
    watch = channels.loc[(channels["modality"] == "watch")
                         & channels["recording_id"].isin(recording_ids)]
    rates = watch.groupby("recording_id")["sample_rate_hz"].first()
    undeclared = sorted(recording_ids - set(rates.dropna().index))
    if undeclared:
        raise ValueError(f"no watch sample rate declared for {undeclared}")
    return {rid: decimation_factor(rate, target_hz) for rid, rate in rates.items()}


def export_selected_recordings(
    root: Path | str,
    manifest: pd.DataFrame,
    selection: ExplorerSelection | Mapping[str, bool],
    destination: Path | str,
    *,
    watch_target_hz: float | None = None,
) -> Path:
    """Create a portable folder containing only the selected recordings.

    With `watch_target_hz`, watch tables above that rate are written
    downsampled (see `convert.downsample`) instead of copied, and the
    subset's manifest, channels and `selection.json` say so. Every other
    table is a byte copy of the published file.
    """

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
    factors: dict[str, int] = {}
    if watch_target_hz is not None and "watch" in modalities:
        watch_ids = {Path(rel).stem for _, rel in plan if Path(rel).parts[0] == "watch"}
        factors = {rid: f for rid, f in _watch_factors(channels, watch_ids, watch_target_hz).items()
                   if f > 1}
        resampled = subset_channels["recording_id"].isin(factors) & (
            subset_channels["modality"] == "watch")
        subset_channels.loc[resampled, "sample_rate_hz"] = float(watch_target_hz)
        rows = subset_manifest["recording_id"].isin(factors)
        for column in ("watch_hz_nominal", "watch_hz_measured"):
            if column in subset_manifest:
                subset_manifest.loc[rows, column] = float(watch_target_hz)
        if "n_samples_watch" in subset_manifest:
            # Every factor-th row survives decimation: ceil(n / factor).
            n = subset_manifest.loc[rows, "n_samples_watch"]
            divisor = subset_manifest.loc[rows, "recording_id"].map(factors)
            subset_manifest.loc[rows, "n_samples_watch"] = -(-n // divisor)

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
            watch_resampled_to_hz=float(watch_target_hz) if factors else None,
        )
        for source, relative_destination in plan:
            target = staging / relative_destination
            target.parent.mkdir(parents=True, exist_ok=True)
            recording_id = source.stem
            if relative_destination.parts[0] == "watch" and recording_id in factors:
                rate = watch_target_hz * factors[recording_id]
                table = downsample(read_table(source), rate, watch_target_hz)
                write_table(table, target, recording_id)
            else:
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
