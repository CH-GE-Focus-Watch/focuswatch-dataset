"""Label derivation on top of a published bundle.

The bundle publishes ground truth as recorded: pen contact events and
observer intervals. Turning them into per-sample training labels is a
labelling policy, not an observation, so it lives here, with every choice an
explicit parameter, and never writes back into the bundle.

Two tasks, two label columns that are never merged (writing and attention are
different axes):

- ``writing``: ``label_writing`` on the watch timeline, from pen contact.
- ``attention``: ``label_attention`` on the head-IMU timeline, from observer
  intervals.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd

from . import schema as S
from .convert import downsample
from .load import load_manifest, load_recording
from .write import write_table

WRITING_DOT_TYPES = ("PEN_DOWN", "PEN_MOVE")

# Why: the ML4SCS merge tolerance - about two watch periods at 50 Hz, so
# sample jitter is absorbed while real pen-free phases stay 0.
DEFAULT_CONTACT_TOLERANCE_S = 0.04

TASKS = {
    "writing": ("watch", "pen", "label_writing"),
    "attention": ("headimu", "attention", "label_attention"),
}


def contact_labels(t_ns: np.ndarray, pen: pd.DataFrame, *, tolerance_s: float,
                   pen_shift_s: float = 0.0) -> np.ndarray:
    """1 where the pen event nearest to a sample is a writing contact.

    Same rule as the ML4SCS merge: the nearest pen event within
    `tolerance_s` decides; no event in range means not writing. `pen_shift_s`
    is added to the pen timestamps before the join.
    """
    events = pen.sort_values("t_ns", kind="stable")
    pen_t = events["t_ns"].to_numpy(dtype=np.int64) + int(round(pen_shift_s * 1e9))
    writing = events["dot_type"].isin(WRITING_DOT_TYPES).to_numpy()
    out = np.zeros(len(t_ns), dtype=np.int8)
    if len(pen_t) == 0:
        return out
    after = np.searchsorted(pen_t, t_ns)
    before = np.clip(after - 1, 0, len(pen_t) - 1)
    after = np.clip(after, 0, len(pen_t) - 1)
    # A tie goes to the earlier event, as pandas merge_asof(direction="nearest") does.
    nearest = np.where(t_ns - pen_t[before] <= pen_t[after] - t_ns, before, after)
    in_range = np.abs(pen_t[nearest] - t_ns) <= tolerance_s * 1e9
    out[in_range & writing[nearest]] = 1
    return out


def close_gaps(label: np.ndarray, t_ns: np.ndarray, *, gap_closing_s: float) -> np.ndarray:
    """Relabel non-writing runs of at most `gap_closing_s` between writing as writing.

    A run's length is the time from its first to its last sample, as in the
    ML4SCS `smooth_labels`. Runs at the start or end of the recording are left
    alone: with context on one side only they may be a real idle phase.
    """
    out = label.astype(np.int8).copy()
    if len(out) < 3 or gap_closing_s <= 0:
        return out
    change = np.flatnonzero(np.diff(out)) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [len(out)]))
    for start, end in zip(starts[1:-1], ends[1:-1]):
        if out[start] == 0 and t_ns[end - 1] - t_ns[start] <= gap_closing_s * 1e9:
            out[start:end] = 1
    return out


def attention_labels(t_ns: np.ndarray, intervals: pd.DataFrame) -> pd.Series:
    """Observer label of the interval containing each sample, else missing."""
    labels = np.full(len(t_ns), None, dtype=object)
    for row in intervals.itertuples(index=False):
        labels[(t_ns >= row.t_start_ns) & (t_ns < row.t_end_ns)] = row.label
    return pd.Series(labels, dtype="string")


@dataclass(frozen=True)
class PreprocessConfig:
    """Every labelling choice, recorded next to the output it produced."""

    task: str
    gap_closing_s: float | None = None
    contact_tolerance_s: float = DEFAULT_CONTACT_TOLERANCE_S
    pen_shift_s: float = 0.0
    target_hz: float | None = None
    include_overlap_only: bool = False

    def __post_init__(self) -> None:
        if self.task not in TASKS:
            raise ValueError(f"unknown task {self.task!r}; expected one of {tuple(TASKS)}")
        if self.task == "writing" and self.gap_closing_s is None:
            # Why: the gap length defines what "writing" means; a silent
            # default would let two papers publish different labels under one name.
            raise ValueError("the writing task needs an explicit gap_closing_s (e.g. 3.0)")


def _check_alignment(row: pd.Series, config: PreprocessConfig) -> None:
    if config.task != "writing":
        return
    alignment = row.get("time_alignment")
    if alignment == S.TIME_ALIGNMENT_OVERLAP_ONLY and not config.include_overlap_only:
        raise ValueError(
            f"{row['recording_id']}: watch and pen run on independent clocks "
            "(time_alignment = overlap_only), so contact labels are approximate; "
            "pass include_overlap_only=True to label it anyway"
        )


def preprocess_recording(root: Path | str, recording_id: str,
                         config: PreprocessConfig) -> pd.DataFrame:
    """One recording's motion table with its task label column added."""
    root = Path(root)
    motion_modality, truth_modality, label_column = TASKS[config.task]
    manifest = load_manifest(root).set_index("recording_id", drop=False)
    if recording_id not in manifest.index:
        raise KeyError(f"recording id {recording_id!r} not found in {root / 'sessions.parquet'}")
    row = manifest.loc[recording_id]
    for modality in (motion_modality, truth_modality):
        if not bool(row.get(f"has_{modality}", False)):
            raise ValueError(f"{recording_id}: the {config.task} task needs {modality}, "
                             f"but has_{modality} is false")
    _check_alignment(row, config)

    motion = load_recording(root, recording_id, motion_modality)
    # Why: stable sort keeps arrival order on tied timestamps, as the ML4SCS merge does.
    motion = motion.sort_values(S.TIME_COLUMN, kind="stable").reset_index(drop=True)
    if config.target_hz is not None:
        rate_column = "watch_hz_measured" if motion_modality == "watch" else "head_hz_measured"
        motion = downsample(motion, float(row[rate_column]), config.target_hz)
    t_ns = motion[S.TIME_COLUMN].to_numpy(dtype=np.int64)
    truth = load_recording(root, recording_id, truth_modality)

    if config.task == "writing":
        raw = contact_labels(t_ns, truth, tolerance_s=config.contact_tolerance_s,
                             pen_shift_s=config.pen_shift_s)
        motion[label_column] = close_gaps(raw, t_ns, gap_closing_s=config.gap_closing_s)
    else:
        motion[label_column] = attention_labels(t_ns, truth).to_numpy()
    motion.insert(0, "recording_id", recording_id)
    return motion


def preprocess_recordings(root: Path | str, recording_ids: list[str],
                          config: PreprocessConfig) -> pd.DataFrame:
    """`preprocess_recording` for several ids, concatenated."""
    if not recording_ids:
        raise ValueError("recording_ids is empty; nothing to preprocess")
    frames = [preprocess_recording(root, rid, config) for rid in recording_ids]
    return pd.concat(frames, ignore_index=True)


def eligible_recordings(manifest: pd.DataFrame, config: PreprocessConfig) -> list[str]:
    """Recording ids that carry both streams the task needs and pass the clock rule."""
    motion_modality, truth_modality, _ = TASKS[config.task]
    keep = manifest[f"has_{motion_modality}"].astype(bool) & manifest[f"has_{truth_modality}"].astype(bool)
    if config.task == "writing" and not config.include_overlap_only and "time_alignment" in manifest:
        keep &= manifest["time_alignment"] != S.TIME_ALIGNMENT_OVERLAP_ONLY
    return manifest.loc[keep, "recording_id"].tolist()


def write_preprocessed(root: Path | str, recording_ids: list[str], config: PreprocessConfig,
                       destination: Path | str, *,
                       on_progress: Callable[[int, int, str], None] | None = None) -> Path:
    """Write one labelled table per recording plus `preprocess.json`.

    The output is a new directory next to, never inside, the bundle; an
    existing destination is refused, and a failure leaves nothing behind.
    """
    root, output = Path(root), Path(destination)
    if output.exists():
        raise FileExistsError(f"destination already exists: {output}")
    if output.resolve().is_relative_to(root.resolve()):
        raise ValueError("preprocessed output must not be written inside the bundle")
    if not recording_ids:
        raise ValueError("recording_ids is empty; nothing to preprocess")
    manifest = load_manifest(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        rows = []
        for n, rid in enumerate(recording_ids, 1):
            if on_progress is not None:
                on_progress(n, len(recording_ids), rid)
            table = preprocess_recording(root, rid, config)
            write_table(table.drop(columns="recording_id"),
                        staging / config.task / f"{rid}.parquet", rid)
            label = table[TASKS[config.task][2]]
            share = (label == 1) if config.task == "writing" else label.notna()
            rows.append({"recording_id": rid, "n_samples": len(table),
                         "labelled_share": round(float(share.mean()), 4)})
        sha = manifest["build_git_sha"].iloc[0] if "build_git_sha" in manifest else None
        record = {
            "config": asdict(config),
            "source_bundle": str(root.resolve()),
            "source_build_git_sha": sha,
            "recordings": rows,
        }
        (staging / "preprocess.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        staging.rename(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output
