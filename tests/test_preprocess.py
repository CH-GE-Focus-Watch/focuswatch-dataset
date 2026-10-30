import json

import numpy as np
import pandas as pd
import pytest

from focuswatch_dataset.build import build_dataset
from focuswatch_dataset.load import load_manifest, load_recording
from focuswatch_dataset.preprocess import (
    PreprocessConfig,
    attention_labels,
    close_gaps,
    contact_labels,
    eligible_recordings,
    preprocess_recording,
    preprocess_recordings,
    write_preprocessed,
)

from .test_build import sources

MS = 1_000_000


def pen_events(*events):
    return pd.DataFrame(events, columns=["t_ns", "dot_type"])


# --- contact_labels --------------------------------------------------------------

def test_contact_needs_a_writing_event_within_the_tolerance():
    t = np.array([0, 30, 100]) * MS
    pen = pen_events((10 * MS, "PEN_MOVE"))
    assert contact_labels(t, pen, tolerance_s=0.04).tolist() == [1, 1, 0]


def test_the_nearest_event_decides_even_if_it_is_a_pen_up():
    t = np.array([20]) * MS
    pen = pen_events((0, "PEN_MOVE"), (25 * MS, "PEN_UP"))
    assert contact_labels(t, pen, tolerance_s=0.04).tolist() == [0]


def test_pen_shift_moves_the_pen_timeline():
    t = np.array([0, 500]) * MS
    pen = pen_events((300 * MS, "PEN_DOWN"))
    assert contact_labels(t, pen, tolerance_s=0.04, pen_shift_s=0.2).tolist() == [0, 1]


def test_contact_labels_match_merge_asof_nearest():
    """The rule must equal the ML4SCS merge: merge_asof(nearest, tolerance 40 ms)."""
    rng = np.random.default_rng(7)
    t = np.sort(rng.integers(0, 60_000, 3000)) * MS
    pen_t = np.sort(rng.integers(0, 60_000, 800)) * MS
    kinds = rng.choice(["PEN_DOWN", "PEN_MOVE", "PEN_UP", "PEN_HOVER"], len(pen_t))
    pen = pd.DataFrame({"t_ns": pen_t, "dot_type": kinds})

    reference = pd.merge_asof(
        pd.DataFrame({"t_ns": t}),
        pen.assign(w=pen["dot_type"].isin(["PEN_DOWN", "PEN_MOVE"]).astype(int)),
        on="t_ns", direction="nearest", tolerance=40 * MS,
    )["w"].fillna(0).astype(int)

    assert contact_labels(t, pen, tolerance_s=0.04).tolist() == reference.tolist()


# --- close_gaps --------------------------------------------------------------------

def _labels_with_gap(gap_s):
    t = np.arange(0, 20_000, 20) * MS  # 50 Hz, 20 s
    label = ((t < 5_000 * MS) | (t >= (5_000 + gap_s * 1000) * MS)).astype(np.int8)
    label[t >= 15_000 * MS] = 0  # trailing idle run
    return label, t


def test_pauses_up_to_the_gap_length_count_as_writing():
    label, t = _labels_with_gap(2.9)
    closed = close_gaps(label, t, gap_closing_s=3.0)
    assert closed[(t >= 5_000 * MS) & (t < 7_900 * MS)].all()


def test_longer_pauses_stay_idle():
    label, t = _labels_with_gap(3.2)
    closed = close_gaps(label, t, gap_closing_s=3.0)
    assert not closed[(t >= 5_000 * MS) & (t < 8_200 * MS)].any()


def test_leading_and_trailing_idle_runs_are_untouched():
    t = np.arange(0, 10) * 100 * MS
    label = np.array([0, 0, 1, 1, 0, 1, 1, 0, 0, 0], dtype=np.int8)
    closed = close_gaps(label, t, gap_closing_s=3.0)
    assert closed.tolist() == [0, 0, 1, 1, 1, 1, 1, 0, 0, 0]


# --- attention_labels ------------------------------------------------------------

def test_attention_labels_come_from_the_containing_interval():
    t = np.array([0, 10, 20, 30])
    intervals = pd.DataFrame({"t_start_ns": [0, 20], "t_end_ns": [20, 30],
                              "label": ["focused", "distracted"]})
    got = attention_labels(t, intervals)
    assert got.iloc[:3].tolist() == ["focused", "focused", "distracted"]
    assert got.isna().iloc[3]


# --- config ------------------------------------------------------------------------

def test_writing_needs_an_explicit_gap_length():
    with pytest.raises(ValueError, match="explicit gap_closing_s"):
        PreprocessConfig(task="writing")


def test_unknown_task_is_refused():
    with pytest.raises(ValueError, match="unknown task"):
        PreprocessConfig(task="sleep")


# --- on a built bundle -------------------------------------------------------------

@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    root = tmp_path_factory.mktemp("preprocess")
    out = root / "bundle"
    build_dataset(sources(root), out)
    return out


WRITING = PreprocessConfig(task="writing", gap_closing_s=3.0)


def test_writing_labels_the_watch_timeline(bundle):
    table = preprocess_recording(bundle, "ML4SCS-S096", WRITING)
    watch = load_recording(bundle, "ML4SCS-S096", "watch")
    assert len(table) == len(watch)
    assert set(table["label_writing"].unique()) <= {0, 1}
    assert table["label_writing"].any()


def test_overlap_only_recordings_need_an_explicit_opt_in(bundle):
    moleskine = load_manifest(bundle).query("cohort == 'MOLESKINE'")["recording_id"].iloc[0]
    with pytest.raises(ValueError, match="independent clocks"):
        preprocess_recording(bundle, moleskine, WRITING)
    opted_in = PreprocessConfig(task="writing", gap_closing_s=3.0, include_overlap_only=True)
    assert "label_writing" in preprocess_recording(bundle, moleskine, opted_in)


def test_eligible_recordings_follow_the_task_and_the_clock_rule(bundle):
    manifest = load_manifest(bundle)
    writing = eligible_recordings(manifest, WRITING)
    assert "ML4SCS-S096" in writing
    assert not any(rid.startswith("MOLESKINE") for rid in writing)
    assert not any(rid.startswith("AIRPODS") for rid in writing)
    attention = eligible_recordings(manifest, PreprocessConfig(task="attention"))
    assert attention and all(rid.startswith("AIRPODS") for rid in attention)


def test_attention_labels_the_head_imu_timeline(bundle):
    table = preprocess_recording(bundle, "AIRPODS-P1", PreprocessConfig(task="attention"))
    assert set(table["label_attention"].dropna()) <= {"focused", "distracted"}
    assert table["label_attention"].notna().any()
    assert "label_writing" not in table


def test_a_recording_without_the_task_streams_is_refused(bundle):
    with pytest.raises(ValueError, match="the writing task needs watch, but has_watch is false"):
        preprocess_recording(bundle, "AIRPODS-P1", WRITING)
    with pytest.raises(ValueError, match="the attention task needs headimu"):
        preprocess_recording(bundle, "ML4SCS-S096", PreprocessConfig(task="attention"))


def test_target_hz_labels_the_downsampled_timeline(bundle):
    native = preprocess_recording(bundle, "ML4SCS-S096", WRITING)
    at_50 = preprocess_recording(
        bundle, "ML4SCS-S096", PreprocessConfig(task="writing", gap_closing_s=3.0, target_hz=50.0))
    assert len(at_50) == -(-len(native) // 2)


def test_preprocess_recordings_concatenates_with_ids(bundle):
    out = preprocess_recordings(bundle, ["ML4SCS-S096", "ZHAW-SL-E2_session6"], WRITING)
    assert set(out["recording_id"]) == {"ML4SCS-S096", "ZHAW-SL-E2_session6"}


def test_write_preprocessed_records_every_choice(bundle, tmp_path):
    out = write_preprocessed(bundle, ["ML4SCS-S096"], WRITING, tmp_path / "labelled")

    table = pd.read_parquet(out / "writing" / "ML4SCS-S096.parquet")
    assert "label_writing" in table
    record = json.loads((out / "preprocess.json").read_text())
    assert record["config"]["gap_closing_s"] == 3.0
    assert record["config"]["pen_shift_s"] == 0.0
    assert record["recordings"][0]["recording_id"] == "ML4SCS-S096"
    assert 0 < record["recordings"][0]["labelled_share"] < 1


def test_write_preprocessed_refuses_existing_or_in_bundle_destinations(bundle, tmp_path):
    existing = tmp_path / "exists"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        write_preprocessed(bundle, ["ML4SCS-S096"], WRITING, existing)
    with pytest.raises(ValueError, match="inside the bundle"):
        write_preprocessed(bundle, ["ML4SCS-S096"], WRITING, bundle / "derived")


def test_write_preprocessed_leaves_nothing_behind_on_failure(bundle, tmp_path):
    destination = tmp_path / "failed"
    with pytest.raises(ValueError, match="the writing task needs watch, but has_watch is false"):
        write_preprocessed(bundle, ["ML4SCS-S096", "AIRPODS-P1"], WRITING, destination)
    assert not destination.exists()
    assert not list(tmp_path.glob(".failed-*"))


# --- fw preprocess -----------------------------------------------------------------

def test_cli_writes_labels_with_progress(bundle, tmp_path, capsys):
    from focuswatch_dataset import cli

    out = tmp_path / "cli-labels"
    code = cli.main(["preprocess", "--dataset", str(bundle), "--task", "writing",
                     "--gap-closing-s", "3", "--cohort", "ML4SCS", "--out", str(out)])

    printed = capsys.readouterr().out
    assert code == 0, printed
    assert "[1/1] ML4SCS-S096" in printed
    assert (out / "writing" / "ML4SCS-S096.parquet").is_file()


def test_cli_requires_the_gap_length_for_writing(bundle, tmp_path, capsys):
    from focuswatch_dataset import cli

    code = cli.main(["preprocess", "--dataset", str(bundle), "--task", "writing",
                     "--out", str(tmp_path / "x")])

    assert code == 1
    assert "explicit gap_closing_s" in capsys.readouterr().out


def test_cli_names_recordings_that_do_not_fit_the_task(bundle, tmp_path, capsys):
    from focuswatch_dataset import cli

    code = cli.main(["preprocess", "--dataset", str(bundle), "--task", "attention",
                     "--recording", "ML4SCS-S096", "--out", str(tmp_path / "x")])

    assert code == 1
    assert "not eligible for the attention task: ['ML4SCS-S096']" in capsys.readouterr().out
