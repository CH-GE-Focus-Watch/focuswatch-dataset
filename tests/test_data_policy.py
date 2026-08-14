from pathlib import Path

import pytest

from scripts.check_no_data import check_paths


def test_rejects_large_file(tmp_path: Path):
    big = tmp_path / "watch.parquet"
    big.write_bytes(b"\0" * (1_048_576 + 1))
    assert check_paths([big]) != []


def test_rejects_known_data_extensions_regardless_of_size(tmp_path: Path):
    for name in ("S008_watch.csv", "P1_airpod_motion_labeled.csv", "sweep_data.zip"):
        f = tmp_path / name
        f.write_text("x")
        assert check_paths([f]) != [], name


def test_allows_small_source_and_fixtures(tmp_path: Path):
    ok = tmp_path / "adapters" / "ege.py"
    ok.parent.mkdir()
    ok.write_text("def load(): ...")
    assert check_paths([ok]) == []


@pytest.mark.parametrize("ext", [".csv", ".parquet", ".zip", ".jsonl"])
def test_blocked_extensions_are_blocked_at_any_size_and_depth(tmp_path: Path, ext: str):
    nested = tmp_path / "a" / "b" / "c"
    nested.mkdir(parents=True)
    f = nested / f"tiny{ext}"
    f.write_bytes(b"x")
    assert check_paths([f]) != []


def test_allows_ordinary_txt_files(tmp_path: Path):
    reqs = tmp_path / "requirements.txt"
    reqs.write_text("pandas\n")
    notes_dir = tmp_path / "docs"
    notes_dir.mkdir()
    notes = notes_dir / "notes.txt"
    notes.write_text("notes")
    assert check_paths([reqs]) == []
    assert check_paths([notes]) == []


def test_rejects_ground_truth_txt(tmp_path: Path):
    f = tmp_path / "P1_ground_truth_2026-04-28_13-40-36.txt"
    f.write_text("x")
    assert check_paths([f]) != []


def test_rejects_large_markdown_via_size_rule(tmp_path: Path):
    big = tmp_path / "notes.md"
    big.write_bytes(b"\0" * (1_048_576 + 1))
    assert check_paths([big]) != []


def test_violation_message_names_the_rule_that_fired(tmp_path: Path):
    csv_file = tmp_path / "capture.csv"
    csv_file.write_text("x")
    txt_file = tmp_path / "P1_protokoll_x.txt"
    txt_file.write_text("x")
    md_file = tmp_path / "big.md"
    md_file.write_bytes(b"\0" * (1_048_576 + 1))

    [csv_violation] = check_paths([csv_file])
    [txt_violation] = check_paths([txt_file])
    [md_violation] = check_paths([md_file])

    assert "type rule" in csv_violation
    assert "name rule" in txt_violation
    assert "size rule" in md_violation
