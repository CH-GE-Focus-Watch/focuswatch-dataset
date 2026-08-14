from pathlib import Path

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
