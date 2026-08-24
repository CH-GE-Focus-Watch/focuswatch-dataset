from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from focuswatch_dataset.build import build_dataset

from .test_build import sources


@pytest.fixture(scope="module")
def synthetic_bundle(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("select-and-load")
    out = root / "bundle"
    build_dataset(sources(root), out)
    return out


def run_example(bundle: Path, *args: str) -> subprocess.CompletedProcess[str]:
    repository = Path(__file__).resolve().parents[1]
    return subprocess.run(
        [sys.executable, str(repository / "examples" / "select_and_load.py"), str(bundle), *args],
        cwd=repository,
        text=True,
        capture_output=True,
        check=False,
    )


def test_documented_example_uses_safe_default_watch_semantics(synthetic_bundle):
    """The default watch pull must select one meaning of acceleration before loading."""
    result = run_example(synthetic_bundle, "--require-pen", "--modality", "watch")

    assert result.returncode == 0, result.stderr
    assert "ML4SCS-S096" in result.stdout
    assert "ETH-SL-E2_session6" in result.stdout
    assert "ETH-EGE-T6" not in result.stdout
    assert "loaded" in result.stdout and "watch sample row(s)" in result.stdout


def test_example_selects_only_recordings_with_the_requested_modality(synthetic_bundle):
    """An attention pull must not send watch-only recording IDs to the loader."""
    result = run_example(synthetic_bundle, "--modality", "attention")

    assert result.returncode == 0, result.stderr
    assert "AIRPODS-P1" in result.stdout
    assert "loaded 3 attention sample row(s)" in result.stdout
