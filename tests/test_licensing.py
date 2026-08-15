"""I12: DESIGN §10 commits to Apache-2.0 for the code and CC BY 4.0 for the
data, plus a CITATION.cff - none of the three existed before this fix.
"""
from __future__ import annotations

import tomllib
from pathlib import Path

from .test_build import sources
from focuswatch_dataset.build import build_dataset

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_repo_has_an_apache_2_license_file():
    text = (REPO_ROOT / "LICENSE").read_text()
    assert "Apache License" in text
    assert "Version 2.0" in text


def test_repo_has_a_citation_file_with_a_clearly_marked_doi_placeholder():
    text = (REPO_ROOT / "CITATION.cff").read_text()
    assert "cff-version" in text
    assert "type: software" in text
    # A real Zenodo DOI cannot be filled in yet - it must not be invented,
    # but the field must exist so the article's citation snippet has
    # somewhere obvious to update it.
    assert "PLACEHOLDER" in text


def test_pyproject_declares_the_code_license():
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    assert data["project"]["license"] == "Apache-2.0"


def test_bundle_carries_a_cc_by_license_file_backing_datapackage_json(tmp_path):
    """datapackage.json's `licenses` field claims CC-BY-4.0; a LICENSE file
    with that claim's actual text must sit beside it in every built bundle,
    not just be asserted in JSON metadata nobody can read as a legal text.
    """
    out = tmp_path / "out"
    build_dataset(sources(tmp_path), out)
    license_text = (out / "LICENSE").read_text()
    assert "CC BY 4.0" in license_text
    assert "creativecommons.org/licenses/by/4.0" in license_text
