"""Shared test helpers."""
from pathlib import Path


def write_bundle_documents(root: Path) -> None:
    """The documents a published bundle always carries beside its tables."""
    (root / "LICENSE").write_text("license\n")
    (root / "data_dictionary.md").write_text("# Data dictionary\n")
