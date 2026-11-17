"""SHA-256 checksum list shipped inside a bundle.

The list uses the `sha256sum` text format, so `sha256sum -c SHA256SUMS` or
`shasum -a 256 -c SHA256SUMS` verifies a downloaded copy without this package.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

CHECKSUM_FILE = "SHA256SUMS"
_CHUNK_BYTES = 1 << 20


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _bundle_files(root: Path) -> list[str]:
    return sorted(
        p.relative_to(root).as_posix() for p in root.rglob("*")
        if p.is_file() and p.relative_to(root).as_posix() != CHECKSUM_FILE
    )


def write_checksums(root: Path) -> None:
    """List every file of `root` except the list itself. Must run after the last write."""
    lines = [f"{_sha256(root / rel)}  {rel}" for rel in _bundle_files(root)]
    (root / CHECKSUM_FILE).write_text("\n".join(lines) + "\n")


def checksum_problems(root: Path) -> list[str]:
    """Every file that is missing, unlisted, or differs from the shipped list."""
    path = root / CHECKSUM_FILE
    if not path.is_file():
        return [f"{CHECKSUM_FILE} is missing"]
    listed: dict[str, str] = {}
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        digest, sep, rel = line.partition("  ")
        if not sep or len(digest) != 64:
            return [f"{CHECKSUM_FILE} line {number} is not '<sha256>  <path>'"]
        listed[rel] = digest
    present = set(_bundle_files(root))
    problems = [f"{rel} is listed but missing" for rel in sorted(set(listed) - present)]
    problems += [f"{rel} is not listed" for rel in sorted(present - set(listed))]
    problems += [f"{rel} does not match its checksum" for rel in sorted(present & set(listed))
                 if _sha256(root / rel) != listed[rel]]
    return problems
