"""Pre-commit guard: keeps participant recordings out of the git tree."""
from __future__ import annotations

import re
import sys
from pathlib import Path

MAX_BYTES = 1_048_576

# Why: this package never legitimately tracks these formats — fixtures live
# only under pytest's tmp_path, never committed. A type rule can't lag the
# corpus the way a denylist of known source filenames would.
BLOCKED_EXTENSIONS = {".csv", ".parquet", ".zip", ".jsonl"}

# Why: .txt is otherwise legitimate (requirements.txt, docs); only these two
# AirPods ground-truth/protocol filename shapes are capture data.
TXT_DATA_PATTERNS = (
    re.compile(r"_ground_truth_"),
    re.compile(r"_protokoll_"),
)


def check_paths(paths: list[Path], max_bytes: int = MAX_BYTES) -> list[str]:
    violations = []
    for p in paths:
        if not p.is_file():
            continue
        suffix = p.suffix.lower()
        if suffix in BLOCKED_EXTENSIONS:
            violations.append(f"{p}: extension {suffix!r} is always blocked (type rule)")
        elif suffix == ".txt" and any(pat.search(p.name) for pat in TXT_DATA_PATTERNS):
            violations.append(f"{p}: filename matches a capture-data pattern (name rule)")
        elif p.stat().st_size > max_bytes:
            violations.append(f"{p}: {p.stat().st_size} bytes exceeds {max_bytes} (size rule)")
    return violations


def main(argv: list[str]) -> int:
    violations = check_paths([Path(a) for a in argv])
    for v in violations:
        print(f"blocked: {v}", file=sys.stderr)
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
