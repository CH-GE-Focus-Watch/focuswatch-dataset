"""Pre-commit guard: keeps participant recordings out of the git tree."""
from __future__ import annotations

import re
import sys
from pathlib import Path

MAX_BYTES = 1_048_576

# Why: names seen in the source corpora. Fixtures must not look like real captures.
DATA_NAME_PATTERNS = (
    re.compile(r"_watch\.csv$"),
    re.compile(r"_pen\.csv$"),
    re.compile(r"_markers\.csv$"),
    re.compile(r"airpod_motion.*\.csv$"),
    re.compile(r"^(WristMotion|Headphone|WatchAccelerometerUncalibrated)\.csv$"),
    re.compile(r"\.zip$"),
    re.compile(r"\.parquet$"),
)


def check_paths(paths: list[Path], max_bytes: int = MAX_BYTES) -> list[str]:
    violations = []
    for p in paths:
        if not p.is_file():
            continue
        if any(pat.search(p.name) for pat in DATA_NAME_PATTERNS):
            violations.append(f"{p}: filename matches a capture-data pattern")
        elif p.stat().st_size > max_bytes:
            violations.append(f"{p}: {p.stat().st_size} bytes exceeds {max_bytes}")
    return violations


def main(argv: list[str]) -> int:
    violations = check_paths([Path(a) for a in argv])
    for v in violations:
        print(f"blocked: {v}", file=sys.stderr)
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
