"""Command line entry point: fw build | validate | report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .build import build_dataset
from .load import load_manifest
from .manifest import check_manifest_consistency
from .redact import RedactionPolicy


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fw")
    sub = parser.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build")
    b.add_argument("--source", action="append", required=True, metavar="NAME=PATH")
    b.add_argument("--out", required=True)
    b.add_argument("--redact", choices=[p.value for p in RedactionPolicy],
                   default=RedactionPolicy.NONE.value)
    b.add_argument("--no-strict", action="store_true")

    v = sub.add_parser("validate")
    v.add_argument("--dataset", required=True)

    r = sub.add_parser("report")
    r.add_argument("--dataset", required=True)

    args = parser.parse_args(argv)

    if args.command == "build":
        roots = dict(s.split("=", 1) for s in args.source)
        try:
            report = build_dataset({k: Path(v) for k, v in roots.items()}, Path(args.out),
                                   RedactionPolicy(args.redact), strict=not args.no_strict)
        except RuntimeError as exc:
            # Why: a physical-check failure, a coverage gap, a manifest
            # inconsistency and a load/discovery failure all surface here as
            # a RuntimeError (see build.py) - this CLI runs in CI, where a
            # raw traceback is a worse failure report than a message + exit
            # code, matching the `validate` branch below.
            print(f"build failed: {exc}")
            return 1
        print(f"{len(report.findings)} checks, {len(report.failed)} failed")
        return 1 if report.failed else 0

    dataset = Path(args.dataset)
    if args.command == "validate":
        try:
            manifest = load_manifest(dataset)
            problems = check_manifest_consistency(manifest, dataset)
            failed = [f for f in json.loads((dataset / "validation_report.json").read_text())
                      if not f["passed"]]
        except Exception as exc:
            # Why (M11): a missing or corrupt bundle (sessions.parquet or
            # validation_report.json absent, truncated, or unparseable) must
            # exit 1 with a message naming what happened, matching the
            # `build` branch above - not a raw traceback in what is meant to
            # run in CI.
            print(f"validate failed: could not read bundle at {dataset}: {exc}")
            return 1
        for p in problems:
            print(f"manifest: {p}")
        for f in failed:
            print(f"physics: {f['recording_id']} {f['check']} observed={f['observed']}")
        return 1 if (problems or failed) else 0

    try:
        manifest = load_manifest(dataset)
    except Exception as exc:
        print(f"report failed: could not read bundle at {dataset}: {exc}")
        return 1
    # Why (item 4, fix round C): "participant_id" is namespaced per cohort, so
    # this count would be unchanged if all three cohorts had recorded the same
    # people - "participants" hands a citing paper an N the data cannot
    # support. Matches the README wording (docs.write_readme), which commit
    # 243ee07 already fixed for this exact reason.
    print(f"{len(manifest)} recordings, "
          f"{manifest['participant_id'].nunique()} participant identifiers")
    print(manifest.groupby("cohort").size().to_string())
    return 0
