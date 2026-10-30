"""Command line entry point: fw build | validate | report | preprocess | explore."""
from __future__ import annotations

import argparse
from pathlib import Path

from .build import build_dataset
from .load import load_manifest
from .preprocess import (
    DEFAULT_CONTACT_TOLERANCE_S,
    TASKS,
    PreprocessConfig,
    eligible_recordings,
    write_preprocessed,
)
from .redact import RedactionPolicy
from .time_origin import TimeOrigin
from .validate import validate_dataset


def _source_roots(specs: list[str]) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for spec in specs:
        name, separator, value = spec.partition("=")
        if not separator or not name.strip() or not value.strip():
            raise ValueError(
                f"invalid --source {spec!r}; expected a non-empty NAME=PATH"
            )
        name = name.strip()
        if name in roots:
            raise ValueError(f"duplicate --source name {name!r}")
        roots[name] = Path(value)
    return roots


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fw")
    sub = parser.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build")
    b.add_argument("--source", action="append", required=True, metavar="NAME=PATH")
    b.add_argument("--out", required=True)
    b.add_argument(
        "--redact", choices=[p.value for p in RedactionPolicy], required=True,
        help="required explicit pen-coordinate policy; use all_xy for a public release",
    )
    b.add_argument(
        "--time-origin", choices=[o.value for o in TimeOrigin], required=True,
        help="required explicit timestamp origin; use recording_start for a public release",
    )
    b.add_argument("--no-strict", action="store_true")

    v = sub.add_parser("validate")
    v.add_argument("--dataset", required=True)

    r = sub.add_parser("report")
    r.add_argument("--dataset", required=True)

    pp = sub.add_parser("preprocess", help="derive per-sample training labels from a bundle")
    pp.add_argument("--dataset", required=True)
    pp.add_argument("--task", choices=sorted(TASKS), required=True)
    pp.add_argument("--out", required=True, help="new directory, outside the bundle")
    pp.add_argument("--gap-closing-s", type=float,
                    help="writing: pauses up to this length between pen contacts count as "
                         "writing; required for --task writing (the project uses 3.0)")
    pp.add_argument("--pen-shift-s", type=float, default=0.0,
                    help="writing: seconds added to pen timestamps before the join (default 0)")
    pp.add_argument("--contact-tolerance-s", type=float, default=DEFAULT_CONTACT_TOLERANCE_S)
    pp.add_argument("--target-hz", type=float, help="downsample the motion stream first")
    pp.add_argument("--include-overlap-only", action="store_true",
                    help="writing: also label recordings whose watch and pen clocks are "
                         "independent (Moleskine); their labels are approximate")
    pp.add_argument("--cohort", action="append", help="restrict to a cohort (repeatable)")
    pp.add_argument("--recording", action="append", help="restrict to a recording id (repeatable)")

    e = sub.add_parser("explore")
    e.add_argument("root", metavar="DATASET_ROOT")
    e.add_argument(
        "--export",
        metavar="DIRECTORY",
        help="directory for selected files (default: Downloads/focuswatch-selection-<timestamp>)",
    )

    args = parser.parse_args(argv)

    if args.command == "build":
        try:
            roots = _source_roots(args.source)
            report = build_dataset(roots, Path(args.out),
                                   RedactionPolicy(args.redact), strict=not args.no_strict,
                                   time_origin=TimeOrigin(args.time_origin))
        except (RuntimeError, ValueError) as exc:
            # Why: a physical-check failure, a coverage gap, a manifest
            # inconsistency and a load/discovery failure all surface here as
            # a RuntimeError (see build.py) - this CLI runs in CI, where a
            # raw traceback is a worse failure report than a message + exit
            # code, matching the `validate` branch below.
            print(f"build failed: {exc}")
            return 1
        print(f"{len(report.findings)} checks, {len(report.failed)} failed")
        return 1 if report.failed else 0

    if args.command == "preprocess":
        return _preprocess(args)

    if args.command == "explore":
        try:
            from .tui import DatasetExplorerApp
        except ModuleNotFoundError as exc:
            if exc.name not in {"textual", "focuswatch_dataset.tui"}:
                raise
            print(
                'Explorer unavailable: install it with pip install '
                '"focuswatch-dataset[tui]"'
            )
            return 1
        try:
            DatasetExplorerApp(
                Path(args.root), export_path=Path(args.export) if args.export else None
            ).run()
        except Exception as exc:
            print(f"explore failed: could not read bundle at {args.root}: {exc}")
            return 1
        return 0

    dataset = Path(args.dataset)
    if args.command == "validate":
        try:
            report = validate_dataset(dataset)
        except Exception as exc:
            # A missing or corrupt archive must exit 1 with a message naming
            # what happened, rather than a traceback in CI.
            print(f"validate failed: could not read bundle at {dataset}: {exc}")
            return 1
        for f in report.failed:
            print(f"physics: {f.recording_id} {f.check} observed={f.observed}")
        return 1 if report.failed else 0

    try:
        manifest = load_manifest(dataset)
    except Exception as exc:
        print(f"report failed: could not read bundle at {dataset}: {exc}")
        return 1
    # participant_id is namespaced per cohort, so
    # this count would be unchanged if every cohort had recorded the same
    # people - "participants" hands a citing paper an N the data cannot
    # support. Matches the README wording (docs.write_readme), which commit
    # 243ee07 already fixed for this exact reason.
    n_identifiers = int(manifest["participant_id"].nunique())
    identifier_noun = "identifier" if n_identifiers == 1 else "identifiers"
    print(f"{len(manifest)} recordings, "
          f"{n_identifiers} participant {identifier_noun}")
    print(manifest.groupby("cohort").size().to_string())
    return 0


def _preprocess(args: argparse.Namespace) -> int:
    try:
        config = PreprocessConfig(
            task=args.task, gap_closing_s=args.gap_closing_s,
            contact_tolerance_s=args.contact_tolerance_s, pen_shift_s=args.pen_shift_s,
            target_hz=args.target_hz, include_overlap_only=args.include_overlap_only,
        )
        manifest = load_manifest(Path(args.dataset))
        ids = eligible_recordings(manifest, config)
        if args.cohort:
            cohorts = manifest.set_index("recording_id")["cohort"]
            ids = [rid for rid in ids if cohorts[rid] in args.cohort]
        if args.recording:
            unknown = sorted(set(args.recording) - set(ids))
            if unknown:
                raise ValueError(f"not eligible for the {args.task} task: {unknown}")
            ids = [rid for rid in ids if rid in args.recording]
        out = write_preprocessed(
            Path(args.dataset), ids, config, Path(args.out),
            on_progress=lambda n, total, rid: print(f"[{n}/{total}] {rid}", flush=True),
        )
    except (FileExistsError, KeyError, ValueError) as exc:
        print(f"preprocess failed: {exc}")
        return 1
    print(f"{len(ids)} recordings labelled for {args.task} -> {out}")
    return 0
