"""Select recordings from the manifest, then explicitly load their samples.

Usage::

    python examples/select_and_load.py DATASET_ROOT --require-pen --accel-semantics user

The first phase reads only ``sessions.parquet``. Sensor rows are not read until
the selected recording IDs are passed explicitly to ``load_recordings``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from focuswatch_dataset import by_flags, load_manifest, load_recordings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select published recordings from the manifest and load samples."
    )
    parser.add_argument(
        "root",
        type=Path,
        metavar="DATASET_ROOT",
        help="path to a published focuswatch-dataset bundle",
    )
    parser.add_argument(
        "--cohort",
        help="select one cohort (for example, ML4SCS or ETH)",
    )
    parser.add_argument(
        "--require-pen",
        action="store_true",
        help="keep only recordings with a published pen table",
    )
    parser.add_argument(
        "--modality",
        default="watch",
        choices=("watch", "watch_rawaccel", "headimu", "pen", "markers", "attention"),
        help="sensor table to load after selection (default: watch)",
    )
    parser.add_argument(
        "--accel-semantics",
        choices=("user", "total"),
        default="user",
        help="watch acceleration semantics to select (default: user)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        metavar="N",
        help="load at most N selected recordings",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be at least 1")

    # Manifest-only phase: this reads sessions.parquet, never a sensor table.
    manifest = load_manifest(args.root)
    flags = {}
    if args.cohort is not None:
        flags["cohort"] = args.cohort
    if args.require_pen:
        flags["has_pen"] = True
    flags[f"has_{args.modality}"] = True
    if args.modality == "watch":
        flags["accel_semantics"] = args.accel_semantics
    selected = by_flags(manifest, **flags)
    recording_ids = selected["recording_id"].tolist()
    if args.limit is not None:
        recording_ids = recording_ids[: args.limit]
    if not recording_ids:
        raise SystemExit("selection matched no recordings; adjust the manifest filters")

    print(f"selected {len(recording_ids)} recording(s): {recording_ids}")

    # Loading is explicit and happens only after the manifest selection.
    samples = load_recordings(args.root, recording_ids, modality=args.modality)
    print(f"loaded {len(samples)} {args.modality} sample row(s)")
    print(samples.head())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
