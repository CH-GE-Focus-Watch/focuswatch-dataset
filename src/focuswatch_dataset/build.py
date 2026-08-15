"""Orchestration: discover, load, validate, write, describe.

Fail-loud choice: any single recording that cannot be discovered or loaded
aborts the *entire* build (the exception propagates, wrapped with which
source/recording it came from, and nothing is written to `out` beyond the
directory itself - no sessions.parquet, no per-modality tables). A strict
build that finds a physical-check failure or a coverage gap behaves the
same way: it writes only `validation_report.json` (the diagnostic, not a
publishable artefact) and raises. The alternative - drop the offending
recording and publish the rest - was rejected: a 66-of-67 bundle looks
exactly like a complete one from the outside, and the missing recording
would only surface if someone thought to check the count against the
corpus manifest kept elsewhere. Loud and total beats quiet and partial.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from . import docs
from .adapters import base
from .adapters.base import RecordingBundle
from .manifest import build_channels, build_manifest, check_manifest_consistency
from .redact import RedactionPolicy, redact_bundle
from .validate import (
    ValidationReport, check_coverage, validate_motion_table, validate_pen_table,
    validate_recording,
)
from .write import write_table

_MOTION_MODALITIES = ("watch", "watch_rawaccel", "headimu")


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return ""


def _validate(bundle: RecordingBundle) -> list:
    findings = []
    for modality in _MOTION_MODALITIES:
        if modality in bundle.tables:
            nominal = bundle.meta.get("head_hz_nominal" if modality == "headimu"
                                      else "watch_hz_nominal")
            findings += validate_motion_table(bundle.tables[modality],
                                              bundle.ref.recording_id, modality, nominal)
    if "pen" in bundle.tables:
        findings += validate_pen_table(bundle.tables["pen"], bundle.ref.recording_id)
    findings += validate_recording(bundle.ref.recording_id, bundle.tables, bundle.meta)
    return findings


def _load_bundle(name: str, root: Path) -> list[RecordingBundle]:
    """Discover and load every recording of one source, tagged with build-time meta.

    Any exception here - a malformed source CSV, an adapter bug, a recording
    with structurally impossible data - is re-raised with the source/
    recording named, then left to propagate: see the module docstring for
    why that aborts the whole build rather than skipping the recording.
    """
    adapter = base.get_adapter(name)
    try:
        refs = adapter.discover(Path(root))
    except Exception as exc:
        raise RuntimeError(f"discovery failed for source '{name}' at {root}: {exc}") from exc

    bundles = []
    for ref in refs:
        try:
            bundle = adapter.load(ref)
        except Exception as exc:
            raise RuntimeError(
                f"failed to load recording '{ref.recording_id}' (source '{name}'): {exc}"
            ) from exc
        bundle.meta["build_git_sha"] = _git_sha()
        bundles.append(bundle)
    return bundles


def build_dataset(source_roots: dict[str, Path], out: Path,
                  policy: RedactionPolicy = RedactionPolicy.NONE,
                  strict: bool = True) -> ValidationReport:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    report = ValidationReport()
    bundles: list[RecordingBundle] = []

    # Why: sorted() on the caller-supplied dict, not the dict's own insertion
    # order - a build must produce identical bytes regardless of the order
    # its source_roots argument happens to have been assembled in (e.g. a
    # dict comprehension over os.listdir), not merely regardless of calling
    # build_dataset twice with the exact same dict object.
    for name, root in sorted(source_roots.items()):
        for bundle in _load_bundle(name, root):
            # Why: redact_bundle, not a hand-rolled apply_redaction + meta
            # stamp - it is the one seam that guarantees the published
            # redaction_policy meta can never disagree with what was
            # actually done to the pen table (see redact.py's docstring).
            bundle = redact_bundle(bundle, policy)
            report.findings += _validate(bundle)
            bundles.append(bundle)

    manifest_preview = build_manifest(bundles)
    gaps = check_coverage(manifest_preview, report.findings)
    if strict and (report.failed or gaps):
        (out / "validation_report.json").write_text(report.to_json())
        detail = f"{len(report.failed)} failed checks"
        if gaps:
            # Why: an unrun check is indistinguishable from a passed one in the
            # report, so coverage gaps abort just as hard as failures.
            detail += f", {len(gaps)} coverage gaps: " + "; ".join(gaps[:5])
        raise RuntimeError(f"validation failed - {detail}; see validation_report.json")

    for bundle in bundles:
        for modality, df in bundle.tables.items():
            write_table(df, out / modality / f"{bundle.ref.recording_id}.parquet",
                        bundle.ref.recording_id)

    manifest = manifest_preview
    channels = build_channels(bundles)
    manifest.to_parquet(out / "sessions.parquet", index=False)
    manifest.to_csv(out / "sessions.csv", index=False)
    channels.to_parquet(out / "channels.parquet", index=False)

    problems = check_manifest_consistency(manifest, out)
    if problems and strict:
        raise RuntimeError("manifest inconsistent: " + "; ".join(problems))

    docs.write_datapackage(out, manifest, channels)
    docs.write_data_dictionary(out, channels)
    (out / "validation_report.json").write_text(report.to_json())
    return report
