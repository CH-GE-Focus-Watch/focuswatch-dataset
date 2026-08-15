"""Orchestration: discover, load, validate, write, describe.

Fail-loud choice: any single recording that cannot be discovered or loaded
aborts the *entire* build (the exception propagates, wrapped with which
source/recording it came from). A strict build that finds a physical-check
failure or a coverage gap behaves the same way: it writes only
`validation_report.json` directly into `out` (the diagnostic, not a
publishable artefact) and raises. The alternative - drop the offending
recording and publish the rest - was rejected: a 66-of-67 bundle looks
exactly like a complete one from the outside, and the missing recording
would only surface if someone thought to check the count against the
corpus manifest kept elsewhere. Loud and total beats quiet and partial.

That guarantee has to survive `out` already existing, not just a fresh
directory: a rebuild with a shrunk or corrected corpus must not leave a
stale `<modality>/<old_id>.parquet` behind (check_manifest_consistency would
catch the mismatch, but only after the new build's own files are already on
disk next to it), and a crash mid-write must not leave a partial manifest.
Both are solved the same way - every publishable artefact is written into a
sibling staging directory first and the whole thing is promoted into `out`
with two directory renames (`out` -> a throwaway backup name, staging ->
`out`) only once every write has succeeded; any failure before that point
touches `out` not at all. The two renames stay on one filesystem because the
staging directory and the backup name are both created next to `out`, not in
the system temp directory.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

from . import docs
from .adapters import base
from .adapters.base import RecordingBundle
from .manifest import build_channels, build_manifest, check_manifest_consistency
from .redact import RedactionPolicy, redact_bundle
from .validate import (
    Finding, ValidationReport, check_coverage, validate_motion_table, validate_pen_table,
    validate_recording,
)
from .write import write_table

_MOTION_MODALITIES = ("watch", "watch_rawaccel", "headimu")

# Why: a directory carrying either of these is recognisably a previous build
# of this package, not merely "something happens to live at this path" - both
# are written only by this module, near the very end of a successful build.
_PRIOR_BUILD_SIGNATURE = ("sessions.parquet", "datapackage.json")


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


def _gap_finding(gap: str) -> Finding:
    """Turn one check_coverage problem string into a failing Finding.

    Why: validation_report.json's schema is (and must stay) a flat list of
    Finding dicts - test_validation_report_is_written_and_all_checks_pass
    reads it that way - so a coverage gap has to become a Finding to be
    findable in the file at all, rather than living only in the console
    message and the raised exception. Best-effort parse of the
    "rid/modality: detail" shape check_coverage.require()/the quaternion
    either-or check both use; a gap that doesn't match still round-trips
    (the full text lands in `observed`), it just loses the structured
    recording_id/modality split.
    """
    rid, sep, rest = gap.partition("/")
    modality, sep2, detail = rest.partition(": ")
    if not sep or not sep2:
        rid, modality, detail = "-", "-", gap
    return Finding("coverage_gap", rid, modality, "-", detail,
                   "check ran and produced a finding", False)


def _report_with_gaps(report: ValidationReport, gaps: list[str]) -> ValidationReport:
    return ValidationReport(report.findings + [_gap_finding(g) for g in gaps])


def _refuse_foreign_directory(out: Path) -> None:
    if not out.exists():
        return
    if not any(out.iterdir()):
        return
    if any((out / f).exists() for f in _PRIOR_BUILD_SIGNATURE):
        return
    raise RuntimeError(
        f"refusing to build into {out}: it is non-empty and carries none of "
        f"{_PRIOR_BUILD_SIGNATURE}, so it does not look like a previous "
        "focuswatch-dataset build - point --out at an empty directory or a "
        "directory this tool already built"
    )


def _promote(staging: Path, out: Path) -> None:
    """Swap `staging` into `out`'s place with two same-filesystem renames.

    `out` is never simultaneously absent and half-populated: right up until
    the first rename it holds whatever it held before (nothing, or a
    complete previous build); after the first rename it is briefly absent;
    after the second it holds the complete new build. A failure on the
    second rename restores the previous `out` from the backup rather than
    leaving the directory missing.
    """
    if not out.exists():
        os.replace(staging, out)
        return
    backup = out.parent / f".{out.name}.replaced-{uuid.uuid4().hex[:8]}"
    os.replace(out, backup)
    try:
        os.replace(staging, out)
    except Exception:
        if out.exists():
            shutil.rmtree(out, ignore_errors=True)
        os.replace(backup, out)
        raise
    shutil.rmtree(backup, ignore_errors=True)


def build_dataset(source_roots: dict[str, Path], out: Path,
                  policy: RedactionPolicy = RedactionPolicy.NONE,
                  strict: bool = True) -> ValidationReport:
    out = Path(out)
    _refuse_foreign_directory(out)
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
        out.mkdir(parents=True, exist_ok=True)
        report_path = out / "validation_report.json"
        # Why: the gaps go into the persisted file in full, not just the
        # truncated console message below - a stranger reading
        # validation_report.json after a coverage-gap abort must be able to
        # see every gap, not just the reason a RuntimeError happened to name.
        report_path.write_text(_report_with_gaps(report, gaps).to_json())
        detail = f"{len(report.failed)} failed checks"
        if gaps:
            detail += f", {len(gaps)} coverage gaps: " + "; ".join(gaps[:5])
            if len(gaps) > 5:
                detail += f" (+{len(gaps) - 5} more)"
        raise RuntimeError(f"validation failed - {detail}; see {report_path} for the full list")

    # Why: a sibling of `out`, not tempfile's default system temp dir - the
    # promotion below is two directory renames, which only stay atomic (and
    # only work at all without copying) when both sides are on one filesystem.
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=out.parent, prefix=f".{out.name}.build-"))
    try:
        for bundle in bundles:
            for modality, df in bundle.tables.items():
                write_table(df, staging / modality / f"{bundle.ref.recording_id}.parquet",
                            bundle.ref.recording_id)

        manifest = manifest_preview
        channels = build_channels(bundles)
        manifest.to_parquet(staging / "sessions.parquet", index=False)
        manifest.to_csv(staging / "sessions.csv", index=False)
        channels.to_parquet(staging / "channels.parquet", index=False)

        problems = check_manifest_consistency(manifest, staging)
        if problems and strict:
            raise RuntimeError("manifest inconsistent: " + "; ".join(problems))

        docs.write_datapackage(staging, manifest, channels)
        docs.write_data_dictionary(staging, channels)
        (staging / "validation_report.json").write_text(
            _report_with_gaps(report, gaps).to_json())
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    _promote(staging, out)
    return report
