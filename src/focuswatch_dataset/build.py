"""Discover, validate, and atomically publish a complete dataset bundle.

Any recording or validation failure aborts the build. Artifacts are staged next
to the destination and promoted only after every write succeeds.
"""
from __future__ import annotations

import importlib.metadata
import os
import shutil
import subprocess
import tempfile
import uuid
from collections import Counter
from pathlib import Path

from . import docs
from . import schema as S
from .adapters import base
from .adapters.base import RecordingBundle
from .manifest import build_channels, build_manifest, check_manifest_consistency
from .redact import RedactionPolicy, redact_bundle
from .validate import (
    Dropout, Finding, ValidationReport, check_coverage, detect_dropouts, validate_bundle,
)
from .write import write_table

# Why: a directory carrying either of these is recognisably a previous build
# of this package, not merely "something happens to live at this path" - both
# are written only by this module, near the very end of a successful build.
_PRIOR_BUILD_SIGNATURE = ("sessions.parquet", "datapackage.json")


def _git_sha() -> str:
    """The SHA of this PACKAGE's own checkout, not whatever the caller's cwd is.

    A missing `cwd` would make `git rev-parse HEAD` run against the caller's working
    directory - "not a git repository" for an unrelated cwd, or a foreign
    repo's SHA for an unrelated one. `-dirty` records uncommitted changes,
    which a bare SHA would otherwise claim as clean. Never an empty string: a
    checkout with no `.git` at all (e.g. installed from a wheel) falls back
    to the installed package's own version, which is still a real, checkable
    provenance fact - unlike "".
    """
    repo_dir = Path(__file__).resolve().parent
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_dir, text=True, stderr=subprocess.DEVNULL,
        ).strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=repo_dir, text=True,
            capture_output=True, check=True,
        ).stdout.strip()
        return f"{sha}-dirty" if dirty else sha
    except Exception:
        try:
            return importlib.metadata.version("focuswatch-dataset")
        except importlib.metadata.PackageNotFoundError:
            return ""


def _load_bundle(name: str, root: Path, build_git_sha: str) -> list[RecordingBundle]:
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
    if not refs:
        raise RuntimeError(f"source '{name}' at {root} discovered no recordings")

    bundles = []
    for ref in refs:
        try:
            bundle = adapter.load(ref)
        except Exception as exc:
            raise RuntimeError(
                f"failed to load recording '{ref.recording_id}' (source '{name}'): {exc}"
            ) from exc
        bundle.meta["build_git_sha"] = build_git_sha
        bundles.append(bundle)
    return bundles


def _require_unique_recording_ids(bundles: list[RecordingBundle]) -> None:
    """Refuse ambiguous output names before validation or staging begins."""
    counts = Counter(bundle.ref.recording_id for bundle in bundles)
    duplicates = sorted(recording_id for recording_id, count in counts.items() if count > 1)
    if duplicates:
        raise RuntimeError(f"duplicate recording_id values: {', '.join(duplicates)}")


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


def _manifest_finding(problem: str) -> Finding:
    """Turn one check_manifest_consistency problem string into a failing Finding.

    Mirrors _gap_finding above: a manifest/file mismatch on the
    permissive (strict=False) path must be discoverable in
    validation_report.json, not just present as a string nobody reads once
    the build proceeds anyway. Best-effort parse of the
    "<modality>/<recording_id>.parquet <detail>" shape
    check_manifest_consistency emits; a string that doesn't match still
    round-trips (the full text lands in `observed`).
    """
    modality, sep, rest = problem.partition("/")
    recording_id, sep2, detail = rest.partition(".parquet ")
    if not sep or not sep2:
        modality, recording_id, detail = "-", "-", problem
    return Finding("manifest_consistency", recording_id, modality, "-", detail,
                   "has_<modality> flag matches file presence", False)


def _refuse_foreign_directory(out: Path) -> None:
    if not out.exists():
        return
    if not out.is_dir():
        # Why: checked before out.iterdir(), which raises NotADirectoryError
        # on a plain file - an error shape that escapes both this function's
        # own RuntimeError contract and cli.py's `except RuntimeError`,
        # turning a controlled refusal into a raw traceback.
        raise RuntimeError(f"refusing to build into {out}: it exists and is not a directory")
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


def _warn_about_interrupted_promotions(out: Path) -> None:
    """Announce, never touch, leftovers from a promotion killed mid-swap.

    `_promote` guarantees `out` itself is left correct even if the process
    dies between its two renames (see that function's docstring) - but the
    `.replaced-*` backup and/or `.build-*` staging sibling from that
    interruption stay on disk with nothing pointing at them. Surfacing them
    here is the entire fix: no automatic reattachment, no deletion - either
    would be a destructive guess layered on top of an already-recovered
    state for a marginal convenience gain.
    """
    parent = out.parent
    if not parent.exists():
        return
    build_prefix = f".{out.name}.build-"
    backup_prefix = f".{out.name}.replaced-"
    try:
        siblings = list(parent.iterdir())
    except OSError:
        # Why: this notice is a courtesy, so a parent we cannot list must cost
        # the notice and nothing else. Letting the error out would abort a build
        # that had not started yet, and as a non-RuntimeError it would bypass
        # the CLI's handler and surface as a traceback.
        return
    leftovers = sorted(
        p for p in siblings
        if p.is_dir() and (p.name.startswith(build_prefix) or p.name.startswith(backup_prefix))
    )
    for p in leftovers:
        kind = ("an unpromoted staging directory from an interrupted build"
                if p.name.startswith(build_prefix) else
                "the previous build, preserved by an interrupted promotion")
        print(f"note: {p} looks like {kind} - left untouched, remove manually if unwanted")


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
    _warn_about_interrupted_promotions(out)
    _refuse_foreign_directory(out)
    report = ValidationReport()
    loaded_bundles: list[RecordingBundle] = []
    bundles: list[RecordingBundle] = []
    build_git_sha = _git_sha()

    # Why: sorted() on the caller-supplied dict, not the dict's own insertion
    # order - a build must produce identical bytes regardless of the order
    # its source_roots argument happens to have been assembled in (e.g. a
    # dict comprehension over os.listdir), not merely regardless of calling
    # build_dataset twice with the exact same dict object.
    for name, root in sorted(source_roots.items()):
        loaded_bundles.extend(_load_bundle(name, root, build_git_sha))

    _require_unique_recording_ids(loaded_bundles)

    dropouts: dict[str, dict[str, Dropout]] = {}
    for bundle in loaded_bundles:
        # Why: redact_bundle, not a hand-rolled apply_redaction + meta
        # stamp - it is the one seam that guarantees the published
        # redaction_policy meta can never disagree with what was
        # actually done to the pen table (see redact.py's docstring).
        bundle = redact_bundle(bundle, policy)
        bundle_dropouts = detect_dropouts(bundle.tables)
        dropouts[bundle.ref.recording_id] = bundle_dropouts
        report.findings += validate_bundle(bundle, bundle_dropouts)
        bundles.append(bundle)

    manifest_preview = build_manifest(bundles, dropouts)
    # Computed once per bundle and passed explicitly rather than
    # re-derived inside check_coverage (which never receives bundle.tables) -
    # the exact same computation manifest.build_manifest already ran to
    # populate issue_codes, so the two can never disagree about which
    # modality is a declared dropout.
    gaps = check_coverage(manifest_preview, report.findings, dropouts)
    # Why: computed once and used everywhere validation_report.json is
    # written or returned - report.findings alone (no synthetic coverage_gap
    # entries) must never diverge from what actually landed on disk, on
    # either the abort path or the strict=False path that proceeds despite
    # gaps (cli.py's summary and exit code read the returned report, not the
    # file, so a mismatch there would silently under-report a real problem).
    final_report = _report_with_gaps(report, gaps)
    if strict and (report.failed or gaps):
        # Invariant: a strict validation failure never creates or changes
        # `out`; its complete report is written to a unique sibling instead.
        out.parent.mkdir(parents=True, exist_ok=True)
        failed_dir = out.parent / f".{out.name}.failed-{uuid.uuid4().hex[:8]}"
        failed_dir.mkdir(parents=True, exist_ok=True)
        report_path = failed_dir / "validation_report.json"
        # Why: the gaps go into the persisted file in full, not just the
        # truncated console message below - a stranger reading
        # validation_report.json after a coverage-gap abort must be able to
        # see every gap, not just the reason a RuntimeError happened to name.
        report_path.write_text(final_report.to_json())
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
        if problems:
            if strict:
                raise RuntimeError("manifest inconsistent: " + "; ".join(problems))
            # On the permissive path these used to be computed and
            # then discarded - neither raised, appended, nor printed. Same
            # principle as _report_with_gaps above: the report that gets
            # persisted and returned must not diverge from what
            # check_manifest_consistency actually found.
            final_report = ValidationReport(
                final_report.findings + [_manifest_finding(p) for p in problems])

        docs.write_license(staging)
        docs.write_datapackage(staging, manifest, channels)
        docs.write_data_dictionary(staging, manifest, channels)
        docs.write_readme(staging, manifest, channels)
        (staging / "validation_report.json").write_text(final_report.to_json())
        # Why: promotion is inside this same try - a failure here (either
        # rename) must trigger the same staging cleanup as a write failure,
        # or a leaked `.<out>.build-*` sibling is exactly the accumulating
        # mess this whole staging design exists to prevent (see the module
        # docstring). _promote's own internal recovery keeps `out` correct
        # either way; this block only ever has staging left to clean up.
        _promote(staging, out)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return final_report
