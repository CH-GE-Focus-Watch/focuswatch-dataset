import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from focuswatch_dataset.build import build_dataset
from focuswatch_dataset.cli import main
from focuswatch_dataset.load import load_manifest
from focuswatch_dataset.manifest import check_manifest_consistency
from focuswatch_dataset.redact import RedactionPolicy
from focuswatch_dataset.write import read_table

from .test_adapter_airpods import write_fixture as write_airpods
from .test_adapter_ege import write_fixture as write_ege
from .test_adapter_ml4scs import write_fixture as write_ml4scs
from .test_adapter_sensorlogger import write_fixture as write_sl


def sources(tmp_path):
    roots = {}
    for name, writer in (("ml4scs", write_ml4scs), ("ege", write_ege),
                         ("sensorlogger", write_sl), ("airpods", write_airpods)):
        d = tmp_path / "src" / name
        d.mkdir(parents=True)
        writer(d)
        roots[name] = d
    return roots


def test_build_produces_all_expected_artefacts(tmp_path):
    out = tmp_path / "out"
    build_dataset(sources(tmp_path), out)
    for f in ("sessions.parquet", "sessions.csv", "channels.parquet",
              "datapackage.json", "data_dictionary.md", "validation_report.json"):
        assert (out / f).exists(), f


def test_manifest_covers_every_cohort(tmp_path):
    out = tmp_path / "out"
    build_dataset(sources(tmp_path), out)
    assert set(load_manifest(out)["cohort"]) == {"ML4SCS", "ETH", "AIRPODS"}


def test_build_is_deterministic(tmp_path):
    src = sources(tmp_path)
    a, b = tmp_path / "a", tmp_path / "b"
    build_dataset(src, a)
    build_dataset(src, b)
    for f in sorted(p.relative_to(a) for p in a.rglob("*.parquet")):
        assert hashlib.sha256((a / f).read_bytes()).hexdigest() == \
               hashlib.sha256((b / f).read_bytes()).hexdigest(), f


def test_validation_report_is_written_and_all_checks_pass(tmp_path):
    out = tmp_path / "out"
    build_dataset(sources(tmp_path), out)
    findings = json.loads((out / "validation_report.json").read_text())
    assert findings
    assert [f for f in findings if not f["passed"]] == []


def test_strict_build_aborts_on_a_physical_failure(tmp_path, monkeypatch):
    from focuswatch_dataset import schema as S
    src = sources(tmp_path)
    # Force every acceleration median into the forbidden gap.
    monkeypatch.setattr(S, "ACCEL_USER_BAND", (0.9, 1.1))
    with pytest.raises(RuntimeError, match="validation failed"):
        build_dataset(src, tmp_path / "out", strict=True)


def test_redaction_policy_is_recorded_in_the_manifest(tmp_path):
    out = tmp_path / "out"
    build_dataset(sources(tmp_path), out, policy=RedactionPolicy.FREE_WRITING_XY)
    assert set(load_manifest(out)["redaction_policy"]) == {"free_writing_xy"}


def test_cli_build_and_validate(tmp_path):
    src = sources(tmp_path)
    out = tmp_path / "out"
    args = ["build", "--out", str(out)]
    for name, path in src.items():
        args += ["--source", f"{name}={path}"]
    assert main(args) == 0
    assert main(["validate", "--dataset", str(out)]) == 0


# --- Coverage-gate: check_coverage must actually gate the build -----------

def test_missing_required_check_aborts_the_build(tmp_path, monkeypatch):
    """A recording that never produced a required check must abort a strict
    build, not merely have the gap reported alongside a green exit.

    Drops the gyro_range finding for every watch table (has_watch is true
    for the ml4scs recording, so check_coverage requires it) without
    touching any other check - report.failed stays empty, so only
    check_coverage's own gap-detection can be responsible for the abort.
    """
    import focuswatch_dataset.build as build_mod

    original = build_mod.validate_motion_table

    def dropped(df, recording_id, modality, nominal_hz):
        findings = original(df, recording_id, modality, nominal_hz)
        if modality == "watch":
            findings = [f for f in findings if f.check != "gyro_range"]
        return findings

    monkeypatch.setattr(build_mod, "validate_motion_table", dropped)
    with pytest.raises(RuntimeError, match="coverage gap"):
        build_dataset(sources(tmp_path), tmp_path / "out", strict=True)


def test_coverage_gap_abort_persists_the_full_gap_list_in_the_report(tmp_path, monkeypatch):
    """fix round 1, item 2: before this fix, validation_report.json on a
    coverage-gap abort held only the physical findings (which can all be
    `passed: true`) - the actual reason for the abort lived solely in the
    RuntimeError message, truncated to five gaps and never written to disk.
    """
    import focuswatch_dataset.build as build_mod

    original = build_mod.validate_motion_table

    def dropped(df, recording_id, modality, nominal_hz):
        findings = original(df, recording_id, modality, nominal_hz)
        if modality == "watch":
            findings = [f for f in findings if f.check != "gyro_range"]
        return findings

    monkeypatch.setattr(build_mod, "validate_motion_table", dropped)
    out = tmp_path / "out"
    with pytest.raises(RuntimeError, match="coverage gap"):
        build_dataset(sources(tmp_path), out, strict=True)

    findings = json.loads((out / "validation_report.json").read_text())
    gap_findings = [f for f in findings if f["check"] == "coverage_gap"]
    assert gap_findings, "the report must not read as if every check passed"
    assert all(not f["passed"] for f in gap_findings)
    assert any(f["modality"] == "watch" and "gyro_range" in f["observed"] for f in gap_findings)


# --- Order-independence: source_roots dict key order must not matter ------

def test_build_is_order_independent_of_source_dict_key_order(tmp_path):
    """Byte-identical output for two *differently ordered* dicts describing
    the same corpus - not just for the same dict object called twice, which
    can never expose an unsorted iteration over the caller's mapping.
    """
    src = sources(tmp_path)
    reversed_src = dict(reversed(list(src.items())))
    a, b = tmp_path / "a", tmp_path / "b"
    build_dataset(src, a)
    build_dataset(reversed_src, b)
    files_a = sorted(p.relative_to(a) for p in a.rglob("*") if p.is_file())
    files_b = sorted(p.relative_to(b) for p in b.rglob("*") if p.is_file())
    assert [f.as_posix() for f in files_a] == [f.as_posix() for f in files_b]
    # Why: every file, not only *.parquet - validation_report.json's finding
    # order follows bundle-processing order, so it is the one artefact an
    # ordering regression could corrupt that a parquet-only hash comparison
    # would miss entirely (write_table's own byte-reproducibility guarantee
    # covers the parquet files regardless of this test).
    for f in files_a:
        assert hashlib.sha256((a / f).read_bytes()).hexdigest() == \
               hashlib.sha256((b / f).read_bytes()).hexdigest(), f


# --- Fail loudly: a single failed recording must abort, never half-write --

def test_recording_load_failure_aborts_without_writing_output(tmp_path, monkeypatch):
    from focuswatch_dataset.adapters import base

    airpods_adapter = base.get_adapter("airpods")

    def boom(ref):
        raise ValueError("simulated corrupt recording")

    monkeypatch.setattr(airpods_adapter, "load", boom)
    out = tmp_path / "out"
    with pytest.raises(RuntimeError, match="failed to load recording"):
        build_dataset(sources(tmp_path), out, strict=True)
    # Why: a build that half-writes and exits zero must be impossible - this
    # failure happens before any publishable artefact is written (staging
    # doesn't get created until every recording has loaded), so `out` must
    # not exist at all, let alone carry a partial set of files.
    assert not out.exists()


# --- Never half-write into a pre-existing `out` (fix round 1, item 1) -----

def test_rebuild_over_a_previous_build_leaves_no_stale_file(tmp_path):
    """Re-running into the same --out with a shrunk corpus must not leave a
    stale <modality>/<old_id>.parquet from the first build lying around next
    to the new, smaller set - the exact "ordinary iteration, not hand-
    corruption" scenario fix round 1 named.
    """
    out = tmp_path / "out"
    build_dataset(sources(tmp_path), out)
    before = {p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()}
    # Sanity: an AIRPODS-prefixed recording file is actually present before
    # the shrink - recording_id, not the source name, is what tags the file
    # (the modality subfolder is named by modality, e.g. attention/, not by
    # source).
    assert any("AIRPODS-" in name for name in before)

    shrunk_root = tmp_path / "src2"
    shrunk_root.mkdir()
    d = shrunk_root / "ml4scs"
    d.mkdir()
    write_ml4scs(d)
    build_dataset({"ml4scs": d}, out)

    after = {p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()}
    assert not any("AIRPODS-" in name for name in after)  # every airpods recording file is gone
    manifest = load_manifest(out)
    assert set(manifest["cohort"]) == {"ML4SCS"}
    # No leftover file outside what the new, smaller manifest declares.
    assert check_manifest_consistency(manifest, out) == []


def test_rebuild_into_a_foreign_nonempty_directory_refuses(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "unrelated.txt").write_text("someone else's data")
    with pytest.raises(RuntimeError, match="refusing to build"):
        build_dataset(sources(tmp_path), out)
    # Refused before writing anything - the foreign file is untouched and
    # nothing new appeared next to it.
    assert (out / "unrelated.txt").read_text() == "someone else's data"
    assert list(out.iterdir()) == [out / "unrelated.txt"]


def test_write_failure_mid_staging_leaves_out_untouched(tmp_path, monkeypatch):
    """A crash after some artefacts are already written to the staging
    directory (write_table succeeded for one bundle, then something breaks)
    must not promote a partial staging directory into `out`, and must not
    disturb a previous good `out` either.
    """
    import focuswatch_dataset.build as build_mod

    out = tmp_path / "out"
    src = sources(tmp_path)
    build_dataset(src, out)
    before = {p.relative_to(out).as_posix(): (out / p).read_bytes()
             for p in out.rglob("*") if p.is_file()}

    original = build_mod.write_table
    calls = {"n": 0}

    def flaky(df, path, recording_id):
        calls["n"] += 1
        if calls["n"] == 3:
            raise OSError("simulated disk failure")
        return original(df, path, recording_id)

    monkeypatch.setattr(build_mod, "write_table", flaky)
    with pytest.raises(OSError, match="simulated disk failure"):
        build_dataset(src, out)

    after = {p.relative_to(out).as_posix(): (out / p).read_bytes()
             for p in out.rglob("*") if p.is_file()}
    assert after == before
    # No leaked staging directory next to `out`.
    leaked = [p for p in out.parent.iterdir() if p.name.startswith(f".{out.name}.build-")]
    assert leaked == []


# --- Fix round 2 -------------------------------------------------------

def test_promotion_failure_does_not_leak_the_staging_directory(tmp_path, monkeypatch):
    """fix round 2, item 1: the old code called _promote(staging, out)
    *outside* the try/except that cleaned up staging, so a failure inside
    promotion itself (as opposed to a failure while populating staging)
    leaked the staging directory forever next to `out` - exactly the
    accumulating-sibling outcome the staging design exists to prevent.
    """
    import focuswatch_dataset.build as build_mod

    def boom(a, b):
        raise OSError("simulated promotion failure")

    monkeypatch.setattr(build_mod.os, "replace", boom)
    out = tmp_path / "out"
    with pytest.raises(OSError, match="simulated promotion failure"):
        build_dataset(sources(tmp_path), out)
    assert not out.exists()
    leaked = [p for p in tmp_path.iterdir() if p.name.startswith(f".{out.name}.build-")]
    assert leaked == []


def test_out_as_a_plain_file_refuses_with_a_runtime_error_not_a_traceback(tmp_path):
    """fix round 2, item 2: out.iterdir() on a plain file raises
    NotADirectoryError, which escapes both this module's own RuntimeError
    contract and cli.py's `except RuntimeError`.
    """
    out = tmp_path / "out"
    out.write_text("i am a file, not a directory")
    with pytest.raises(RuntimeError, match="refusing to build"):
        build_dataset(sources(tmp_path), out)
    assert out.read_text() == "i am a file, not a directory"


def test_cli_build_reports_out_as_a_file_without_a_traceback(tmp_path, capsys):
    src = sources(tmp_path)
    out = tmp_path / "out"
    out.write_text("occupied")
    args = ["build", "--out", str(out)]
    for name, path in src.items():
        args += ["--source", f"{name}={path}"]
    assert main(args) == 1
    assert "build failed" in capsys.readouterr().out
    assert out.read_text() == "occupied"


def test_permissive_build_returns_the_same_report_it_persists(tmp_path, monkeypatch):
    """fix round 2, item 3: a strict=False build with real coverage gaps
    used to persist a validation_report.json carrying synthetic
    coverage_gap findings while returning the original report object,
    which did not - cli.py's summary/exit code read the returned report,
    not the file, so the two disagreed about whether anything failed.
    """
    import focuswatch_dataset.build as build_mod

    original = build_mod.validate_motion_table

    def dropped(df, recording_id, modality, nominal_hz):
        findings = original(df, recording_id, modality, nominal_hz)
        if modality == "watch":
            findings = [f for f in findings if f.check != "gyro_range"]
        return findings

    monkeypatch.setattr(build_mod, "validate_motion_table", dropped)
    out = tmp_path / "out"
    report = build_dataset(sources(tmp_path), out, strict=False)

    assert any(f.check == "coverage_gap" for f in report.findings)
    persisted = json.loads((out / "validation_report.json").read_text())
    assert [f.check for f in report.findings] == [p["check"] for p in persisted]
    assert [f.passed for f in report.findings] == [p["passed"] for p in persisted]


def test_interrupted_promotion_leftovers_are_announced_not_touched(tmp_path, capsys):
    """fix round 2, item 4: a promotion killed between its two renames
    leaves `out` correct (see _promote's docstring) but its `.build-*`/
    `.replaced-*` siblings unrecoverable-by-inspection unless something
    names them. Neither must be touched - only reported.
    """
    out = tmp_path / "out"
    leftover_build = tmp_path / f".{out.name}.build-deadbeef"
    leftover_backup = tmp_path / f".{out.name}.replaced-deadbeef"
    leftover_build.mkdir()
    (leftover_build / "marker.txt").write_text("unpromoted staging content")
    leftover_backup.mkdir()
    (leftover_backup / "marker.txt").write_text("previous good build")

    build_dataset(sources(tmp_path), out)

    printed = capsys.readouterr().out
    assert str(leftover_build) in printed
    assert str(leftover_backup) in printed
    # Untouched: still present, content unchanged, and the real build still
    # landed correctly at `out` alongside them.
    assert (leftover_build / "marker.txt").read_text() == "unpromoted staging content"
    assert (leftover_backup / "marker.txt").read_text() == "previous good build"
    assert (out / "sessions.parquet").exists()


def test_unlistable_parent_costs_the_notice_and_not_the_build(tmp_path, monkeypatch):
    """The leftover notice is a courtesy and must never fail a build. Listing
    the parent can raise OSError, which is not a RuntimeError and would reach
    the user as a traceback past the CLI's handler.
    """
    out = tmp_path / "out"
    real_iterdir = Path.iterdir

    def refuse_parent(self):
        if self == tmp_path:
            raise PermissionError(13, "Permission denied", str(self))
        return real_iterdir(self)

    monkeypatch.setattr(Path, "iterdir", refuse_parent)
    build_dataset(sources(tmp_path), out)
    assert (out / "sessions.parquet").exists()


# --- redact_bundle actually has a call site --------------------------------

def test_redaction_actually_blanks_the_written_pen_table(tmp_path):
    out = tmp_path / "out"
    build_dataset(sources(tmp_path), out, policy=RedactionPolicy.ALL_XY)
    manifest = load_manifest(out)
    pen_rows = manifest.loc[manifest["has_pen"]]
    assert len(pen_rows) > 0
    for rid in pen_rows["recording_id"]:
        pen = read_table(out / "pen" / f"{rid}.parquet")
        assert pen[["x", "y"]].isna().all().all(), rid


def test_none_policy_leaves_pen_xy_untouched(tmp_path):
    out = tmp_path / "out"
    build_dataset(sources(tmp_path), out, policy=RedactionPolicy.NONE)
    manifest = load_manifest(out)
    pen_rows = manifest.loc[manifest["has_pen"]]
    assert len(pen_rows) > 0
    for rid in pen_rows["recording_id"]:
        pen = read_table(out / "pen" / f"{rid}.parquet")
        assert pen[["x", "y"]].notna().any().any(), rid


# --- CLI: fw report (fix round 1, item 4) ----------------------------------

def test_cli_report(tmp_path, capsys):
    src = sources(tmp_path)
    out = tmp_path / "out"
    args = ["build", "--out", str(out)]
    for name, path in src.items():
        args += ["--source", f"{name}={path}"]
    assert main(args) == 0
    capsys.readouterr()  # discard the build command's own output

    assert main(["report", "--dataset", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "recordings" in printed
    assert "participants" in printed
    for cohort in ("ML4SCS", "ETH", "AIRPODS"):
        assert cohort in printed


def test_cli_build_prints_a_message_and_returns_1_on_failure(tmp_path, capsys):
    from focuswatch_dataset import schema as S

    src = sources(tmp_path)
    out = tmp_path / "out"
    args = ["build", "--out", str(out)]
    for name, path in src.items():
        args += ["--source", f"{name}={path}"]

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(S, "ACCEL_USER_BAND", (0.9, 1.1))
        rc = main(args)
    assert rc == 1
    printed = capsys.readouterr().out
    assert "build failed" in printed
