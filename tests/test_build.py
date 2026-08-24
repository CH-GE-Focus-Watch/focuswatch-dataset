import hashlib
import json
import re
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

from focuswatch_dataset.build import build_dataset
from focuswatch_dataset.cli import main
from focuswatch_dataset.load import load_manifest
from focuswatch_dataset.manifest import check_manifest_consistency
from focuswatch_dataset.redact import RedactionPolicy
from focuswatch_dataset.write import read_table, write_table
from focuswatch_dataset.adapters.base import RecordingBundle, RecordingRef

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


def test_build_refuses_a_requested_source_that_discovers_no_recordings(tmp_path):
    """An explicitly requested source must not quietly yield an empty archive."""
    empty = tmp_path / "empty"
    empty.mkdir()

    with pytest.raises(RuntimeError, match="airpods.*no recordings"):
        build_dataset({"airpods": empty}, tmp_path / "out")


def test_build_refuses_duplicate_recording_ids_before_writing(tmp_path, monkeypatch):
    """Duplicate IDs must fail before validation, manifest construction, or staging."""
    import focuswatch_dataset.build as build_mod

    duplicate = RecordingBundle(
        RecordingRef("ML4SCS-S001", "P", "ML4SCS", "ml4scs", Path(".")), {}, {}
    )
    monkeypatch.setattr(build_mod, "_load_bundle", lambda name, root: [duplicate, duplicate])

    with pytest.raises(RuntimeError, match="duplicate recording_id"):
        build_dataset({"ml4scs": tmp_path}, tmp_path / "out")

    assert not (tmp_path / "out").exists()


def test_build_produces_all_expected_artefacts(tmp_path):
    out = tmp_path / "out"
    build_dataset(sources(tmp_path), out)
    for f in ("sessions.parquet", "sessions.csv", "channels.parquet",
              "datapackage.json", "data_dictionary.md", "validation_report.json",
              "README.md", "LICENSE"):
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


def test_delta_is_never_applied_or_published(tmp_path):
    """I7: delta_applied had no test at all - flipping manifest.py's default
    to True left 213/213 green. Folds in C2's other D5 guarantee: pen_delta_s
    (the estimated offset itself) must stay NaN for every row, not just the
    estimated_delta cohort - publishing it risks irreparable mislabeling.
    """
    out = tmp_path / "out"
    build_dataset(sources(tmp_path), out)
    manifest = load_manifest(out)
    assert not manifest["delta_applied"].any()
    assert manifest["pen_delta_s"].isna().all()


def test_strict_build_aborts_on_a_physical_failure(tmp_path, monkeypatch):
    from focuswatch_dataset import schema as S
    src = sources(tmp_path)
    # Force every acceleration median into the forbidden gap.
    monkeypatch.setattr(S, "ACCEL_USER_BAND", (0.9, 1.1))
    with pytest.raises(RuntimeError, match="validation failed"):
        build_dataset(src, tmp_path / "out", strict=True)


# --- I2: a failed strict build must not write into `out` -----------------

def test_strict_abort_does_not_overwrite_a_preexisting_good_bundles_report(tmp_path):
    """Before this fix, build.py wrote the abort's validation_report.json
    straight into `out` - overwriting a previous good build's report with
    failures that describe a validation run over different (mutated-band)
    data, not the bundle sitting beside it on disk.
    """
    from focuswatch_dataset import schema as S

    out = tmp_path / "out"
    src = sources(tmp_path)
    build_dataset(src, out)
    good_report = (out / "validation_report.json").read_text()
    good_files = {p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()}

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(S, "ACCEL_USER_BAND", (0.9, 1.1))
        with pytest.raises(RuntimeError, match="validation failed"):
            build_dataset(src, out, strict=True)

    assert (out / "validation_report.json").read_text() == good_report
    assert {p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()} == good_files


def test_strict_abort_on_a_fresh_out_does_not_block_the_retry(tmp_path):
    """The old bug's second half: `out.mkdir(...)` plus a bare
    validation_report.json (neither `sessions.parquet` nor
    `datapackage.json`) left a directory `_refuse_foreign_directory` then
    refused to build into on the very next attempt.
    """
    from focuswatch_dataset import schema as S

    out = tmp_path / "out"
    src = sources(tmp_path)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(S, "ACCEL_USER_BAND", (0.9, 1.1))
        with pytest.raises(RuntimeError, match="validation failed"):
            build_dataset(src, out, strict=True)

    assert not out.exists()
    # A corrected retry must not be refused as a "foreign" directory.
    build_dataset(src, out, strict=True)
    assert (out / "sessions.parquet").exists()


def test_strict_abort_report_lands_in_a_sibling_directory_named_in_the_error(tmp_path):
    from focuswatch_dataset import schema as S

    out = tmp_path / "out"
    src = sources(tmp_path)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(S, "ACCEL_USER_BAND", (0.9, 1.1))
        with pytest.raises(RuntimeError, match="validation failed") as excinfo:
            build_dataset(src, out, strict=True)

    message = str(excinfo.value)
    failed_dirs = [p for p in tmp_path.iterdir()
                  if p.is_dir() and p.name.startswith(f".{out.name}.failed-")]
    assert len(failed_dirs) == 1, failed_dirs
    report_path = failed_dirs[0] / "validation_report.json"
    assert report_path.exists()
    assert str(report_path) in message


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


def test_cli_validate_rejects_a_table_with_the_wrong_embedded_recording_id(tmp_path):
    """The Parquet footer is part of the published archive contract."""
    out = tmp_path / "out"
    build_dataset(sources(tmp_path), out)
    path = next((out / "watch").glob("*.parquet"))
    write_table(read_table(path), path, "WRONG-ID")

    assert main(["validate", "--dataset", str(out)]) == 1


def test_cli_validate_rejects_a_present_but_corrupt_motion_table(tmp_path):
    """A readable historical report cannot mask corrupt published data."""
    out = tmp_path / "out"
    build_dataset(sources(tmp_path), out)
    path = next((out / "watch").glob("*.parquet"))
    path.write_bytes(b"not parquet")

    assert main(["validate", "--dataset", str(out)]) == 1


def test_cli_validate_rejects_a_table_with_the_wrong_embedded_schema_version(tmp_path):
    """Footer schema metadata must agree with the published manifest contract."""
    out = tmp_path / "out"
    build_dataset(sources(tmp_path), out)
    path = next((out / "watch").glob("*.parquet"))
    table = pq.read_table(path)
    metadata = dict(table.schema.metadata or {})
    metadata[b"schema_version"] = b"WRONG-VERSION"
    pq.write_table(table.replace_schema_metadata(metadata), path)

    assert main(["validate", "--dataset", str(out)]) == 1


def test_cli_validate_rejects_duplicate_manifest_recording_ids(tmp_path):
    """A duplicated ID makes every manifest-to-table mapping ambiguous."""
    out = tmp_path / "out"
    build_dataset(sources(tmp_path), out)
    manifest = load_manifest(out)
    pd.concat([manifest, manifest.iloc[[0]]], ignore_index=True).to_parquet(
        out / "sessions.parquet", index=False
    )

    assert main(["validate", "--dataset", str(out)]) == 1


def test_cli_validate_rechecks_the_stored_table_physics(tmp_path):
    """The archive table, rather than validation_report.json, is authoritative."""
    out = tmp_path / "out"
    build_dataset(sources(tmp_path), out)
    path = next((out / "watch").glob("*.parquet"))
    frame = read_table(path)
    frame.loc[1, "t_ns"] = frame.loc[0, "t_ns"] - 1
    write_table(frame, path, path.stem)

    assert main(["validate", "--dataset", str(out)]) == 1


def test_cli_validate_requires_the_complete_published_manifest_contract(tmp_path):
    """An omitted published manifest fact cannot be treated as adapter-only metadata."""
    out = tmp_path / "out"
    build_dataset(sources(tmp_path), out)
    load_manifest(out).drop(columns=["issue_codes"]).to_parquet(
        out / "sessions.parquet", index=False
    )

    assert main(["validate", "--dataset", str(out)]) == 1


def test_cli_validate_recomputes_table_derived_manifest_values(tmp_path):
    """Stored tables, not mutable summaries, define the derived manifest facts."""
    out = tmp_path / "out"
    build_dataset(sources(tmp_path), out)
    manifest = load_manifest(out)
    watch_row = manifest.index[manifest["has_watch"]][0]
    task_row = manifest.index[manifest["n_writing_tasks"].notna()][0]
    mutations = (
        (watch_row, "has_quaternion", False),
        (watch_row, "watch_hz_measured", manifest.loc[watch_row, "watch_hz_measured"] + 1),
        (watch_row, "n_samples_watch", manifest.loc[watch_row, "n_samples_watch"] + 1),
        (watch_row, "t_start_ns", manifest.loc[watch_row, "t_start_ns"] + 1),
        (watch_row, "duration_s", manifest.loc[watch_row, "duration_s"] + 1),
        (task_row, "n_writing_tasks", manifest.loc[task_row, "n_writing_tasks"] + 1),
        (watch_row, "issue_codes", "tampered"),
    )
    for row, column, value in mutations:
        tampered = manifest.copy()
        tampered.loc[row, column] = value
        tampered.to_parquet(out / "sessions.parquet", index=False)

        assert main(["validate", "--dataset", str(out)]) == 1, column


def test_cli_validate_rejects_small_published_float_changes(tmp_path):
    """Rounded archive summaries must not accept nearby tampered float values."""
    out = tmp_path / "out"
    src = sources(tmp_path)
    build_dataset(src, out)
    manifest = load_manifest(out)
    # The fixture's 619.96 s duration uses the same near-value mutation as
    # 8713.6 -> 8713.65: both fall inside numpy.isclose's default relative band.
    duration_row = manifest.index[manifest["recording_id"] == "AIRPODS-P1"][0]
    manifest.loc[duration_row, "duration_s"] = 619.965
    manifest.to_parquet(out / "sessions.parquet", index=False)
    assert main(["validate", "--dataset", str(out)]) == 1

    build_dataset(src, out)
    channels = pd.read_parquet(out / "channels.parquet")
    # The fixture's 100 Hz stream exercises the same relative-tolerance bug
    # as 50 -> 50.0004.
    rate_row = channels.index[channels["sample_rate_hz"] == 100][0]
    channels.loc[rate_row, "sample_rate_hz"] = 100.0004
    channels.to_parquet(out / "channels.parquet", index=False)
    assert main(["validate", "--dataset", str(out)]) == 1


def test_cli_validate_rejects_orphan_modality_tables(tmp_path):
    """Every archived modality file must have exactly one declared manifest key."""
    out = tmp_path / "out"
    build_dataset(sources(tmp_path), out)
    source = next((out / "watch").glob("*.parquet"))
    (out / "watch" / "ORPHAN.parquet").write_bytes(source.read_bytes())

    assert main(["validate", "--dataset", str(out)]) == 1


def test_cli_validate_rejects_channels_for_unknown_or_undeclared_tables(tmp_path):
    """Channel descriptors cannot introduce an archive table key on their own."""
    out = tmp_path / "out"
    build_dataset(sources(tmp_path), out)
    channels = pd.read_parquet(out / "channels.parquet")
    template = channels.iloc[0].copy()
    mutations = (
        ("recording_id", "UNKNOWN-ID"),
        ("modality", "watch_rawaccel"),
    )
    for column, value in mutations:
        tampered = channels.copy()
        row = template.copy()
        row[column] = value
        tampered.loc[len(tampered)] = row
        tampered.to_parquet(out / "channels.parquet", index=False)

        assert main(["validate", "--dataset", str(out)]) == 1, f"{column}={value}"


def test_cli_validate_requires_and_rechecks_channel_descriptors(tmp_path):
    """Published descriptor values are validated wherever archive data determines them."""
    out = tmp_path / "out"
    src = sources(tmp_path)
    build_dataset(src, out)
    channels = pd.read_parquet(out / "channels.parquet")
    channels.drop(columns=["time_domain"]).to_parquet(out / "channels.parquet", index=False)
    assert main(["validate", "--dataset", str(out)]) == 1

    build_dataset(src, out)
    channels = pd.read_parquet(out / "channels.parquet")
    row = channels.index[channels["column"] == "t_ns"][0]
    channels.loc[row, "quantity"] = "tampered"
    channels.to_parquet(out / "channels.parquet", index=False)
    assert main(["validate", "--dataset", str(out)]) == 1


# --- Coverage-gate: check_coverage must actually gate the build -----------

def test_missing_required_check_aborts_the_build(tmp_path, monkeypatch):
    """A recording that never produced a required check must abort a strict
    build, not merely have the gap reported alongside a green exit.

    Drops the gyro_range finding for every watch table (has_watch is true
    for the ml4scs recording, so check_coverage requires it) without
    touching any other check - report.failed stays empty, so only
    check_coverage's own gap-detection can be responsible for the abort.
    """
    import focuswatch_dataset.validate as validate_mod

    original = validate_mod.validate_motion_table

    def dropped(df, recording_id, modality, nominal_hz):
        findings = original(df, recording_id, modality, nominal_hz)
        if modality == "watch":
            findings = [f for f in findings if f.check != "gyro_range"]
        return findings

    monkeypatch.setattr(validate_mod, "validate_motion_table", dropped)
    with pytest.raises(RuntimeError, match="coverage gap"):
        build_dataset(sources(tmp_path), tmp_path / "out", strict=True)


def test_coverage_gap_abort_persists_the_full_gap_list_in_the_report(tmp_path, monkeypatch):
    """fix round 1, item 2: before this fix, validation_report.json on a
    coverage-gap abort held only the physical findings (which can all be
    `passed: true`) - the actual reason for the abort lived solely in the
    RuntimeError message, truncated to five gaps and never written to disk.
    """
    import focuswatch_dataset.validate as validate_mod

    original = validate_mod.validate_motion_table

    def dropped(df, recording_id, modality, nominal_hz):
        findings = original(df, recording_id, modality, nominal_hz)
        if modality == "watch":
            findings = [f for f in findings if f.check != "gyro_range"]
        return findings

    monkeypatch.setattr(validate_mod, "validate_motion_table", dropped)
    out = tmp_path / "out"
    with pytest.raises(RuntimeError, match="coverage gap"):
        build_dataset(sources(tmp_path), out, strict=True)

    # Why (I2): the abort report no longer lands in `out` itself - see
    # test_strict_abort_report_lands_in_a_sibling_directory_named_in_the_error.
    failed_dirs = [p for p in tmp_path.iterdir()
                  if p.is_dir() and p.name.startswith(f".{out.name}.failed-")]
    assert len(failed_dirs) == 1, failed_dirs
    findings = json.loads((failed_dirs[0] / "validation_report.json").read_text())
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
    import focuswatch_dataset.validate as validate_mod

    original = validate_mod.validate_motion_table

    def dropped(df, recording_id, modality, nominal_hz):
        findings = original(df, recording_id, modality, nominal_hz)
        if modality == "watch":
            findings = [f for f in findings if f.check != "gyro_range"]
        return findings

    monkeypatch.setattr(validate_mod, "validate_motion_table", dropped)
    out = tmp_path / "out"
    report = build_dataset(sources(tmp_path), out, strict=False)

    assert any(f.check == "coverage_gap" for f in report.findings)
    persisted = json.loads((out / "validation_report.json").read_text())
    assert [f.check for f in report.findings] == [p["check"] for p in persisted]
    assert [f.passed for f in report.findings] == [p["passed"] for p in persisted]


# --- M1: strict=False must not silently discard manifest inconsistencies --

def test_permissive_build_surfaces_manifest_inconsistencies_as_findings(tmp_path, monkeypatch):
    """`problems = check_manifest_consistency(...)` used to be computed and
    then discarded on the strict=False path - neither raised, appended to
    the report, nor printed, contradicting the same "returned report must
    not diverge from disk" principle the coverage-gap fix established.
    """
    import focuswatch_dataset.build as build_mod

    def fake_problems(manifest, staging):
        return ["watch/FAKE-001.parquet declared but missing"]

    monkeypatch.setattr(build_mod, "check_manifest_consistency", fake_problems)
    out = tmp_path / "out"
    report = build_dataset(sources(tmp_path), out, strict=False)

    manifest_findings = [f for f in report.findings if f.check == "manifest_consistency"]
    assert manifest_findings, "manifest-consistency problems must surface as findings"
    assert all(not f.passed for f in manifest_findings)
    assert any(f.recording_id == "FAKE-001" for f in manifest_findings)

    persisted = json.loads((out / "validation_report.json").read_text())
    assert [f.check for f in report.findings] == [p["check"] for p in persisted]
    assert [f.passed for f in report.findings] == [p["passed"] for p in persisted]


def test_strict_build_still_raises_on_a_manifest_inconsistency(tmp_path, monkeypatch):
    """The strict=True branch must be unaffected by the M1 fix - it still
    raises before any of the permissive-path finding-collection code runs.
    """
    import focuswatch_dataset.build as build_mod

    def fake_problems(manifest, staging):
        return ["watch/FAKE-001.parquet declared but missing"]

    monkeypatch.setattr(build_mod, "check_manifest_consistency", fake_problems)
    with pytest.raises(RuntimeError, match="manifest inconsistent"):
        build_dataset(sources(tmp_path), tmp_path / "out", strict=True)


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


# --- I11: build_git_sha resolves against the package's own repo -----------

def test_git_sha_resolves_against_package_location_not_cwd(tmp_path, monkeypatch):
    """The pre-fix `_git_sha()` shelled `git rev-parse HEAD` with no `cwd`, so
    it read whatever repository the CALLER happened to be sitting in (or none
    at all) instead of this package's own checkout. Chdir to a bare, non-git
    tmp_path and confirm the SHA is still this repo's real SHA, not "".
    """
    from focuswatch_dataset import build
    monkeypatch.chdir(tmp_path)
    sha = build._git_sha()
    core = sha.removesuffix("-dirty")
    assert len(core) == 40 and all(c in "0123456789abcdef" for c in core)


def test_git_sha_falls_back_to_the_installed_version_not_an_empty_string(monkeypatch):
    """A checkout with no .git at all (e.g. installed from a wheel) must still
    publish a real, checkable provenance fact - not "", which is
    indistinguishable from "the lookup ran and found nothing".
    """
    from focuswatch_dataset import build

    def _raise(*a, **k):
        raise FileNotFoundError("git not found")

    monkeypatch.setattr(build.subprocess, "check_output", _raise)
    assert build._git_sha() != ""


def test_git_sha_records_a_dirty_working_tree(monkeypatch):
    """Fix round C item 5: test_git_sha_resolves_against_package_location_not_cwd
    strips the `-dirty` suffix unconditionally (`sha.removesuffix("-dirty")`),
    so it cannot tell a clean SHA from a dirty one - a mutation dropping the
    suffix left 293/293 green. Force `git status --porcelain` to report a
    modification and require the suffix to survive; a bundle built from a
    modified tree must not publish a bare, falsely-clean SHA.
    """
    from focuswatch_dataset import build

    monkeypatch.setattr(build.subprocess, "check_output", lambda *a, **k: "abc123\n")
    monkeypatch.setattr(
        build.subprocess, "run",
        lambda *a, **k: build.subprocess.CompletedProcess(a, 0, stdout=" M some_file.py\n", stderr=""))
    assert build._git_sha() == "abc123-dirty"


def test_git_sha_omits_the_dirty_suffix_on_a_clean_tree(monkeypatch):
    """The companion case: an empty `git status --porcelain` must NOT get the
    suffix. Without this, a mutation that always appended `-dirty` would
    survive undetected alongside the test above.
    """
    from focuswatch_dataset import build

    monkeypatch.setattr(build.subprocess, "check_output", lambda *a, **k: "abc123\n")
    monkeypatch.setattr(
        build.subprocess, "run",
        lambda *a, **k: build.subprocess.CompletedProcess(a, 0, stdout="", stderr=""))
    assert build._git_sha() == "abc123"


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
    assert "participant identifiers" in printed
    for cohort in ("ML4SCS", "ETH", "AIRPODS"):
        assert cohort in printed


def test_cli_report_does_not_claim_more_distinct_people_than_it_can_show(tmp_path, capsys):
    """Fix round C item 4: `fw report` still printed "N participants" -
    exactly the claim commit 243ee07 removed from the README (`participant_id`
    is namespaced per cohort, so the count would be identical if all three
    cohorts had recorded the same people). It survived in this sibling
    surface; both must use the same wording so neither self-statement can
    drift from the other again.
    """
    out = tmp_path / "out"
    build_dataset(sources(tmp_path), out)
    readme = (out / "README.md").read_text()
    capsys.readouterr()

    assert main(["report", "--dataset", str(out)]) == 0
    printed = capsys.readouterr().out
    assert not re.search(r"\d+\s+participants\b", printed), (
        "fw report states a bare participant count"
    )
    assert re.search(r"\d+ participant identifiers", printed)
    assert re.search(r"\d+ participant identifiers", printed).group() in readme


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


# --- M11: `fw validate` / `fw report` on a missing or corrupt bundle ------

def test_cli_validate_on_a_missing_bundle_prints_a_message_not_a_traceback(tmp_path, capsys):
    rc = main(["validate", "--dataset", str(tmp_path / "does-not-exist")])
    assert rc == 1
    assert "validate failed" in capsys.readouterr().out


def test_cli_report_on_a_missing_bundle_prints_a_message_not_a_traceback(tmp_path, capsys):
    rc = main(["report", "--dataset", str(tmp_path / "does-not-exist")])
    assert rc == 1
    assert "report failed" in capsys.readouterr().out


def test_cli_validate_ignores_a_corrupt_historical_report(tmp_path, capsys):
    src = sources(tmp_path)
    out = tmp_path / "out"
    args = ["build", "--out", str(out)]
    for name, path in src.items():
        args += ["--source", f"{name}={path}"]
    assert main(args) == 0
    capsys.readouterr()  # discard the build command's own output

    (out / "validation_report.json").write_text("{not valid json")
    rc = main(["validate", "--dataset", str(out)])
    assert rc == 0
    assert "validate failed" not in capsys.readouterr().out


def test_annotation_tables_publish_one_schema_across_cohorts(tmp_path):
    """I15: markers arrived with two shapes - ML4SCS had seven columns, ETH nine
    (`src_payload`, `src_t_session_ms`), and `task_index` was float64 with NaN in
    one cohort and int64 with a -1 sentinel in the other. A reuser concatenating
    them got a ragged frame in which `task_index >= 0` filtered differently per
    cohort, and the package descriptor could state no single schema per
    modality. Removing either ML4SCS column left 41/41 green before this.

    Scoped to the annotation tables, and the exclusion is not a weakening. A
    motion table's columns are a statement about what the device measured: Ege's
    head stream carries no gyroscope and no gravity, declared through
    has_head_gyro and enforced by check_coverage. Flattening those to one schema
    would publish empty channels asserting sensors that never existed.
    """
    out = tmp_path / "out"
    build_dataset(sources(tmp_path), out)

    annotation = {"markers", "attention", "pen"}
    checked = 0
    for d in sorted(p for p in out.iterdir() if p.is_dir() and p.name in annotation):
        shapes = {}
        for f in sorted(d.glob("*.parquet")):
            df = read_table(f)
            shapes.setdefault(
                (tuple(df.columns), tuple(str(t) for t in df.dtypes)), []).append(f.name)
        assert len(shapes) == 1, (
            f"{d.name}: {len(shapes)} schemas across cohorts -> "
            + " | ".join(f"{cols} in {names}" for (cols, _), names in shapes.items())
        )
        checked += 1
    assert checked == len(annotation), f"only {checked} annotation modalities were built"


# --- I4: bundle README + a valid datapackage.json --------------------------

_FRICTIONLESS_NAME = re.compile(r"^[a-z0-9._-]+$")


def test_datapackage_resource_names_are_valid_frictionless_slugs(tmp_path):
    """Resource names were `f"{modality}/{f.stem}"` -> e.g. "watch/ETH-EGE-T6",
    invalid twice over: the "/" and the upper case. The Frictionless spec
    allows lowercase alphanumeric plus `.`, `-`, `_` only.
    """
    out = tmp_path / "out"
    build_dataset(sources(tmp_path), out)
    package = json.loads((out / "datapackage.json").read_text())
    names = [r["name"] for r in package["resources"]]
    assert names, "datapackage.json must list at least one resource"
    for name in names:
        assert _FRICTIONLESS_NAME.match(name), name
    assert len(names) == len(set(names)), "resource names must be unique"


def test_datapackage_lists_the_four_previously_missing_resources(tmp_path):
    """Before this fix, `datapackage.json` listed only sessions.csv and the
    per-recording parquet files - a Frictionless consumer never saw
    channels.parquet, the one place a column's unit/semantics is declared.
    """
    out = tmp_path / "out"
    build_dataset(sources(tmp_path), out)
    package = json.loads((out / "datapackage.json").read_text())
    paths = {r["path"] for r in package["resources"]}
    for expected in ("sessions.parquet", "channels.parquet",
                     "validation_report.json", "data_dictionary.md"):
        assert expected in paths, expected


def test_readme_is_generated_from_the_actual_build(tmp_path):
    """The README must be DERIVED from what this build actually produced, not
    a hand-maintained file list - assert every number in it is read straight
    back out of the manifest/disk it describes.
    """
    out = tmp_path / "out"
    build_dataset(sources(tmp_path), out)
    manifest = load_manifest(out)
    text = (out / "README.md").read_text()

    assert f"{len(manifest)} recordings" in text
    for cohort, n in manifest.groupby("cohort").size().items():
        assert f"| {cohort} | {n} |" in text
    for modality in ("watch", "pen", "markers"):
        n_files = len(list((out / modality).glob("*.parquet")))
        assert f"{n_files} files" in text
    # The recipe chapter must use the real, importable public API (I3), not
    # a stale snippet a reuser would copy-paste into an ImportError.
    assert "from focuswatch_dataset import load_manifest, load_recording, by_flags" in text


# --- I5: the unit vocabulary ships defined, and pen units are disambiguated -

def test_data_dictionary_ships_the_unit_vocabulary_glossary(tmp_path):
    """The `unit` column's non-physical values (category/ordinal/device_native/
    source_native/n/a) were previously defined only in docs/DESIGN.md (German,
    never shipped) and a schema.py code comment - a reuser with only the
    archive had no glossary at all.
    """
    out = tmp_path / "out"
    build_dataset(sources(tmp_path), out)
    text = (out / "data_dictionary.md").read_text()
    assert "## Unit vocabulary" in text
    for value in ("category", "ordinal", "device_native", "source_native", "n/a"):
        assert f"`{value}`" in text


def test_every_channels_unit_value_is_defined_in_the_glossary(tmp_path):
    """Item 2 (fix round C): the glossary above asserts a FIXED five-value set,
    which does not prove the built channels.parquet only ever contains those
    five - whole-branch-review-2.md finding 2 measured six MORE values
    (ncode_grid, moleskine_raw, webapp_raw, webapp_force, sl_webapp_raw,
    sl_webapp_force) reaching the `unit` column with no definition anywhere
    in the bundle. Derive the check from channels.parquet itself, the same
    place the cells come from, so the two cannot drift apart again.
    """
    from focuswatch_dataset import schema as S
    out = tmp_path / "out"
    build_dataset(sources(tmp_path), out)
    channels = read_table(out / "channels.parquet")
    text = (out / "data_dictionary.md").read_text()

    physical_units = set(S.UNITS.values()) | {"ns"}
    glossary_values = {"category", "ordinal", "device_native", "source_native", "n/a"}
    for value in channels["unit"].unique():
        if value in physical_units:
            continue
        assert value in glossary_values, (
            f"{value!r} is neither a physical unit nor in the glossary vocabulary"
        )
        assert f"`{value}`" in text


def test_every_overridden_time_domain_value_is_defined_in_the_glossary(tmp_path):
    """Item 1 (fix round C): the same discipline as the unit-vocabulary test
    above, applied to time_domain_by_column's override terms - derive the
    check from the built channels.parquet itself, not a second hardcoded
    list, so the "Time domain vocabulary" section cannot drift from what the
    adapters actually declared. A modality's own default clock name (e.g.
    watch_capture_clock) is open-ended by design and excluded; only the
    closed override vocabulary (schema.TIME_DOMAIN_*) must be defined.
    """
    from focuswatch_dataset import schema as S
    out = tmp_path / "out"
    build_dataset(sources(tmp_path), out)
    channels = read_table(out / "channels.parquet")
    text = (out / "data_dictionary.md").read_text()

    override_values = {
        v for k, v in vars(S).items()
        if k.startswith("TIME_DOMAIN_") and isinstance(v, str)
    }
    assert override_values, "no schema.TIME_DOMAIN_* constants found"
    for value in set(channels["time_domain"].unique()) & override_values:
        assert f"`{value}`" in text


def test_data_dictionary_disambiguates_pen_x_by_recording(tmp_path):
    """The channels summary lists `pen | x` once per distinct unit value with
    no way to tell which recording carries which - a per-recording table,
    read straight from the manifest's own pen_xy_unit/pen_pressure_scale
    columns, closes that gap.
    """
    out = tmp_path / "out"
    build_dataset(sources(tmp_path), out)
    manifest = load_manifest(out)
    text = (out / "data_dictionary.md").read_text()

    assert "## Pen coordinate/pressure units by recording" in text
    pen_rows = manifest.loc[manifest["has_pen"]]
    assert len(pen_rows) > 0
    for _, r in pen_rows.iterrows():
        assert f"| {r.recording_id} | {r.cohort} | {r.pen_xy_unit} | {r.pen_pressure_scale} |" in text


def test_data_dictionary_states_the_generation_a_vs_b_measured_facts(tmp_path):
    """Two measured facts from whole-branch-review-findings.md's open
    questions 4 and 6 must be quoted, not paraphrased or invented: the real
    Ege pen_events.csv header (tilt/pen-device clock exist only for
    generation-B ETH recordings), and the observed pen x/y range that shows
    the ~256x scale jump - with an explicit refusal to invent a millimetre
    conversion.
    """
    out = tmp_path / "out"
    build_dataset(sources(tmp_path), out)
    text = (out / "data_dictionary.md").read_text()

    assert "id, session_id, t_ms, t_session_ms, type, x, y, force, created_at" in text
    assert "generation-B" in text
    assert "E1_session3" in text and "5.69" in text and "16416.00" in text
    assert "pen_paper_info" in text
    assert "millimetre conversion" in text


def test_readme_does_not_claim_more_distinct_people_than_it_can_show(tmp_path):
    """`participant_id` is namespaced per cohort (`ML4SCS-P01`, `AIRPODS-P1`),
    so its cardinality would be unchanged if all three cohorts had recorded the
    same people. Calling that number "participants" hands a citing paper an N
    the data cannot support, which is exactly the class of false self-statement
    this bundle must not contain.
    """
    out = tmp_path / "out"
    build_dataset(sources(tmp_path), out)
    readme = (out / "README.md").read_text()

    assert "participant identifiers" in readme
    assert not re.search(r"\d+\s+participants\b", readme), (
        "README states a bare participant count"
    )
    assert "no" in readme and "mapping across cohorts" in readme
