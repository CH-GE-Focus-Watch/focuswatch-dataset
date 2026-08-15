import hashlib
import json

import pandas as pd
import pytest

from focuswatch_dataset.build import build_dataset
from focuswatch_dataset.cli import main
from focuswatch_dataset.load import load_manifest
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
    # Why: a build that half-writes and exits zero must be impossible - the
    # directory may exist (mkdir happens up front) but must carry none of
    # the publishable artefacts a caller would mistake for a finished build.
    for f in ("sessions.parquet", "sessions.csv", "channels.parquet",
              "datapackage.json", "data_dictionary.md", "validation_report.json"):
        assert not (out / f).exists(), f
    assert not any(out.rglob("*.parquet"))


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
