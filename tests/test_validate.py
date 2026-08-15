import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from focuswatch_dataset import schema as S
from focuswatch_dataset.time_axis import median_rate_hz
from focuswatch_dataset.validate import (
    ValidationReport, check_coverage, validate_motion_table, validate_pen_table,
    validate_recording,
)


def make_watch(n=2000, fs=100.0, accel_scale=0.04, gravity_scale=1.0, seed=0):
    rng = np.random.default_rng(seed)
    rots = Rotation.random(n, random_state=seed)
    q = rots.as_quat()
    grav = rots.inv().apply([0.0, 0.0, -1.0]) * gravity_scale
    return pd.DataFrame({
        "t_ns": np.arange(n, dtype=np.int64) * int(1e9 / fs),
        **dict(zip(S.COLUMNS[S.Quantity.ACCEL_USER], rng.normal(0, accel_scale, (n, 3)).T)),
        **dict(zip(S.COLUMNS[S.Quantity.GYRO], rng.normal(0, 0.15, (n, 3)).T)),
        **dict(zip(S.COLUMNS[S.Quantity.GRAVITY], grav.T)),
        **dict(zip(S.COLUMNS[S.Quantity.QUAT], q.T)),
    })


def failed_checks(findings):
    return {f.check for f in findings if not f.passed}


def test_clean_table_passes():
    assert failed_checks(validate_motion_table(make_watch(), "R1", "watch", 100.0)) == set()


def test_gravity_left_in_ms2_fails():
    df = make_watch(gravity_scale=S.G_TO_MS2)
    assert "gravity_norm" in failed_checks(validate_motion_table(df, "R1", "watch", 100.0))


def test_accel_in_the_forbidden_gap_fails_and_names_the_cause():
    # A constant 0.3 per axis gives a norm of 0.52, inside the (0.2, 0.9) gap.
    df = make_watch(accel_scale=0.0)
    df[list(S.COLUMNS[S.Quantity.ACCEL_USER])] = 0.3
    findings = validate_motion_table(df, "R1", "watch", 100.0)
    bad = next(f for f in findings if f.check == "accel_semantic_band")
    assert not bad.passed
    assert "user/total" in bad.expected


def test_accel_in_si_units_is_diagnosed_as_such():
    df = make_watch(accel_scale=0.0)
    df[list(S.COLUMNS[S.Quantity.ACCEL_USER])] = S.G_TO_MS2 / np.sqrt(3)
    bad = next(f for f in validate_motion_table(df, "R1", "watch", 100.0)
               if f.check == "accel_semantic_band")
    assert not bad.passed
    assert "m/s2" in bad.expected


def test_gyro_in_degrees_per_second_fails():
    df = make_watch()
    df[list(S.COLUMNS[S.Quantity.GYRO])] *= 180.0 / np.pi * 50
    assert "gyro_range" in failed_checks(validate_motion_table(df, "R1", "watch", 100.0))


def test_scalar_first_quaternion_fails_the_gravity_crosscheck():
    df = make_watch()
    q = df[list(S.COLUMNS[S.Quantity.QUAT])].to_numpy()
    df[list(S.COLUMNS[S.Quantity.QUAT])] = np.roll(q, 1, axis=1)
    assert "quat_gravity_agreement" in failed_checks(validate_motion_table(df, "R1", "watch", 100.0))


def test_rate_mismatch_fails():
    assert "sample_rate" in failed_checks(validate_motion_table(make_watch(fs=100.0), "R1", "watch", 50.0))


def test_non_monotonic_time_fails():
    df = make_watch()
    df.loc[10, "t_ns"] = 0
    assert "time_monotonic" in failed_checks(validate_motion_table(df, "R1", "watch", 100.0))


def test_pen_vocabulary_is_checked():
    df = pd.DataFrame({"t_ns": [1, 2], "dot_type": ["PEN_DOWN", "SCRIBBLE"],
                       "x": [1.0, 2.0], "y": [1.0, 2.0]})
    assert "dot_type_vocabulary" in failed_checks(validate_pen_table(df, "R1"))


def test_pen_framing_sentinel_is_allowed():
    df = pd.DataFrame({"t_ns": [1, 2], "dot_type": ["PEN_DOWN", "PEN_UP"],
                       "x": [-1.0, 2.0], "y": [-1.0, 2.0]})
    assert failed_checks(validate_pen_table(df, "R1")) == set()


def test_report_serialises_and_reports_failure():
    r = ValidationReport(validate_motion_table(make_watch(gravity_scale=S.G_TO_MS2), "R1", "watch", 100.0))
    assert r.failed
    assert '"check"' in r.to_json()


# --- The Ege case: quaternion present, gravity column absent -------------------

def make_ege_like(n=3000, seed=7, still_fraction=0.2, roll_quaternion=False):
    """Total acceleration plus quaternion, no gravity column, with quiet stretches."""
    rng = np.random.default_rng(seed)
    rots = Rotation.random(n, random_state=seed)
    q = rots.as_quat()
    grav = rots.inv().apply([0.0, 0.0, -1.0])
    gyro = rng.normal(0, 0.15, (n, 3))
    n_still = int(n * still_fraction)
    gyro[:n_still] = rng.normal(0, 0.001, (n_still, 3))
    total = grav + rng.normal(0, 0.005, (n, 3))
    stored_q = np.roll(q, 1, axis=1) if roll_quaternion else q
    return pd.DataFrame({
        "t_ns": np.arange(n, dtype=np.int64) * 10_000_000,
        **dict(zip(S.COLUMNS[S.Quantity.ACCEL_TOTAL], total.T)),
        **dict(zip(S.COLUMNS[S.Quantity.GYRO], gyro.T)),
        **dict(zip(S.COLUMNS[S.Quantity.QUAT], stored_q.T)),
    })


def test_still_window_quaternion_check_runs_without_a_gravity_column():
    findings = validate_motion_table(make_ege_like(), "ETH-EGE-T6", "watch", 100.0)
    checks = {f.check for f in findings}
    assert "quat_still_agreement" in checks
    assert failed_checks(findings) == set()


def test_still_window_check_catches_a_scalar_first_quaternion():
    findings = validate_motion_table(make_ege_like(roll_quaternion=True), "ETH-EGE-T6", "watch", 100.0)
    assert "quat_still_agreement" in failed_checks(findings)


def test_nan_quaternion_rows_are_excluded_not_poisoning():
    # ML4SCS quaternion capture is forward-only; older sessions carry empty columns.
    df = make_watch()
    df.loc[:200, list(S.COLUMNS[S.Quantity.QUAT])] = np.nan
    findings = validate_motion_table(df, "R1", "watch", 100.0)
    agreement = next(f for f in findings if f.check == "quat_gravity_agreement")
    assert agreement.passed
    assert not np.isnan(float(agreement.observed))


def test_fully_missing_quaternion_reports_skipped_not_passed():
    df = make_watch()
    df[list(S.COLUMNS[S.Quantity.QUAT])] = np.nan
    checks = {f.check for f in validate_motion_table(df, "R1", "watch", 100.0)}
    assert "quat_gravity_agreement" not in checks


# --- Recording level -----------------------------------------------------------

def test_streams_that_do_not_overlap_fail():
    watch = make_watch(n=500)
    pen = pd.DataFrame({"t_ns": watch["t_ns"].max() + 10 ** 12 + np.arange(5, dtype=np.int64),
                        "dot_type": ["PEN_DOWN"] * 5, "x": 1.0, "y": 1.0})
    findings = validate_recording("R1", {"watch": watch, "pen": pen}, {})
    assert "streams_overlap" in failed_checks(findings)


def test_samples_far_before_the_session_start_fail():
    watch = make_watch(n=500)
    declared_start = int(watch["t_ns"].min())
    watch.loc[0, "t_ns"] = declared_start - 300 * 10 ** 9
    findings = validate_recording("R1", {"watch": watch}, {"session_start_ns": declared_start})
    assert "spill_guard" in failed_checks(findings)


def test_spill_guard_is_skipped_when_the_source_declares_no_start():
    # Not every source records a session start; the coverage matrix must not
    # demand a check the data cannot support.
    checks = {f.check for f in validate_recording("R1", {"watch": make_watch(n=500)}, {})}
    assert "spill_guard" not in checks


def test_declared_time_unit_must_match_the_magnitude():
    watch = make_watch(n=100)
    watch["t_ns"] = np.arange(100, dtype=np.int64)          # session-relative, not an epoch
    findings = validate_recording("R1", {"watch": watch}, {"time_domain": "backend_wall_clock"})
    assert "time_magnitude" in failed_checks(findings)


# --- Coverage matrix -----------------------------------------------------------

def test_coverage_accepts_a_complete_report():
    manifest = pd.DataFrame([{"recording_id": "R1", "has_watch": True, "has_quaternion": True,
                              "has_gravity": True, "has_headimu": False, "has_pen": False,
                              "has_watch_rawaccel": False, "has_markers": False,
                              "has_attention": False}])
    findings = validate_motion_table(make_watch(), "R1", "watch", 100.0)
    findings += validate_recording("R1", {"watch": make_watch()}, {})
    assert check_coverage(manifest, findings) == []


def test_coverage_rejects_a_silently_skipped_quaternion_check():
    manifest = pd.DataFrame([{"recording_id": "R1", "has_watch": True, "has_quaternion": True,
                              "has_gravity": True, "has_headimu": False, "has_pen": False,
                              "has_watch_rawaccel": False, "has_markers": False,
                              "has_attention": False}])
    # An adapter that dropped the quaternion columns produces no quat findings at all.
    df = make_watch().drop(columns=list(S.COLUMNS[S.Quantity.QUAT]))
    findings = validate_motion_table(df, "R1", "watch", 100.0)
    findings += validate_recording("R1", {"watch": df}, {})
    problems = check_coverage(manifest, findings)
    assert any("quat_norm" in p for p in problems)


def test_coverage_rejects_a_silently_skipped_gyro_check():
    manifest = pd.DataFrame([{"recording_id": "R1", "has_watch": True, "has_quaternion": True,
                              "has_gravity": True, "has_headimu": False, "has_pen": False,
                              "has_watch_rawaccel": False, "has_markers": False,
                              "has_attention": False}])
    # An adapter that dropped the gyro columns produces no gyro_range finding at all,
    # even though gravity and quaternion are still present.
    df = make_watch().drop(columns=list(S.COLUMNS[S.Quantity.GYRO]))
    findings = validate_motion_table(df, "R1", "watch", 100.0)
    findings += validate_recording("R1", {"watch": df}, {})
    problems = check_coverage(manifest, findings)
    assert any("gyro_range" in p for p in problems)


def test_coverage_requires_a_rawaccel_table_that_never_arrived():
    manifest = pd.DataFrame([{"recording_id": "R1", "has_watch": False, "has_quaternion": False,
                              "has_gravity": False, "has_headimu": False, "has_pen": False,
                              "has_watch_rawaccel": True, "has_markers": False,
                              "has_attention": False}])
    # No findings at all: an adapter that dropped the rawaccel table entirely.
    problems = check_coverage(manifest, [])
    assert any("time_monotonic" in p for p in problems)
    assert any("sample_rate" in p for p in problems)
    assert any("accel_semantic_band" in p for p in problems)
    assert any("gyro_range" in p for p in problems)
    assert any("time_magnitude" in p for p in problems)


def test_sample_rate_still_emitted_with_no_nominal_rate_declared():
    df = make_watch(fs=100.0)
    findings = validate_motion_table(df, "R1", "watch", None)
    f = next(x for x in findings if x.check == "sample_rate")
    measured = median_rate_hz(df["t_ns"].to_numpy(dtype=np.int64))
    assert f.passed
    assert f.observed == round(measured, 3)


def test_coverage_requires_time_magnitude_for_a_pen_only_recording():
    manifest = pd.DataFrame([{"recording_id": "R1", "has_watch": False, "has_quaternion": False,
                              "has_gravity": False, "has_headimu": False, "has_pen": True,
                              "has_watch_rawaccel": False, "has_markers": False,
                              "has_attention": False}])
    # An adapter that produced pen findings but never ran the cross-table
    # recording-level checks (e.g. crashed before calling validate_recording).
    df = pd.DataFrame({"t_ns": [1, 2], "dot_type": ["PEN_DOWN", "PEN_UP"], "x": [1.0, 2.0], "y": [1.0, 2.0]})
    findings = validate_pen_table(df, "R1")
    problems = check_coverage(manifest, findings)
    assert any("time_magnitude" in p for p in problems)


# --- Coverage matrix: per-modality scoping (fix-round-1 items 2/3) -------------
#
# check_coverage used to key its "which checks ran" set on recording_id alone,
# so one modality's finding (e.g. watch's own time_magnitude) could silently
# satisfy a requirement declared for a different, absent modality of the same
# recording (e.g. headimu). These pin the (recording_id, modality) scoping.

def test_coverage_time_magnitude_is_scoped_per_modality_not_per_recording():
    manifest = pd.DataFrame([{"recording_id": "R1", "has_watch": True, "has_quaternion": False,
                              "has_gravity": False, "has_headimu": True, "has_pen": False,
                              "has_watch_rawaccel": False, "has_markers": False,
                              "has_attention": False, "has_head_gravity": False,
                              "has_head_quaternion": False}])
    watch = make_watch()
    findings = validate_motion_table(watch, "R1", "watch", 100.0)
    # has_headimu is true but no headimu table ever reached validate_recording -
    # watch's own time_magnitude finding must not cover for it.
    findings += validate_recording("R1", {"watch": watch}, {})
    problems = check_coverage(manifest, findings)
    assert any("time_magnitude" in p and "headimu" in p for p in problems)


def test_coverage_requires_head_gravity_norm_specifically_on_headimu():
    manifest = pd.DataFrame([{"recording_id": "R1", "has_watch": True, "has_quaternion": False,
                              "has_gravity": False, "has_headimu": True, "has_pen": False,
                              "has_watch_rawaccel": False, "has_markers": False,
                              "has_attention": False, "has_head_gravity": True,
                              "has_head_quaternion": False}])
    # Watch's own gravity_norm finding (from make_watch's gravity column) must
    # not satisfy has_head_gravity - no headimu table was ever validated.
    watch = make_watch()
    findings = validate_motion_table(watch, "R1", "watch", 100.0)
    findings += validate_recording("R1", {"watch": watch}, {})
    problems = check_coverage(manifest, findings)
    assert any("gravity_norm" in p and "headimu" in p for p in problems)


def test_coverage_requires_head_quat_norm_specifically_on_headimu():
    manifest = pd.DataFrame([{"recording_id": "R1", "has_watch": True, "has_quaternion": True,
                              "has_gravity": True, "has_headimu": True, "has_pen": False,
                              "has_watch_rawaccel": False, "has_markers": False,
                              "has_attention": False, "has_head_gravity": False,
                              "has_head_quaternion": True}])
    watch = make_watch()
    findings = validate_motion_table(watch, "R1", "watch", 100.0)
    findings += validate_recording("R1", {"watch": watch}, {})
    problems = check_coverage(manifest, findings)
    assert any("quat_norm" in p and "headimu" in p for p in problems)
