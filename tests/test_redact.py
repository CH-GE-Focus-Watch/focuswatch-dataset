from pathlib import Path

import numpy as np
import pandas as pd

from focuswatch_dataset.adapters.base import RecordingBundle, RecordingRef
from focuswatch_dataset.manifest import build_manifest
from focuswatch_dataset.redact import RedactionPolicy, apply_redaction, redact_bundle


def pen():
    return pd.DataFrame({"t_ns": [10, 20, 30, 40], "dot_type": ["PEN_DOWN"] * 4,
                         "x": [1.0, 2.0, 3.0, 4.0], "y": [1.0, 2.0, 3.0, 4.0],
                         "pressure": [300.0] * 4})


def markers():
    return pd.DataFrame({
        "t_ns": [5, 25, 26, 45],
        "event": ["task_start", "task_end", "task_start", "task_end"],
        "task_id": ["abschreiben", "abschreiben", "free_writing", "free_writing"],
        "task_index": [0, 0, 1, 1], "task_category": ["writing"] * 4,
        "protocol_id": ["ml4scs_v2"] * 4,
    })


def test_none_is_the_default_and_changes_nothing():
    pd.testing.assert_frame_equal(apply_redaction(pen(), markers(), RedactionPolicy.NONE), pen())


def test_free_writing_xy_blanks_only_the_free_writing_block():
    out = apply_redaction(pen(), markers(), RedactionPolicy.FREE_WRITING_XY)
    assert out.loc[:1, ["x", "y"]].notna().all().all()
    assert out.loc[2:, ["x", "y"]].isna().all().all()
    # Non-positional channels survive: the label only needs dot_type and time.
    assert out["pressure"].notna().all()
    assert out["dot_type"].tolist() == ["PEN_DOWN"] * 4


def test_all_xy_blanks_everything_positional():
    out = apply_redaction(pen(), markers(), RedactionPolicy.ALL_XY)
    assert out[["x", "y"]].isna().all().all()


def test_free_writing_without_markers_falls_back_to_all_xy():
    out = apply_redaction(pen(), None, RedactionPolicy.FREE_WRITING_XY)
    assert out[["x", "y"]].isna().all().all()


def test_markers_without_task_structure_also_fall_back():
    # The ETH sources emit phase events with empty task ids. Matching nothing
    # would silently publish every coordinate under a redaction policy.
    structureless = pd.DataFrame({
        "t_ns": [5, 45], "event": ["session_start", "session_end"],
        "task_id": ["", ""], "task_index": [-1, -1],
        "task_category": ["", ""], "protocol_id": ["eth_ege_web"] * 2,
    })
    out = apply_redaction(pen(), structureless, RedactionPolicy.FREE_WRITING_XY)
    assert out[["x", "y"]].isna().all().all()


def test_row_count_never_changes():
    for p in RedactionPolicy:
        assert len(apply_redaction(pen(), markers(), p)) == 4


# --- redact_bundle: the apply-and-record seam a build pipeline calls -------
#
# apply_redaction is a pure table transform; nothing above ties its output to
# what a published manifest claims. redact_bundle is that seam - it applies
# the policy to a bundle's pen table and stamps meta["redaction_policy"] with
# the same policy, so build_manifest (which trusts meta for this field, see
# its docstring) cannot be fed a lie. The tests below go through the public
# RecordingBundle / build_manifest path, matching test_manifest.py's style.


def _bundle(**meta):
    return RecordingBundle(
        RecordingRef("COHORT-R1", "COHORT-P1", "COHORT", "pipeline", Path(".")),
        {"pen": pen(), "markers": markers()},
        meta,
    )


def test_redact_bundle_default_is_none_and_leaves_pen_untouched():
    out = redact_bundle(_bundle())
    pd.testing.assert_frame_equal(out.tables["pen"], pen())
    assert out.meta["redaction_policy"] == "none"


def test_redact_bundle_manifest_declares_the_requested_policy():
    """The manifest's redaction_policy must equal what redact_bundle was
    asked to apply - the declaration half of the provenance guarantee.
    """
    for policy in RedactionPolicy:
        out = redact_bundle(_bundle(), policy)
        row = build_manifest([out]).iloc[0]
        assert row["redaction_policy"] == policy.value


def test_redact_bundle_actually_blanks_coordinates_iff_a_policy_is_declared():
    """The dangerous direction, standalone: a manifest could declare a
    policy was applied while the coordinates were never touched. This must
    stay a standalone check - if it were ever folded back into the
    declaration test above, a future edit that trims that shared block
    could silently drop the only assertion catching that failure mode.
    """
    for policy in RedactionPolicy:
        out = redact_bundle(_bundle(), policy)
        any_blanked = out.tables["pen"][["x", "y"]].isna().any().any()
        if policy == RedactionPolicy.NONE:
            assert not any_blanked
        else:
            assert any_blanked


def test_redact_bundle_accepts_the_raw_string_form_of_a_policy():
    """StrEnum members compare equal to their plain string value, so a
    caller wired from a CLI flag or config value - which passes a plain str,
    not a RedactionPolicy member - must still get the correct behaviour.
    An `is` comparison in apply_redaction would silently miss this and fall
    through toward a weaker policy instead of applying the requested one.
    """
    out = redact_bundle(_bundle(), "all_xy")
    assert out.tables["pen"][["x", "y"]].isna().all().all()
    assert out.meta["redaction_policy"] == "all_xy"


def test_redact_bundle_without_a_pen_table_still_records_the_policy():
    b = RecordingBundle(
        RecordingRef("COHORT-R2", "COHORT-P1", "COHORT", "pipeline", Path(".")),
        {"markers": markers()}, {},
    )
    out = redact_bundle(b, RedactionPolicy.ALL_XY)
    assert "pen" not in out.tables
    assert out.meta["redaction_policy"] == "all_xy"
