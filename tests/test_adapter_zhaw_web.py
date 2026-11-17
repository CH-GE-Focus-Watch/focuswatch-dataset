import pytest

from focuswatch_dataset.adapters.zhaw_web import handedness_from_payload, participant_for_session_code


def test_handedness_defaults_to_unknown_when_the_source_is_silent():
    assert handedness_from_payload({}) == "unknown"


def test_handedness_is_normalised_to_lowercase():
    assert handedness_from_payload({"handedness": " Right "}) == "right"


def test_handedness_rejects_an_unrecognised_value():
    """This raise (adapters/zhaw_web.py) had no
    test at all - a source payload with a stray handedness value (typo, a new
    app option) would otherwise publish an unvalidated string into a manifest
    column reusers group by, instead of failing loudly.
    """
    with pytest.raises(ValueError, match="unrecognised handedness value"):
        handedness_from_payload({"handedness": "ambidextrous"})


def test_session_codes_map_to_the_two_zhaw_participants():
    assert {participant_for_session_code(c) for c in ("T6", "T7", "E3")} == {"ZHAW-P01"}
    assert {participant_for_session_code(c) for c in (
        "E1", "E2", "S3_SENSOR_LOGGER", "T8", "T9", "T10")} == {"ZHAW-P02"}


def test_an_unknown_session_code_fails_the_build():
    with pytest.raises(ValueError, match="'T11' has no known participant"):
        participant_for_session_code("T11")
