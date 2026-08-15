import pytest

from focuswatch_dataset.adapters.eth_web import handedness_from_payload


def test_handedness_defaults_to_unknown_when_the_source_is_silent():
    assert handedness_from_payload({}) == "unknown"


def test_handedness_is_normalised_to_lowercase():
    assert handedness_from_payload({"handedness": " Right "}) == "right"


def test_handedness_rejects_an_unrecognised_value():
    """Fix round C item 6 audit: this raise (adapters/eth_web.py) had no
    test at all - a source payload with a stray handedness value (typo, a new
    app option) would otherwise publish an unvalidated string into a manifest
    column reusers group by, instead of failing loudly.
    """
    with pytest.raises(ValueError, match="unrecognised handedness value"):
        handedness_from_payload({"handedness": "ambidextrous"})
