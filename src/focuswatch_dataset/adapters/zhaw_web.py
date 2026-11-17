"""Shared handling of the ZHAW web app's event payloads.

Both ZHAW pipelines export the same `session_start` payload — the watch-app export
carries it as a JSON string in `events.csv`, SensorLogger as a nested object in
the session JSON — so the redaction rule and the handedness lookup live here
once. Applying them in only one adapter is what let `user_agent` and `screen`
reach the published watch-app markers after the SensorLogger ones were cleaned.
"""
from __future__ import annotations

import json

from .. import schema as S


# The web app's participant_id field holds a session code, not a person: nine
# codes came from two people. The code -> person table is the ZHAW recording
# team's statement of 2026-11-14, corrected on 2026-11-16 (E3 moved to P01);
# it cannot be derived from the exports. The correction is consistent with
# the SensorLogger metadata: E3 is the only session recorded on a different
# phone, and it ran in parallel with E1 on the same day.
_PARTICIPANT_BY_SESSION_CODE = {
    "T6": "ZHAW-P01", "T7": "ZHAW-P01", "E3": "ZHAW-P01",
    "E1": "ZHAW-P02", "E2": "ZHAW-P02", "S3_SENSOR_LOGGER": "ZHAW-P02",
    "T8": "ZHAW-P02", "T9": "ZHAW-P02", "T10": "ZHAW-P02",
}


def participant_for_session_code(code: str) -> str:
    """Return the published participant id for a ZHAW session code."""
    try:
        return _PARTICIPANT_BY_SESSION_CODE[code]
    except KeyError:
        raise ValueError(
            f"ZHAW session code {code!r} has no known participant; "
            "add it to _PARTICIPANT_BY_SESSION_CODE in adapters/zhaw_web.py"
        ) from None


def redact_payload(payload: dict) -> tuple[dict, int]:
    """Keep only allow-listed keys; return the survivors and the drop count.

    An allow-list, not a deny-list: a deny-list fails open, and the next export
    generation adding a field would publish it. The count is reported so a new
    field is visible rather than silently discarded.
    """
    allowed = {k: v for k, v in payload.items() if k in S.MARKER_PAYLOAD_ALLOWED_KEYS}
    return allowed, len(payload) - len(allowed)


def redact_payload_json(text: object) -> tuple[str, int]:
    """`redact_payload` for a payload that arrives as a JSON string.

    A value that is not an object — absent, empty, or a scalar — has no keys to
    filter and passes through as an empty payload rather than raising: the
    column is optional in the source schema.
    """
    if not isinstance(text, str) or not text.strip():
        return "", 0
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return "", 0
    if not isinstance(payload, dict):
        return "", 0
    allowed, dropped = redact_payload(payload)
    return json.dumps(allowed), dropped


def handedness_from_payload(payload: dict) -> str:
    """Read `handedness`, validated against the closed vocabulary.

    Defaults to "unknown" when the source is silent, but rejects a value it does
    not recognise instead of publishing it — an unvalidated string here would
    become a manifest column a reuser groups by.
    """
    value = payload.get("handedness")
    if value is None:
        return "unknown"
    value = str(value).strip().lower()
    if value not in S.HANDEDNESS_VALUES:
        raise ValueError(
            f"unrecognised handedness value {value!r}; expected one of {S.HANDEDNESS_VALUES}"
        )
    return value


def pen_model_from_payload(payload: dict) -> str:
    """Map `pen_name`, the pen's Bluetooth name, to the pen model it stands for.

    A null or empty name (logged when no pen was connected at session start)
    is "unknown"; a name not seen before fails rather than being published.
    """
    value = payload.get("pen_name")
    if value is None or not str(value).strip():
        return S.DEVICE_MODEL_UNKNOWN
    value = str(value).strip()
    if value not in S.ZHAW_PEN_MODEL_BY_NAME:
        raise ValueError(
            f"unrecognised pen_name {value!r}; expected one of {tuple(S.ZHAW_PEN_MODEL_BY_NAME)}"
        )
    return S.ZHAW_PEN_MODEL_BY_NAME[value]
