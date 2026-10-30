"""Shared handling of the ZHAW web app's event payloads.

Both ZHAW pipelines export the same `session_start` payload — the Ege export
carries it as a JSON string in `events.csv`, SensorLogger as a nested object in
the session JSON — so the redaction rule and the handedness lookup live here
once. Applying them in only one adapter is what let `user_agent` and `screen`
reach the published Ege markers after the SensorLogger ones were cleaned.
"""
from __future__ import annotations

import json

from .. import schema as S


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
