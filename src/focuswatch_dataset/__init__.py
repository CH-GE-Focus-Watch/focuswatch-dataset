"""Public consumer API for a published focuswatch-dataset bundle.

See docs/DESIGN.md §7.1 for the intended usage. Everything here is a thin
re-export - the implementations live in `load.py`, `select.py` and
`convert.py`, which stay importable on their own for internal callers
(`build.py`, `cli.py`) that would otherwise pay for this package's `__init__`
to run before the bundle it is building even exists.
"""
from __future__ import annotations

from .convert import to_user_acceleration
from .load import load_channels, load_manifest, load_recording, load_recordings
from .select import by_flags

__all__ = [
    "load_manifest",
    "load_channels",
    "load_recording",
    "load_recordings",
    "by_flags",
    "to_user_acceleration",
]
