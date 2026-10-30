"""Public consumer API for a focuswatch-dataset bundle.

This module is a thin re-export. Internal build modules import the underlying
implementations directly so a build does not require this public API to load.
"""
from __future__ import annotations

from .convert import to_user_acceleration
from .explore import ExplorerSelection, filter_manifest, selection_query
from .load import load_channels, load_manifest, load_recording, load_recordings
from .select import by_flags

__all__ = [
    "load_manifest",
    "load_channels",
    "load_recording",
    "load_recordings",
    "by_flags",
    "to_user_acceleration",
    "ExplorerSelection",
    "filter_manifest",
    "selection_query",
]
