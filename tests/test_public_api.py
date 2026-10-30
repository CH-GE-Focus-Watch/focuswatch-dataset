"""I3: the package's __init__.py must actually export the consumer API DESIGN
§7.1 advertises. Before this fix `from focuswatch_dataset import
load_manifest` raised ImportError - __init__.py was 0 bytes.
"""
from __future__ import annotations

import focuswatch_dataset as fw
from focuswatch_dataset import (
    ExplorerSelection,
    by_flags,
    filter_manifest,
    load_channels,
    load_manifest,
    load_recording,
    load_recordings,
    to_user_acceleration,
    selection_query,
)
from focuswatch_dataset.convert import to_user_acceleration as _to_user_acceleration
from focuswatch_dataset.load import (
    load_channels as _load_channels,
    load_manifest as _load_manifest,
    load_recording as _load_recording,
    load_recordings as _load_recordings,
)
from focuswatch_dataset.select import by_flags as _by_flags
from focuswatch_dataset.explore import (
    ExplorerSelection as _ExplorerSelection,
    filter_manifest as _filter_manifest,
    selection_query as _selection_query,
)


def test_top_level_import_exposes_every_advertised_symbol():
    """The advertised import must work, not merely a submodule
    import - a reuser who copies that snippet gets what it says, not
    ImportError."""
    assert fw.load_manifest is _load_manifest
    assert fw.load_channels is _load_channels
    assert fw.load_recording is _load_recording
    assert fw.load_recordings is _load_recordings
    assert fw.by_flags is _by_flags
    assert fw.to_user_acceleration is _to_user_acceleration
    assert fw.ExplorerSelection is _ExplorerSelection
    assert fw.filter_manifest is _filter_manifest
    assert fw.selection_query is _selection_query


def test_star_import_names_match_the_real_implementations():
    assert load_manifest is _load_manifest
    assert load_channels is _load_channels
    assert load_recording is _load_recording
    assert load_recordings is _load_recordings
    assert by_flags is _by_flags
    assert to_user_acceleration is _to_user_acceleration
    assert ExplorerSelection is _ExplorerSelection
    assert filter_manifest is _filter_manifest
    assert selection_query is _selection_query


def test_all_lists_exactly_the_public_surface():
    assert set(fw.__all__) == {
        "load_manifest", "load_channels", "load_recording", "load_recordings",
        "by_flags", "to_user_acceleration",
        "ExplorerSelection", "filter_manifest", "selection_query",
    }
