from pathlib import Path

import pandas as pd
import pytest

from focuswatch_dataset.adapters.base import (
    Adapter, RecordingBundle, RecordingRef, all_adapters, get_adapter, register,
)


class _Dummy:
    name = "dummy"

    def discover(self, root: Path) -> list[RecordingRef]:
        return [RecordingRef("D-1", "D-P1", "dummy", "dummy", root)]

    def load(self, ref: RecordingRef) -> RecordingBundle:
        return RecordingBundle(ref, {"watch": pd.DataFrame({"t_ns": [1]})}, {})


def test_dummy_satisfies_the_protocol():
    assert isinstance(_Dummy(), Adapter)


def test_register_and_retrieve(tmp_path):
    register(_Dummy())
    assert get_adapter("dummy").discover(tmp_path)[0].recording_id == "D-1"
    assert "dummy" in {a.name for a in all_adapters()}


def test_bundle_rejects_unknown_modality():
    ref = RecordingRef("D-1", "D-P1", "dummy", "dummy", Path("."))
    with pytest.raises(ValueError, match="unknown modality"):
        RecordingBundle(ref, {"telepathy": pd.DataFrame()}, {})
