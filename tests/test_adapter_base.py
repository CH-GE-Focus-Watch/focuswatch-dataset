from pathlib import Path

import pandas as pd
import pytest

from focuswatch_dataset.adapters import base
from focuswatch_dataset.adapters.base import (
    Adapter, RecordingBundle, RecordingRef, all_adapters, get_adapter, register,
)


@pytest.fixture(autouse=True)
def _isolated_registry():
    # Why: register() mutates a module-level dict; without this, adapters
    # registered by one test leak into every later test in the same session.
    snapshot = dict(base._REGISTRY)
    yield
    base._REGISTRY.clear()
    base._REGISTRY.update(snapshot)


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


def test_pin_isolation_step1_registers_a_probe():
    register(_Dummy())
    assert "dummy" in {a.name for a in all_adapters()}


def test_pin_isolation_step2_registry_does_not_leak():
    assert "dummy" not in {a.name for a in all_adapters()}
