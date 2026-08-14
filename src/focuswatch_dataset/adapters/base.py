"""Adapter contract. Each source pipeline implements discover() and load()."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

import pandas as pd

from .. import schema as S


@dataclass(frozen=True)
class RecordingRef:
    recording_id: str
    participant_id: str
    cohort: str
    pipeline: str
    path: Path


@dataclass
class RecordingBundle:
    ref: RecordingRef
    tables: dict[str, pd.DataFrame]
    meta: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        unknown = set(self.tables) - set(S.MODALITIES)
        if unknown:
            raise ValueError(f"unknown modality: {sorted(unknown)}")


@runtime_checkable
class Adapter(Protocol):
    name: str

    def discover(self, root: Path) -> list[RecordingRef]: ...
    def load(self, ref: RecordingRef) -> RecordingBundle: ...


_REGISTRY: dict[str, Adapter] = {}


def register(adapter: Adapter) -> None:
    _REGISTRY[adapter.name] = adapter


def get_adapter(name: str) -> Adapter:
    return _REGISTRY[name]


def all_adapters() -> list[Adapter]:
    return list(_REGISTRY.values())
