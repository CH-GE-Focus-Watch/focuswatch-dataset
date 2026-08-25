from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
import pytest

from focuswatch_dataset import schema as S
from focuswatch_dataset.adapters.moleskine import MoleskineAdapter
from focuswatch_dataset.manifest import build_channels, build_manifest
from focuswatch_dataset.validate import validate_bundle


def write_fixture(root, *, draw_types=(17, 18, 18, 20), page_ids=None):
    start = pd.Timestamp("2024-08-22T14:12:07.869+02:00")
    n = 500
    times = pd.date_range(start, periods=n, freq="20ms")
    rng = np.random.default_rng(0)
    accel = np.column_stack((
        rng.normal(0, 0.02, n),
        rng.normal(0, 0.02, n),
        rng.normal(-1, 0.02, n),
    ))
    watch = pd.DataFrame({
        "loggingTime(txt)": times.astype(str),
        "accelerometerTimestamp_sinceReboot(s)": 62_608.0 + np.arange(n) / 50.0,
        "accelerometerAccelerationX(G)": accel[:, 0],
        "accelerometerAccelerationY(G)": accel[:, 1],
        "accelerometerAccelerationZ(G)": accel[:, 2],
    })
    sensorlog = root / "sensorlog"
    sensorlog.mkdir(parents=True)
    # The source label must never become a public recording or participant id.
    watch.to_csv(sensorlog / "2024-08-22_14_12_07_Apple Watch_Dani.csv", index=False)

    database_dir = root / "moleskine" / "backup-db"
    database_dir.mkdir(parents=True)
    database = database_dir / "note.db"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE Stroke (id INTEGER PRIMARY KEY, pageID INTEGER NOT NULL)")
        connection.execute(
            """CREATE TABLE draw_point (
                id INTEGER PRIMARY KEY, stokeID INTEGER NOT NULL, pageX REAL NOT NULL,
                pageY REAL NOT NULL, drawType INTEGER NOT NULL,
                penPressure INTEGER NOT NULL, date INTEGER NOT NULL
            )"""
        )
        pages = page_ids or [3] * len(draw_types)
        for stroke_id, page_id in enumerate(pages, start=1):
            connection.execute("INSERT INTO Stroke (id, pageID) VALUES (?, ?)",
                               (stroke_id, page_id))
        first_ms = start.tz_convert("UTC").value // 1_000_000 + 1_000
        connection.executemany(
            """INSERT INTO draw_point
               (id, stokeID, pageX, pageY, drawType, penPressure, date)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            [
                (i, i, 10.0 + i, 20.0 + i, draw_type, 40 + i, first_ms + i * 20)
                for i, draw_type in enumerate(draw_types, start=1)
            ],
        )
    return root


def test_discovery_uses_a_safe_utc_id_and_does_not_publish_source_names(tmp_path):
    write_fixture(tmp_path)
    [ref] = MoleskineAdapter().discover(tmp_path)

    assert ref.recording_id == "MOLESKINE-20240822T121207869Z"
    assert ref.participant_id == "MOLESKINE-UNKNOWN"
    assert "Dani" not in ref.recording_id
    assert "Dani" not in ref.participant_id


def test_load_maps_watch_and_pen_columns_without_inventing_gyro(tmp_path):
    write_fixture(tmp_path)
    adapter = MoleskineAdapter()
    bundle = adapter.load(adapter.discover(tmp_path)[0])

    watch = bundle.tables["watch"]
    assert set(S.COLUMNS[S.Quantity.ACCEL_TOTAL]) <= set(watch.columns)
    assert not set(S.COLUMNS[S.Quantity.GYRO]) & set(watch.columns)
    assert watch["t_ns"].dtype == np.int64
    assert bundle.meta["accel_semantics"] == "total"

    pen = bundle.tables["pen"]
    assert pen["dot_type"].tolist() == ["PEN_DOWN", "PEN_MOVE", "PEN_MOVE", "PEN_UP"]
    assert pen[["tilt_x", "tilt_y"]].isna().all().all()
    assert pen[["src_timestamp", "src_t_session_ms"]].isna().all().all()
    assert pen["t_ns"].between(watch["t_ns"].min(), watch["t_ns"].max()).all()


def test_manifest_exposes_accelerometer_only_capability_and_alignment_warning(tmp_path):
    write_fixture(tmp_path)
    adapter = MoleskineAdapter()
    bundle = adapter.load(adapter.discover(tmp_path)[0])

    row = build_manifest([bundle]).iloc[0]
    assert bool(row["has_watch"])
    assert not bool(row["has_watch_gyro"])
    assert row["time_alignment"] == "overlap_only"
    assert "paired only" in row["alignment_note"]
    assert "approximate" in row["alignment_note"]

    domains = build_channels([bundle]).set_index(["modality", "column"])["time_domain"]
    assert domains.loc[("watch", "t_ns")] == "watch_device_wall_clock"
    assert domains.loc[("pen", "t_ns")] == "moleskine_app_wall_clock"
    assert domains.loc[("watch", "src_accelerometer_timestamp_s")] == \
        S.TIME_DOMAIN_DEVICE_MONOTONIC_CLOCK


def test_bundle_passes_all_applicable_validation(tmp_path):
    write_fixture(tmp_path)
    adapter = MoleskineAdapter()
    bundle = adapter.load(adapter.discover(tmp_path)[0])

    failed = [finding for finding in validate_bundle(bundle) if not finding.passed]
    assert failed == []


def test_load_rejects_ambiguous_page_overlap(tmp_path):
    write_fixture(tmp_path, draw_types=(17, 20), page_ids=[3, 4])
    adapter = MoleskineAdapter()
    with pytest.raises(ValueError, match="overlaps 2 notebook pages"):
        adapter.load(adapter.discover(tmp_path)[0])


def test_load_rejects_unknown_draw_type(tmp_path):
    write_fixture(tmp_path, draw_types=(17, 99, 20))
    adapter = MoleskineAdapter()
    with pytest.raises(ValueError, match="unknown Moleskine drawType"):
        adapter.load(adapter.discover(tmp_path)[0])
