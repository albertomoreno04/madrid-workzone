"""Tests for the detector-reliability audit."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from madrid_twin.data.audit import (
    AuditConfig,
    aggregate_snapshots,
    filter_usable_detectors,
    load_snapshot_files,
    write_audit_outputs,
)
from madrid_twin.data.open_data import DetectorReading


def _reading(
    det_id: str,
    *,
    intensity: float | None = 100.0,
    status: str | None = "0",
    ts: datetime | None = None,
) -> DetectorReading:
    return DetectorReading(
        detector_id=det_id,
        timestamp=ts or datetime(2026, 5, 13, 8, 0, 0),
        intensity_veh_h=intensity,
        occupancy_pct=None,
        load_pct=None,
        service_status=status,
    )


class TestAuditConfig:
    @pytest.mark.parametrize("bad", [-0.1, 1.1, 2.0])
    def test_invalid_fraction_raises(self, bad: float) -> None:
        with pytest.raises(ValueError):
            AuditConfig(min_healthy_fraction=bad)

    def test_negative_min_intensity_raises(self) -> None:
        with pytest.raises(ValueError):
            AuditConfig(min_mean_intensity_veh_h=-1.0)


class TestAggregateSnapshots:
    def test_empty_input(self) -> None:
        assert aggregate_snapshots([]) == {}

    def test_single_detector_one_snapshot(self) -> None:
        snap = [_reading("d1", intensity=200.0)]
        result = aggregate_snapshots([snap])
        assert "d1" in result
        r = result["d1"]
        assert r.n_snapshots == 1
        assert r.n_healthy == 1
        assert r.n_with_intensity == 1
        assert r.healthy_fraction == 1.0
        assert r.intensity_fraction == 1.0
        assert r.mean_intensity_veh_h == 200.0
        assert r.std_intensity_veh_h == 0.0

    def test_multiple_snapshots_aggregate(self) -> None:
        snaps = [
            [_reading("d1", intensity=100.0)],
            [_reading("d1", intensity=200.0)],
            [_reading("d1", intensity=300.0)],
        ]
        r = aggregate_snapshots(snaps)["d1"]
        assert r.n_snapshots == 3
        assert r.mean_intensity_veh_h == 200.0  # (100+200+300)/3
        # std of [100, 200, 300] = sqrt(((100-200)^2 + 0 + (100)^2) / 3) ~ 81.65
        assert r.std_intensity_veh_h == pytest.approx(81.65, abs=0.1)

    def test_unhealthy_detector(self) -> None:
        snaps = [
            [_reading("d1", status="0")],
            [_reading("d1", status="2")],  # failure code
            [_reading("d1", status="2")],
        ]
        r = aggregate_snapshots(snaps)["d1"]
        assert r.healthy_fraction == pytest.approx(1.0 / 3.0)

    def test_treats_missing_status_as_healthy(self) -> None:
        snaps = [
            [_reading("d1", status=None)],
            [_reading("d1", status="")],
            [_reading("d1", status="0")],
        ]
        r = aggregate_snapshots(snaps)["d1"]
        assert r.healthy_fraction == 1.0

    def test_null_intensity_drops_only_from_intensity_count(self) -> None:
        snaps = [
            [_reading("d1", intensity=100.0)],
            [_reading("d1", intensity=None)],
        ]
        r = aggregate_snapshots(snaps)["d1"]
        assert r.n_snapshots == 2
        assert r.n_with_intensity == 1
        assert r.intensity_fraction == 0.5
        assert r.mean_intensity_veh_h == 100.0  # only the non-null counted

    def test_detector_missing_from_some_snapshots(self) -> None:
        snaps = [
            [_reading("d1", intensity=10.0), _reading("d2", intensity=20.0)],
            [_reading("d1", intensity=30.0)],  # d2 missing
        ]
        result = aggregate_snapshots(snaps)
        assert result["d1"].n_snapshots == 2
        assert result["d2"].n_snapshots == 1

    def test_timestamps_first_last(self) -> None:
        snaps = [
            [_reading("d1", ts=datetime(2026, 1, 1, 0, 0, 0))],
            [_reading("d1", ts=datetime(2026, 1, 3, 0, 0, 0))],
            [_reading("d1", ts=datetime(2026, 1, 2, 0, 0, 0))],
        ]
        r = aggregate_snapshots(snaps)["d1"]
        assert r.first_seen_iso == "2026-01-01T00:00:00"
        assert r.last_seen_iso == "2026-01-03T00:00:00"


class TestFilterUsableDetectors:
    def test_keeps_only_passing(self) -> None:
        # d_good passes; d_unhealthy fails healthy_fraction; d_low fails intensity
        snaps = [
            [
                _reading("d_good", intensity=100.0, status="0"),
                _reading("d_unhealthy", intensity=100.0, status="2"),
                _reading("d_zero", intensity=0.0, status="0"),
            ]
        ] * 10
        rels = aggregate_snapshots(snaps)
        usable = filter_usable_detectors(rels.values())
        assert "d_good" in usable
        assert "d_unhealthy" not in usable
        assert "d_zero" not in usable

    def test_returns_sorted(self) -> None:
        snaps = [
            [
                _reading("d_z", intensity=100.0),
                _reading("d_a", intensity=100.0),
            ]
        ] * 5
        rels = aggregate_snapshots(snaps)
        usable = filter_usable_detectors(rels.values())
        assert usable == sorted(usable)


class TestLoadSnapshotFiles:
    def test_missing_directory_returns_empty(self, tmp_path: Path) -> None:
        assert load_snapshot_files(tmp_path / "does-not-exist") == []

    def test_loads_pm_xml_files(self, tmp_path: Path, traffic_intensity_sample_xml: str) -> None:
        (tmp_path / "pm_20260513T080000Z.xml").write_text(
            traffic_intensity_sample_xml, encoding="utf-8"
        )
        (tmp_path / "pm_20260513T080500Z.xml").write_text(
            traffic_intensity_sample_xml, encoding="utf-8"
        )
        # Non-matching name must be skipped.
        (tmp_path / "other.xml").write_text("garbage", encoding="utf-8")

        snaps = load_snapshot_files(tmp_path)
        assert len(snaps) == 2
        # Each parses to 3 detectors (per the fixture).
        assert all(len(s) == 3 for s in snaps)


class TestWriteAuditOutputs:
    def test_writes_both_files(self, tmp_path: Path) -> None:
        snaps = [[_reading("d_a", intensity=100.0)]] * 5
        rels = aggregate_snapshots(snaps)
        usable = filter_usable_detectors(rels.values())

        table_path, usable_path = write_audit_outputs(rels, usable, tmp_path)
        assert table_path.exists()
        assert usable_path.exists()

        import json as _json

        table = _json.loads(table_path.read_text())
        assert "d_a" in table
        usable_json = _json.loads(usable_path.read_text())
        assert usable_json["detector_ids"] == ["d_a"]
