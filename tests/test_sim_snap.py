"""Tests for the detector-to-edge snapper.

The snap algorithm is dependency-injected: it takes any object that
implements ``NetworkAdapter`` and any callable that projects UTM
coordinates. The tests pass tiny in-memory fakes, so neither ``sumolib``
nor ``pyproj`` is required to run them.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from madrid_twin.data.open_data import DetectorReading
from madrid_twin.sim.snap import (
    DetectorEdgeMatch,
    SnapConfig,
    SnapReport,
    filter_to_ids,
    load_usable_detector_ids,
    snap_detectors_to_edges,
    write_snap_outputs,
)


# ---------------------------------------------------------------------------
# Fakes that stand in for sumolib + pyproj.
# ---------------------------------------------------------------------------


class FakeNetwork:
    """A minimal NetworkAdapter built from an explicit list of edges.

    Each edge is ``(edge_id, x, y)`` — a single representative point.
    ``neighboring_edges`` returns every edge within ``radius_m`` using
    Euclidean distance from ``(x, y)`` to the edge's representative
    point. ``convert_lonlat_to_xy`` is the identity — tests feed lon/lat
    values that are already in the fake network's coordinate system.
    """

    def __init__(self, edges: list[tuple[str, float, float]]) -> None:
        self.edges = edges

    def convert_lonlat_to_xy(self, lon: float, lat: float) -> tuple[float, float]:
        return float(lon), float(lat)

    def neighboring_edges(self, x: float, y: float, radius_m: float) -> list[tuple[str, float]]:
        result: list[tuple[str, float]] = []
        for edge_id, ex, ey in self.edges:
            d = math.hypot(ex - x, ey - y)
            if d <= radius_m:
                result.append((edge_id, d))
        return result


def identity_utm_to_lonlat(x: float, y: float) -> tuple[float, float]:
    """Stand-in for pyproj: tests use UTM == lon/lat == network-XY."""
    return x, y


# ---------------------------------------------------------------------------
# Reading fixtures.
# ---------------------------------------------------------------------------


def _reading(
    detector_id: str,
    x: float | None = 100.0,
    y: float | None = 200.0,
    description: str | None = None,
) -> DetectorReading:
    return DetectorReading(
        detector_id=detector_id,
        timestamp=None,
        intensity_veh_h=500.0,
        occupancy_pct=10.0,
        load_pct=20.0,
        service_status="0",
        description=description,
        x_utm=x,
        y_utm=y,
    )


# ---------------------------------------------------------------------------
# snap_detectors_to_edges()
# ---------------------------------------------------------------------------


class TestSnap:
    def test_matches_to_nearest_edge(self) -> None:
        # Three edges; the detector at (100, 200) should snap to e2.
        net = FakeNetwork(
            [
                ("e1", 0.0, 0.0),
                ("e2", 110.0, 195.0),  # closest to (100, 200)
                ("e3", 500.0, 500.0),
            ]
        )
        readings = [_reading("d1", x=100.0, y=200.0)]
        matches, report = snap_detectors_to_edges(
            readings,
            network=net,
            utm_to_lonlat=identity_utm_to_lonlat,
        )
        assert len(matches) == 1
        assert matches[0].edge_id == "e2"
        assert matches[0].detector_id == "d1"
        assert matches[0].distance_m == pytest.approx(math.hypot(10.0, 5.0))
        assert report.n_input == 1
        assert report.n_matched == 1
        assert report.n_skipped_no_coords == 0
        assert report.n_skipped_no_edge_within_radius == 0

    def test_detector_without_coords_is_skipped(self) -> None:
        net = FakeNetwork([("e1", 0.0, 0.0)])
        readings = [_reading("d_with", x=0.0, y=0.0), _reading("d_no", x=None, y=None)]
        matches, report = snap_detectors_to_edges(
            readings,
            network=net,
            utm_to_lonlat=identity_utm_to_lonlat,
        )
        assert {m.detector_id for m in matches} == {"d_with"}
        assert report.n_skipped_no_coords == 1

    def test_detector_beyond_max_distance_is_skipped(self) -> None:
        net = FakeNetwork([("e1", 0.0, 0.0)])
        # Detector is 200 m away — outside the default 100 m radius.
        readings = [_reading("d_far", x=200.0, y=0.0)]
        matches, report = snap_detectors_to_edges(
            readings,
            network=net,
            utm_to_lonlat=identity_utm_to_lonlat,
            config=SnapConfig(max_distance_m=100.0, initial_radius_m=25.0),
        )
        assert matches == []
        assert report.n_skipped_no_edge_within_radius == 1

    def test_expanding_radius_eventually_finds_far_edge(self) -> None:
        net = FakeNetwork([("e1", 0.0, 0.0)])
        # Detector is 80 m away — the initial 25 m radius misses, but
        # one doubling (50, then 100) catches it.
        readings = [_reading("d_mid", x=80.0, y=0.0)]
        matches, _ = snap_detectors_to_edges(
            readings,
            network=net,
            utm_to_lonlat=identity_utm_to_lonlat,
            config=SnapConfig(max_distance_m=200.0, initial_radius_m=25.0),
        )
        assert len(matches) == 1
        assert matches[0].edge_id == "e1"

    def test_matches_are_sorted_by_detector_id(self) -> None:
        net = FakeNetwork([("e1", 0.0, 0.0)])
        readings = [
            _reading("d_zebra", x=0.0, y=0.0),
            _reading("d_alpha", x=0.0, y=0.0),
            _reading("d_mango", x=0.0, y=0.0),
        ]
        matches, _ = snap_detectors_to_edges(
            readings,
            network=net,
            utm_to_lonlat=identity_utm_to_lonlat,
        )
        assert [m.detector_id for m in matches] == ["d_alpha", "d_mango", "d_zebra"]

    def test_report_distance_stats(self) -> None:
        net = FakeNetwork([("e1", 0.0, 0.0)])
        readings = [
            _reading("d1", x=10.0, y=0.0),
            _reading("d2", x=20.0, y=0.0),
            _reading("d3", x=30.0, y=0.0),
        ]
        _, report = snap_detectors_to_edges(
            readings,
            network=net,
            utm_to_lonlat=identity_utm_to_lonlat,
        )
        assert report.median_distance_m == pytest.approx(20.0)
        assert report.max_distance_m == pytest.approx(30.0)

    def test_description_passed_through(self) -> None:
        net = FakeNetwork([("e1", 0.0, 0.0)])
        readings = [_reading("d1", x=0.0, y=0.0, description="M-30 S-W")]
        matches, _ = snap_detectors_to_edges(
            readings,
            network=net,
            utm_to_lonlat=identity_utm_to_lonlat,
        )
        assert matches[0].description == "M-30 S-W"

    def test_utm_to_lonlat_called(self) -> None:
        net = FakeNetwork([("e1", 0.0, 0.0)])
        readings = [_reading("d1", x=100.0, y=200.0)]
        seen: list[tuple[float, float]] = []

        def spy(x: float, y: float) -> tuple[float, float]:
            seen.append((x, y))
            return 0.0, 0.0

        snap_detectors_to_edges(readings, network=net, utm_to_lonlat=spy)
        assert seen == [(100.0, 200.0)]

    def test_invalid_config_raises(self) -> None:
        net = FakeNetwork([("e1", 0.0, 0.0)])
        with pytest.raises(ValueError):
            snap_detectors_to_edges(
                [_reading("d1")],
                network=net,
                utm_to_lonlat=identity_utm_to_lonlat,
                config=SnapConfig(max_distance_m=0.0),
            )
        with pytest.raises(ValueError):
            snap_detectors_to_edges(
                [_reading("d1")],
                network=net,
                utm_to_lonlat=identity_utm_to_lonlat,
                config=SnapConfig(initial_radius_m=0.0),
            )


# ---------------------------------------------------------------------------
# I/O helpers.
# ---------------------------------------------------------------------------


class TestIo:
    def test_load_usable_detector_ids(self, tmp_path: Path) -> None:
        path = tmp_path / "usable.json"
        path.write_text(json.dumps({"detector_ids": ["a", "b", "c"]}), encoding="utf-8")
        assert load_usable_detector_ids(path) == {"a", "b", "c"}

    def test_filter_to_ids(self) -> None:
        readings = [_reading("a"), _reading("b"), _reading("c")]
        kept = filter_to_ids(readings, {"a", "c"})
        assert [r.detector_id for r in kept] == ["a", "c"]

    def test_write_snap_outputs_roundtrip(self, tmp_path: Path) -> None:
        matches = [
            DetectorEdgeMatch("d1", "e1", 5.0, description="x", x_utm=1.0, y_utm=2.0),
            DetectorEdgeMatch("d2", "e2", 7.5),
        ]
        report = SnapReport(
            n_input=2,
            n_matched=2,
            n_skipped_no_coords=0,
            n_skipped_no_edge_within_radius=0,
            median_distance_m=5.0,
            max_distance_m=7.5,
        )
        mapping_path, report_path = write_snap_outputs(matches, report, tmp_path)
        assert mapping_path.exists()
        assert report_path.exists()

        mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
        assert set(mapping.keys()) == {"d1", "d2"}
        assert mapping["d1"]["edge_id"] == "e1"
        assert mapping["d1"]["distance_m"] == 5.0

        report_dict = json.loads(report_path.read_text(encoding="utf-8"))
        assert report_dict["n_input"] == 2
