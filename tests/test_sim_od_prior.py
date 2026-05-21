"""Tests for the gravity OD prior."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from madrid_twin.sim.edgedata import EdgeObservation
from madrid_twin.sim.od_prior import (
    OdPriorReport,
    ZoneStats,
    aggregate_zone_stats,
    build_od_prior_report,
    gravity_od_prior,
    load_od_prior,
    save_od_prior,
)

# ---------------------------------------------------------------------------
# aggregate_zone_stats()
# ---------------------------------------------------------------------------


class TestAggregateZones:
    def test_single_zone_single_edge(self) -> None:
        edge_to_zone = {"e1": 0}
        centroids = {"e1": (10.0, 20.0)}
        obs = {"e1": EdgeObservation("e1", 500.0, None, None, 1, ("d1",))}
        stats = aggregate_zone_stats(edge_to_zone, centroids, obs)
        assert set(stats) == {0}
        z = stats[0]
        assert z.centroid_x == 10.0
        assert z.centroid_y == 20.0
        assert z.n_edges == 1
        assert z.n_detector_equipped_edges == 1
        assert z.total_observed_flow_veh_h == 500.0

    def test_zone_centroid_is_mean_of_edge_centroids(self) -> None:
        edge_to_zone = {"e1": 0, "e2": 0}
        centroids = {"e1": (0.0, 0.0), "e2": (10.0, 10.0)}
        obs: dict[str, EdgeObservation] = {}  # no detector coverage
        stats = aggregate_zone_stats(edge_to_zone, centroids, obs)
        assert stats[0].centroid_x == 5.0
        assert stats[0].centroid_y == 5.0
        assert stats[0].n_edges == 2
        assert stats[0].n_detector_equipped_edges == 0
        assert stats[0].total_observed_flow_veh_h == 0.0

    def test_flows_are_summed_across_edges_in_zone(self) -> None:
        edge_to_zone = {"e1": 0, "e2": 0}
        centroids = {"e1": (0.0, 0.0), "e2": (1.0, 1.0)}
        obs = {
            "e1": EdgeObservation("e1", 300.0, None, None, 1, ("d1",)),
            "e2": EdgeObservation("e2", 400.0, None, None, 1, ("d2",)),
        }
        stats = aggregate_zone_stats(edge_to_zone, centroids, obs)
        assert stats[0].total_observed_flow_veh_h == 700.0
        assert stats[0].n_detector_equipped_edges == 2

    def test_edges_without_centroid_are_skipped(self) -> None:
        edge_to_zone = {"e1": 0, "e2": 0}
        centroids = {"e1": (0.0, 0.0)}  # e2 missing
        obs: dict[str, EdgeObservation] = {}
        stats = aggregate_zone_stats(edge_to_zone, centroids, obs)
        assert stats[0].n_edges == 1


# ---------------------------------------------------------------------------
# gravity_od_prior()
# ---------------------------------------------------------------------------


class TestGravity:
    def test_row_sums_equal_productions(self) -> None:
        zones = [
            ZoneStats(0, 0.0, 0.0, 10, 5, 1000.0),
            ZoneStats(1, 1000.0, 0.0, 10, 5, 2000.0),
            ZoneStats(2, 0.0, 1000.0, 10, 5, 1500.0),
        ]
        T = gravity_od_prior(zones, beta_per_km=0.1)
        assert T.shape == (3, 3)
        row_sums = T.sum(axis=1)
        assert row_sums[0] == pytest.approx(1000.0)
        assert row_sums[1] == pytest.approx(2000.0)
        assert row_sums[2] == pytest.approx(1500.0)

    def test_diagonal_is_zero_by_default(self) -> None:
        zones = [
            ZoneStats(0, 0.0, 0.0, 10, 5, 500.0),
            ZoneStats(1, 100.0, 0.0, 10, 5, 500.0),
        ]
        T = gravity_od_prior(zones)
        assert T[0, 0] == 0.0
        assert T[1, 1] == 0.0

    def test_closer_zones_get_more_demand(self) -> None:
        # Three zones: 0 close to 1 (100m apart) and far from 2 (10km).
        # All have the same observed flow. Zone 0's demand should mostly
        # go to zone 1.
        zones = [
            ZoneStats(0, 0.0, 0.0, 10, 5, 1000.0),
            ZoneStats(1, 100.0, 0.0, 10, 5, 1000.0),
            ZoneStats(2, 10000.0, 0.0, 10, 5, 1000.0),
        ]
        T = gravity_od_prior(zones, beta_per_km=0.5)
        assert T[0, 1] > T[0, 2]

    def test_zone_with_zero_production_emits_zero_row(self) -> None:
        zones = [
            ZoneStats(0, 0.0, 0.0, 10, 0, 0.0),  # no observed flow
            ZoneStats(1, 1000.0, 0.0, 10, 5, 500.0),
        ]
        T = gravity_od_prior(zones)
        assert (T[0, :] == 0.0).all()

    def test_empty_zones_returns_empty_matrix(self) -> None:
        T = gravity_od_prior([])
        assert T.shape == (0, 0)

    def test_negative_beta_raises(self) -> None:
        with pytest.raises(ValueError):
            gravity_od_prior([ZoneStats(0, 0.0, 0.0, 1, 1, 100.0)], beta_per_km=-0.1)

    def test_zero_diagonal_false_keeps_self_attractions(self) -> None:
        zones = [
            ZoneStats(0, 0.0, 0.0, 10, 5, 100.0),
            ZoneStats(1, 10000.0, 0.0, 10, 5, 100.0),
        ]
        T = gravity_od_prior(zones, beta_per_km=1.0, zero_diagonal=False)
        # At 10 km separation with beta=1/km, the cross terms are
        # exp(-10) ≈ 0 — so the diagonals capture most of the demand.
        assert T[0, 0] > T[0, 1]


# ---------------------------------------------------------------------------
# Report + I/O.
# ---------------------------------------------------------------------------


class TestReport:
    def test_report_counts_zero_flow_zones(self) -> None:
        T = np.array(
            [
                [0.0, 0.0, 0.0],
                [10.0, 0.0, 5.0],
                [0.0, 0.0, 0.0],
            ]
        )
        report = build_od_prior_report(T, beta_per_km=0.1)
        assert isinstance(report, OdPriorReport)
        assert report.n_zones == 3
        assert report.n_zones_without_flow == 2
        assert report.total_demand_veh_h == 15.0


class TestIo:
    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        T = np.array([[0.0, 1.0], [2.0, 0.0]])
        zones = [
            ZoneStats(0, 0.0, 0.0, 10, 5, 100.0),
            ZoneStats(1, 1.0, 1.0, 10, 5, 200.0),
        ]
        matrix_path, zones_path, _ = save_od_prior(T, zones, tmp_path, beta_per_km=0.1)
        T2, zones2 = load_od_prior(matrix_path, zones_path)
        np.testing.assert_array_equal(T2, T)
        assert zones2 == zones
