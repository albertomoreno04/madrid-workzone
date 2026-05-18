"""Tests for the OD-matrix → SUMO trips emitter."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from madrid_twin.sim.demand import (
    ODMatrixSpec,
    apply_adversarial_perturbation,
    apply_demand_multiplier,
    sample_trips,
    trips_to_sumo_routes_xml,
    write_sumo_routes,
)


def _spec() -> ODMatrixSpec:
    return ODMatrixSpec(
        origins=("e_a", "e_b"),
        destinations=("e_c", "e_d"),
        counts=np.array([[10.0, 5.0], [2.0, 8.0]]),
        begin_s=0.0,
        end_s=600.0,
    )


class TestODMatrixSpec:
    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            ODMatrixSpec(
                origins=("a",),
                destinations=("b", "c"),
                counts=np.zeros((2, 2)),
                begin_s=0.0,
                end_s=10.0,
            )

    def test_negative_counts_raise(self) -> None:
        with pytest.raises(ValueError):
            ODMatrixSpec(
                origins=("a",),
                destinations=("b",),
                counts=np.array([[-1.0]]),
                begin_s=0.0,
                end_s=10.0,
            )

    def test_inverted_times_raise(self) -> None:
        with pytest.raises(ValueError):
            ODMatrixSpec(
                origins=("a",),
                destinations=("b",),
                counts=np.array([[1.0]]),
                begin_s=10.0,
                end_s=0.0,
            )


class TestApplyDemandMultiplier:
    def test_scales_counts(self) -> None:
        scaled = apply_demand_multiplier(_spec(), 1.5)
        np.testing.assert_allclose(scaled.counts, _spec().counts * 1.5)

    def test_negative_multiplier_raises(self) -> None:
        with pytest.raises(ValueError):
            apply_demand_multiplier(_spec(), -0.5)


class TestApplyAdversarialPerturbation:
    def test_adds_along_rows(self) -> None:
        spec = _spec()
        perturbed = apply_adversarial_perturbation(spec, np.array([3.0, -1.0]))
        # Row 0 starts with [10, 5] (sum 15); +3 spread proportionally:
        # +3 * 10/15 = 2 -> 12; +3 * 5/15 = 1 -> 6.
        np.testing.assert_allclose(perturbed.counts[0], [12.0, 6.0])
        # Row 1 starts with [2, 8] (sum 10); -1 spread proportionally:
        # -1 * 2/10 = -0.2 -> 1.8; -1 * 8/10 = -0.8 -> 7.2.
        np.testing.assert_allclose(perturbed.counts[1], [1.8, 7.2])

    def test_floors_at_zero(self) -> None:
        spec = _spec()
        perturbed = apply_adversarial_perturbation(spec, np.array([-100.0, 0.0]))
        assert (perturbed.counts[0] >= 0).all()

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            apply_adversarial_perturbation(_spec(), np.zeros(3))


class TestSampleTrips:
    def test_count_close_to_expected(self) -> None:
        # Total expected count from the fixture spec: 10 + 5 + 2 + 8 = 25.
        spec = _spec()
        trips = sample_trips(spec, seed=42)
        # Stochastic rounding can wander +/-3 around the expectation.
        assert 22 <= len(trips) <= 28

    def test_departures_sorted(self) -> None:
        trips = sample_trips(_spec(), seed=0)
        depart_times = [t[2] for t in trips]
        assert depart_times == sorted(depart_times)

    def test_departures_within_window(self) -> None:
        spec = _spec()
        trips = sample_trips(spec, seed=0)
        for _, _, t in trips:
            assert spec.begin_s <= t < spec.end_s


class TestRoutesXML:
    def test_emits_well_formed_xml(self) -> None:
        trips = [("e_a", "e_c", 10.0), ("e_b", "e_d", 20.0)]
        xml = trips_to_sumo_routes_xml(trips)
        root = ET.fromstring(xml.split("?>", 1)[-1])
        assert root.tag == "routes"
        trip_elements = root.findall("trip")
        assert len(trip_elements) == 2
        assert trip_elements[0].attrib["from"] == "e_a"
        assert trip_elements[0].attrib["to"] == "e_c"

    def test_write_sumo_routes(self, tmp_path: Path) -> None:
        spec = _spec()
        out = tmp_path / "trips.rou.xml"
        path = write_sumo_routes(spec, out, seed=0)
        assert path == out
        assert out.exists()
        assert out.read_text(encoding="utf-8").startswith("<?xml")
