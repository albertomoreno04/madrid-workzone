"""Tests for the domain-randomization sampler."""

from __future__ import annotations

import pytest

from madrid_twin.control.domain_randomization import (
    DomainRandomizationConfig,
    DomainRandomizer,
)
from madrid_twin.sim.workzone import LaneClosure, Workzone


@pytest.fixture()
def base_workzone() -> Workzone:
    return Workzone(
        id="wz_test",
        closures=(LaneClosure(edge_id="e1", lane_index=0),),
        start_s=0.0,
        end_s=3600.0,
        capacity_drop_pct=0.20,
        description="test",
    )


class TestConfigValidation:
    def test_inverted_range_raises(self) -> None:
        with pytest.raises(ValueError):
            DomainRandomizationConfig(capacity_drop_range=(0.5, 0.1))

    def test_capacity_drop_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError):
            DomainRandomizationConfig(capacity_drop_range=(0.0, 1.0))

    def test_incident_probability_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError):
            DomainRandomizationConfig(incident_probability=1.5)


class TestDomainRandomizer:
    def test_sample_returns_valid_workzone(self, base_workzone: Workzone) -> None:
        dr = DomainRandomizer(base_workzone, seed=42)
        sc = dr.sample()
        # Capacity drop must lie inside the default config range.
        cfg = DomainRandomizationConfig()
        lo, hi = cfg.capacity_drop_range
        assert lo <= sc.workzone.capacity_drop_pct <= hi
        # Demand multiplier inside its range.
        dlo, dhi = cfg.demand_multiplier_range
        assert dlo <= sc.demand_multiplier <= dhi
        # Sensor noise non-negative.
        assert sc.sensor_noise_std >= 0.0
        # Incident flag is a bool.
        assert isinstance(sc.incident, bool)
        # Workzone id is derived from the base.
        assert sc.workzone.id.startswith("wz_test")

    def test_seed_reproducibility(self, base_workzone: Workzone) -> None:
        a = DomainRandomizer(base_workzone, seed=7).sample()
        b = DomainRandomizer(base_workzone, seed=7).sample()
        assert a.workzone.capacity_drop_pct == b.workzone.capacity_drop_pct
        assert a.demand_multiplier == b.demand_multiplier
        assert a.sensor_noise_std == b.sensor_noise_std
        assert a.incident == b.incident

    def test_different_seeds_yield_different_samples(self, base_workzone: Workzone) -> None:
        a = DomainRandomizer(base_workzone, seed=1).sample()
        b = DomainRandomizer(base_workzone, seed=2).sample()
        # Highly unlikely all three random draws match across seeds.
        assert (
            a.workzone.capacity_drop_pct != b.workzone.capacity_drop_pct
            or a.demand_multiplier != b.demand_multiplier
            or a.sensor_noise_std != b.sensor_noise_std
        )
