"""Tests for the composite reward."""

from __future__ import annotations

import numpy as np
import pytest

from madrid_twin.control.rewards import (
    NetworkState,
    RewardConfig,
    composite_reward,
    emissions,
    equity_penalty,
    gini,
    mean_delay,
    spillback_penalty,
    throughput,
)


def _state(**overrides):
    defaults = {
        "delays_s": np.array([10.0, 20.0, 30.0]),
        "throughput_veh": 2,
        "queue_lengths_m": np.array([50.0, 100.0]),
        "link_lengths_m": np.array([200.0, 200.0]),
        "emissions_g": 15.0,
        "travel_time_impact_per_zone_s": np.array([10.0, 20.0, 30.0]),
    }
    defaults.update(overrides)
    return NetworkState(**defaults)


class TestRewardConfig:
    def test_negative_weight_raises(self) -> None:
        with pytest.raises(ValueError):
            RewardConfig(w_delay=-1.0)


class TestNetworkState:
    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            NetworkState(
                delays_s=np.zeros(0),
                throughput_veh=0,
                queue_lengths_m=np.array([10.0]),
                link_lengths_m=np.array([10.0, 20.0]),
                emissions_g=0.0,
            )

    def test_zero_link_length_raises(self) -> None:
        with pytest.raises(ValueError):
            NetworkState(
                delays_s=np.zeros(0),
                throughput_veh=0,
                queue_lengths_m=np.array([10.0]),
                link_lengths_m=np.array([0.0]),
                emissions_g=0.0,
            )

    def test_negative_throughput_raises(self) -> None:
        with pytest.raises(ValueError):
            NetworkState(
                delays_s=np.zeros(0),
                throughput_veh=-1,
                queue_lengths_m=np.array([10.0]),
                link_lengths_m=np.array([100.0]),
                emissions_g=0.0,
            )


class TestPerTerm:
    def test_mean_delay(self) -> None:
        assert mean_delay(_state()) == pytest.approx(20.0)

    def test_mean_delay_empty(self) -> None:
        assert mean_delay(_state(delays_s=np.zeros(0))) == 0.0

    def test_throughput(self) -> None:
        assert throughput(_state()) == 2.0

    def test_emissions(self) -> None:
        assert emissions(_state()) == 15.0

    def test_spillback_no_violation(self) -> None:
        assert spillback_penalty(_state(), RewardConfig()) == 0.0

    def test_spillback_with_violation(self) -> None:
        s = _state(
            queue_lengths_m=np.array([180.0, 100.0]),
            link_lengths_m=np.array([200.0, 200.0]),
        )
        assert spillback_penalty(s, RewardConfig()) == pytest.approx(0.05)


class TestGini:
    def test_perfect_equality(self) -> None:
        assert gini(np.array([1.0, 1.0, 1.0, 1.0])) == pytest.approx(0.0)

    def test_perfect_inequality_approaches_one(self) -> None:
        v = np.array([0.0, 0.0, 0.0, 0.0, 100.0])
        assert gini(v) == pytest.approx(0.8, abs=0.01)

    def test_empty_returns_zero(self) -> None:
        assert gini(np.zeros(0)) == 0.0

    def test_all_zero_returns_zero(self) -> None:
        assert gini(np.zeros(5)) == 0.0

    def test_negative_values_shifted_non_negative(self) -> None:
        g = gini(np.array([-1.0, 0.0, 1.0]))
        assert 0.0 <= g <= 1.0
        assert g > 0.0


class TestEquityPenalty:
    def test_uniform_impact_is_zero(self) -> None:
        assert equity_penalty(_state(travel_time_impact_per_zone_s=np.full(5, 10.0))) == 0.0

    def test_concentrated_impact_is_positive(self) -> None:
        s = _state(travel_time_impact_per_zone_s=np.array([0.0, 0.0, 0.0, 100.0]))
        assert equity_penalty(s) > 0.5


class TestComposite:
    def test_composite_is_finite(self) -> None:
        r = composite_reward(_state())
        assert np.isfinite(r)

    def test_throughput_increases_reward(self) -> None:
        low = composite_reward(_state(throughput_veh=0))
        high = composite_reward(_state(throughput_veh=100))
        assert high > low

    def test_delay_decreases_reward(self) -> None:
        r_less_delay = composite_reward(_state(delays_s=np.array([5.0])))
        r_more_delay = composite_reward(_state(delays_s=np.array([500.0])))
        assert r_less_delay > r_more_delay
