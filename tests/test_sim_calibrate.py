"""Tests for the W-SPSA OD calibration algorithm."""

from __future__ import annotations

import numpy as np
import pytest

from madrid_twin.sim.calibrate import (
    WSPSAConfig,
    WSPSAReport,
    count_matching_loss,
    wspsa,
)


class TestCountMatchingLoss:
    def test_zero_on_match(self) -> None:
        sim = np.array([100.0, 200.0, 300.0])
        assert count_matching_loss(sim, sim) == pytest.approx(0.0)

    def test_squared_error(self) -> None:
        sim = np.array([100.0, 200.0])
        obs = np.array([110.0, 190.0])
        # SE: 100 + 100 = 200; mean weighted (uniform) = 100.
        assert count_matching_loss(sim, obs) == pytest.approx(100.0)

    def test_weighted(self) -> None:
        sim = np.array([100.0, 100.0])
        obs = np.array([110.0, 90.0])
        w = np.array([1.0, 3.0])
        # numerator = 1*100 + 3*100 = 400; denom = 4 -> 100.
        assert count_matching_loss(sim, obs, weights=w) == pytest.approx(100.0)

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            count_matching_loss(np.zeros(3), np.zeros(2))

    def test_negative_weights_raise(self) -> None:
        with pytest.raises(ValueError):
            count_matching_loss(
                np.zeros(2), np.zeros(2), weights=np.array([1.0, -1.0])
            )


class TestWSPSAConfig:
    def test_negative_a_raises(self) -> None:
        with pytest.raises(ValueError):
            WSPSAConfig(a=-1.0)


class TestWSPSA:
    def test_converges_on_quadratic(self) -> None:
        # Synthetic: f(theta) = ||theta - target||^2 -- a textbook convex
        # objective. W-SPSA should descend toward the target.
        target = np.array([3.0, -2.0, 5.0])

        def loss(theta: np.ndarray) -> float:
            return float(((theta - target) ** 2).sum())

        theta0 = np.zeros(3)
        cfg = WSPSAConfig(a=0.3, c=0.05, max_iter=200, seed=0)
        theta_star, report = wspsa(theta0, loss, cfg)

        assert isinstance(report, WSPSAReport)
        # Final loss must be substantially below initial loss.
        assert report.final_loss < 0.5 * report.initial_loss
        # We should have moved meaningfully toward the target.
        assert float(np.linalg.norm(theta_star - target)) < float(
            np.linalg.norm(theta0 - target)
        )

    def test_records_trace(self) -> None:
        target = np.array([1.0])

        def loss(t: np.ndarray) -> float:
            return float(((t - target) ** 2).sum())

        cfg = WSPSAConfig(max_iter=5, seed=1)
        _, report = wspsa(np.zeros(1), loss, cfg)
        # Trace has initial + max_iter entries.
        assert len(report.losses) == 6
        assert report.n_iter == 5

    def test_rejects_non_1d_theta(self) -> None:
        with pytest.raises(ValueError):
            wspsa(np.zeros((2, 2)), lambda t: 0.0, WSPSAConfig(max_iter=1))

    def test_explicit_weights_used(self) -> None:
        # Same problem, but pass tiny weights so we move very little.
        target = np.array([10.0])

        def loss(t: np.ndarray) -> float:
            return float(((t - target) ** 2).sum())

        cfg = WSPSAConfig(max_iter=50, seed=2)
        theta_tiny, _ = wspsa(np.zeros(1), loss, cfg, weights=np.array([1e-6]))
        # With near-zero weights we should barely move.
        assert abs(float(theta_tiny[0])) < 1.0

    def test_min_value_floor(self) -> None:
        target = np.array([-5.0])

        def loss(t: np.ndarray) -> float:
            return float(((t - target) ** 2).sum())

        cfg = WSPSAConfig(max_iter=50, min_value=0.0, seed=3)
        theta_star, _ = wspsa(np.array([2.0]), loss, cfg)
        # Floor must hold even when target is negative.
        assert float(theta_star[0]) >= 0.0
