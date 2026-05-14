"""Tests for the LWR conservation residual."""

from __future__ import annotations

import numpy as np
import pytest

from madrid_twin.predict.physics import lwr_residual, lwr_residual_loss


class TestLWRResidual:
    def test_zero_residual_on_consistent_state(self) -> None:
        # Density rises by exactly the dt/L * (in - out) amount each step.
        T = 3
        N = 2
        L = np.array([100.0, 200.0])
        dt = 1.0

        inflow = np.array([[2.0, 1.0]] * T)
        outflow = np.array([[1.0, 0.5]] * T)
        # delta_rho per step = dt/L * (inflow - outflow)
        # node 0: 1/100 = 0.01; node 1: 0.5/200 = 0.0025
        density = np.zeros((T + 1, N))
        for t in range(T):
            density[t + 1, 0] = density[t, 0] + 0.01
            density[t + 1, 1] = density[t, 1] + 0.0025

        residual = lwr_residual(density, inflow, outflow, segment_length_m=L, dt_s=dt)
        np.testing.assert_allclose(residual, 0.0, atol=1e-12)

    def test_inconsistent_state_yields_nonzero_residual(self) -> None:
        # All flows zero but density grows -> residual = delta_rho.
        T = 2
        N = 1
        density = np.array([[0.0], [0.5], [1.0]])
        inflow = np.zeros((T, N))
        outflow = np.zeros((T, N))
        residual = lwr_residual(density, inflow, outflow, segment_length_m=10.0, dt_s=1.0)
        np.testing.assert_allclose(residual.flatten(), [0.5, 0.5])

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            lwr_residual(
                density=np.zeros((3, 2)),
                inflow=np.zeros((3, 2)),  # should be T = 2
                outflow=np.zeros((3, 2)),
                segment_length_m=1.0,
                dt_s=1.0,
            )

    def test_non_positive_segment_length_raises(self) -> None:
        with pytest.raises(ValueError):
            lwr_residual(
                density=np.zeros((2, 1)),
                inflow=np.zeros((1, 1)),
                outflow=np.zeros((1, 1)),
                segment_length_m=0.0,
                dt_s=1.0,
            )

    def test_non_positive_dt_raises(self) -> None:
        with pytest.raises(ValueError):
            lwr_residual(
                density=np.zeros((2, 1)),
                inflow=np.zeros((1, 1)),
                outflow=np.zeros((1, 1)),
                segment_length_m=10.0,
                dt_s=0.0,
            )


class TestLWRResidualLoss:
    def test_mean_abs_zero_on_consistent_state(self) -> None:
        T = 4
        N = 2
        L = np.array([100.0, 100.0])
        inflow = np.ones((T, N))
        outflow = np.ones((T, N))  # net zero
        density = np.zeros((T + 1, N))  # stays flat
        assert lwr_residual_loss(density, inflow, outflow, segment_length_m=L, dt_s=1.0) == 0.0

    def test_mean_abs_positive_on_violation(self) -> None:
        density = np.array([[0.0], [1.0]])
        loss = lwr_residual_loss(
            density,
            np.zeros((1, 1)),
            np.zeros((1, 1)),
            segment_length_m=10.0,
            dt_s=1.0,
        )
        assert loss == pytest.approx(1.0)

    def test_mean_sq_reduction(self) -> None:
        density = np.array([[0.0], [2.0]])  # residual = 2
        loss = lwr_residual_loss(
            density,
            np.zeros((1, 1)),
            np.zeros((1, 1)),
            segment_length_m=10.0,
            dt_s=1.0,
            reduction="mean_sq",
        )
        assert loss == pytest.approx(4.0)

    def test_unknown_reduction_raises(self) -> None:
        with pytest.raises(ValueError):
            lwr_residual_loss(
                np.zeros((2, 1)),
                np.zeros((1, 1)),
                np.zeros((1, 1)),
                segment_length_m=1.0,
                dt_s=1.0,
                reduction="nope",
            )
