"""Tests for the classical forecaster baselines."""

from __future__ import annotations

import numpy as np
import pytest

from madrid_twin.predict.baselines import (
    AR1Forecaster,
    ForecastBundle,
    Forecaster,
    HistoricalAverage,
    NaiveLastValue,
)


@pytest.fixture()
def small_history() -> np.ndarray:
    # 10 timesteps, 3 nodes; node 0 is constant, node 1 linear, node 2 noisy.
    rng = np.random.default_rng(0)
    T, N = 10, 3
    h = np.zeros((T, N))
    h[:, 0] = 5.0
    h[:, 1] = np.arange(T, dtype=float)
    h[:, 2] = rng.normal(0.0, 1.0, size=T)
    return h


class TestForecastBundle:
    def test_with_intervals_validates_shape(self) -> None:
        bundle = ForecastBundle(point=np.zeros((3, 5)), horizons_s=(60, 120, 180))
        with pytest.raises(ValueError):
            bundle.with_intervals(np.zeros((2, 5)), np.zeros((3, 5)))

    def test_with_intervals_attaches_bounds(self) -> None:
        bundle = ForecastBundle(point=np.zeros((2, 4)), horizons_s=(60, 120))
        lo = np.full((2, 4), -1.0)
        hi = np.full((2, 4), 1.0)
        new = bundle.with_intervals(lo, hi)
        np.testing.assert_array_equal(new.lower, lo)
        np.testing.assert_array_equal(new.upper, hi)


class TestHistoricalAverage:
    def test_flat_mean(self, small_history: np.ndarray) -> None:
        m = HistoricalAverage()
        m.fit(small_history)
        out = m.predict(small_history, horizons_s=[60, 120, 180])
        assert out.point.shape == (3, 3)
        # Node 0 mean is 5.0 by construction.
        assert out.point[0, 0] == pytest.approx(5.0)
        assert out.point[2, 0] == pytest.approx(5.0)
        # Node 1 mean is mean(0..9) = 4.5.
        np.testing.assert_allclose(out.point[:, 1], 4.5)

    def test_seasonal_mean(self) -> None:
        # 8 timesteps with period 4; node 0 has phase pattern [1, 2, 3, 4]
        h = np.tile(np.array([[1.0], [2.0], [3.0], [4.0]]), (2, 1))
        m = HistoricalAverage(seasonality_steps=4)
        m.fit(h)
        # Context covers 5 steps -> next phases are 1, 2, 3.
        out = m.predict(h[:5], horizons_s=[60, 120, 180])
        # phase 5%4=1, 6%4=2, 7%4=3
        np.testing.assert_allclose(out.point[:, 0], [2.0, 3.0, 4.0])

    def test_predict_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError):
            HistoricalAverage().predict(np.zeros((1, 1)), horizons_s=[60])


class TestNaiveLastValue:
    def test_repeats_last_observation(self, small_history: np.ndarray) -> None:
        m = NaiveLastValue()
        m.fit(small_history)
        out = m.predict(small_history, horizons_s=[60, 120, 180])
        assert out.point.shape == (3, 3)
        # All horizons should equal the last row.
        for i in range(3):
            np.testing.assert_array_equal(out.point[i], small_history[-1])

    def test_predict_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError):
            NaiveLastValue().predict(np.zeros((1, 1)), horizons_s=[60])


class TestAR1Forecaster:
    def test_recovers_known_coefficients(self) -> None:
        # Build a series x_{t+1} = 0.6 * x_t + 1.0 ; converges to b/(1-a) = 2.5.
        T = 200
        x = np.zeros((T, 1))
        x[0, 0] = 0.0
        for t in range(1, T):
            x[t, 0] = 0.6 * x[t - 1, 0] + 1.0
        m = AR1Forecaster()
        m.fit(x)
        assert m._a is not None and m._b is not None
        assert m._a[0] == pytest.approx(0.6, abs=1e-6)
        assert m._b[0] == pytest.approx(1.0, abs=1e-6)

    def test_handles_constant_series(self) -> None:
        h = np.full((20, 2), 7.0)
        m = AR1Forecaster()
        m.fit(h)
        # a=0 for constant series; predictions stay at the mean (= 7).
        out = m.predict(h, horizons_s=[60, 120])
        np.testing.assert_allclose(out.point, 7.0)

    def test_too_short_history_raises(self) -> None:
        m = AR1Forecaster()
        with pytest.raises(ValueError):
            m.fit(np.zeros((2, 1)))


class TestForecasterProtocol:
    def test_all_baselines_satisfy_protocol(self) -> None:
        for m in (HistoricalAverage(), NaiveLastValue(), AR1Forecaster()):
            assert isinstance(m, Forecaster)
