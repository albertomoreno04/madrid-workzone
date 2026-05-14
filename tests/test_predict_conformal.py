"""Tests for the split-conformal prediction wrapper.

The key correctness property is nominal coverage: across a large
calibration set drawn from the same distribution as the test data,
the (1-alpha)% empirical coverage should be close to (1-alpha).
"""

from __future__ import annotations

import numpy as np
import pytest

from madrid_twin.predict.baselines import (
    HistoricalAverage,
    NaiveLastValue,
)
from madrid_twin.predict.conformal import SplitConformalPredictor


class TestSplitConformalSanity:
    def test_alpha_validation(self) -> None:
        with pytest.raises(ValueError):
            SplitConformalPredictor(base=NaiveLastValue(), alpha=0.0)
        with pytest.raises(ValueError):
            SplitConformalPredictor(base=NaiveLastValue(), alpha=1.0)

    def test_predict_before_calibrate_raises(self) -> None:
        m = SplitConformalPredictor(base=NaiveLastValue(), alpha=0.1)
        with pytest.raises(RuntimeError):
            m.predict(np.zeros((1, 1)), horizons_s=[60])

    def test_horizons_must_match_calibration(self) -> None:
        base = NaiveLastValue()
        base.fit(np.zeros((5, 1)))
        cp = SplitConformalPredictor(base=base, alpha=0.1)
        ctx = np.zeros((1, 1))
        tgt = np.zeros((1, 1))
        cp.calibrate(contexts=[ctx], targets=[tgt], horizons_s=[60])
        with pytest.raises(ValueError):
            cp.predict(ctx, horizons_s=[120])


class TestSplitConformalIntervals:
    def test_intervals_contain_point_forecast(self) -> None:
        rng = np.random.default_rng(0)
        T = 50
        h = rng.normal(0.0, 1.0, size=(T, 1))
        base = NaiveLastValue()
        base.fit(h)

        # Build (context, target) pairs by sliding a window.
        contexts = [h[i : i + 5] for i in range(T - 6)]
        targets = [h[i + 5 : i + 6] for i in range(T - 6)]

        cp = SplitConformalPredictor(base=base, alpha=0.1)
        cp.calibrate(contexts=contexts, targets=targets, horizons_s=[60])

        out = cp.predict(h[-5:], horizons_s=[60])
        assert out.lower is not None and out.upper is not None
        assert np.all(out.lower <= out.point)
        assert np.all(out.upper >= out.point)

    def test_nominal_coverage_on_synthetic_data(self) -> None:
        # Generate stationary noise; HA is the right model, residuals are ~N(0, 1).
        rng = np.random.default_rng(42)
        T_total = 2000
        h = rng.normal(0.0, 1.0, size=(T_total, 1))

        # Train HA on first chunk, calibrate on second, evaluate on third.
        T_train = 800
        T_cal = 600
        train, cal, test = (
            h[:T_train],
            h[T_train : T_train + T_cal],
            h[T_train + T_cal :],
        )

        ctx_len = 5
        base = HistoricalAverage()
        base.fit(train)

        cal_ctx = [cal[i : i + ctx_len] for i in range(len(cal) - ctx_len - 1)]
        cal_tgt = [cal[i + ctx_len : i + ctx_len + 1] for i in range(len(cal) - ctx_len - 1)]

        cp = SplitConformalPredictor(base=base, alpha=0.1, per_horizon_per_node=False)
        cp.calibrate(contexts=cal_ctx, targets=cal_tgt, horizons_s=[1])

        # Evaluate coverage on the test split.
        covered = 0
        n_eval = 0
        for i in range(len(test) - ctx_len - 1):
            ctx = test[i : i + ctx_len]
            true_next = test[i + ctx_len, 0]
            out = cp.predict(ctx, horizons_s=[1])
            lo, hi = float(out.lower[0, 0]), float(out.upper[0, 0])
            if lo <= true_next <= hi:
                covered += 1
            n_eval += 1

        coverage = covered / n_eval
        # Nominal target is 0.9; allow generous tolerance to absorb finite-sample noise.
        assert coverage >= 0.85, f"coverage {coverage:.3f} fell below 0.85"
        assert coverage <= 1.0
