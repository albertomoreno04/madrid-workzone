"""Split-conformal prediction wrapper.

Given any base :class:`Forecaster`, this module produces prediction
intervals with a distribution-free, finite-sample coverage guarantee:

    P( y_true in [lower, upper] )  >=  1 - alpha

The construction is split-conformal (Vovk & Shafer; popularised in the
ML community by Romano et al.): split the training data into a fit
set and a calibration set, fit the base forecaster on the former,
collect residuals on the latter, take the empirical (1-alpha)-quantile
of those residuals, and use it as the half-width of the interval.

We additionally support **adaptive split-conformal**: a separate
calibration quantile per horizon and per node. This is the right
choice for traffic, where heteroskedasticity across nodes and across
forecast horizons is dramatic (long-horizon variance is much larger
than 5-minute variance).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from madrid_twin.predict.baselines import ForecastBundle, Forecaster


def _safe_quantile(values: np.ndarray, alpha: float) -> float:
    """Empirical (1-alpha)-quantile with the standard conformal correction.

    Vovk's correction adds 1 to numerator and denominator to keep the
    quantile valid under finite-sample exchangeability:

        q_{(n+1)(1-alpha)} / n
    """
    if values.size == 0:
        raise ValueError("Cannot compute quantile on empty residual array")
    n = values.size
    q_level = min(1.0, math_ceil((1.0 - alpha) * (n + 1)) / n)
    # numpy.quantile uses linear interpolation by default; that's fine here.
    return float(np.quantile(values, q_level))


def math_ceil(x: float) -> int:
    """Standalone ceiling — avoids pulling math just for one function."""
    i = int(x)
    return i if x == i else i + 1


@dataclass
class SplitConformalPredictor:
    """Wraps a base ``Forecaster`` with coverage-guaranteed intervals.

    Parameters
    ----------
    base
        The point-forecast model. Must already implement the
        ``Forecaster`` protocol.
    alpha
        Mis-coverage level. ``alpha=0.1`` means 90% coverage.
    per_horizon_per_node
        When True (default), one calibration quantile is computed for
        each (horizon, node) pair. When False, a single global quantile
        is used.
    """

    base: Forecaster
    alpha: float = 0.1
    per_horizon_per_node: bool = True
    _quantile: np.ndarray | float | None = None
    _horizons_s: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1); got {self.alpha}")

    def calibrate(
        self,
        contexts: Sequence[np.ndarray],
        targets: Sequence[np.ndarray],
        horizons_s: Sequence[int],
    ) -> None:
        """Collect residuals on held-out (context, target) pairs.

        Each ``contexts[i]`` is the history slice fed to the model; each
        ``targets[i]`` has shape ``(len(horizons_s), n_nodes)`` and
        contains the true future values at the requested horizons.
        """
        if len(contexts) != len(targets):
            raise ValueError(f"len(contexts)={len(contexts)} != len(targets)={len(targets)}")
        if not contexts:
            raise ValueError("Need at least one calibration pair")

        horizons = tuple(int(h) for h in horizons_s)

        # First-pass shape check using the very first prediction.
        residuals: list[np.ndarray] = []
        for ctx, tgt in zip(contexts, targets, strict=True):
            bundle = self.base.predict(ctx, horizons)
            if tgt.shape != bundle.point.shape:
                raise ValueError(
                    f"Target shape {tgt.shape} doesn't match point-forecast "
                    f"shape {bundle.point.shape}"
                )
            residuals.append(np.abs(tgt - bundle.point))

        stacked = np.stack(residuals, axis=0)  # (n_cal, n_horizons, n_nodes)

        if self.per_horizon_per_node:
            # One quantile per (horizon, node).
            n_cal = stacked.shape[0]
            # We need the (1-alpha) quantile along axis 0.
            n = n_cal
            q_level = min(1.0, math_ceil((1.0 - self.alpha) * (n + 1)) / n)
            self._quantile = np.quantile(stacked, q_level, axis=0)
        else:
            self._quantile = _safe_quantile(stacked.ravel(), self.alpha)

        self._horizons_s = horizons

    def predict(self, context: np.ndarray, horizons_s: Sequence[int]) -> ForecastBundle:
        """Forecast with calibrated intervals attached."""
        if self._quantile is None:
            raise RuntimeError("calibrate() must be called before predict()")

        horizons = tuple(int(h) for h in horizons_s)
        if self._horizons_s is not None and horizons != self._horizons_s:
            raise ValueError(
                f"Predict horizons {horizons} differ from calibration horizons {self._horizons_s}"
            )

        bundle = self.base.predict(context, horizons)

        if isinstance(self._quantile, np.ndarray):
            q = self._quantile
            if q.shape != bundle.point.shape:
                raise RuntimeError(
                    f"Calibration quantile shape {q.shape} "
                    f"doesn't match point shape {bundle.point.shape}"
                )
        else:
            q = np.full_like(bundle.point, float(self._quantile))

        return bundle.with_intervals(
            lower=bundle.point - q,
            upper=bundle.point + q,
        )


__all__ = ["SplitConformalPredictor"]
