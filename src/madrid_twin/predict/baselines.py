"""Classical forecasting baselines.

These are the non-deep baselines every ST-GNN paper compares against:
Historical Average (HA), Naive Last Value (the "persistence" baseline)
and a simple first-order autoregressive model AR(1). They are useful in
their own right — for the short horizons we operate at, HA is often
hard to beat in dense urban traffic — and they establish a floor for
the deep ST-GNN models implemented later in :mod:`madrid_twin.predict.stgnn`.

The module also defines the :class:`Forecaster` protocol that every
predictor in MADTwin implements, plus the :class:`ForecastBundle`
return type carrying point forecasts and (optionally) coverage
intervals from a downstream conformal wrapper.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass
class ForecastBundle:
    """The result of a forecast call.

    Attributes
    ----------
    point
        Shape ``(horizons, n_nodes)`` — point forecasts per future
        step and per graph node.
    horizons_s
        Forecast horizons (seconds into the future) corresponding to
        the rows of ``point``.
    lower, upper
        Optional coverage interval bounds (same shape as ``point``).
        Set by a downstream conformal wrapper; left as None by the
        raw baseline forecasters.
    """

    point: np.ndarray
    horizons_s: tuple[int, ...]
    lower: np.ndarray | None = None
    upper: np.ndarray | None = None
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def n_nodes(self) -> int:
        return int(self.point.shape[1])

    @property
    def n_horizons(self) -> int:
        return int(self.point.shape[0])

    def with_intervals(self, lower: np.ndarray, upper: np.ndarray) -> ForecastBundle:
        """Return a copy with the given coverage bounds attached."""
        if lower.shape != self.point.shape or upper.shape != self.point.shape:
            raise ValueError(
                f"Interval shape mismatch: point={self.point.shape}, "
                f"lower={lower.shape}, upper={upper.shape}"
            )
        return ForecastBundle(
            point=self.point,
            horizons_s=self.horizons_s,
            lower=lower,
            upper=upper,
            metadata=dict(self.metadata),
        )


@runtime_checkable
class Forecaster(Protocol):
    """The contract every MADTwin predictor satisfies."""

    def fit(self, history: np.ndarray) -> None:
        """Train the predictor on a 2-D (T, n_nodes) historical series."""
        ...

    def predict(self, context: np.ndarray, horizons_s: Sequence[int]) -> ForecastBundle:
        """Forecast at the requested horizons, given the recent context."""
        ...


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _validate_history(history: np.ndarray) -> np.ndarray:
    if history.ndim != 2:
        raise ValueError(f"history must be 2-D (T, n_nodes); got shape {history.shape}")
    if history.shape[0] < 1:
        raise ValueError("history must contain at least one timestep")
    return np.asarray(history, dtype=np.float64)


def _validate_context(context: np.ndarray, n_nodes: int) -> np.ndarray:
    if context.ndim != 2:
        raise ValueError(f"context must be 2-D (T_ctx, n_nodes); got shape {context.shape}")
    if context.shape[1] != n_nodes:
        raise ValueError(f"context has {context.shape[1]} nodes, expected {n_nodes}")
    return np.asarray(context, dtype=np.float64)


# ---------------------------------------------------------------------------
# Historical Average baseline.
# ---------------------------------------------------------------------------


@dataclass
class HistoricalAverage:
    """Predict each node's per-step historical mean.

    A surprisingly strong baseline on urban traffic at short horizons —
    DCRNN's own paper reports HA within a few percent of the best deep
    model on PEMS-BAY at 5-minute horizons.
    """

    seasonality_steps: int | None = None
    _means: np.ndarray | None = None

    def fit(self, history: np.ndarray) -> None:
        h = _validate_history(history)
        if self.seasonality_steps is None or self.seasonality_steps <= 0:
            # Flat mean per node, broadcast across all horizons.
            self._means = h.mean(axis=0, keepdims=True)  # shape (1, n_nodes)
        else:
            # One mean per phase-of-cycle (e.g. minute-of-day index modulo).
            S = self.seasonality_steps
            T, N = h.shape
            buckets = np.full((S, N), np.nan, dtype=np.float64)
            counts = np.zeros((S, N), dtype=np.int64)
            for t in range(T):
                phase = t % S
                buckets[phase] = np.where(
                    np.isnan(buckets[phase]),
                    h[t],
                    buckets[phase] + h[t],
                )
                counts[phase] += 1
            buckets = np.where(counts > 0, buckets / np.maximum(counts, 1), 0.0)
            self._means = buckets

    def predict(self, context: np.ndarray, horizons_s: Sequence[int]) -> ForecastBundle:
        if self._means is None:
            raise RuntimeError("HistoricalAverage.fit must be called before predict")
        n_nodes = self._means.shape[1]
        _validate_context(context, n_nodes)
        horizons = tuple(int(h) for h in horizons_s)

        if self.seasonality_steps is None or self.seasonality_steps <= 0:
            point = np.repeat(self._means, len(horizons), axis=0)
        else:
            S = self.seasonality_steps
            base = context.shape[0]
            point = np.stack([self._means[(base + i) % S] for i in range(len(horizons))])

        return ForecastBundle(
            point=point,
            horizons_s=horizons,
            metadata={"model": "HistoricalAverage"},
        )


# ---------------------------------------------------------------------------
# Naive last value (persistence).
# ---------------------------------------------------------------------------


@dataclass
class NaiveLastValue:
    """Predict that the last observed value repeats at every future horizon."""

    _n_nodes: int | None = None

    def fit(self, history: np.ndarray) -> None:
        h = _validate_history(history)
        self._n_nodes = int(h.shape[1])

    def predict(self, context: np.ndarray, horizons_s: Sequence[int]) -> ForecastBundle:
        if self._n_nodes is None:
            raise RuntimeError("NaiveLastValue.fit must be called before predict")
        c = _validate_context(context, self._n_nodes)
        if c.shape[0] < 1:
            raise ValueError("context must have at least one row")
        horizons = tuple(int(h) for h in horizons_s)
        last = c[-1:, :]  # shape (1, n_nodes)
        point = np.repeat(last, len(horizons), axis=0)
        return ForecastBundle(
            point=point,
            horizons_s=horizons,
            metadata={"model": "NaiveLastValue"},
        )


# ---------------------------------------------------------------------------
# AR(1) per node.
# ---------------------------------------------------------------------------


@dataclass
class AR1Forecaster:
    """Per-node first-order autoregressive: ``x_{t+1} = a*x_t + b``.

    Coefficients are fit by OLS on the lagged training pairs. Forecasts
    are produced by iterating the recursion forward — i.e. for horizon
    ``h`` we apply the AR(1) update ``h`` times.
    """

    _a: np.ndarray | None = None  # shape (n_nodes,)
    _b: np.ndarray | None = None  # shape (n_nodes,)

    def fit(self, history: np.ndarray) -> None:
        h = _validate_history(history)
        if h.shape[0] < 3:
            raise ValueError("AR1Forecaster needs at least 3 timesteps")
        x_prev = h[:-1, :]  # (T-1, n)
        x_next = h[1:, :]  # (T-1, n)

        # Closed-form OLS per node:
        #   a = cov(x_prev, x_next) / var(x_prev)
        #   b = mean(x_next) - a * mean(x_prev)
        var_prev = x_prev.var(axis=0)
        # Avoid division by zero for constant series.
        safe_var = np.where(var_prev > 1e-12, var_prev, 1.0)
        cov = ((x_prev - x_prev.mean(axis=0)) * (x_next - x_next.mean(axis=0))).mean(axis=0)
        a = np.where(var_prev > 1e-12, cov / safe_var, 0.0)
        b = x_next.mean(axis=0) - a * x_prev.mean(axis=0)
        self._a = a
        self._b = b

    def predict(self, context: np.ndarray, horizons_s: Sequence[int]) -> ForecastBundle:
        if self._a is None or self._b is None:
            raise RuntimeError("AR1Forecaster.fit must be called before predict")
        n_nodes = self._a.shape[0]
        c = _validate_context(context, n_nodes)
        if c.shape[0] < 1:
            raise ValueError("context must have at least one row")
        horizons = tuple(int(h) for h in horizons_s)
        n_steps = len(horizons)

        # Iterate the recursion forward n_steps from the last observation.
        out = np.empty((n_steps, n_nodes), dtype=np.float64)
        x = c[-1, :].copy()
        for i in range(n_steps):
            x = self._a * x + self._b
            out[i, :] = x

        return ForecastBundle(
            point=out,
            horizons_s=horizons,
            metadata={"model": "AR1"},
        )


__all__ = [
    "AR1Forecaster",
    "ForecastBundle",
    "Forecaster",
    "HistoricalAverage",
    "NaiveLastValue",
]
