"""W-SPSA OD calibration.

Weighted Simultaneous Perturbation Stochastic Approximation, the workhorse
algorithm for offline SUMO demand calibration (Tympakianaki et al., 2015).
The problem: find an Origin-Destination matrix theta that, when fed to a
simulator, reproduces observed detector counts. The OD has thousands of
unknowns; counts give tens-to-hundreds of constraints — heavily
underdetermined. W-SPSA threads the needle by:

  1. Drawing one random ±delta vector across **all** OD entries jointly.
  2. Running the simulator twice (theta + c*delta, theta - c*delta).
  3. Forming a *two-sided* gradient estimate from the count-mismatch
     difference, scaled by 1/(2*c*delta_i) per entry.
  4. Taking a step theta -= a_k * grad_hat.

The "weighted" part rescales the gradient per OD entry by the inverse of
its baseline magnitude, so we update small-flow ODs less aggressively
than large-flow ones — which is essential to avoid oscillation on the
long tail of small-volume cells.

This module exposes the algorithm in a sim-agnostic form: the simulator
appears as a callable ``loss_fn(theta) -> float``. A test on a synthetic
quadratic objective verifies convergence to a known optimum; the
end-to-end SUMO version plugs in once Phase 4's :mod:`madrid_twin.sim.runner`
is wired up.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

LossFn = Callable[[np.ndarray], float]


@dataclass(frozen=True)
class WSPSAConfig:
    """Tunables for the W-SPSA loop.

    The Spall-recommended defaults (a, c, alpha, gamma, A) are widely
    cited; we expose them so the experiment harness can sweep.
    """

    # Step-size schedule a_k = a / (k + 1 + A)^alpha.
    a: float = 0.05
    A: float = 10.0
    alpha: float = 0.602

    # Perturbation magnitude schedule c_k = c / (k + 1)^gamma.
    c: float = 0.1
    gamma: float = 0.101

    # Number of iterations.
    max_iter: int = 200

    # Floor on OD entries (vehicles); demand can't go negative.
    min_value: float = 0.0

    # Random seed (Bernoulli sign vectors come from this).
    seed: int = 0

    def __post_init__(self) -> None:
        for name, val in (
            ("a", self.a),
            ("c", self.c),
            ("alpha", self.alpha),
            ("gamma", self.gamma),
        ):
            if val <= 0:
                raise ValueError(f"WSPSAConfig.{name} must be > 0, got {val}")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be positive")


@dataclass
class WSPSAReport:
    """Per-iteration trace of a calibration run."""

    initial_loss: float
    final_loss: float
    losses: list[float] = field(default_factory=list)
    n_iter: int = 0
    converged_by_max_iter: bool = False


def wspsa(
    theta0: np.ndarray,
    loss_fn: LossFn,
    config: WSPSAConfig | None = None,
    *,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, WSPSAReport]:
    """Run W-SPSA to minimise ``loss_fn`` starting from ``theta0``.

    Parameters
    ----------
    theta0
        Shape ``(n_od,)``. Initial OD vector (typically a gravity-model
        prior).
    loss_fn
        Maps a candidate ``theta`` to a scalar loss. In production this
        runs the simulator and computes the count-matching error; in
        tests it's a synthetic quadratic.
    config
        Hyperparameters. Defaults to :class:`WSPSAConfig`.
    weights
        Per-entry gradient weights (shape ``(n_od,)``). When omitted,
        defaults to ``1 / max(theta0_i, 1)`` — the W-SPSA recipe that
        scales small-flow ODs proportionally less.

    Returns
    -------
    A pair ``(theta_star, report)`` with the calibrated OD and a per-
    iteration loss trace.
    """
    cfg = config or WSPSAConfig()
    rng = np.random.default_rng(cfg.seed)

    theta = np.asarray(theta0, dtype=np.float64).copy()
    if theta.ndim != 1:
        raise ValueError(f"theta0 must be 1-D, got shape {theta.shape}")

    n = theta.size
    if weights is None:
        weights = 1.0 / np.maximum(theta, 1.0)
    else:
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != theta.shape:
            raise ValueError(f"weights shape {weights.shape} must match theta {theta.shape}")

    initial_loss = float(loss_fn(theta))
    losses: list[float] = [initial_loss]

    for k in range(cfg.max_iter):
        a_k = cfg.a / ((k + 1 + cfg.A) ** cfg.alpha)
        c_k = cfg.c / ((k + 1) ** cfg.gamma)

        # Bernoulli ±1 perturbation vector — the SPSA defining trick.
        delta = rng.choice(np.array([-1.0, 1.0]), size=n)

        theta_plus = np.maximum(cfg.min_value, theta + c_k * delta)
        theta_minus = np.maximum(cfg.min_value, theta - c_k * delta)

        loss_plus = float(loss_fn(theta_plus))
        loss_minus = float(loss_fn(theta_minus))

        # Two-sided SPSA gradient estimate.
        grad_hat = (loss_plus - loss_minus) / (2.0 * c_k * delta)

        # Apply per-entry weights then take the step.
        theta = theta - a_k * weights * grad_hat
        theta = np.maximum(cfg.min_value, theta)

        losses.append(float(loss_fn(theta)))

    return theta, WSPSAReport(
        initial_loss=initial_loss,
        final_loss=losses[-1],
        losses=losses,
        n_iter=cfg.max_iter,
        converged_by_max_iter=True,
    )


# ---------------------------------------------------------------------------
# Convenience: count-matching loss for use with a SUMO simulator wrapper.
# ---------------------------------------------------------------------------


def count_matching_loss(
    simulated_counts: np.ndarray,
    observed_counts: np.ndarray,
    *,
    weights: np.ndarray | None = None,
) -> float:
    """Weighted squared error between simulated and observed detector counts.

    Parameters
    ----------
    simulated_counts, observed_counts
        Same shape ``(n_detectors,)``. Vehicles-per-hour averages over the
        calibration window.
    weights
        Per-detector weights, e.g. inverse-variance from the upstream
        detector. Defaults to uniform.

    Returns
    -------
    The scalar loss ``sum(w_i * (sim_i - obs_i)^2) / sum(w_i)``.
    """
    s = np.asarray(simulated_counts, dtype=np.float64)
    o = np.asarray(observed_counts, dtype=np.float64)
    if s.shape != o.shape:
        raise ValueError(f"simulated_counts {s.shape} must match observed_counts {o.shape}")
    if weights is None:
        w = np.ones_like(s)
    else:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != s.shape:
            raise ValueError(f"weights {w.shape} must match counts {s.shape}")
        if (w < 0).any():
            raise ValueError("weights must be non-negative")
    total_w = float(w.sum())
    if total_w == 0:
        return 0.0
    return float((w * (s - o) ** 2).sum() / total_w)


__all__ = [
    "WSPSAConfig",
    "WSPSAReport",
    "count_matching_loss",
    "wspsa",
]
