"""RARL-style adversarial demand perturbation.

Pinto et al.'s Robust Adversarial RL (2017) recipe is: train a
protagonist policy and an adversary policy in alternation, where the
adversary injects worst-case demand perturbations within a bounded
budget. The protagonist learns a policy that is robust to those
perturbations — which translates, in our setting, to a controller that
holds up under demand shifts, sensor degradation, and incident-like
disturbances.

This module supplies the *perturbation primitive* used by both the
adversary's policy network and by smoke-test scripts that exercise a
hand-coded adversary. The actual learned adversary network is part of
the MAPPO training loop in :mod:`madrid_twin.control.train_mappo`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AdversaryConfig:
    """Bounds on the perturbation the adversary is allowed to apply.

    A protagonist policy trained against an adversary with **unbounded**
    perturbation budget collapses to passivity (everything looks
    adversarial, so the protagonist gives up). The published guidance —
    Pinto 2017 plus subsequent work — is that the per-step L_inf budget
    should be a small fraction of the natural scale.
    """

    # Maximum absolute perturbation as a fraction of the baseline demand
    # at each origin.
    max_relative_perturbation: float = 0.20

    # Optional total L1 budget across origins, also as a fraction of
    # baseline demand. None disables the global constraint.
    max_total_relative_budget: float | None = 0.50

    def __post_init__(self) -> None:
        if not 0.0 <= self.max_relative_perturbation <= 1.0:
            raise ValueError(
                f"max_relative_perturbation must be in [0, 1], got {self.max_relative_perturbation}"
            )
        if self.max_total_relative_budget is not None and self.max_total_relative_budget < 0:
            raise ValueError("max_total_relative_budget must be non-negative or None")


def project_perturbation(
    raw: np.ndarray,
    baseline_demand: np.ndarray,
    config: AdversaryConfig,
) -> np.ndarray:
    """Project a raw perturbation vector into the feasible adversarial set.

    Two-stage projection:
      1. Per-element clip: ``|delta_i| <= max_relative_perturbation * baseline_i``.
      2. Global L1 rescale (if a total budget is set): scale the whole
         vector down to satisfy
         ``sum |delta_i| <= max_total_relative_budget * sum baseline_i``.

    Returns the projected perturbation, ready to be added to the baseline
    demand inside the scenario runner.
    """
    if raw.shape != baseline_demand.shape:
        raise ValueError(f"raw {raw.shape} must match baseline_demand {baseline_demand.shape}")
    baseline = np.asarray(baseline_demand, dtype=np.float64)
    if (baseline < 0).any():
        raise ValueError("baseline_demand must be non-negative everywhere")

    delta = np.asarray(raw, dtype=np.float64).copy()

    # Stage 1: per-element clip.
    elem_cap = config.max_relative_perturbation * baseline
    delta = np.clip(delta, -elem_cap, elem_cap)

    # Stage 2: optional global L1 rescale.
    if config.max_total_relative_budget is not None:
        total_cap = config.max_total_relative_budget * baseline.sum()
        current_l1 = float(np.abs(delta).sum())
        if current_l1 > total_cap > 0:
            delta = delta * (total_cap / current_l1)
        elif total_cap == 0:
            delta = np.zeros_like(delta)

    return delta


def apply_perturbation(baseline_demand: np.ndarray, perturbation: np.ndarray) -> np.ndarray:
    """Return ``baseline_demand + perturbation``, floored at zero.

    Demand can't go negative — that would imply vehicles being sucked
    out of the network, which has no physical meaning. The floor is
    applied after projection.
    """
    result = baseline_demand + perturbation
    return np.maximum(0.0, result)


def random_adversary_action(
    baseline_demand: np.ndarray,
    config: AdversaryConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample a random perturbation inside the feasible set.

    Used as the *initial* adversary policy at the start of training, and
    as a sanity baseline in tests. Real adversaries are policy networks
    trained jointly with the protagonist in MAPPO.
    """
    raw = (
        rng.uniform(
            -config.max_relative_perturbation,
            config.max_relative_perturbation,
            size=baseline_demand.shape,
        )
        * baseline_demand
    )
    return project_perturbation(raw, baseline_demand, config)


__all__ = [
    "AdversaryConfig",
    "apply_perturbation",
    "project_perturbation",
    "random_adversary_action",
]
