"""Composite reward for the MARL controller.

Combines five terms — network delay, throughput, queue-spillback penalty,
emissions, and a Gini-based equity penalty across census zones — into a
single scalar reward signal. Weights are exposed through
:class:`RewardConfig` so the equity vs. efficiency trade-off can be
ablated at training time.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class RewardConfig:
    """Per-term weights for the composite reward."""

    w_delay: float = 1.0
    w_throughput: float = 1.0
    w_spillback: float = 5.0
    w_emissions: float = 0.1
    w_equity: float = 2.0

    spillback_queue_fraction: float = 0.8

    def __post_init__(self) -> None:
        for name, val in self.__dict__.items():
            if val < 0:
                raise ValueError(f"RewardConfig.{name} must be non-negative, got {val}")


@dataclass
class NetworkState:
    """Snapshot of the network at one decision step."""

    delays_s: np.ndarray
    throughput_veh: int
    queue_lengths_m: np.ndarray
    link_lengths_m: np.ndarray
    emissions_g: float
    travel_time_impact_per_zone_s: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.float64)
    )

    def __post_init__(self) -> None:
        if self.queue_lengths_m.shape != self.link_lengths_m.shape:
            raise ValueError(
                f"queue_lengths_m {self.queue_lengths_m.shape} must match "
                f"link_lengths_m {self.link_lengths_m.shape}"
            )
        if (self.link_lengths_m <= 0).any():
            raise ValueError("link_lengths_m must be strictly positive everywhere")
        if self.throughput_veh < 0:
            raise ValueError("throughput_veh must be non-negative")
        if self.emissions_g < 0:
            raise ValueError("emissions_g must be non-negative")


def mean_delay(state: NetworkState) -> float:
    """Mean delay per vehicle (seconds). Zero when no vehicles are in the system."""
    if state.delays_s.size == 0:
        return 0.0
    return float(state.delays_s.mean())


def throughput(state: NetworkState) -> float:
    return float(state.throughput_veh)


def spillback_penalty(state: NetworkState, cfg: RewardConfig) -> float:
    """Mean violation of the spillback fraction across links."""
    ratios = state.queue_lengths_m / state.link_lengths_m
    excess = np.maximum(0.0, ratios - cfg.spillback_queue_fraction)
    return float(excess.mean()) if excess.size else 0.0


def emissions(state: NetworkState) -> float:
    return state.emissions_g


def gini(values: np.ndarray) -> float:
    """Gini coefficient of a non-negative 1-D array."""
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return 0.0
    if (v < 0).any():
        v = v - v.min()
    if v.sum() == 0:
        return 0.0
    mad = np.abs(v[:, None] - v[None, :]).mean()
    return float(mad / (2.0 * v.mean()))


def equity_penalty(state: NetworkState) -> float:
    return gini(state.travel_time_impact_per_zone_s)


def composite_reward(state: NetworkState, cfg: RewardConfig | None = None) -> float:
    """The single scalar that MAPPO maximises."""
    if cfg is None:
        cfg = RewardConfig()
    return (
        -cfg.w_delay * mean_delay(state)
        + cfg.w_throughput * throughput(state)
        - cfg.w_spillback * spillback_penalty(state, cfg)
        - cfg.w_emissions * emissions(state)
        - cfg.w_equity * equity_penalty(state)
    )


__all__ = [
    "NetworkState",
    "RewardConfig",
    "composite_reward",
    "emissions",
    "equity_penalty",
    "gini",
    "mean_delay",
    "spillback_penalty",
    "throughput",
]
