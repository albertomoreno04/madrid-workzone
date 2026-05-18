"""Gymnasium observation & action spaces for the heterogeneous agents.

Three agent classes:

- **Signal agents** observe local queues + predicted travel time +
  conformal uncertainty, and choose a discrete phase index per step.
- **Lane-manager agents** observe local saturation and choose a
  discrete lane re-allocation among a small set of pre-validated
  physically-feasible configurations.
- **VMS / detour-advisor agents** observe upstream queue + downstream
  free-flow, and choose which (pre-validated) detour message to display.

The spaces themselves are tiny numpy-shape declarations. Gymnasium is
lazy-imported so this module can be inspected without the [control]
extras installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Default observation feature counts, documented so they match the env.
SIGNAL_OBS_DIM: int = 8
LANE_OBS_DIM: int = 4
VMS_OBS_DIM: int = 4


@dataclass(frozen=True)
class AgentSpec:
    """A single agent's class + its discrete action cardinality."""

    agent_id: str
    agent_class: str  # "signal" | "lane" | "vms"
    n_actions: int

    def __post_init__(self) -> None:
        if self.agent_class not in ("signal", "lane", "vms"):
            raise ValueError(
                f"agent_class must be 'signal' | 'lane' | 'vms', got {self.agent_class!r}"
            )
        if self.n_actions <= 0:
            raise ValueError("n_actions must be positive")


def _obs_dim_for_class(agent_class: str) -> int:
    return {
        "signal": SIGNAL_OBS_DIM,
        "lane": LANE_OBS_DIM,
        "vms": VMS_OBS_DIM,
    }[agent_class]


def build_observation_space(agent_class: str) -> Any:
    """Return a gymnasium ``Box`` space of the right dimensionality.

    Imports ``gymnasium`` lazily so the module is importable without
    the [control] extras. Raises a clear error if the extras are
    missing.
    """
    try:
        import numpy as np
        from gymnasium import spaces  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - dev env guard
        raise ImportError(
            'Gymnasium not available. Install with: pip install -e ".[control]"'
        ) from exc
    dim = _obs_dim_for_class(agent_class)
    return spaces.Box(low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float32)


def build_action_space(n_actions: int) -> Any:
    """Return a gymnasium ``Discrete`` space."""
    try:
        from gymnasium import spaces  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            'Gymnasium not available. Install with: pip install -e ".[control]"'
        ) from exc
    if n_actions <= 0:
        raise ValueError("n_actions must be positive")
    return spaces.Discrete(n_actions)


__all__ = [
    "LANE_OBS_DIM",
    "SIGNAL_OBS_DIM",
    "VMS_OBS_DIM",
    "AgentSpec",
    "build_action_space",
    "build_observation_space",
]
