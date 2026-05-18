"""Control layer — heterogeneous multi-agent reinforcement learning.

Module-load surface is intentionally narrow: only the pure-numpy items
are re-exported here. The PettingZoo env, the gymnasium space builders,
and the RLlib MAPPO trainer are reached through their submodules
(`madrid_twin.control.env`, `.spaces`, `.train_mappo`) so the package
imports cleanly without the [control] extras.
"""

from __future__ import annotations

from madrid_twin.control.adversary import (
    AdversaryConfig,
    apply_perturbation,
    project_perturbation,
    random_adversary_action,
)
from madrid_twin.control.baselines import (
    FixedTimeController,
    IntersectionState,
    MaxPressureController,
)
from madrid_twin.control.domain_randomization import (
    DomainRandomizationConfig,
    DomainRandomizer,
    RandomizedScenario,
)
from madrid_twin.control.rewards import (
    NetworkState,
    RewardConfig,
    composite_reward,
    equity_penalty,
    gini,
    mean_delay,
    spillback_penalty,
    throughput,
)
from madrid_twin.control.spaces import (
    LANE_OBS_DIM,
    SIGNAL_OBS_DIM,
    VMS_OBS_DIM,
    AgentSpec,
)

__all__ = [
    "AdversaryConfig",
    "AgentSpec",
    "DomainRandomizationConfig",
    "DomainRandomizer",
    "FixedTimeController",
    "IntersectionState",
    "LANE_OBS_DIM",
    "MaxPressureController",
    "NetworkState",
    "RandomizedScenario",
    "RewardConfig",
    "SIGNAL_OBS_DIM",
    "VMS_OBS_DIM",
    "apply_perturbation",
    "composite_reward",
    "equity_penalty",
    "gini",
    "mean_delay",
    "project_perturbation",
    "random_adversary_action",
    "spillback_penalty",
    "throughput",
]
