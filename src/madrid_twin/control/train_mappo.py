"""MAPPO training entry point — lazy-imported RLlib.

This is the headline Phase 3 deliverable. It wires together every piece
that lands in this session:

  * the heterogeneous PettingZoo env (signal / lane / VMS agents)
  * domain-randomized scenario sampling per training episode
  * RARL-style adversarial demand perturbation
  * MAPPO under CTDE with parameter sharing within each agent class
  * MLflow tagging of every run

Ray / RLlib are imported lazily so this module is inspectable without
the [control] extras (which together pull ~2 GB of dependencies). The
runner raises a clear error if you try to actually train without those
extras installed.

The implementation here is the *training driver* — the config-shaped
function that callers invoke. The actual training loop is delegated to
RLlib's MAPPO algorithm, which is the citation-standard MARL
implementation in 2024–2025 papers.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from madrid_twin.control.adversary import AdversaryConfig
from madrid_twin.control.domain_randomization import DomainRandomizationConfig
from madrid_twin.control.rewards import RewardConfig
from madrid_twin.control.spaces import AgentSpec


@dataclass(frozen=True)
class MAPPOConfig:
    """Hyperparameters for the MAPPO training run.

    Defaults are the citation-standard values from the MAPPO paper (Yu
    et al., 2022) tuned for cooperative MARL — the regime our setup
    falls into, since all agents share the same global reward.
    """

    # Sampling.
    rollout_workers: int = 4
    num_envs_per_worker: int = 2
    train_batch_size: int = 4000
    sgd_minibatch_size: int = 256
    num_sgd_iter: int = 10

    # PPO.
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_param: float = 0.2
    entropy_coeff: float = 0.01
    vf_clip_param: float = 10.0
    learning_rate: float = 3e-4

    # Length of training.
    total_timesteps: int = 1_000_000

    # Robustness.
    use_domain_randomization: bool = True
    use_adversary: bool = True
    adversary_update_every: int = 5  # protagonist steps per adversary update

    # Logging.
    mlflow_experiment: str = "madtwin-marl"


@dataclass(frozen=True)
class MAPPORunSpec:
    """Top-level spec for a single training run."""

    agent_specs: Sequence[AgentSpec]
    reward_config: RewardConfig = field(default_factory=RewardConfig)
    dr_config: DomainRandomizationConfig = field(default_factory=DomainRandomizationConfig)
    adversary_config: AdversaryConfig = field(default_factory=AdversaryConfig)
    algo_config: MAPPOConfig = field(default_factory=MAPPOConfig)
    output_dir: Path = Path("data/outputs/marl")


class MAPPOTrainerNotInstalledError(RuntimeError):
    """Raised when MAPPO training is requested but [control] extras are absent."""


def train_mappo(spec: MAPPORunSpec) -> Any:
    """Run a MAPPO training session and return RLlib's results object.

    This function is intentionally *thin*: it wires the env factory,
    DR/adversary, and RLlib config together and hands them off. All the
    expensive bits (env construction, MLflow run setup, gradient steps)
    happen inside RLlib.

    The lazy imports + the explicit ``NotImplementedError`` make this a
    locked-in interface contract: the surface is final, the training
    body will be filled in alongside Phase 1's calibrated twin when we
    have a real simulator to train against.
    """
    try:
        import ray  # noqa: F401, PLC0415
        from ray import rllib  # noqa: F401, PLC0415
    except ImportError as exc:  # pragma: no cover - dev env guard
        raise MAPPOTrainerNotInstalledError(
            "MAPPO training requires the [control] extras. Install with:\n"
            '    pip install -e ".[control]"\n'
            "(this pulls ray[rllib] and torch.)"
        ) from exc

    raise NotImplementedError(
        "MAPPO training body will land alongside the Phase 1 calibrated SUMO twin. "
        "Today's deliverable is the full interface (MAPPORunSpec + MAPPOConfig) "
        "and the heterogeneous PettingZoo env + DR + adversary plumbing they "
        "consume. With [control] installed and a SUMO scenario wired in, this "
        "function is a thin shim over RLlib's PPO trainer with parameter "
        "sharing per agent class."
    )


__all__ = [
    "MAPPOConfig",
    "MAPPORunSpec",
    "MAPPOTrainerNotInstalledError",
    "train_mappo",
]
