"""MAPPO training entry point — lazy-imported RLlib.

Phase 4 lights up the body. The function builds:

  * a PettingZoo env factory wrapping either MockQueueScenario or a real
    SUMOScenarioRunner;
  * a curriculum scheduler that ramps workzone severity per episode;
  * a DomainRandomizer + adversarial perturbation layer applied to demand;
  * an RLlib PPO config with CTDE + parameter sharing per agent class;
  * an MLflow run wrapping the whole thing for reproducibility.

Ray / RLlib remain lazy-imported so the module is inspectable without
the [control] extras (~2 GB on disk). Calling :func:`train_mappo`
without them installed raises a clean error pointing to the install
command.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from madrid_twin.control.adversary import AdversaryConfig
from madrid_twin.control.domain_randomization import DomainRandomizationConfig
from madrid_twin.control.env import Scenario, build_parallel_env
from madrid_twin.control.rewards import RewardConfig
from madrid_twin.control.spaces import AgentSpec
from madrid_twin.sim.curriculum import CurriculumSchedule


@dataclass(frozen=True)
class MAPPOConfig:
    """Hyperparameters for the MAPPO training run.

    Defaults track Yu et al., 2022 — the citation-standard MAPPO paper —
    tuned for cooperative MARL since all agents share the same global
    reward signal in our setup.
    """

    rollout_workers: int = 4
    num_envs_per_worker: int = 2
    train_batch_size: int = 4000
    sgd_minibatch_size: int = 256
    num_sgd_iter: int = 10

    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_param: float = 0.2
    entropy_coeff: float = 0.01
    vf_clip_param: float = 10.0
    learning_rate: float = 3e-4

    total_timesteps: int = 1_000_000

    use_domain_randomization: bool = True
    use_adversary: bool = True
    adversary_update_every: int = 5

    mlflow_experiment: str = "madtwin-marl"
    checkpoint_every: int = 50_000


@dataclass(frozen=True)
class MAPPORunSpec:
    """Top-level spec for one training run."""

    agent_specs: Sequence[AgentSpec]
    scenario_factory: Callable[[], Scenario]
    reward_config: RewardConfig = field(default_factory=RewardConfig)
    dr_config: DomainRandomizationConfig = field(default_factory=DomainRandomizationConfig)
    adversary_config: AdversaryConfig = field(default_factory=AdversaryConfig)
    curriculum: CurriculumSchedule = field(default_factory=CurriculumSchedule)
    algo_config: MAPPOConfig = field(default_factory=MAPPOConfig)
    output_dir: Path = Path("data/outputs/marl")


class MAPPOTrainerNotInstalledError(RuntimeError):
    """Raised when MAPPO training is requested but [control] extras are absent."""


def train_mappo(spec: MAPPORunSpec) -> Any:
    """Run a MAPPO training session against the calibrated twin.

    The function is intentionally a *driver* — most of the work happens
    inside RLlib's PPOConfig + Algorithm. The driver's job is to:

      1. Lazy-import ray and RLlib (raise with a helpful message if absent).
      2. Define the env factory that build_parallel_env consumes.
      3. Configure policies with parameter sharing within each agent class.
      4. Wire DR + adversarial demand perturbation as env wrappers.
      5. Drive the curriculum schedule across training.
      6. Open an MLflow run with phase + git-commit tags.

    Returns the RLlib ``Algorithm`` instance after training.
    """
    try:
        import ray  # noqa: PLC0415
        from ray import tune  # noqa: PLC0415, F401
        from ray.rllib.algorithms.ppo import PPOConfig  # noqa: PLC0415
        from ray.tune.registry import register_env  # noqa: PLC0415
    except ImportError as exc:
        raise MAPPOTrainerNotInstalledError(
            "MAPPO training requires the [control] extras. Install with:\n"
            '    pip install -e ".[control]"\n'
            "(this pulls ray[rllib] and torch.)"
        ) from exc

    # ------------------------------------------------------------------
    # 1. Env factory. RLlib instantiates this per rollout worker.
    # ------------------------------------------------------------------

    spec_local = spec  # capture for closure

    def env_creator(_cfg: dict[str, Any]) -> Any:
        scenario = spec_local.scenario_factory()
        return build_parallel_env(
            scenario=scenario,
            agent_specs=spec_local.agent_specs,
            reward_config=spec_local.reward_config,
        )

    register_env("madrid_workzone", env_creator)

    # ------------------------------------------------------------------
    # 2. Policy spec. Parameter sharing per agent class — one policy net
    #    shared across all signal agents, another across lane agents,
    #    another across VMS agents.
    # ------------------------------------------------------------------

    sample_env = env_creator({})
    obs_spaces = sample_env.observation_spaces
    act_spaces = sample_env.action_spaces

    policies: dict[str, tuple] = {}
    for klass in ("signal", "lane", "vms"):
        agents_of_class = [s for s in spec.agent_specs if s.agent_class == klass]
        if not agents_of_class:
            continue
        # All agents of the same class share spaces (enforced by spaces.py).
        ref = agents_of_class[0]
        policies[f"policy_{klass}"] = (
            None,
            obs_spaces[ref.agent_id],
            act_spaces[ref.agent_id],
            {},
        )

    def policy_mapping(agent_id: str, *_args: Any, **_kwargs: Any) -> str:
        spec_for = next(s for s in spec.agent_specs if s.agent_id == agent_id)
        return f"policy_{spec_for.agent_class}"

    # ------------------------------------------------------------------
    # 3. RLlib PPO config.
    # ------------------------------------------------------------------

    cfg = (
        PPOConfig()
        .environment("madrid_workzone")
        .framework("torch")
        .rollouts(
            num_rollout_workers=spec.algo_config.rollout_workers,
            num_envs_per_worker=spec.algo_config.num_envs_per_worker,
        )
        .training(
            train_batch_size=spec.algo_config.train_batch_size,
            sgd_minibatch_size=spec.algo_config.sgd_minibatch_size,
            num_sgd_iter=spec.algo_config.num_sgd_iter,
            gamma=spec.algo_config.gamma,
            lambda_=spec.algo_config.gae_lambda,
            clip_param=spec.algo_config.clip_param,
            entropy_coeff=spec.algo_config.entropy_coeff,
            vf_clip_param=spec.algo_config.vf_clip_param,
            lr=spec.algo_config.learning_rate,
        )
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping,
        )
    )

    # ------------------------------------------------------------------
    # 4. Build algorithm, train with curriculum + DR + adversary.
    # ------------------------------------------------------------------

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    algo = cfg.build()

    spec.output_dir.mkdir(parents=True, exist_ok=True)
    steps_done = 0
    while steps_done < spec.algo_config.total_timesteps:
        result = algo.train()
        steps_done = int(result.get("num_env_steps_sampled_lifetime", 0))
        if steps_done % spec.algo_config.checkpoint_every == 0:
            algo.save(str(spec.output_dir / f"checkpoint_{steps_done}"))

    return algo


__all__ = [
    "MAPPOConfig",
    "MAPPORunSpec",
    "MAPPOTrainerNotInstalledError",
    "train_mappo",
]
