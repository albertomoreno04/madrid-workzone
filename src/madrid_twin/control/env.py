"""PettingZoo ``ParallelEnv`` wrapping a mock queue scenario.

Phase 3 ships the *environment skeleton* — observation/action spaces,
step/reset semantics, reward dispatch — over a deterministic mock
network. The real SUMO-backed scenario plugs into the same
:class:`MockQueueScenario` abstraction in Phase 4, when Phase 1's
calibrated twin is producing trajectories.

PettingZoo + Gymnasium are lazy-imported so the module is inspectable
without the [control] extras.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from madrid_twin.control.baselines import IntersectionState
from madrid_twin.control.rewards import NetworkState, RewardConfig, composite_reward
from madrid_twin.control.spaces import (
    AgentSpec,
    build_action_space,
    build_observation_space,
)


@dataclass
class MockQueueScenario:
    """A tiny deterministic queue model for end-to-end env tests."""

    n_intersections: int = 4
    n_phases_per_intersection: int = 4
    arrival_rate: float = 5.0
    service_rate: float = 8.0
    queue_capacity_m: float = 200.0
    veh_length_m: float = 7.5
    step_s: float = 5.0
    seed: int = 0

    _queues: np.ndarray | None = None
    _rng: np.random.Generator | None = None
    _step: int = 0

    def reset(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        self._queues = np.zeros(
            (self.n_intersections, self.n_phases_per_intersection),
            dtype=np.float64,
        )
        self._step = 0

    def step(self, phase_per_intersection: Sequence[int]) -> NetworkState:
        if self._queues is None or self._rng is None:
            raise RuntimeError("MockQueueScenario.reset() must be called first")

        noise = self._rng.normal(0.0, 0.5, size=self._queues.shape)
        self._queues += np.maximum(0.0, self.arrival_rate + noise)

        throughput = 0
        for i, phase in enumerate(phase_per_intersection):
            served = min(self._queues[i, phase], self.service_rate)
            self._queues[i, phase] -= served
            throughput += int(round(served))

        n_links = self.n_intersections * self.n_phases_per_intersection
        queue_lengths_m = self._queues.ravel() * self.veh_length_m
        link_lengths_m = np.full(n_links, self.queue_capacity_m)

        vehicles_waiting = self._queues.sum()
        delays_s = np.full(int(vehicles_waiting), self.step_s)
        emissions_g = 0.5 * vehicles_waiting
        per_zone_impact = self._queues.mean(axis=1)

        self._step += 1

        return NetworkState(
            delays_s=delays_s,
            throughput_veh=throughput,
            queue_lengths_m=queue_lengths_m,
            link_lengths_m=link_lengths_m,
            emissions_g=emissions_g,
            travel_time_impact_per_zone_s=per_zone_impact,
        )

    def intersection_states(self) -> list[IntersectionState]:
        if self._queues is None:
            raise RuntimeError("MockQueueScenario.reset() must be called first")
        states: list[IntersectionState] = []
        for i in range(self.n_intersections):
            inc = self._queues[i].copy()
            out = np.zeros_like(inc)
            states.append(
                IntersectionState(
                    intersection_id=f"int_{i}",
                    n_phases=self.n_phases_per_intersection,
                    incoming_queues=inc,
                    outgoing_queues=out,
                )
            )
        return states


def build_parallel_env(
    scenario: MockQueueScenario,
    agent_specs: Sequence[AgentSpec],
    reward_config: RewardConfig | None = None,
    *,
    max_steps: int = 720,
) -> Any:
    """Construct a PettingZoo ``ParallelEnv`` over a scenario + agent set."""
    try:
        from pettingzoo import ParallelEnv
    except ImportError as exc:
        raise ImportError(
            'PettingZoo not available. Install with: pip install -e ".[control]"'
        ) from exc

    cfg = reward_config or RewardConfig()
    specs = list(agent_specs)
    signal_agents = [s for s in specs if s.agent_class == "signal"]
    if not signal_agents:
        raise ValueError("At least one signal agent is required")

    class MadridParallelEnv(ParallelEnv):
        metadata = {"render_modes": [], "name": "madrid-workzone-v0"}

        def __init__(self) -> None:
            super().__init__()
            self.possible_agents = [s.agent_id for s in specs]
            self.agents: list[str] = []
            self.observation_spaces = {
                s.agent_id: build_observation_space(s.agent_class) for s in specs
            }
            self.action_spaces = {s.agent_id: build_action_space(s.n_actions) for s in specs}
            self._scenario = scenario
            self._step = 0

        def reset(self, seed=None, options=None):
            self._scenario.reset()
            self._step = 0
            self.agents = list(self.possible_agents)
            obs = {a: self._obs_for(a) for a in self.agents}
            return obs, {a: {} for a in self.agents}

        def step(self, actions):
            phases = []
            for signal in signal_agents:
                phases.append(int(actions.get(signal.agent_id, 0)))
            while len(phases) < scenario.n_intersections:
                phases.append(0)

            state = self._scenario.step(phases[: scenario.n_intersections])
            reward = composite_reward(state, cfg)

            self._step += 1
            truncated = self._step >= max_steps
            terminated = False

            rewards = {a: float(reward) for a in self.agents}
            observations = {a: self._obs_for(a) for a in self.agents}
            terminations = dict.fromkeys(self.agents, terminated)
            truncations = dict.fromkeys(self.agents, truncated)
            infos = {a: {} for a in self.agents}

            if terminated or truncated:
                self.agents = []

            return observations, rewards, terminations, truncations, infos

        def _obs_for(self, agent_id: str) -> np.ndarray:
            spec = next(s for s in specs if s.agent_id == agent_id)
            dim = self.observation_spaces[agent_id].shape[0]
            obs = np.zeros(dim, dtype=np.float32)
            if spec.agent_class == "signal":
                idx = self.possible_agents.index(agent_id) % scenario.n_intersections
                if self._scenario._queues is not None:
                    q = self._scenario._queues[idx]
                    obs[: q.size] = q[:dim]
            return obs

    return MadridParallelEnv()


__all__ = [
    "MockQueueScenario",
    "build_parallel_env",
]
