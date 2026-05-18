"""Phase 3 smoke: roll out the mock queue scenario with the baseline controllers.

For each of FixedTimeController and MaxPressureController, runs N steps
against the deterministic MockQueueScenario, accumulates per-step
composite reward, and writes a small JSON report. Pure numpy — no SUMO
and no MARL extras needed.

Serves as the smoke-test endpoint for the Phase 3 control surface: if
this runs end to end and produces non-degenerate rewards, the
intersection-state -> controller -> reward dispatch wiring is intact.
"""

from __future__ import annotations

import json

import numpy as np

from madrid_twin.config import DATA_OUTPUTS
from madrid_twin.control.baselines import (
    FixedTimeController,
    IntersectionState,
    MaxPressureController,
)
from madrid_twin.control.env import MockQueueScenario
from madrid_twin.control.rewards import RewardConfig, composite_reward


def _build_states(scenario: MockQueueScenario) -> list[IntersectionState]:
    return scenario.intersection_states()


def _roll_out(scenario: MockQueueScenario, controller, n_steps: int) -> dict[str, float]:
    scenario.reset()
    cfg = RewardConfig()
    rewards: list[float] = []
    throughputs: list[int] = []
    for _ in range(n_steps):
        states = _build_states(scenario)
        actions_by_int = controller.act(states)
        phases = [actions_by_int[s.intersection_id] for s in states]
        net = scenario.step(phases)
        rewards.append(composite_reward(net, cfg))
        throughputs.append(net.throughput_veh)
    return {
        "mean_reward": float(np.mean(rewards)),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
        "mean_throughput": float(np.mean(throughputs)),
        "total_throughput": int(sum(throughputs)),
    }


def main() -> None:
    scenario = MockQueueScenario(seed=0)
    n_steps = 200

    # Fixed-time: equal 4-step duration for each of 4 phases.
    fixed_schedule = {f"int_{i}": (4, 4, 4, 4) for i in range(scenario.n_intersections)}
    fixed = FixedTimeController(phase_durations=fixed_schedule)
    fixed_report = _roll_out(scenario, fixed, n_steps)

    # Max-pressure: phase chosen each step from queue differentials.
    mp = MaxPressureController()
    mp_report = _roll_out(scenario, mp, n_steps)

    report = {
        "n_steps": n_steps,
        "FixedTimeController": fixed_report,
        "MaxPressureController": mp_report,
        "delta_mean_reward_max_minus_fixed": (
            mp_report["mean_reward"] - fixed_report["mean_reward"]
        ),
    }

    out_dir = DATA_OUTPUTS / "control"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "smoke_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nWritten: {out_path}")


if __name__ == "__main__":
    main()
