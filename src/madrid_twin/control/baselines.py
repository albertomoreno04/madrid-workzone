"""Classical signal-control baselines.

Every MARL traffic paper compares against (a) fixed-time cycles
(current Madrid practice) and (b) max-pressure control (Varaiya 2013,
the de-facto theoretical optimum for fully-actuated single-agent
control). Implementing both here gives the experiment harness a fair,
peer-review-grade reference to evaluate the MARL policy against.

Both controllers consume the same :class:`IntersectionState` (a
lightweight numpy-only abstraction over what the env exposes) and emit
a phase index per intersection. They are deterministic and fully
testable without any heavyweight RL dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class IntersectionState:
    """Lightweight, sim-agnostic view of one signalised intersection.

    Attributes
    ----------
    intersection_id
        Stable identifier.
    n_phases
        Number of discrete phases available.
    incoming_queues
        Shape ``(n_phases,)`` — queue length (vehicles) on the lanes
        served by each phase. Max-pressure uses this directly.
    outgoing_queues
        Shape ``(n_phases,)`` — queue length on the destination links
        for vehicles served by each phase. Max-pressure needs the
        differential incoming - outgoing.
    """

    intersection_id: str
    n_phases: int
    incoming_queues: np.ndarray
    outgoing_queues: np.ndarray

    def __post_init__(self) -> None:
        if self.n_phases <= 0:
            raise ValueError("n_phases must be positive")
        if self.incoming_queues.shape != (self.n_phases,):
            raise ValueError(
                f"incoming_queues shape {self.incoming_queues.shape} must be ({self.n_phases},)"
            )
        if self.outgoing_queues.shape != (self.n_phases,):
            raise ValueError(
                f"outgoing_queues shape {self.outgoing_queues.shape} must be ({self.n_phases},)"
            )


# ---------------------------------------------------------------------------
# Fixed-time controller.
# ---------------------------------------------------------------------------


@dataclass
class FixedTimeController:
    """Round-robin cycle through phases on a fixed schedule.

    Mimics Madrid current-practice signal plans: each phase holds for a
    fixed duration, then advances to the next. The schedule is stored as
    a per-intersection list of phase durations (in simulation steps).
    """

    phase_durations: dict[str, tuple[int, ...]]
    _step: int = 0

    def reset(self) -> None:
        self._step = 0

    def act(self, states: list[IntersectionState]) -> dict[str, int]:
        actions: dict[str, int] = {}
        for s in states:
            schedule = self.phase_durations.get(s.intersection_id)
            if schedule is None or not schedule:
                actions[s.intersection_id] = 0
                continue
            cycle = sum(schedule)
            t = self._step % cycle
            cumulative = 0
            chosen = 0
            for phase_idx, dur in enumerate(schedule):
                cumulative += dur
                if t < cumulative:
                    chosen = phase_idx
                    break
            actions[s.intersection_id] = chosen
        self._step += 1
        return actions


# ---------------------------------------------------------------------------
# Max-pressure controller (Varaiya, 2013).
# ---------------------------------------------------------------------------


@dataclass
class MaxPressureController:
    """At each intersection, serve the phase with the greatest pressure.

    Pressure for phase ``p`` is defined as the differential between
    incoming and outgoing queue lengths on the lanes that phase serves:

        pressure_p = max(0, incoming_p - outgoing_p)

    Varaiya proved this rule is *stable* — it keeps the network in
    equilibrium whenever any stabilising policy exists — under mild
    conditions. Ties are broken by phase index.
    """

    def act(self, states: list[IntersectionState]) -> dict[str, int]:
        actions: dict[str, int] = {}
        for s in states:
            pressures = np.maximum(0.0, s.incoming_queues - s.outgoing_queues)
            # If all pressures are zero, hold phase 0.
            if pressures.sum() == 0:
                actions[s.intersection_id] = 0
            else:
                actions[s.intersection_id] = int(np.argmax(pressures))
        return actions


__all__ = [
    "FixedTimeController",
    "IntersectionState",
    "MaxPressureController",
]
