"""Curriculum learning scheduler for workzone severity.

MARL training is notoriously unstable when the protagonist policy faces
its full target task from step one. The recipe that consistently works
in the traffic-control literature — and matches our own MAPPO setup —
is *curriculum learning*: start with an easy version of the task
(no workzone, gentle demand), and ramp severity over training.

This module exposes a stateless scheduler that maps a training step to
a severity scalar in ``[0, 1]``. Downstream layers consume the scalar
to set the workzone's capacity-drop magnitude and the demand multiplier
each episode.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CurriculumSchedule:
    """Linear ramp from ``start_severity`` to ``end_severity`` across training.

    The default is a "no workzone → strong workzone" ramp over 30% of
    training, then a plateau at full severity for the remaining 70%.
    """

    start_severity: float = 0.0
    end_severity: float = 1.0
    ramp_end_fraction: float = 0.3  # fraction of total_steps used for the ramp
    total_steps: int = 1_000_000

    def __post_init__(self) -> None:
        if not 0.0 <= self.start_severity <= 1.0:
            raise ValueError(f"start_severity must be in [0, 1], got {self.start_severity}")
        if not 0.0 <= self.end_severity <= 1.0:
            raise ValueError(f"end_severity must be in [0, 1], got {self.end_severity}")
        if not 0.0 < self.ramp_end_fraction <= 1.0:
            raise ValueError(f"ramp_end_fraction must be in (0, 1], got {self.ramp_end_fraction}")
        if self.total_steps <= 0:
            raise ValueError("total_steps must be positive")

    def severity_at(self, step: int) -> float:
        """Linear ramp, clamped to ``[start_severity, end_severity]``."""
        if step < 0:
            return self.start_severity
        ramp_end_step = int(self.ramp_end_fraction * self.total_steps)
        if step >= ramp_end_step:
            return self.end_severity
        if ramp_end_step == 0:
            return self.end_severity
        progress = step / ramp_end_step
        return float(self.start_severity + progress * (self.end_severity - self.start_severity))


@dataclass(frozen=True)
class PiecewiseSchedule:
    """Discrete piecewise schedule — e.g., stage 1 / stage 2 / stage 3 curricula.

    ``stages`` is a tuple of ``(step_at_or_after, severity)`` pairs,
    ordered by step. The severity is the value at the most recent stage
    whose threshold has been crossed.
    """

    stages: tuple[tuple[int, float], ...] = (
        (0, 0.0),
        (200_000, 0.3),
        (500_000, 0.7),
        (800_000, 1.0),
    )

    def __post_init__(self) -> None:
        if not self.stages:
            raise ValueError("PiecewiseSchedule must contain at least one stage")
        steps = [s[0] for s in self.stages]
        if steps != sorted(steps):
            raise ValueError("PiecewiseSchedule.stages must be sorted by step")
        for s, v in self.stages:
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"severity at step {s} must be in [0, 1], got {v}")

    def severity_at(self, step: int) -> float:
        current = self.stages[0][1]
        for s, v in self.stages:
            if step >= s:
                current = v
            else:
                break
        return float(current)


def derive_workzone_params(
    severity: float,
    *,
    base_capacity_drop: float = 0.20,
    base_demand_multiplier: float = 1.0,
    severe_capacity_drop: float = 0.45,
    severe_demand_multiplier: float = 1.3,
) -> tuple[float, float]:
    """Map a severity scalar to concrete workzone parameters.

    Returns ``(capacity_drop_pct, demand_multiplier)`` — the two scalars
    the control layer consumes per episode.
    """
    if not 0.0 <= severity <= 1.0:
        raise ValueError(f"severity must be in [0, 1], got {severity}")
    cap = base_capacity_drop + severity * (severe_capacity_drop - base_capacity_drop)
    dem = base_demand_multiplier + severity * (severe_demand_multiplier - base_demand_multiplier)
    return float(cap), float(dem)


__all__ = [
    "CurriculumSchedule",
    "PiecewiseSchedule",
    "derive_workzone_params",
]
