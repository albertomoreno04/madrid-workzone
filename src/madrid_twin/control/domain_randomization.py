"""Domain randomization sampler.

At the start of every training episode the protagonist MARL policy
faces a fresh, randomly-perturbed scenario: capacity drop magnitude,
demand multiplier, sensor noise level, and incident probability are
all resampled from configurable ranges. This is the *first* of MADTwin's
two robustness mechanisms (the second being the RARL adversary in
:mod:`madrid_twin.control.adversary`).

The ranges are deliberately conservative — they match the spread of
real-world variation reviewers are likely to push back on, rather than
the much wider ranges that would trivially destabilise training.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from madrid_twin.sim.workzone import Workzone


@dataclass(frozen=True)
class DomainRandomizationConfig:
    """Range for each randomised quantity. All ranges are inclusive."""

    # Capacity drop on the workzone, expressed as fraction in [0, 1).
    # HCM 2016 reports 0.10-0.30 on urban arterials.
    capacity_drop_range: tuple[float, float] = (0.10, 0.30)

    # Demand multiplier applied to the baseline OD matrix.
    demand_multiplier_range: tuple[float, float] = (0.8, 1.2)

    # Standard deviation of additive Gaussian sensor noise on detector
    # intensity readings, in veh/h.
    sensor_noise_std_range: tuple[float, float] = (0.0, 50.0)

    # Per-episode probability of injecting a random incident on a
    # workzone-adjacent link.
    incident_probability: float = 0.05

    def __post_init__(self) -> None:
        for name, rng in (
            ("capacity_drop_range", self.capacity_drop_range),
            ("demand_multiplier_range", self.demand_multiplier_range),
            ("sensor_noise_std_range", self.sensor_noise_std_range),
        ):
            lo, hi = rng
            if lo > hi:
                raise ValueError(f"{name} must satisfy lo <= hi, got ({lo}, {hi})")
        if not 0.0 <= self.incident_probability <= 1.0:
            raise ValueError(
                f"incident_probability must be in [0, 1], got {self.incident_probability}"
            )
        # Capacity drop must stay in the Workzone-allowed [0, 1) range.
        if not (self.capacity_drop_range[0] >= 0.0 and self.capacity_drop_range[1] < 1.0):
            raise ValueError(
                f"capacity_drop_range must lie in [0, 1), got {self.capacity_drop_range}"
            )


@dataclass(frozen=True)
class RandomizedScenario:
    """A single randomised draw produced by :class:`DomainRandomizer`."""

    workzone: Workzone
    demand_multiplier: float
    sensor_noise_std: float
    incident: bool


class DomainRandomizer:
    """Stateful sampler — keep one instance per training run.

    Constructor takes a base :class:`Workzone` template (typically the
    realistic central scenario) plus a config; :meth:`sample` returns a
    :class:`RandomizedScenario` with every relevant attribute perturbed.

    Reproducibility runs through the ``seed`` argument; per-call
    randomness is drawn from a ``numpy.random.Generator`` so test
    fixtures stay deterministic.
    """

    def __init__(
        self,
        base_workzone: Workzone,
        config: DomainRandomizationConfig | None = None,
        *,
        seed: int = 0,
    ) -> None:
        self.base_workzone = base_workzone
        self.config = config or DomainRandomizationConfig()
        self.rng = np.random.default_rng(seed)

    def sample(self) -> RandomizedScenario:
        cap_lo, cap_hi = self.config.capacity_drop_range
        capacity_drop = float(self.rng.uniform(cap_lo, cap_hi))

        dem_lo, dem_hi = self.config.demand_multiplier_range
        demand_multiplier = float(self.rng.uniform(dem_lo, dem_hi))

        noise_lo, noise_hi = self.config.sensor_noise_std_range
        sensor_noise_std = float(self.rng.uniform(noise_lo, noise_hi))

        incident = bool(self.rng.random() < self.config.incident_probability)

        wz = self.base_workzone
        randomized_workzone = Workzone(
            id=f"{wz.id}__dr_{int(self.rng.integers(0, 2**31 - 1))}",
            closures=wz.closures,
            start_s=wz.start_s,
            end_s=wz.end_s,
            capacity_drop_pct=capacity_drop,
            description=f"DR-perturbed: {wz.description}".strip(": "),
        )

        return RandomizedScenario(
            workzone=randomized_workzone,
            demand_multiplier=demand_multiplier,
            sensor_noise_std=sensor_noise_std,
            incident=incident,
        )


__all__ = [
    "DomainRandomizationConfig",
    "DomainRandomizer",
    "RandomizedScenario",
]
