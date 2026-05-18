"""OD-matrix → SUMO trips XML emitter.

SUMO consumes demand as a ``<routes>`` file containing one ``<trip>`` per
vehicle: an origin edge, a destination edge, and a depart time. We build
that file from an OD matrix plus a list of edge identifiers, with
optional demand multiplier and adversarial perturbation applied first.

Departure times are sampled uniformly within the OD time bin, which is
the SUMO recipe when no finer-grained arrival profile is available. For
peak-hour calibration this is adequate; for events / surges the profile
would need to come from the data layer instead.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from xml.dom import minidom

import numpy as np


@dataclass(frozen=True)
class ODMatrixSpec:
    """A single OD-matrix slice for one time bin.

    Attributes
    ----------
    origins, destinations
        Edge identifiers, length ``n_origins`` / ``n_destinations``.
    counts
        Shape ``(n_origins, n_destinations)``. Expected vehicles to be
        emitted from origin ``i`` to destination ``j`` during the bin.
    begin_s, end_s
        Time bin window in simulator seconds (``end_s`` exclusive).
    """

    origins: tuple[str, ...]
    destinations: tuple[str, ...]
    counts: np.ndarray
    begin_s: float
    end_s: float

    def __post_init__(self) -> None:
        if self.counts.shape != (len(self.origins), len(self.destinations)):
            raise ValueError(
                f"counts shape {self.counts.shape} must match "
                f"({len(self.origins)}, {len(self.destinations)})"
            )
        if (self.counts < 0).any():
            raise ValueError("counts must be non-negative everywhere")
        if self.begin_s >= self.end_s:
            raise ValueError(f"begin_s ({self.begin_s}) must be < end_s ({self.end_s})")


def apply_demand_multiplier(spec: ODMatrixSpec, multiplier: float) -> ODMatrixSpec:
    """Return an :class:`ODMatrixSpec` with counts scaled by ``multiplier``."""
    if multiplier < 0:
        raise ValueError(f"multiplier must be non-negative, got {multiplier}")
    return ODMatrixSpec(
        origins=spec.origins,
        destinations=spec.destinations,
        counts=spec.counts * multiplier,
        begin_s=spec.begin_s,
        end_s=spec.end_s,
    )


def apply_adversarial_perturbation(
    spec: ODMatrixSpec, perturbation_per_origin: np.ndarray
) -> ODMatrixSpec:
    """Add a per-origin perturbation across all destinations.

    The perturbation is distributed across destinations proportionally to
    the *current* row distribution — so an aggressive injection on
    origin 3 spreads across the destinations 3 already serves. Counts
    are floored at zero.
    """
    if perturbation_per_origin.shape != (len(spec.origins),):
        raise ValueError(
            f"perturbation_per_origin shape {perturbation_per_origin.shape} "
            f"must be ({len(spec.origins)},)"
        )
    counts = spec.counts.copy()
    for i, delta in enumerate(perturbation_per_origin):
        row = counts[i]
        row_sum = row.sum()
        if row_sum > 0:
            counts[i] = np.maximum(0.0, row + delta * (row / row_sum))
        else:
            # No baseline flow on this origin -> spread uniformly.
            n_dest = counts.shape[1]
            counts[i] = np.maximum(0.0, np.full(n_dest, delta / n_dest))
    return ODMatrixSpec(
        origins=spec.origins,
        destinations=spec.destinations,
        counts=counts,
        begin_s=spec.begin_s,
        end_s=spec.end_s,
    )


def sample_trips(spec: ODMatrixSpec, *, seed: int = 0) -> list[tuple[str, str, float]]:
    """Sample concrete (origin, destination, depart_s) trip tuples.

    Counts are rounded stochastically: a count of 3.7 produces 3 or 4
    vehicles with the appropriate probability. Departures are uniform
    over the bin window.
    """
    rng = np.random.default_rng(seed)
    trips: list[tuple[str, str, float]] = []

    for i, origin in enumerate(spec.origins):
        for j, dest in enumerate(spec.destinations):
            count_f = float(spec.counts[i, j])
            base = int(np.floor(count_f))
            frac = count_f - base
            n_veh = base + (1 if rng.random() < frac else 0)
            if n_veh == 0:
                continue
            depart_times = rng.uniform(spec.begin_s, spec.end_s, size=n_veh)
            for t in sorted(depart_times):
                trips.append((origin, dest, float(t)))

    trips.sort(key=lambda x: x[2])
    return trips


def trips_to_sumo_routes_xml(
    trips: Sequence[tuple[str, str, float]],
    *,
    vehicle_type: str = "passenger",
) -> str:
    """Render a SUMO ``<routes>`` XML fragment from a trip list."""
    root = ET.Element("routes")

    ET.SubElement(
        root,
        "vType",
        attrib={
            "id": vehicle_type,
            "vClass": "passenger",
            "length": "5.0",
            "maxSpeed": "13.89",
        },
    )

    for idx, (origin, dest, depart) in enumerate(trips):
        ET.SubElement(
            root,
            "trip",
            attrib={
                "id": f"veh_{idx}",
                "type": vehicle_type,
                "depart": f"{depart:.2f}",
                "from": origin,
                "to": dest,
            },
        )

    rough = ET.tostring(root, encoding="unicode")
    return minidom.parseString(rough).toprettyxml(indent="  ")


def write_sumo_routes(
    spec: ODMatrixSpec,
    output_path: Path,
    *,
    seed: int = 0,
    vehicle_type: str = "passenger",
) -> Path:
    """End-to-end: sample trips from an OD spec and write a SUMO routes file."""
    trips = sample_trips(spec, seed=seed)
    xml_text = trips_to_sumo_routes_xml(trips, vehicle_type=vehicle_type)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(xml_text, encoding="utf-8")
    return output_path


__all__ = [
    "ODMatrixSpec",
    "apply_adversarial_perturbation",
    "apply_demand_multiplier",
    "sample_trips",
    "trips_to_sumo_routes_xml",
    "write_sumo_routes",
]
