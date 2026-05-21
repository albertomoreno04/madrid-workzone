"""Gravity-model OD prior for the W-SPSA calibration starting point.

W-SPSA refines an initial OD matrix to match observed counts. The
starting matrix matters: a smart prior shrinks the number of iterations
to convergence and helps the optimizer avoid local optima. The classic
choice is a doubly-constrained gravity model fed by census-based
productions and attractions, but Madrid's open data doesn't expose that
directly. We approximate by using *observed detector flow aggregated to
each zone* as a proxy for both production and attraction (the
steady-state assumption: average production equals average attraction).

The singly-constrained gravity model used here is

    T_ij = P_i * A_j * f(d_ij) / sum_k [A_k * f(d_ik)]

with ``f(d) = exp(-beta * d)`` deterrence and Euclidean inter-centroid
distance ``d``. Row sums equal ``P_i`` by construction, so total demand
out of zone ``i`` matches the observed flow there.

Diagonal entries (intra-zonal trips) are zeroed by default — they're a
mess to calibrate against link-count observations and W-SPSA can pick
them up later if needed.

Why this is a *prior* and not the final answer: the gravity model
spreads trips based purely on size and distance. It ignores network
topology, congestion, time-of-day routing preferences, and a dozen
other realities. W-SPSA's job is to perturb this matrix until simulated
detector counts match observations. The prior just gives the search a
warm start.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from madrid_twin.sim.edgedata import EdgeObservation

# ---------------------------------------------------------------------------
# Data classes.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ZoneStats:
    """Aggregate stats for a single TAZ.

    The centroid is the arithmetic mean of edge centroids in the zone
    (matches how the TAZ decomposition computes spatial position).
    Productions and attractions are both proxied by
    ``total_observed_flow_veh_h``.
    """

    zone_id: int
    centroid_x: float
    centroid_y: float
    n_edges: int
    n_detector_equipped_edges: int
    total_observed_flow_veh_h: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OdPriorReport:
    """Summary of the prior matrix."""

    n_zones: int
    beta_per_km: float
    total_demand_veh_h: float
    median_cell_veh_h: float
    max_cell_veh_h: float
    n_zones_without_flow: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Zone aggregation.
# ---------------------------------------------------------------------------


def aggregate_zone_stats(
    edge_to_zone: Mapping[str, int],
    edge_centroids: Mapping[str, tuple[float, float]],
    edge_observations: Mapping[str, EdgeObservation],
) -> dict[int, ZoneStats]:
    """Roll edge-level observations and centroids up to per-zone stats.

    Parameters
    ----------
    edge_to_zone
        Output of :func:`build_taz_by_kmeans`.
    edge_centroids
        ``{edge_id: (cx, cy)}`` — local Cartesian XY centroids.
    edge_observations
        Output of :func:`aggregate_observations_by_edge`.
    """
    buckets: dict[int, dict[str, list]] = {}
    for edge_id, zone in edge_to_zone.items():
        centroid = edge_centroids.get(edge_id)
        if centroid is None:
            continue
        bucket = buckets.setdefault(
            zone,
            {"xs": [], "ys": [], "detector_flows": [], "n_edges": [0]},
        )
        bucket["xs"].append(centroid[0])
        bucket["ys"].append(centroid[1])
        bucket["n_edges"][0] += 1
        obs = edge_observations.get(edge_id)
        if obs is not None:
            bucket["detector_flows"].append(obs.intensity_veh_h)

    out: dict[int, ZoneStats] = {}
    for zone, b in buckets.items():
        xs: list[float] = b["xs"]
        ys: list[float] = b["ys"]
        flows: list[float] = b["detector_flows"]
        out[zone] = ZoneStats(
            zone_id=zone,
            centroid_x=sum(xs) / len(xs),
            centroid_y=sum(ys) / len(ys),
            n_edges=b["n_edges"][0],
            n_detector_equipped_edges=len(flows),
            total_observed_flow_veh_h=sum(flows),
        )
    return out


# ---------------------------------------------------------------------------
# Gravity model.
# ---------------------------------------------------------------------------


def gravity_od_prior(
    zones: Sequence[ZoneStats],
    *,
    beta_per_km: float = 0.1,
    zero_diagonal: bool = True,
) -> np.ndarray:
    """Singly-constrained gravity OD matrix.

    ``T[i, j] = P_i * A_j * exp(-beta * d_ij)
                / sum_k [A_k * exp(-beta * d_ik)]``

    with productions ``P_i`` and attractions ``A_j`` both equal to
    ``zones[i].total_observed_flow_veh_h``. The row sums equal
    ``P_i`` by construction.

    Distances are computed in **kilometres** from the Cartesian
    centroids. ``beta_per_km`` is therefore in 1/km. The default
    ``0.1`` is a typical value for urban deterrence; W-SPSA will
    refine this implicitly via the OD perturbations.

    Zones with zero observed flow contribute zero-flow rows AND
    columns — they have no trips in the prior. W-SPSA can light them
    up if the data demands it.
    """
    n = len(zones)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float64)
    if beta_per_km < 0:
        raise ValueError("beta_per_km must be non-negative")

    productions = np.array([z.total_observed_flow_veh_h for z in zones], dtype=np.float64)
    attractions = productions.copy()

    coords = np.array([(z.centroid_x, z.centroid_y) for z in zones], dtype=np.float64)
    # Pairwise Euclidean distance in metres -> km.
    diffs = coords[:, None, :] - coords[None, :, :]
    d_km = np.linalg.norm(diffs, axis=-1) / 1000.0
    deterrence = np.exp(-beta_per_km * d_km)
    if zero_diagonal:
        np.fill_diagonal(deterrence, 0.0)

    # Weighted attractions: A_j * f(d_ij) for each i.
    weighted = attractions[None, :] * deterrence  # shape (n, n)
    row_sums = weighted.sum(axis=1, keepdims=True)
    # Guard against zero rows (zone with no reachable destinations) —
    # divide by 1 there; the row will be all zeros anyway because the
    # corresponding production is zero (or it shouldn't be, but...).
    row_sums = np.where(row_sums > 0, row_sums, 1.0)

    T = productions[:, None] * weighted / row_sums
    return T


def build_od_prior_report(matrix: np.ndarray, beta_per_km: float) -> OdPriorReport:
    """Compute summary statistics of an OD prior matrix."""
    n = matrix.shape[0]
    total = float(matrix.sum())
    non_diag = matrix[~np.eye(n, dtype=bool)]
    median_cell = float(np.median(non_diag)) if non_diag.size else 0.0
    max_cell = float(matrix.max()) if matrix.size else 0.0
    n_zero_rows = int((matrix.sum(axis=1) == 0).sum())
    return OdPriorReport(
        n_zones=n,
        beta_per_km=beta_per_km,
        total_demand_veh_h=total,
        median_cell_veh_h=median_cell,
        max_cell_veh_h=max_cell,
        n_zones_without_flow=n_zero_rows,
    )


# ---------------------------------------------------------------------------
# I/O helpers.
# ---------------------------------------------------------------------------


def save_od_prior(
    matrix: np.ndarray,
    zones: Sequence[ZoneStats],
    output_dir: Path,
    *,
    beta_per_km: float,
) -> tuple[Path, Path, Path]:
    """Persist the OD prior matrix + zone metadata + report.

    Writes three files:
      ``od_prior.npy``         — the ``(K, K)`` numpy matrix
      ``zones.json``           — ordered zone metadata (row index = zone position)
      ``od_prior_report.json`` — summary statistics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix_path = output_dir / "od_prior.npy"
    np.save(matrix_path, matrix)

    zones_path = output_dir / "zones.json"
    zones_path.write_text(
        json.dumps([z.to_dict() for z in zones], indent=2),
        encoding="utf-8",
    )

    report_path = output_dir / "od_prior_report.json"
    report = build_od_prior_report(matrix, beta_per_km)
    report_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")

    return matrix_path, zones_path, report_path


def load_od_prior(matrix_path: Path, zones_path: Path) -> tuple[np.ndarray, list[ZoneStats]]:
    """Reverse of :func:`save_od_prior`. Zone order is preserved."""
    matrix = np.load(matrix_path)
    payload = json.loads(Path(zones_path).read_text(encoding="utf-8"))
    zones = [ZoneStats(**entry) for entry in payload]
    return matrix, zones


__all__ = [
    "OdPriorReport",
    "ZoneStats",
    "aggregate_zone_stats",
    "build_od_prior_report",
    "gravity_od_prior",
    "load_od_prior",
    "save_od_prior",
]
