"""Simulation layer — the SUMO-based calibrated digital twin."""

from __future__ import annotations

from madrid_twin.sim.calibrate import (
    WSPSAConfig,
    WSPSAReport,
    count_matching_loss,
    wspsa,
)
from madrid_twin.sim.curriculum import (
    CurriculumSchedule,
    PiecewiseSchedule,
    derive_workzone_params,
)
from madrid_twin.sim.demand import (
    ODMatrixSpec,
    apply_adversarial_perturbation,
    apply_demand_multiplier,
    sample_trips,
    trips_to_sumo_routes_xml,
    write_sumo_routes,
)
from madrid_twin.sim.network import (
    DEFAULT_ROAD_TYPES,
    NetconvertError,
    NetconvertResult,
    build_network_from_osm,
)
from madrid_twin.sim.workzone import (
    LaneClosure,
    Workzone,
    workzone_from_lanes,
)

__all__ = [
    "CurriculumSchedule",
    "DEFAULT_ROAD_TYPES",
    "LaneClosure",
    "NetconvertError",
    "NetconvertResult",
    "ODMatrixSpec",
    "PiecewiseSchedule",
    "WSPSAConfig",
    "WSPSAReport",
    "Workzone",
    "apply_adversarial_perturbation",
    "apply_demand_multiplier",
    "build_network_from_osm",
    "count_matching_loss",
    "derive_workzone_params",
    "sample_trips",
    "trips_to_sumo_routes_xml",
    "workzone_from_lanes",
    "write_sumo_routes",
    "wspsa",
]
