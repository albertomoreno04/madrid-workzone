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
from madrid_twin.sim.edgedata import (
    EdgeObservation,
    aggregate_observations_by_edge,
    edgedata_to_xml,
    load_detector_edge_mapping,
    write_edgedata_xml,
)
from madrid_twin.sim.network import (
    DEFAULT_ROAD_TYPES,
    NetconvertError,
    NetconvertResult,
    build_network_from_osm,
)
from madrid_twin.sim.snap import (
    DetectorEdgeMatch,
    SnapConfig,
    SnapReport,
    filter_to_ids,
    load_sumo_network,
    load_usable_detector_ids,
    pyproj_utm25830_to_wgs84,
    snap_detectors_to_edges,
    write_snap_outputs,
)
from madrid_twin.sim.taz import (
    EdgeCentroid,
    TazReport,
    build_taz_by_kmeans,
    extract_edge_centroids_from_sumo_network,
    taz_to_sumo_xml,
    write_taz_xml,
)
from madrid_twin.sim.workzone import (
    LaneClosure,
    Workzone,
    workzone_from_lanes,
)

__all__ = [
    "CurriculumSchedule",
    "DEFAULT_ROAD_TYPES",
    "DetectorEdgeMatch",
    "EdgeCentroid",
    "EdgeObservation",
    "LaneClosure",
    "NetconvertError",
    "NetconvertResult",
    "ODMatrixSpec",
    "PiecewiseSchedule",
    "SnapConfig",
    "SnapReport",
    "TazReport",
    "WSPSAConfig",
    "WSPSAReport",
    "Workzone",
    "aggregate_observations_by_edge",
    "apply_adversarial_perturbation",
    "apply_demand_multiplier",
    "build_network_from_osm",
    "build_taz_by_kmeans",
    "count_matching_loss",
    "derive_workzone_params",
    "edgedata_to_xml",
    "extract_edge_centroids_from_sumo_network",
    "filter_to_ids",
    "load_detector_edge_mapping",
    "load_sumo_network",
    "load_usable_detector_ids",
    "pyproj_utm25830_to_wgs84",
    "sample_trips",
    "snap_detectors_to_edges",
    "taz_to_sumo_xml",
    "trips_to_sumo_routes_xml",
    "workzone_from_lanes",
    "write_edgedata_xml",
    "write_snap_outputs",
    "write_sumo_routes",
    "write_taz_xml",
    "wspsa",
]
