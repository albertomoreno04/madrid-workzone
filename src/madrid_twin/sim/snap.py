"""Snap real-world detectors to SUMO network edges.

The Ayuntamiento feed gives every detector a UTM coordinate in EPSG:25830
(peninsular Spain). The SUMO network is in its own local Cartesian
system — ``netconvert`` projects the OSM data into UTM then shifts the
origin so coordinates start near ``(0, 0)``. We bridge the two via
WGS84 lon/lat as a neutral intermediate, then use ``sumolib``'s built-in
geometric proximity query to find the closest edge to each detector.

The output is a JSON mapping ``{detector_id: {edge_id, distance_m,
description, ...}}`` that the calibration pipeline uses to look up which
edge each observed flow target applies to.

The module is structured around dependency injection: the actual SUMO
``Net`` object and the UTM→lon/lat transformer are passed in, so unit
tests can run without ``sumolib`` or ``pyproj`` installed.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from madrid_twin.data.open_data import DetectorReading

# ---------------------------------------------------------------------------
# Data classes.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DetectorEdgeMatch:
    """One successful detector→edge snap."""

    detector_id: str
    edge_id: str
    distance_m: float
    description: str | None = None
    x_utm: float | None = None
    y_utm: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SnapConfig:
    """Tuning knobs for the snapping process."""

    # Maximum distance from detector to edge we consider a valid match.
    # 100 m absorbs typical OSM/Ayuntamiento georeferencing slack while
    # still rejecting detectors that fell outside the network crop.
    max_distance_m: float = 100.0
    # Starting search radius. Each miss doubles it up to max_distance_m.
    initial_radius_m: float = 25.0


@dataclass(frozen=True)
class SnapReport:
    """Tally of what happened during a snap run."""

    n_input: int
    n_matched: int
    n_skipped_no_coords: int
    n_skipped_no_edge_within_radius: int
    median_distance_m: float | None
    max_distance_m: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Adapter protocols — what we need from sumolib and pyproj.
# ---------------------------------------------------------------------------


class NetworkAdapter(Protocol):
    """The slice of the SUMO Net object we depend on."""

    def convert_lonlat_to_xy(self, lon: float, lat: float) -> tuple[float, float]:
        """Project (lon, lat) in WGS84 to the network's local Cartesian XY."""

    def neighboring_edges(self, x: float, y: float, radius_m: float) -> list[tuple[str, float]]:
        """Return ``(edge_id, distance_m)`` for every edge within ``radius_m``."""


UtmToLonLat = Callable[[float, float], tuple[float, float]]


# ---------------------------------------------------------------------------
# Concrete adapters (lazy-import their heavy dependencies).
# ---------------------------------------------------------------------------


class SumolibNetworkAdapter:
    """Adapter around ``sumolib.net.Net`` exposing only the methods we use."""

    def __init__(self, net: Any) -> None:
        self._net = net

    def convert_lonlat_to_xy(self, lon: float, lat: float) -> tuple[float, float]:
        x, y = self._net.convertLonLat2XY(lon, lat)
        return float(x), float(y)

    def neighboring_edges(self, x: float, y: float, radius_m: float) -> list[tuple[str, float]]:
        return [
            (edge.getID(), float(dist))
            for edge, dist in self._net.getNeighboringEdges(x, y, radius_m)
        ]


def load_sumo_network(net_xml_path: Path) -> SumolibNetworkAdapter:
    """Read a SUMO ``.net.xml`` and wrap it in the adapter."""
    import sumolib  # local import keeps tests independent.

    net = sumolib.net.readNet(str(net_xml_path))
    return SumolibNetworkAdapter(net)


def pyproj_utm25830_to_wgs84() -> UtmToLonLat:
    """Return a callable that projects EPSG:25830 (x, y) to (lon, lat)."""
    from pyproj import Transformer  # local import keeps tests independent.

    transformer = Transformer.from_crs("EPSG:25830", "EPSG:4326", always_xy=True)

    def _project(x_utm: float, y_utm: float) -> tuple[float, float]:
        lon, lat = transformer.transform(x_utm, y_utm)
        return float(lon), float(lat)

    return _project


# ---------------------------------------------------------------------------
# Core algorithm.
# ---------------------------------------------------------------------------


def snap_detectors_to_edges(
    detectors: Iterable[DetectorReading],
    *,
    network: NetworkAdapter,
    utm_to_lonlat: UtmToLonLat,
    config: SnapConfig | None = None,
) -> tuple[list[DetectorEdgeMatch], SnapReport]:
    """Snap every detector with valid UTM coords to its closest SUMO edge.

    The expanding-radius search starts at ``config.initial_radius_m`` and
    doubles until a hit or until ``config.max_distance_m`` is exceeded.
    The hit with minimum distance within the first non-empty radius wins.

    Detectors with no coordinates are skipped (counted in the report).
    Detectors whose closest edge is farther than ``max_distance_m`` are
    also skipped — typically these are outside the network crop or on
    edge types that ``netconvert`` filtered out (service roads etc).
    """
    cfg = config or SnapConfig()
    if cfg.max_distance_m <= 0:
        raise ValueError("max_distance_m must be positive")
    if cfg.initial_radius_m <= 0:
        raise ValueError("initial_radius_m must be positive")

    matches: list[DetectorEdgeMatch] = []
    n_input = 0
    n_no_coords = 0
    n_no_edge = 0

    for d in detectors:
        n_input += 1
        if d.x_utm is None or d.y_utm is None:
            n_no_coords += 1
            continue

        lon, lat = utm_to_lonlat(d.x_utm, d.y_utm)
        x, y = network.convert_lonlat_to_xy(lon, lat)

        # Expanding-radius search.
        best: tuple[str, float] | None = None
        radius = cfg.initial_radius_m
        while radius <= cfg.max_distance_m:
            candidates = network.neighboring_edges(x, y, radius)
            if candidates:
                best = min(candidates, key=lambda c: c[1])
                # The radius is in metres; if best[1] > max_distance_m we
                # still drop it. (Shouldn't happen for non-degenerate
                # geometry but guard against it.)
                if best[1] <= cfg.max_distance_m:
                    break
                best = None
            radius *= 2.0

        if best is None:
            n_no_edge += 1
            continue

        edge_id, distance_m = best
        matches.append(
            DetectorEdgeMatch(
                detector_id=d.detector_id,
                edge_id=edge_id,
                distance_m=distance_m,
                description=d.description,
                x_utm=d.x_utm,
                y_utm=d.y_utm,
            )
        )

    matches.sort(key=lambda m: m.detector_id)

    distances = [m.distance_m for m in matches]
    if distances:
        sorted_d = sorted(distances)
        median = sorted_d[len(sorted_d) // 2]
        max_d = max(distances)
    else:
        median = None
        max_d = None

    report = SnapReport(
        n_input=n_input,
        n_matched=len(matches),
        n_skipped_no_coords=n_no_coords,
        n_skipped_no_edge_within_radius=n_no_edge,
        median_distance_m=median,
        max_distance_m=max_d,
    )
    return matches, report


# ---------------------------------------------------------------------------
# I/O helpers.
# ---------------------------------------------------------------------------


def load_usable_detector_ids(path: Path) -> set[str]:
    """Read ``data/processed/detectors/usable_detectors.json``."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return set(payload.get("detector_ids", []))


def filter_to_ids(detectors: Iterable[DetectorReading], ids: set[str]) -> list[DetectorReading]:
    """Return only the detectors whose id is in ``ids``."""
    return [d for d in detectors if d.detector_id in ids]


def write_snap_outputs(
    matches: Sequence[DetectorEdgeMatch],
    report: SnapReport,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Write the per-detector mapping + the summary report to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mapping_path = output_dir / "detector_edge_mapping.json"
    mapping_payload = {m.detector_id: m.to_dict() for m in matches}
    mapping_path.write_text(json.dumps(mapping_payload, indent=2), encoding="utf-8")

    report_path = output_dir / "snap_report.json"
    report_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")

    return mapping_path, report_path


__all__ = [
    "DetectorEdgeMatch",
    "NetworkAdapter",
    "SnapConfig",
    "SnapReport",
    "SumolibNetworkAdapter",
    "UtmToLonLat",
    "filter_to_ids",
    "load_sumo_network",
    "load_usable_detector_ids",
    "pyproj_utm25830_to_wgs84",
    "snap_detectors_to_edges",
    "write_snap_outputs",
]
