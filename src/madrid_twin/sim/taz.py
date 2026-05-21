"""Traffic Analysis Zone (TAZ) decomposition for the SUMO network.

A TAZ partitions the network into groups of edges that share an
origin/destination. Without TAZ, the OD matrix would be |edges|² —
14 589² ≈ 213 M cells for inner Madrid, intractable. With K zones, it's
K² (e.g. 60² = 3600), small enough for W-SPSA to optimize.

Two strategies are exposed:

- :func:`build_taz_by_kmeans` clusters edge centroids in the network's
  local Cartesian XY using K-means++. Every edge gets a zone, the result
  is deterministic given a seed. Defensible as a first-pass spatial
  decomposition before more elaborate methods (Voronoi on detector
  locations, administrative subareas) come into play.

- (Future) administrative subareas from the Ayuntamiento ``subarea``
  field; for edges without detectors, fall back to the nearest subarea.

The output of either method is the same shape: a mapping from
``edge_id`` to an integer zone index, plus a small report. The XML
emitter writes that mapping in the SUMO TAZ format that ``duarouter``
and the OD-from-TAZ tools expect.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

# A representative point for an edge: (edge_id, centroid_x, centroid_y).
EdgeCentroid = tuple[str, float, float]


# ---------------------------------------------------------------------------
# Data classes.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TazReport:
    """Aggregate statistics about a TAZ decomposition."""

    n_edges: int
    n_zones_requested: int
    n_zones_populated: int
    min_edges_per_zone: int
    max_edges_per_zone: int
    median_edges_per_zone: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Network helpers.
# ---------------------------------------------------------------------------


def extract_edge_centroids_from_sumo_network(net_xml_path: Path) -> list[EdgeCentroid]:
    """Read a SUMO ``.net.xml`` and return ``[(edge_id, cx, cy), ...]``.

    The centroid is the arithmetic mean of the edge's polyline points.
    Edges with empty shape (rare — would indicate a malformed net) are
    skipped silently.
    """
    import sumolib  # local import keeps the module test-friendly.

    net = sumolib.net.readNet(str(net_xml_path))
    out: list[EdgeCentroid] = []
    for edge in net.getEdges():
        shape = edge.getShape()
        if not shape:
            continue
        n = len(shape)
        cx = sum(p[0] for p in shape) / n
        cy = sum(p[1] for p in shape) / n
        out.append((edge.getID(), float(cx), float(cy)))
    return out


# ---------------------------------------------------------------------------
# K-means clustering.
# ---------------------------------------------------------------------------


def build_taz_by_kmeans(
    edges: Sequence[EdgeCentroid],
    *,
    n_zones: int = 60,
    seed: int = 42,
    iter_count: int = 50,
) -> tuple[dict[str, int], TazReport]:
    """Cluster edges into ``n_zones`` TAZs via K-means++ on centroids.

    Parameters
    ----------
    edges
        Output of :func:`extract_edge_centroids_from_sumo_network`.
    n_zones
        Target number of zones. Must be >= 1 and <= ``len(edges)``.
    seed
        Random seed for K-means++ initialization. Same seed = same zones.
    iter_count
        Lloyd iterations after initialization. 50 converges on city-scale
        networks; bigger doesn't help much.

    Returns
    -------
    ``({edge_id: zone_index}, TazReport)``. ``zone_index`` is 0-based.
    """
    if n_zones < 1:
        raise ValueError("n_zones must be at least 1")
    if not edges:
        raise ValueError("edges must be non-empty")
    if n_zones > len(edges):
        raise ValueError(f"n_zones ({n_zones}) cannot exceed number of edges ({len(edges)})")

    # Lazy import so tests of the dataclasses + XML emit don't need scipy.
    import numpy as np
    from scipy.cluster.vq import kmeans2

    coords = np.array([(cx, cy) for _, cx, cy in edges], dtype=np.float64)
    _, labels = kmeans2(coords, n_zones, iter=iter_count, minit="++", seed=seed)
    label_ints: list[int] = [int(label) for label in labels.tolist()]

    mapping: dict[str, int] = {edge_id: label_ints[i] for i, (edge_id, _, _) in enumerate(edges)}

    return mapping, _build_report(mapping, n_zones)


def _build_report(mapping: Mapping[str, int], n_zones_requested: int) -> TazReport:
    counts = Counter(mapping.values())
    edge_counts = sorted(counts.values())
    n = len(edge_counts)
    median = edge_counts[n // 2] if n else 0
    return TazReport(
        n_edges=len(mapping),
        n_zones_requested=n_zones_requested,
        n_zones_populated=len(counts),
        min_edges_per_zone=min(edge_counts) if edge_counts else 0,
        max_edges_per_zone=max(edge_counts) if edge_counts else 0,
        median_edges_per_zone=median,
    )


# ---------------------------------------------------------------------------
# XML emission.
# ---------------------------------------------------------------------------


def taz_to_sumo_xml(mapping: Mapping[str, int]) -> str:
    """Serialize an edge→zone mapping to SUMO TAZ XML.

    SUMO accepts TAZs as ``<additional><taz id=... edges="e1 e2 ..."/></additional>``.
    ``duarouter`` and friends read this format directly.
    """
    zones: dict[int, list[str]] = defaultdict(list)
    for edge_id, zone in mapping.items():
        zones[zone].append(edge_id)

    root = ET.Element("additional")
    for zone in sorted(zones.keys()):
        taz_id = f"taz_{zone:03d}"
        edge_ids = sorted(zones[zone])
        ET.SubElement(
            root,
            "taz",
            {
                "id": taz_id,
                "edges": " ".join(edge_ids),
            },
        )

    ET.indent(root, space="  ")
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(root, encoding="unicode")


def write_taz_xml(
    mapping: Mapping[str, int],
    output_path: Path,
) -> Path:
    """Write the TAZ XML to disk and return the path written."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(taz_to_sumo_xml(mapping), encoding="utf-8")
    return output_path


__all__ = [
    "EdgeCentroid",
    "TazReport",
    "build_taz_by_kmeans",
    "extract_edge_centroids_from_sumo_network",
    "taz_to_sumo_xml",
    "write_taz_xml",
]
