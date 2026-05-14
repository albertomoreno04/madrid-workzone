"""Graph features for the predictive layer.

Parses a SUMO ``.net.xml`` into a directed road-network graph and produces
the adjacency-matrix flavours the downstream ST-GNN forecasters need.
Three weightings are supported: binary connectivity, inverse-distance
(thresholded Gaussian kernel) and inverse-free-flow-travel-time. The
latter two follow the DCRNN / GraphWaveNet convention for traffic
forecasting on road networks.

The module deliberately depends only on numpy + stdlib so it imports
without the ``[predict]`` extras installed.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class RoadEdge:
    """One directed road edge in the SUMO network."""

    edge_id: str
    from_junction: str
    to_junction: str
    length_m: float
    speed_mps: float

    @property
    def free_flow_travel_time_s(self) -> float:
        """Length / free-flow speed (seconds). Guards against zero speed."""
        speed = max(self.speed_mps, 1e-3)
        return self.length_m / speed


@dataclass
class RoadGraph:
    """A directed road-network graph extracted from SUMO."""

    edges: list[RoadEdge]
    # Index mapping: edge_id -> index in `edges` list.
    edge_index: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.edge_index:
            self.edge_index = {e.edge_id: i for i, e in enumerate(self.edges)}

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    def edge_ids(self) -> list[str]:
        return [e.edge_id for e in self.edges]


AdjacencyKind = Literal["binary", "distance", "travel_time"]


# ---------------------------------------------------------------------------
# Parsing.
# ---------------------------------------------------------------------------


def parse_sumo_network(net_xml_path: Path) -> RoadGraph:
    """Parse a SUMO ``.net.xml`` into a :class:`RoadGraph`.

    Only ``<edge>`` elements with ``function != "internal"`` are kept (we
    skip SUMO's internal junction edges, which carry no length/speed of
    interest for forecasting). Each edge's length and speed are taken
    from its first ``<lane>`` child — the standard convention.
    """
    net_xml_path = Path(net_xml_path)
    tree = ET.parse(net_xml_path)
    root = tree.getroot()

    edges: list[RoadEdge] = []
    for edge in root.findall("edge"):
        if edge.attrib.get("function") == "internal":
            continue
        edge_id = edge.attrib.get("id")
        if not edge_id:
            continue
        from_j = edge.attrib.get("from")
        to_j = edge.attrib.get("to")
        if not from_j or not to_j:
            continue

        lane = edge.find("lane")
        if lane is None:
            continue
        length = float(lane.attrib.get("length", "0") or 0.0)
        speed = float(lane.attrib.get("speed", "13.89") or 13.89)  # ~50 km/h default

        edges.append(
            RoadEdge(
                edge_id=edge_id,
                from_junction=from_j,
                to_junction=to_j,
                length_m=length,
                speed_mps=speed,
            )
        )

    return RoadGraph(edges=edges)


# ---------------------------------------------------------------------------
# Adjacency.
# ---------------------------------------------------------------------------


def adjacency_matrix(
    graph: RoadGraph,
    kind: AdjacencyKind = "binary",
    *,
    sigma_s: float | None = None,
    threshold: float = 0.1,
) -> np.ndarray:
    """Build a square adjacency matrix over ``graph.edges``.

    Two edges ``i`` and ``j`` are connected when edge ``i``'s downstream
    junction equals edge ``j``'s upstream junction — i.e. ``i`` feeds
    into ``j``.

    Parameters
    ----------
    kind
        ``"binary"``      Unweighted 0/1 connectivity.
        ``"distance"``    Gaussian kernel ``exp(-(length / sigma)^2)`` on
                          the upstream edge's length. Values below
                          ``threshold`` are zeroed (a sparsification
                          standard in DCRNN).
        ``"travel_time"`` Same kernel but on free-flow travel time.
    sigma_s
        Bandwidth of the Gaussian kernel. When ``None``, the kernel uses
        the standard deviation of the relevant quantity across edges,
        matching the DCRNN preprocessing recipe.
    threshold
        Values below this (after kernelisation) are set to zero. Ignored
        for ``kind="binary"``.
    """
    n = graph.n_edges
    A = np.zeros((n, n), dtype=np.float64)

    # Build a junction -> list-of-(downstream-edge-index) lookup once.
    out_by_junction: dict[str, list[int]] = {}
    for j, e in enumerate(graph.edges):
        out_by_junction.setdefault(e.from_junction, []).append(j)

    for i, e_up in enumerate(graph.edges):
        downstream = out_by_junction.get(e_up.to_junction, [])
        for j in downstream:
            A[i, j] = 1.0

    if kind == "binary":
        return A

    # Quantity associated with each upstream edge for the kernel.
    if kind == "distance":
        values = np.array([e.length_m for e in graph.edges], dtype=np.float64)
    else:  # travel_time
        values = np.array([e.free_flow_travel_time_s for e in graph.edges], dtype=np.float64)

    if sigma_s is None:
        # Fall back to std-dev across edges; protect against degenerate networks.
        std = float(values.std())
        sigma_s = std if std > 1e-9 else 1.0

    inv_sigma_sq = 1.0 / (sigma_s * sigma_s)
    for i in range(n):
        for j in range(n):
            if A[i, j] == 0.0:
                continue
            A[i, j] = math.exp(-(values[i] ** 2) * inv_sigma_sq)

    A[threshold > A] = 0.0
    return A


# ---------------------------------------------------------------------------
# Detector → node mapping.
# ---------------------------------------------------------------------------


def detector_to_node_indices(
    graph: RoadGraph,
    detector_to_edge: dict[str, str],
) -> dict[str, int]:
    """Map detector ids to graph node (edge) indices.

    Detectors whose declared SUMO edge is absent from ``graph`` are
    silently skipped — they may live on internal edges or on edges
    filtered out during network construction.
    """
    return {
        det_id: graph.edge_index[edge_id]
        for det_id, edge_id in detector_to_edge.items()
        if edge_id in graph.edge_index
    }


__all__ = [
    "AdjacencyKind",
    "RoadEdge",
    "RoadGraph",
    "adjacency_matrix",
    "detector_to_node_indices",
    "parse_sumo_network",
]
