"""Tests for the road-graph features module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from madrid_twin.predict.features import (
    RoadEdge,
    RoadGraph,
    adjacency_matrix,
    detector_to_node_indices,
    parse_sumo_network,
)

# A minimal SUMO-style network: A -> B -> C plus an internal edge that
# should be filtered out.
SAMPLE_NET_XML = """<?xml version="1.0" encoding="UTF-8"?>
<net version="1.20">
  <edge id="e_ab" from="A" to="B">
    <lane id="e_ab_0" length="100.0" speed="13.89"/>
  </edge>
  <edge id="e_bc" from="B" to="C">
    <lane id="e_bc_0" length="200.0" speed="20.0"/>
  </edge>
  <edge id=":B_0" function="internal">
    <lane id=":B_0_0" length="3.0" speed="13.89"/>
  </edge>
</net>
"""


@pytest.fixture()
def sample_graph(tmp_path: Path) -> RoadGraph:
    net = tmp_path / "small.net.xml"
    net.write_text(SAMPLE_NET_XML, encoding="utf-8")
    return parse_sumo_network(net)


class TestParseSumoNetwork:
    def test_filters_internal_edges(self, sample_graph: RoadGraph) -> None:
        assert sample_graph.n_edges == 2
        assert sample_graph.edge_ids() == ["e_ab", "e_bc"]

    def test_extracts_lengths_and_speeds(self, sample_graph: RoadGraph) -> None:
        ab, bc = sample_graph.edges
        assert ab.length_m == 100.0
        assert ab.speed_mps == pytest.approx(13.89)
        assert bc.length_m == 200.0
        assert bc.speed_mps == 20.0

    def test_free_flow_travel_time(self) -> None:
        e = RoadEdge("x", "A", "B", length_m=100.0, speed_mps=10.0)
        assert e.free_flow_travel_time_s == 10.0

    def test_zero_speed_is_handled(self) -> None:
        e = RoadEdge("x", "A", "B", length_m=10.0, speed_mps=0.0)
        # Must not divide by zero; result is large but finite.
        assert e.free_flow_travel_time_s > 0
        assert e.free_flow_travel_time_s < 1e6


class TestAdjacencyMatrix:
    def test_binary_chain(self, sample_graph: RoadGraph) -> None:
        A = adjacency_matrix(sample_graph, kind="binary")
        # e_ab (idx 0) feeds e_bc (idx 1).
        expected = np.array([[0.0, 1.0], [0.0, 0.0]])
        np.testing.assert_array_equal(A, expected)

    def test_distance_weighting_is_positive_on_edges(self, sample_graph: RoadGraph) -> None:
        A = adjacency_matrix(sample_graph, kind="distance", sigma_s=200.0, threshold=0.0)
        assert A[0, 1] > 0.0
        assert A[1, 0] == 0.0
        # Diagonal stays zero in a binary-induced graph.
        assert A[0, 0] == 0.0
        assert A[1, 1] == 0.0

    def test_travel_time_weighting(self, sample_graph: RoadGraph) -> None:
        A = adjacency_matrix(sample_graph, kind="travel_time", sigma_s=50.0, threshold=0.0)
        # e_ab (idx 0) feeds e_bc (idx 1), travel time on e_ab is ~7.2s.
        # exp(-(7.2/50)^2) ~ 0.98
        assert A[0, 1] == pytest.approx(np.exp(-((100.0 / 13.89) ** 2) / (50.0**2)))

    def test_threshold_zeroes_small_entries(self, sample_graph: RoadGraph) -> None:
        # Set sigma very small so the kernel value is tiny.
        A = adjacency_matrix(sample_graph, kind="distance", sigma_s=1.0, threshold=0.1)
        assert A[0, 1] == 0.0


class TestDetectorMapping:
    def test_maps_known_and_skips_unknown(self, sample_graph: RoadGraph) -> None:
        mapping = detector_to_node_indices(
            sample_graph,
            detector_to_edge={
                "det_1": "e_ab",
                "det_2": "e_bc",
                "det_unknown": "e_xy",  # not in the graph
            },
        )
        assert mapping == {"det_1": 0, "det_2": 1}
