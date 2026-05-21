"""Tests for the TAZ decomposition.

We test:
  * Pure XML emission — no scipy needed.
  * K-means clustering on a small deterministic point set — needs
    scipy, but it's a core project dependency so always available.
"""

from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

from madrid_twin.sim.taz import (
    TazReport,
    build_taz_by_kmeans,
    taz_to_sumo_xml,
    write_taz_xml,
)


# ---------------------------------------------------------------------------
# taz_to_sumo_xml()
# ---------------------------------------------------------------------------


class TestXml:
    def test_basic_structure(self) -> None:
        mapping = {"e1": 0, "e2": 0, "e3": 1, "e4": 2}
        xml = taz_to_sumo_xml(mapping)
        root = ET.fromstring(xml)
        assert root.tag == "additional"

        tazs = list(root.findall("taz"))
        assert len(tazs) == 3
        ids = [t.attrib["id"] for t in tazs]
        # Zone IDs are zero-padded and sorted.
        assert ids == ["taz_000", "taz_001", "taz_002"]

    def test_edges_per_zone(self) -> None:
        mapping = {"e1": 0, "e2": 0, "e3": 1}
        xml = taz_to_sumo_xml(mapping)
        root = ET.fromstring(xml)
        z0 = next(t for t in root.findall("taz") if t.attrib["id"] == "taz_000")
        z1 = next(t for t in root.findall("taz") if t.attrib["id"] == "taz_001")
        assert sorted(z0.attrib["edges"].split()) == ["e1", "e2"]
        assert z1.attrib["edges"] == "e3"

    def test_empty_mapping_is_empty_xml(self) -> None:
        xml = taz_to_sumo_xml({})
        root = ET.fromstring(xml)
        assert root.tag == "additional"
        assert list(root) == []

    def test_xml_declaration_present(self) -> None:
        xml = taz_to_sumo_xml({"e1": 0})
        assert xml.startswith('<?xml version="1.0" encoding="UTF-8"?>')


# ---------------------------------------------------------------------------
# write_taz_xml()
# ---------------------------------------------------------------------------


class TestWrite:
    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        out = tmp_path / "nested" / "deep" / "taz.xml"
        mapping = {"e1": 0, "e2": 1}
        result = write_taz_xml(mapping, out)
        assert result == out
        assert out.exists()
        # Round-trip parse to confirm the file is well-formed XML.
        root = ET.fromstring(out.read_text(encoding="utf-8"))
        assert len(list(root.findall("taz"))) == 2


# ---------------------------------------------------------------------------
# build_taz_by_kmeans()
# ---------------------------------------------------------------------------


class TestKmeans:
    def test_obvious_two_cluster_split(self) -> None:
        # Two well-separated blobs — K-means++ must find them.
        edges = [
            ("e1", 0.0, 0.0),
            ("e2", 1.0, 0.0),
            ("e3", 0.0, 1.0),
            ("e4", 100.0, 100.0),
            ("e5", 101.0, 100.0),
            ("e6", 100.0, 101.0),
        ]
        mapping, report = build_taz_by_kmeans(edges, n_zones=2, seed=0)
        # Edges in the first blob should share a zone, distinct from
        # edges in the second blob. The actual zone indices are
        # arbitrary, so test that {e1,e2,e3} ∩ {e4,e5,e6} = empty in
        # terms of assigned zone.
        first_blob = {mapping["e1"], mapping["e2"], mapping["e3"]}
        second_blob = {mapping["e4"], mapping["e5"], mapping["e6"]}
        assert len(first_blob) == 1
        assert len(second_blob) == 1
        assert first_blob != second_blob

        assert report.n_edges == 6
        assert report.n_zones_requested == 2
        assert report.n_zones_populated == 2

    def test_deterministic_with_same_seed(self) -> None:
        edges = [(f"e{i}", float(i % 10), float(i // 10)) for i in range(50)]
        m1, _ = build_taz_by_kmeans(edges, n_zones=5, seed=123)
        m2, _ = build_taz_by_kmeans(edges, n_zones=5, seed=123)
        assert m1 == m2

    def test_every_edge_is_assigned(self) -> None:
        edges = [(f"e{i}", float(i), float(i * 2)) for i in range(20)]
        mapping, _ = build_taz_by_kmeans(edges, n_zones=4, seed=42)
        assert set(mapping.keys()) == {e[0] for e in edges}

    def test_zero_or_negative_n_zones_raises(self) -> None:
        edges = [("e1", 0.0, 0.0)]
        with pytest.raises(ValueError):
            build_taz_by_kmeans(edges, n_zones=0)
        with pytest.raises(ValueError):
            build_taz_by_kmeans(edges, n_zones=-1)

    def test_empty_edges_raises(self) -> None:
        with pytest.raises(ValueError):
            build_taz_by_kmeans([], n_zones=1)

    def test_too_many_zones_raises(self) -> None:
        edges = [("e1", 0.0, 0.0), ("e2", 1.0, 1.0)]
        with pytest.raises(ValueError):
            build_taz_by_kmeans(edges, n_zones=3)

    def test_report_edge_counts(self) -> None:
        # 8 edges into 2 clusters; both clusters should have a few edges.
        edges = [
            ("e1", 0.0, 0.0),
            ("e2", 0.5, 0.0),
            ("e3", 0.0, 0.5),
            ("e4", 0.5, 0.5),
            ("e5", 100.0, 100.0),
            ("e6", 100.5, 100.0),
            ("e7", 100.0, 100.5),
            ("e8", 100.5, 100.5),
        ]
        _, report = build_taz_by_kmeans(edges, n_zones=2, seed=0)
        assert isinstance(report, TazReport)
        assert report.n_edges == 8
        # With this geometry both zones get 4 edges.
        assert report.min_edges_per_zone == 4
        assert report.max_edges_per_zone == 4
