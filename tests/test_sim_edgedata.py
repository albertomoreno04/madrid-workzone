"""Tests for the edgeData XML emitter."""

from __future__ import annotations

import json
from pathlib import Path
from xml.etree import ElementTree as ET

from madrid_twin.data.open_data import DetectorReading
from madrid_twin.sim.edgedata import (
    EdgeObservation,
    aggregate_observations_by_edge,
    edgedata_to_xml,
    load_detector_edge_mapping,
    write_edgedata_xml,
)


def _reading(
    detector_id: str,
    intensity: float | None = 500.0,
    occupancy: float | None = 10.0,
    load: float | None = 20.0,
    status: str | None = "0",
) -> DetectorReading:
    return DetectorReading(
        detector_id=detector_id,
        timestamp=None,
        intensity_veh_h=intensity,
        occupancy_pct=occupancy,
        load_pct=load,
        service_status=status,
    )


# ---------------------------------------------------------------------------
# aggregate_observations_by_edge()
# ---------------------------------------------------------------------------


class TestAggregate:
    def test_single_detector_per_edge(self) -> None:
        readings = [_reading("d1", intensity=400.0)]
        mapping = {"d1": "e1"}
        obs = aggregate_observations_by_edge(readings, mapping)
        assert set(obs) == {"e1"}
        assert obs["e1"].intensity_veh_h == 400.0
        assert obs["e1"].n_detectors == 1
        assert obs["e1"].detector_ids == ("d1",)

    def test_multiple_detectors_per_edge_are_averaged(self) -> None:
        readings = [
            _reading("d1", intensity=400.0, occupancy=10.0),
            _reading("d2", intensity=600.0, occupancy=14.0),
        ]
        mapping = {"d1": "e1", "d2": "e1"}
        obs = aggregate_observations_by_edge(readings, mapping)
        assert obs["e1"].intensity_veh_h == 500.0
        assert obs["e1"].occupancy_pct == 12.0
        assert obs["e1"].n_detectors == 2
        assert obs["e1"].detector_ids == ("d1", "d2")

    def test_unmapped_detector_is_silently_dropped(self) -> None:
        readings = [_reading("d1", intensity=400.0), _reading("d2", intensity=900.0)]
        mapping = {"d1": "e1"}
        obs = aggregate_observations_by_edge(readings, mapping)
        assert set(obs) == {"e1"}
        assert obs["e1"].intensity_veh_h == 400.0

    def test_failed_detector_is_dropped(self) -> None:
        readings = [
            _reading("d1", intensity=500.0, status="0"),
            _reading("d2", intensity=900.0, status="1"),  # failure
            _reading("d3", intensity=700.0, status="2"),  # failure
        ]
        mapping = {"d1": "e1", "d2": "e1", "d3": "e1"}
        obs = aggregate_observations_by_edge(readings, mapping)
        assert obs["e1"].intensity_veh_h == 500.0
        assert obs["e1"].n_detectors == 1

    def test_null_intensity_detector_is_dropped(self) -> None:
        readings = [
            _reading("d1", intensity=500.0),
            _reading("d2", intensity=None),
        ]
        mapping = {"d1": "e1", "d2": "e1"}
        obs = aggregate_observations_by_edge(readings, mapping)
        assert obs["e1"].intensity_veh_h == 500.0
        assert obs["e1"].n_detectors == 1

    def test_missing_occupancy_treated_as_absent(self) -> None:
        readings = [
            _reading("d1", intensity=500.0, occupancy=10.0),
            _reading("d2", intensity=500.0, occupancy=None),
        ]
        mapping = {"d1": "e1", "d2": "e1"}
        obs = aggregate_observations_by_edge(readings, mapping)
        # Only the one with occupancy contributes to the mean.
        assert obs["e1"].occupancy_pct == 10.0
        assert obs["e1"].n_detectors == 2

    def test_edges_with_only_failed_detectors_are_omitted(self) -> None:
        readings = [_reading("d1", status="1")]
        mapping = {"d1": "e1"}
        obs = aggregate_observations_by_edge(readings, mapping)
        assert obs == {}


# ---------------------------------------------------------------------------
# edgedata_to_xml()
# ---------------------------------------------------------------------------


class TestXml:
    def test_basic_structure(self) -> None:
        obs = {
            "e1": EdgeObservation("e1", 500.0, 10.0, 20.0, 1, ("d1",)),
            "e2": EdgeObservation("e2", 300.0, None, None, 1, ("d2",)),
        }
        xml = edgedata_to_xml(obs, interval_id="t0", begin_s=0.0, end_s=900.0)
        root = ET.fromstring(xml)
        assert root.tag == "meandata"
        intervals = list(root.findall("interval"))
        assert len(intervals) == 1
        assert intervals[0].attrib["id"] == "t0"
        assert intervals[0].attrib["begin"] == "0.0"
        assert intervals[0].attrib["end"] == "900.0"

        edges = list(intervals[0].findall("edge"))
        ids = sorted(e.attrib["id"] for e in edges)
        assert ids == ["e1", "e2"]

    def test_edge_attributes(self) -> None:
        obs = {"e1": EdgeObservation("e1", 500.0, 10.0, 20.0, 2, ("d1", "d2"))}
        xml = edgedata_to_xml(obs)
        edge = ET.fromstring(xml).find("interval/edge")
        assert edge is not None
        assert edge.attrib["id"] == "e1"
        assert edge.attrib["intensity_veh_h"] == "500.00"
        assert edge.attrib["occupancy_pct"] == "10.00"
        assert edge.attrib["load_pct"] == "20.00"
        assert edge.attrib["n_detectors"] == "2"

    def test_none_attributes_are_omitted(self) -> None:
        obs = {"e1": EdgeObservation("e1", 500.0, None, None, 1, ("d1",))}
        xml = edgedata_to_xml(obs)
        edge = ET.fromstring(xml).find("interval/edge")
        assert edge is not None
        assert "occupancy_pct" not in edge.attrib
        assert "load_pct" not in edge.attrib

    def test_xml_declaration_present(self) -> None:
        xml = edgedata_to_xml({})
        assert xml.startswith('<?xml version="1.0" encoding="UTF-8"?>')


# ---------------------------------------------------------------------------
# write_edgedata_xml() + loader
# ---------------------------------------------------------------------------


class TestRoundtrip:
    def test_write_roundtrip(self, tmp_path: Path) -> None:
        obs = {"e1": EdgeObservation("e1", 500.0, 10.0, 20.0, 1, ("d1",))}
        out = tmp_path / "nested" / "obs.xml"
        result = write_edgedata_xml(obs, out)
        assert result == out
        assert out.exists()
        root = ET.fromstring(out.read_text(encoding="utf-8"))
        assert root.find("interval/edge").attrib["id"] == "e1"  # type: ignore[union-attr]

    def test_load_detector_edge_mapping(self, tmp_path: Path) -> None:
        mapping_path = tmp_path / "map.json"
        mapping_path.write_text(
            json.dumps(
                {
                    "d1": {"edge_id": "e1", "distance_m": 5.0},
                    "d2": {"edge_id": "e2", "distance_m": 2.0},
                }
            ),
            encoding="utf-8",
        )
        mapping = load_detector_edge_mapping(mapping_path)
        assert mapping == {"d1": "e1", "d2": "e2"}
