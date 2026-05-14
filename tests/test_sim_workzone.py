"""Tests for the parametric workzone descriptor."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from madrid_twin.sim.workzone import LaneClosure, Workzone, workzone_from_lanes

# ---------------------------------------------------------------------------
# LaneClosure.
# ---------------------------------------------------------------------------


class TestLaneClosure:
    def test_happy_path(self) -> None:
        c = LaneClosure(edge_id="edge_a", lane_index=2)
        assert c.edge_id == "edge_a"
        assert c.lane_index == 2

    def test_empty_edge_id_raises(self) -> None:
        with pytest.raises(ValueError):
            LaneClosure(edge_id="", lane_index=0)

    def test_negative_lane_index_raises(self) -> None:
        with pytest.raises(ValueError):
            LaneClosure(edge_id="edge_a", lane_index=-1)

    def test_is_hashable(self) -> None:
        # Frozen dataclasses must be hashable so we can put them in sets.
        a = LaneClosure(edge_id="e", lane_index=0)
        b = LaneClosure(edge_id="e", lane_index=0)
        assert {a, b} == {a}


# ---------------------------------------------------------------------------
# Workzone validation.
# ---------------------------------------------------------------------------


def _wz(**overrides):
    defaults = {
        "id": "wz1",
        "closures": (LaneClosure(edge_id="e1", lane_index=0),),
        "start_s": 0.0,
        "end_s": 3600.0,
    }
    defaults.update(overrides)
    return Workzone(**defaults)


class TestWorkzoneValidation:
    def test_happy_path(self) -> None:
        wz = _wz()
        assert wz.id == "wz1"
        assert wz.duration_s == 3600.0
        assert wz.affected_edges == frozenset({"e1"})

    def test_empty_id_raises(self) -> None:
        with pytest.raises(ValueError):
            _wz(id="")

    def test_no_closures_raises(self) -> None:
        with pytest.raises(ValueError):
            _wz(closures=())

    def test_inverted_times_raises(self) -> None:
        with pytest.raises(ValueError):
            _wz(start_s=3600.0, end_s=0.0)

    def test_equal_times_raises(self) -> None:
        with pytest.raises(ValueError):
            _wz(start_s=600.0, end_s=600.0)

    def test_negative_times_raises(self) -> None:
        with pytest.raises(ValueError):
            _wz(start_s=-1.0, end_s=100.0)

    @pytest.mark.parametrize("bad", [-0.1, 1.0, 1.5])
    def test_invalid_capacity_drop_raises(self, bad: float) -> None:
        with pytest.raises(ValueError):
            _wz(capacity_drop_pct=bad)


# ---------------------------------------------------------------------------
# Workzone serialization.
# ---------------------------------------------------------------------------


class TestWorkzoneRoundTrip:
    def test_to_from_dict_roundtrip(self) -> None:
        wz = _wz(
            closures=(
                LaneClosure(edge_id="e1", lane_index=0),
                LaneClosure(edge_id="e2", lane_index=1),
            ),
            capacity_drop_pct=0.15,
            description="M-30 sur, calzada derecha",
        )
        restored = Workzone.from_dict(wz.to_dict())
        assert restored == wz

    def test_json_roundtrip(self) -> None:
        import json as _json

        wz = _wz()
        restored = Workzone.from_dict(_json.loads(wz.to_json()))
        assert restored == wz


# ---------------------------------------------------------------------------
# SUMO additional-file emission.
# ---------------------------------------------------------------------------


class TestSumoAdditionalEmission:
    def test_produces_valid_xml(self) -> None:
        wz = _wz()
        xml_text = wz.to_sumo_additional_xml()
        # Must parse without error and have an <additional> root.
        # The XML is pretty-printed and starts with a declaration; strip it
        # so ET can re-parse cleanly.
        root = ET.fromstring(xml_text.split("?>", 1)[-1])
        assert root.tag == "additional"

    def test_one_rerouter_per_edge(self) -> None:
        wz = _wz(
            closures=(
                LaneClosure(edge_id="e1", lane_index=0),
                LaneClosure(edge_id="e1", lane_index=1),
                LaneClosure(edge_id="e2", lane_index=0),
            ),
        )
        root = ET.fromstring(wz.to_sumo_additional_xml().split("?>", 1)[-1])
        rerouters = root.findall("rerouter")
        edges_seen = {r.attrib["edges"] for r in rerouters}
        assert edges_seen == {"e1", "e2"}
        # The e1 rerouter should have two closingLaneReroute children.
        e1 = next(r for r in rerouters if r.attrib["edges"] == "e1")
        closures = e1.find("interval").findall("closingLaneReroute")  # type: ignore[union-attr]
        assert len(closures) == 2

    def test_interval_window_matches(self) -> None:
        wz = _wz(start_s=600.0, end_s=4200.0)
        root = ET.fromstring(wz.to_sumo_additional_xml().split("?>", 1)[-1])
        interval = root.find("rerouter/interval")
        assert interval is not None
        assert interval.attrib["begin"] == "600"
        assert interval.attrib["end"] == "4200"

    def test_write_sumo_additional(self, tmp_path: Path) -> None:
        wz = _wz()
        out = tmp_path / "additional.xml"
        path = wz.write_sumo_additional(out)
        assert path == out
        assert out.exists()
        assert out.read_text(encoding="utf-8").startswith("<?xml")


# ---------------------------------------------------------------------------
# workzone_from_lanes() helper.
# ---------------------------------------------------------------------------


class TestWorkzoneFromLanes:
    def test_builds_correct_closures(self) -> None:
        wz = workzone_from_lanes(
            id="wz_demo",
            edges_and_lanes={"e1": [0, 1], "e2": [2]},
            start_s=0.0,
            end_s=600.0,
        )
        assert len(wz.closures) == 3
        assert wz.affected_edges == {"e1", "e2"}
