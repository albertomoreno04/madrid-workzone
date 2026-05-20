"""Tests for the Madrid Ayuntamiento open-data client.

HTTP is fully mocked via ``httpx.MockTransport`` so the tests are
deterministic and never touch the network.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import httpx
import pytest

from madrid_twin.data.open_data import (
    TRAFFIC_INTENSITY_URL,
    DetectorReading,
    fetch_traffic_intensity,
    fetch_traffic_intensity_xml,
    filter_valid_readings,
    parse_traffic_intensity_xml,
)

MALFORMED_XML = "<pms><pm><idelem>3501</idelem></pm" + "<<<malformed"


# ---------------------------------------------------------------------------
# parse_traffic_intensity_xml() — legacy fixture shape.
# ---------------------------------------------------------------------------


class TestParse:
    def test_parses_all_detectors(self, traffic_intensity_sample_xml: str) -> None:
        readings = parse_traffic_intensity_xml(traffic_intensity_sample_xml)
        assert len(readings) == 3
        assert {r.detector_id for r in readings} == {"3501", "3502", "3503"}

    def test_numeric_fields_are_parsed(self, traffic_intensity_sample_xml: str) -> None:
        readings = parse_traffic_intensity_xml(traffic_intensity_sample_xml)
        d1 = next(r for r in readings if r.detector_id == "3501")
        assert d1.intensity_veh_h == 1240.0
        # Spanish decimal "12,5" must become 12.5.
        assert d1.occupancy_pct == 12.5
        assert d1.load_pct == 34.0
        assert d1.service_status == "0"

    def test_empty_numerics_become_none(self, traffic_intensity_sample_xml: str) -> None:
        readings = parse_traffic_intensity_xml(traffic_intensity_sample_xml)
        d3 = next(r for r in readings if r.detector_id == "3503")
        assert d3.intensity_veh_h is None
        assert d3.occupancy_pct is None
        assert d3.load_pct is None
        assert d3.service_status == "2"

    def test_timestamp_parsed_in_spanish_locale(self, traffic_intensity_sample_xml: str) -> None:
        readings = parse_traffic_intensity_xml(traffic_intensity_sample_xml)
        d1 = next(r for r in readings if r.detector_id == "3501")
        assert d1.timestamp == datetime(2026, 5, 13, 8, 30, 0)

    def test_malformed_xml_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            parse_traffic_intensity_xml(MALFORMED_XML)

    def test_empty_pms_returns_empty_list(self) -> None:
        readings = parse_traffic_intensity_xml("<pms></pms>")
        assert readings == []


# ---------------------------------------------------------------------------
# parse_traffic_intensity_xml() — live-feed-shaped fixture.
# ---------------------------------------------------------------------------


class TestParseLiveFeed:
    def test_extracts_all_detectors(self, real_traffic_intensity_sample_xml: str) -> None:
        readings = parse_traffic_intensity_xml(real_traffic_intensity_sample_xml)
        assert len(readings) == 3
        assert {r.detector_id for r in readings} == {"9841", "9842", "9999"}

    def test_root_fecha_hora_propagates_to_all_readings(
        self, real_traffic_intensity_sample_xml: str
    ) -> None:
        readings = parse_traffic_intensity_xml(real_traffic_intensity_sample_xml)
        expected = datetime(2026, 5, 20, 12, 50, 19)
        for r in readings:
            assert r.timestamp == expected

    def test_error_flag_N_maps_to_healthy(self, real_traffic_intensity_sample_xml: str) -> None:
        readings = parse_traffic_intensity_xml(real_traffic_intensity_sample_xml)
        d1 = next(r for r in readings if r.detector_id == "9841")
        assert d1.service_status == "0"

    def test_error_flag_Y_maps_to_failure(self, real_traffic_intensity_sample_xml: str) -> None:
        readings = parse_traffic_intensity_xml(real_traffic_intensity_sample_xml)
        broken = next(r for r in readings if r.detector_id == "9999")
        assert broken.service_status == "1"
        assert broken.intensity_veh_h is None  # also empty in the fixture

    def test_description_extracted(self, real_traffic_intensity_sample_xml: str) -> None:
        readings = parse_traffic_intensity_xml(real_traffic_intensity_sample_xml)
        d1 = next(r for r in readings if r.detector_id == "9841")
        assert d1.description == "Valle de Mena S-E - Acc.Ramon Castroviejo"

    def test_saturation_intensity_extracted(self, real_traffic_intensity_sample_xml: str) -> None:
        readings = parse_traffic_intensity_xml(real_traffic_intensity_sample_xml)
        d1 = next(r for r in readings if r.detector_id == "9841")
        assert d1.intensity_sat_veh_h == 3100.0

    def test_service_level_extracted(self, real_traffic_intensity_sample_xml: str) -> None:
        readings = parse_traffic_intensity_xml(real_traffic_intensity_sample_xml)
        d1 = next(r for r in readings if r.detector_id == "9841")
        assert d1.service_level == "0"

    def test_subarea_extracted(self, real_traffic_intensity_sample_xml: str) -> None:
        readings = parse_traffic_intensity_xml(real_traffic_intensity_sample_xml)
        d1 = next(r for r in readings if r.detector_id == "9841")
        assert d1.subarea == "0328"

    def test_utm_coordinates_extracted_with_spanish_decimals(
        self, real_traffic_intensity_sample_xml: str
    ) -> None:
        readings = parse_traffic_intensity_xml(real_traffic_intensity_sample_xml)
        d1 = next(r for r in readings if r.detector_id == "9841")
        # The fixture uses Spanish decimal commas; the parser must
        # convert them to floats.
        assert d1.x_utm == pytest.approx(438339.375874991)
        assert d1.y_utm == pytest.approx(4480454.96970565)

    def test_per_pm_fecha_hora_overrides_root(self) -> None:
        # When both root and per-pm timestamps are present, the per-pm
        # one wins. (Defensive in case a future feed format inverts the
        # convention.)
        xml = """<?xml version="1.0" encoding="UTF-8"?>
        <pms>
          <fecha_hora>01/01/2026 00:00:00</fecha_hora>
          <pm>
            <idelem>1</idelem>
            <fecha_hora>02/02/2026 12:34:56</fecha_hora>
            <intensidad>100</intensidad>
            <error>N</error>
          </pm>
        </pms>"""
        readings = parse_traffic_intensity_xml(xml)
        assert readings[0].timestamp == datetime(2026, 2, 2, 12, 34, 56)


# ---------------------------------------------------------------------------
# filter_valid_readings()
# ---------------------------------------------------------------------------


class TestFilterValid:
    def test_drops_failed_and_missing(self) -> None:
        readings = [
            DetectorReading("a", None, 1000.0, 10.0, 20.0, "0"),
            DetectorReading("b", None, None, None, None, "2"),  # service failure
            DetectorReading("c", None, 500.0, 5.0, 10.0, "1"),  # service failure
            DetectorReading("d", None, 800.0, 8.0, 15.0, None),  # missing status -> healthy
            DetectorReading("e", None, None, None, None, "0"),  # healthy but null intensity
        ]
        kept = filter_valid_readings(readings)
        assert {r.detector_id for r in kept} == {"a", "d"}


# ---------------------------------------------------------------------------
# fetch_traffic_intensity_xml() — mocked HTTP.
# ---------------------------------------------------------------------------


def _mock_client(
    handler: Callable[[httpx.Request], httpx.Response],
) -> httpx.Client:
    transport = httpx.MockTransport(handler)
    return httpx.Client(transport=transport, timeout=5.0)


class TestFetch:
    def test_happy_path(self, tmp_path: Path, traffic_intensity_sample_xml: str) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert str(request.url) == TRAFFIC_INTENSITY_URL
            assert "madrid-workzone" in request.headers["user-agent"]
            return httpx.Response(200, text=traffic_intensity_sample_xml)

        with _mock_client(handler) as client:
            cache = tmp_path / "snapshot.xml"
            xml = fetch_traffic_intensity_xml(client=client, cache_path=cache)
            assert "idelem" in xml
            assert cache.exists()
            assert cache.read_text(encoding="utf-8") == traffic_intensity_sample_xml

    def test_retries_on_5xx_then_succeeds(self, traffic_intensity_sample_xml: str) -> None:
        attempts: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            attempts.append(1)
            if len(attempts) < 3:
                return httpx.Response(503, text="busy")
            return httpx.Response(200, text=traffic_intensity_sample_xml)

        slept: list[float] = []
        with _mock_client(handler) as client:
            xml = fetch_traffic_intensity_xml(
                client=client,
                max_retries=5,
                backoff_s=2.0,
                sleep=slept.append,
            )
        assert "idelem" in xml
        assert len(attempts) == 3
        # We slept twice between the three attempts.
        assert len(slept) == 2

    def test_raises_after_max_retries(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="boom")

        with _mock_client(handler) as client, pytest.raises(RuntimeError):
            fetch_traffic_intensity_xml(client=client, max_retries=2, sleep=lambda _s: None)

    def test_fetch_traffic_intensity_returns_parsed_readings(
        self, traffic_intensity_sample_xml: str
    ) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text=traffic_intensity_sample_xml)

        with _mock_client(handler) as client:
            readings = fetch_traffic_intensity(client=client)
        assert len(readings) == 3
        assert {r.detector_id for r in readings} == {"3501", "3502", "3503"}
