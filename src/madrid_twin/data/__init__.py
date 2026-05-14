"""Data layer — ingest, version and stage all inputs to the digital twin."""

from __future__ import annotations

from madrid_twin.data.open_data import (
    TRAFFIC_INTENSITY_URL,
    DetectorReading,
    fetch_traffic_intensity,
    fetch_traffic_intensity_xml,
    filter_valid_readings,
    parse_traffic_intensity_xml,
)

__all__ = [
    "TRAFFIC_INTENSITY_URL",
    "DetectorReading",
    "fetch_traffic_intensity",
    "fetch_traffic_intensity_xml",
    "filter_valid_readings",
    "parse_traffic_intensity_xml",
]
