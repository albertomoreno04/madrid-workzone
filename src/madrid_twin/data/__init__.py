"""Data layer — ingest, version and stage all inputs to the digital twin."""

from __future__ import annotations

from madrid_twin.data.audit import (
    AuditConfig,
    DetectorReliability,
    aggregate_snapshots,
    filter_usable_detectors,
    load_snapshot_files,
    write_audit_outputs,
)
from madrid_twin.data.open_data import (
    TRAFFIC_INTENSITY_URL,
    DetectorReading,
    fetch_traffic_intensity,
    fetch_traffic_intensity_xml,
    filter_valid_readings,
    parse_traffic_intensity_xml,
)

__all__ = [
    "AuditConfig",
    "DetectorReading",
    "DetectorReliability",
    "TRAFFIC_INTENSITY_URL",
    "aggregate_snapshots",
    "fetch_traffic_intensity",
    "fetch_traffic_intensity_xml",
    "filter_usable_detectors",
    "filter_valid_readings",
    "load_snapshot_files",
    "parse_traffic_intensity_xml",
    "write_audit_outputs",
]
