"""SUMO edgeData XML emission from observed detector readings.

A SUMO ``edgeData`` (a.k.a. *meandata*) file is XML that sumo-gui can
load and use to color the network by any per-edge attribute. We emit one
such file from a live detector snapshot, joining each reading to its
mapped edge via the output of :mod:`madrid_twin.sim.snap`.

This is the first time real Ayuntamiento data shows up on the SUMO
network. It's the visual sanity-check before the W-SPSA calibration loop:
high-flow corridors should align with the M-30 ring and the major
arterials (Castellana, Alcalá, Princesa). If they don't, something's
wrong with the snapping or the CRS conversion.

When more than one detector maps to the same edge — typically because
two adjacent cross-sections of the same road both have stations — we
average the intensity and occupancy and track how many detectors
contributed in ``n_detectors``. Averaging matches the underlying physics
(both detectors are measuring the same flow at different points);
summing would double-count.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from madrid_twin.data.open_data import DetectorReading

# ---------------------------------------------------------------------------
# Data classes.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EdgeObservation:
    """Aggregated observation for a single SUMO edge."""

    edge_id: str
    intensity_veh_h: float
    occupancy_pct: float | None
    load_pct: float | None
    n_detectors: int
    # Whatever detector IDs contributed (for tracing).
    detector_ids: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "intensity_veh_h": self.intensity_veh_h,
            "occupancy_pct": self.occupancy_pct,
            "load_pct": self.load_pct,
            "n_detectors": self.n_detectors,
            "detector_ids": list(self.detector_ids),
        }


# ---------------------------------------------------------------------------
# Aggregation.
# ---------------------------------------------------------------------------


def _mean_or_none(values: list[float]) -> float | None:
    """Return the arithmetic mean of ``values`` or ``None`` if empty."""
    return sum(values) / len(values) if values else None


def aggregate_observations_by_edge(
    readings: Iterable[DetectorReading],
    mapping: Mapping[str, str],
) -> dict[str, EdgeObservation]:
    """Group detector readings by their mapped SUMO edge and aggregate.

    Parameters
    ----------
    readings
        Output of :func:`parse_traffic_intensity_xml`.
    mapping
        ``{detector_id: edge_id}`` — the per-detector edge from the snap
        step. Detectors not in the mapping are silently dropped.

    Returns
    -------
    ``{edge_id: EdgeObservation}``. Edges with zero healthy contributing
    detectors are omitted (no observation = nothing to color).
    """
    buckets: dict[str, dict[str, list[float] | list[str]]] = {}
    for r in readings:
        edge_id = mapping.get(r.detector_id)
        if edge_id is None:
            continue
        # Only healthy detectors with an intensity contribute. Otherwise
        # the aggregate would mix observed flow with sensor failures.
        # The Ayuntamiento feed occasionally uses -1 as a "no data"
        # sentinel rather than leaving <intensidad> empty; treat those
        # as missing too.
        if r.service_status not in (None, "", "0"):
            continue
        if r.intensity_veh_h is None or r.intensity_veh_h < 0:
            continue

        bucket = buckets.setdefault(
            edge_id,
            {"intensity": [], "occupancy": [], "load": [], "ids": []},
        )
        bucket["intensity"].append(r.intensity_veh_h)  # type: ignore[arg-type]
        if r.occupancy_pct is not None:
            bucket["occupancy"].append(r.occupancy_pct)  # type: ignore[arg-type]
        if r.load_pct is not None:
            bucket["load"].append(r.load_pct)  # type: ignore[arg-type]
        bucket["ids"].append(r.detector_id)  # type: ignore[arg-type]

    out: dict[str, EdgeObservation] = {}
    for edge_id, b in buckets.items():
        intensities: list[float] = b["intensity"]  # type: ignore[assignment]
        occupancies: list[float] = b["occupancy"]  # type: ignore[assignment]
        loads: list[float] = b["load"]  # type: ignore[assignment]
        ids: list[str] = b["ids"]  # type: ignore[assignment]
        out[edge_id] = EdgeObservation(
            edge_id=edge_id,
            intensity_veh_h=sum(intensities) / len(intensities),
            occupancy_pct=_mean_or_none(occupancies),
            load_pct=_mean_or_none(loads),
            n_detectors=len(intensities),
            detector_ids=tuple(sorted(ids)),
        )

    return out


# ---------------------------------------------------------------------------
# XML emission.
# ---------------------------------------------------------------------------


def edgedata_to_xml(
    observations: Mapping[str, EdgeObservation],
    *,
    interval_id: str = "observed",
    begin_s: float = 0.0,
    end_s: float = 3600.0,
) -> str:
    """Serialize observations to a SUMO ``edgeData`` XML string.

    The format is the one ``sumo-gui`` loads via *File → Load Edge Data*
    (also accessible by the ``-G`` start flag). Once loaded, *View → Edit
    Visualization → Streets → Color by* exposes every numeric attribute
    as a coloring option.
    """
    root = ET.Element("meandata")
    interval = ET.SubElement(
        root,
        "interval",
        {
            "id": interval_id,
            "begin": f"{begin_s:.1f}",
            "end": f"{end_s:.1f}",
        },
    )
    for edge_id, obs in sorted(observations.items()):
        attrs: dict[str, str] = {
            "id": edge_id,
            "intensity_veh_h": f"{obs.intensity_veh_h:.2f}",
            "n_detectors": str(obs.n_detectors),
        }
        if obs.occupancy_pct is not None:
            attrs["occupancy_pct"] = f"{obs.occupancy_pct:.2f}"
        if obs.load_pct is not None:
            attrs["load_pct"] = f"{obs.load_pct:.2f}"
        # Saturation fraction is a useful derived attribute. Defending
        # against zero capacity = NaN by guarding here.
        # (We don't have intensity_sat_veh_h on the EdgeObservation
        # because per-edge capacity isn't well-defined when multiple
        # detectors with different capacities contribute. Save that for
        # a future refinement.)
        ET.SubElement(interval, "edge", attrs)

    ET.indent(root, space="  ")
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(root, encoding="unicode")


def write_edgedata_xml(
    observations: Mapping[str, EdgeObservation],
    output_path: Path,
    *,
    interval_id: str = "observed",
    begin_s: float = 0.0,
    end_s: float = 3600.0,
) -> Path:
    """Write the edgeData XML to disk and return the path written."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    xml_text = edgedata_to_xml(observations, interval_id=interval_id, begin_s=begin_s, end_s=end_s)
    output_path.write_text(xml_text, encoding="utf-8")
    return output_path


# ---------------------------------------------------------------------------
# Mapping I/O.
# ---------------------------------------------------------------------------


def load_detector_edge_mapping(mapping_json_path: Path) -> dict[str, str]:
    """Load the ``detector_edge_mapping.json`` produced by snap_detectors."""
    import json

    payload = json.loads(Path(mapping_json_path).read_text(encoding="utf-8"))
    # Payload shape: {detector_id: {edge_id, ...}}
    return {det_id: entry["edge_id"] for det_id, entry in payload.items()}


__all__ = [
    "EdgeObservation",
    "aggregate_observations_by_edge",
    "edgedata_to_xml",
    "load_detector_edge_mapping",
    "write_edgedata_xml",
]
