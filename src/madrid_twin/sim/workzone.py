"""Parametric workzone descriptor.

A :class:`Workzone` is a small, hashable, JSON-roundtrippable description
of a temporary road closure: which edges are affected, which lanes are
closed, when the closure is active, and what the capacity-drop magnitude
is. Capacity drop is treated as a free parameter rather than a hard
value because the Highway Capacity Manual 2016 itself gives only a
range (typically 10–30% on urban arterials, larger on motorway lane
drops); during MARL training we sample inside that range as the basis
for domain randomization.

Two operational entry points are provided. ``to_sumo_additional_xml``
emits a SUMO additional-file that closes the affected lanes by
disallowing all vehicle classes — the standard SUMO recipe for a
workzone — for use with statically scheduled experiments. The runtime
companion (TraCI-side) is intentionally left to the scenario runner;
this module only carries the *description*, not the IO/simulator
coupling.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from xml.dom import minidom


@dataclass(frozen=True)
class LaneClosure:
    """One lane on one edge, closed for the duration of the workzone."""

    edge_id: str
    lane_index: int

    def __post_init__(self) -> None:
        if not self.edge_id:
            raise ValueError("LaneClosure.edge_id must be non-empty")
        if self.lane_index < 0:
            raise ValueError(f"LaneClosure.lane_index must be >= 0, got {self.lane_index}")


@dataclass(frozen=True)
class Workzone:
    """A scheduled, parameterized workzone on a SUMO network.

    Parameters
    ----------
    id
        Stable identifier — used in SUMO additional-file ids and in
        DVC/MLflow tags so the same workzone can be referenced across
        experiment runs.
    closures
        Tuple of lane closures. At least one is required.
    start_s, end_s
        Activation window in simulator seconds (``end_s`` exclusive).
        ``start_s < end_s`` is enforced.
    capacity_drop_pct
        Expected fractional throughput drop on adjacent open lanes,
        in ``[0, 1)``. This is informational metadata; the closure
        itself is implemented by disallowing traffic on ``closures``.
        Default is 0.20 (20%) which sits in the middle of the HCM 2016
        range for urban arterials.
    description
        Free-form human-readable note (e.g. "M-30 sur, carril derecho,
        obra de pavimentación").
    """

    id: str
    closures: tuple[LaneClosure, ...]
    start_s: float
    end_s: float
    capacity_drop_pct: float = 0.20
    description: str = ""

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("Workzone.id must be non-empty")
        if not self.closures:
            raise ValueError("Workzone.closures must contain at least one LaneClosure")
        if self.start_s < 0 or self.end_s < 0:
            raise ValueError("Workzone start/end must be non-negative")
        if self.start_s >= self.end_s:
            raise ValueError(
                f"Workzone start_s ({self.start_s}) must be strictly less than end_s ({self.end_s})"
            )
        if not 0.0 <= self.capacity_drop_pct < 1.0:
            raise ValueError(
                f"Workzone.capacity_drop_pct must be in [0, 1), got {self.capacity_drop_pct}"
            )

    # ------------------------------------------------------------------
    # Convenience properties.
    # ------------------------------------------------------------------

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    @property
    def affected_edges(self) -> frozenset[str]:
        return frozenset(c.edge_id for c in self.closures)

    # ------------------------------------------------------------------
    # Serialization.
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict) -> Workzone:
        closures = tuple(LaneClosure(**c) for c in data["closures"])
        return cls(
            id=data["id"],
            closures=closures,
            start_s=float(data["start_s"]),
            end_s=float(data["end_s"]),
            capacity_drop_pct=float(data.get("capacity_drop_pct", 0.20)),
            description=str(data.get("description", "")),
        )

    # ------------------------------------------------------------------
    # SUMO additional-file emission.
    # ------------------------------------------------------------------

    def to_sumo_additional_xml(self) -> str:
        """Render a SUMO additional-file XML closing the affected lanes.

        Implementation uses ``<closingLaneReroute>`` inside a
        ``<rerouter>`` element scheduled for ``[start_s, end_s)``. The
        rerouter is attached to the affected edges so vehicles already
        en route can switch to an alternative; the closure itself
        operates by disallowing all known vehicle classes on the listed
        lanes.

        Returns the XML as a pretty-printed string. Writers can either
        `print()` it or hand it to :meth:`write_sumo_additional`.
        """
        additional = ET.Element("additional")

        # One rerouter per affected edge, gathering its lane closures.
        edge_to_lanes: dict[str, list[int]] = {}
        for c in self.closures:
            edge_to_lanes.setdefault(c.edge_id, []).append(c.lane_index)

        for edge_id, lane_indices in edge_to_lanes.items():
            rerouter = ET.SubElement(
                additional,
                "rerouter",
                attrib={
                    "id": f"{self.id}__{edge_id}",
                    "edges": edge_id,
                },
            )
            interval = ET.SubElement(
                rerouter,
                "interval",
                attrib={"begin": f"{self.start_s:g}", "end": f"{self.end_s:g}"},
            )
            for lane_idx in sorted(lane_indices):
                ET.SubElement(
                    interval,
                    "closingLaneReroute",
                    attrib={
                        "id": f"{edge_id}_{lane_idx}",
                        "disallow": "all",
                    },
                )

        rough = ET.tostring(additional, encoding="unicode")
        return minidom.parseString(rough).toprettyxml(indent="  ")

    def write_sumo_additional(self, path: Path) -> Path:
        """Write the SUMO additional file to ``path`` and return ``path``."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_sumo_additional_xml(), encoding="utf-8")
        return path


# ---------------------------------------------------------------------------
# Convenience constructors.
# ---------------------------------------------------------------------------


def workzone_from_lanes(
    *,
    id: str,
    edges_and_lanes: dict[str, list[int]],
    start_s: float,
    end_s: float,
    capacity_drop_pct: float = 0.20,
    description: str = "",
) -> Workzone:
    """Build a :class:`Workzone` from a friendlier ``{edge: [lane, ...]}`` map."""
    closures: list[LaneClosure] = []
    for edge_id, lane_idxs in edges_and_lanes.items():
        for lane_idx in lane_idxs:
            closures.append(LaneClosure(edge_id=edge_id, lane_index=lane_idx))

    return Workzone(
        id=id,
        closures=tuple(closures),
        start_s=start_s,
        end_s=end_s,
        capacity_drop_pct=capacity_drop_pct,
        description=description,
    )


__all__ = [
    "LaneClosure",
    "Workzone",
    "workzone_from_lanes",
]
