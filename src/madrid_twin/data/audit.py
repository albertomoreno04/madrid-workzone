"""Detector-reliability audit over accumulated open-data snapshots.

After running ``make probe`` on a schedule for a few days, this module
walks ``data/raw/traffic_intensity/`` and produces a per-detector
reliability table: how often the detector was healthy, how often it
reported a non-null intensity, mean/std of its intensity readings, and
the first / last timestamps it was seen.

The output drives the rest of the pipeline. Detectors that fail the
reliability bar (configurable; the defaults match the Wisconsin DOT
calibration recommendations) are dropped from the OD calibration
targets in Phase 4 and from the predictive layer's training set in
Phase 5.

This module is pure stdlib + numpy and operates on already-parsed
:class:`DetectorReading` instances — no HTTP, no file system except
the snapshot directory walk.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from madrid_twin.data.open_data import (
    DetectorReading,
    parse_traffic_intensity_xml,
)


@dataclass(frozen=True)
class DetectorReliability:
    """Per-detector aggregate statistics over a window of snapshots."""

    detector_id: str
    n_snapshots: int
    n_healthy: int  # service_status in ("0", None, "")
    n_with_intensity: int  # intensity_veh_h is not None
    healthy_fraction: float
    intensity_fraction: float
    mean_intensity_veh_h: float | None
    std_intensity_veh_h: float | None
    first_seen_iso: str | None
    last_seen_iso: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AuditConfig:
    """Reliability thresholds for marking a detector 'usable'."""

    min_healthy_fraction: float = 0.80
    min_intensity_fraction: float = 0.80
    min_mean_intensity_veh_h: float = 1.0  # filters out always-zero detectors

    def __post_init__(self) -> None:
        for name in ("min_healthy_fraction", "min_intensity_fraction"):
            val = getattr(self, name)
            if not 0.0 <= val <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {val}")
        if self.min_mean_intensity_veh_h < 0:
            raise ValueError("min_mean_intensity_veh_h must be non-negative")


def aggregate_snapshots(
    snapshots: Sequence[Sequence[DetectorReading]],
) -> dict[str, DetectorReliability]:
    """Aggregate a sequence of parsed snapshots into per-detector stats.

    Each ``snapshots[i]`` is the list of readings from one probe run.
    Detector IDs are unioned across all snapshots; a detector missing
    from one snapshot doesn't increment its ``n_snapshots`` count for
    that snapshot — only detectors actually present do.
    """
    # First pass: accumulate per-detector lists of (status, intensity, timestamp).
    per_detector: dict[str, dict[str, list]] = {}
    for snap in snapshots:
        for r in snap:
            row = per_detector.setdefault(
                r.detector_id,
                {"statuses": [], "intensities": [], "timestamps": []},
            )
            row["statuses"].append(r.service_status)
            row["intensities"].append(r.intensity_veh_h)
            row["timestamps"].append(r.timestamp)

    out: dict[str, DetectorReliability] = {}
    for det_id, row in per_detector.items():
        statuses = row["statuses"]
        intensities = [v for v in row["intensities"] if v is not None]
        timestamps = [t for t in row["timestamps"] if t is not None]

        n = len(statuses)
        n_healthy = sum(1 for s in statuses if s in (None, "", "0"))
        n_with_intensity = sum(1 for v in row["intensities"] if v is not None)

        if intensities:
            arr = np.array(intensities, dtype=np.float64)
            mean_i = float(arr.mean())
            std_i = float(arr.std(ddof=0))
        else:
            mean_i = None
            std_i = None

        first_seen = min(timestamps).isoformat() if timestamps else None
        last_seen = max(timestamps).isoformat() if timestamps else None

        out[det_id] = DetectorReliability(
            detector_id=det_id,
            n_snapshots=n,
            n_healthy=n_healthy,
            n_with_intensity=n_with_intensity,
            healthy_fraction=n_healthy / n if n else 0.0,
            intensity_fraction=n_with_intensity / n if n else 0.0,
            mean_intensity_veh_h=mean_i,
            std_intensity_veh_h=std_i,
            first_seen_iso=first_seen,
            last_seen_iso=last_seen,
        )

    return out


def filter_usable_detectors(
    reliabilities: Iterable[DetectorReliability],
    config: AuditConfig | None = None,
) -> list[str]:
    """Return detector IDs meeting all reliability thresholds."""
    cfg = config or AuditConfig()
    out: list[str] = []
    for r in reliabilities:
        if r.healthy_fraction < cfg.min_healthy_fraction:
            continue
        if r.intensity_fraction < cfg.min_intensity_fraction:
            continue
        if r.mean_intensity_veh_h is None:
            continue
        if r.mean_intensity_veh_h < cfg.min_mean_intensity_veh_h:
            continue
        out.append(r.detector_id)
    return sorted(out)


def load_snapshot_files(snapshot_dir: Path) -> list[list[DetectorReading]]:
    """Walk a snapshot directory and parse every ``pm_*.xml`` it contains."""
    snapshot_dir = Path(snapshot_dir)
    if not snapshot_dir.exists():
        return []
    snapshots: list[list[DetectorReading]] = []
    for path in sorted(snapshot_dir.glob("pm_*.xml")):
        try:
            text = path.read_text(encoding="utf-8")
            snapshots.append(parse_traffic_intensity_xml(text))
        except (OSError, ValueError):
            # Skip unreadable / malformed snapshots — they'll show up
            # in the per-detector n_snapshots column as missing.
            continue
    return snapshots


def write_audit_outputs(
    reliabilities: dict[str, DetectorReliability],
    usable_ids: Sequence[str],
    output_dir: Path,
) -> tuple[Path, Path]:
    """Write the reliability table + usable-detectors list to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    table_path = output_dir / "detector_reliability.json"
    table_payload = {det_id: r.to_dict() for det_id, r in sorted(reliabilities.items())}
    table_path.write_text(json.dumps(table_payload, indent=2), encoding="utf-8")

    usable_path = output_dir / "usable_detectors.json"
    usable_path.write_text(
        json.dumps({"detector_ids": list(usable_ids)}, indent=2),
        encoding="utf-8",
    )

    return table_path, usable_path


__all__ = [
    "AuditConfig",
    "DetectorReliability",
    "aggregate_snapshots",
    "filter_usable_detectors",
    "load_snapshot_files",
    "write_audit_outputs",
]
