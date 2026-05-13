"""Parse a SUMO output bundle into a structured ``BaselineMetrics`` object.

This module is the library-level home of the logic that originally lived in
``scripts/analyze_baseline.py``. Everything is pure-Python and depends only
on the standard library, which keeps the test surface tiny and the import
graph cheap — useful because this is the one ``madrid_twin`` module that
gets imported from CI smoke tests.

Public API
----------

``parse_summary(path)``      Parse a ``summary.xml`` step trace.
``parse_tripinfo(path)``     Parse a ``tripinfo.xml`` per-trip dump.
``parse_statistics(path)``   Parse the optional ``statistics.xml`` run report.
``compute_baseline_metrics`` Top-level entry point that orchestrates the
                              three parsers and returns ``BaselineMetrics``.
``BaselineMetrics``          Dataclass holding the metrics we report.
"""

from __future__ import annotations

import json
import statistics
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Data classes & small numeric helpers.
# ---------------------------------------------------------------------------


@dataclass
class BaselineMetrics:
    """Headline KPIs extracted from a single SUMO scenario run."""

    simulation_duration_s: float | None

    avg_step_mean_speed_mps: float | None
    max_step_mean_speed_mps: float | None
    min_step_mean_speed_mps: float | None

    avg_step_mean_waiting_time_s: float | None
    max_step_mean_waiting_time_s: float | None

    avg_step_mean_travel_time_s: float | None
    max_step_mean_travel_time_s: float | None

    avg_step_running_vehicles: float | None
    max_step_running_vehicles: float | None

    total_inserted_vehicles: int | None
    total_ended_vehicles: int | None
    completion_ratio: float | None

    avg_trip_duration_s: float | None
    median_trip_duration_s: float | None
    p95_trip_duration_s: float | None

    avg_trip_route_length_m: float | None
    avg_trip_waiting_time_s: float | None
    median_trip_waiting_time_s: float | None
    p95_trip_waiting_time_s: float | None

    avg_trip_time_loss_s: float | None
    median_trip_time_loss_s: float | None
    p95_trip_time_loss_s: float | None

    avg_depart_delay_s: float | None
    p95_depart_delay_s: float | None

    avg_waiting_count_per_trip: float | None
    avg_stop_count_per_trip: float | None

    throughput_veh_per_hour: float | None

    spillback_proxy_peak_halting: int | None
    spillback_proxy_avg_halting: float | None

    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


def safe_mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def safe_median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def percentile(values: list[float], q: float) -> float | None:
    """Linear-interpolation percentile. ``q`` is in [0, 1]."""
    if not values:
        return None
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"q must be in [0, 1], got {q!r}")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


# ---------------------------------------------------------------------------
# Parsers.
# ---------------------------------------------------------------------------


def parse_summary(summary_path: Path) -> dict[str, Any]:
    """Walk the ``<step>`` records in a SUMO ``summary.xml`` file."""
    tree = ET.parse(summary_path)
    root = tree.getroot()

    step_mean_speeds: list[float] = []
    step_mean_waiting_times: list[float] = []
    step_mean_travel_times: list[float] = []
    step_running: list[float] = []
    step_halting: list[float] = []

    total_inserted = 0
    total_ended = 0
    sim_duration: float | None = None

    for step in root.findall("step"):
        attrs = step.attrib

        if "meanSpeed" in attrs:
            step_mean_speeds.append(float(attrs["meanSpeed"]))
        if "meanWaitingTime" in attrs:
            step_mean_waiting_times.append(float(attrs["meanWaitingTime"]))
        if "meanTravelTime" in attrs:
            step_mean_travel_times.append(float(attrs["meanTravelTime"]))
        if "running" in attrs:
            step_running.append(float(attrs["running"]))
        if "halting" in attrs:
            step_halting.append(float(attrs["halting"]))
        if "inserted" in attrs:
            total_inserted += int(float(attrs["inserted"]))
        if "ended" in attrs:
            total_ended += int(float(attrs["ended"]))
        if "time" in attrs:
            sim_duration = float(attrs["time"])

    return {
        "simulation_duration_s": sim_duration,
        "step_mean_speeds": step_mean_speeds,
        "step_mean_waiting_times": step_mean_waiting_times,
        "step_mean_travel_times": step_mean_travel_times,
        "step_running": step_running,
        "step_halting": step_halting,
        "total_inserted": total_inserted,
        "total_ended": total_ended,
    }


def parse_tripinfo(tripinfo_path: Path) -> dict[str, list[float]]:
    """Walk the ``<tripinfo>`` records in a SUMO ``tripinfo.xml`` file."""
    tree = ET.parse(tripinfo_path)
    root = tree.getroot()

    durations: list[float] = []
    route_lengths: list[float] = []
    waiting_times: list[float] = []
    time_losses: list[float] = []
    depart_delays: list[float] = []
    waiting_counts: list[float] = []
    stop_counts: list[float] = []

    for trip in root.findall("tripinfo"):
        attrs = trip.attrib
        if "duration" in attrs:
            durations.append(float(attrs["duration"]))
        if "routeLength" in attrs:
            route_lengths.append(float(attrs["routeLength"]))
        if "waitingTime" in attrs:
            waiting_times.append(float(attrs["waitingTime"]))
        if "timeLoss" in attrs:
            time_losses.append(float(attrs["timeLoss"]))
        if "departDelay" in attrs:
            depart_delays.append(float(attrs["departDelay"]))
        if "waitingCount" in attrs:
            waiting_counts.append(float(attrs["waitingCount"]))
        if "stopTime" in attrs:
            stop_counts.append(float(attrs["stopTime"]))

    return {
        "durations": durations,
        "route_lengths": route_lengths,
        "waiting_times": waiting_times,
        "time_losses": time_losses,
        "depart_delays": depart_delays,
        "waiting_counts": waiting_counts,
        "stop_counts": stop_counts,
    }


def parse_statistics(statistics_path: Path) -> dict[str, str | None]:
    """Parse the optional ``statistics.xml`` SUMO run report.

    Returns an empty dict when the file is absent — the run-level report is
    not always emitted, so callers should treat missing data as benign.
    """
    if not statistics_path.exists():
        return {}

    tree = ET.parse(statistics_path)
    root = tree.getroot()

    out: dict[str, str | None] = {}
    vehicles = root.find("vehicles")
    if vehicles is not None:
        out["vehicles_loaded"] = vehicles.attrib.get("loaded")
        out["vehicles_inserted"] = vehicles.attrib.get("inserted")
        out["vehicles_running"] = vehicles.attrib.get("running")
        out["vehicles_waiting"] = vehicles.attrib.get("waiting")

    teleports = root.find("teleports")
    if teleports is not None:
        out["teleports_total"] = teleports.attrib.get("total")

    safety = root.find("safety")
    if safety is not None:
        out["collisions"] = safety.attrib.get("collisions")
        out["emergency_stops"] = safety.attrib.get("emergencyStops")

    return out


# ---------------------------------------------------------------------------
# Top-level orchestrator.
# ---------------------------------------------------------------------------


def compute_baseline_metrics(
    summary_path: Path,
    tripinfo_path: Path,
    statistics_path: Path | None = None,
) -> BaselineMetrics:
    """Parse a SUMO run bundle and assemble the headline ``BaselineMetrics``.

    Parameters
    ----------
    summary_path
        Path to ``summary.xml`` (required).
    tripinfo_path
        Path to ``tripinfo.xml`` (required — add ``tripinfo-output`` to your
        SUMO config if it is missing).
    statistics_path
        Optional path to ``statistics.xml``. If absent or ``None``, run-level
        warnings (teleports, collisions, emergency stops) will not be added
        to ``notes``.
    """
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing {summary_path}")
    if not tripinfo_path.exists():
        raise FileNotFoundError(
            f"Missing {tripinfo_path}. Add tripinfo-output to your SUMO config first."
        )

    summary = parse_summary(summary_path)
    tripinfo = parse_tripinfo(tripinfo_path)
    stats = parse_statistics(statistics_path) if statistics_path is not None else {}

    sim_duration = summary["simulation_duration_s"]
    total_inserted: int = summary["total_inserted"]
    total_ended: int = summary["total_ended"]

    completion_ratio: float | None = None
    if total_inserted > 0:
        completion_ratio = total_ended / total_inserted

    throughput_veh_per_hour: float | None = None
    if sim_duration and sim_duration > 0:
        throughput_veh_per_hour = total_ended * 3600.0 / sim_duration

    metrics = BaselineMetrics(
        simulation_duration_s=sim_duration,
        avg_step_mean_speed_mps=safe_mean(summary["step_mean_speeds"]),
        max_step_mean_speed_mps=(
            max(summary["step_mean_speeds"]) if summary["step_mean_speeds"] else None
        ),
        min_step_mean_speed_mps=(
            min(summary["step_mean_speeds"]) if summary["step_mean_speeds"] else None
        ),
        avg_step_mean_waiting_time_s=safe_mean(summary["step_mean_waiting_times"]),
        max_step_mean_waiting_time_s=(
            max(summary["step_mean_waiting_times"]) if summary["step_mean_waiting_times"] else None
        ),
        avg_step_mean_travel_time_s=safe_mean(summary["step_mean_travel_times"]),
        max_step_mean_travel_time_s=(
            max(summary["step_mean_travel_times"]) if summary["step_mean_travel_times"] else None
        ),
        avg_step_running_vehicles=safe_mean(summary["step_running"]),
        max_step_running_vehicles=(
            max(summary["step_running"]) if summary["step_running"] else None
        ),
        total_inserted_vehicles=total_inserted,
        total_ended_vehicles=total_ended,
        completion_ratio=completion_ratio,
        avg_trip_duration_s=safe_mean(tripinfo["durations"]),
        median_trip_duration_s=safe_median(tripinfo["durations"]),
        p95_trip_duration_s=percentile(tripinfo["durations"], 0.95),
        avg_trip_route_length_m=safe_mean(tripinfo["route_lengths"]),
        avg_trip_waiting_time_s=safe_mean(tripinfo["waiting_times"]),
        median_trip_waiting_time_s=safe_median(tripinfo["waiting_times"]),
        p95_trip_waiting_time_s=percentile(tripinfo["waiting_times"], 0.95),
        avg_trip_time_loss_s=safe_mean(tripinfo["time_losses"]),
        median_trip_time_loss_s=safe_median(tripinfo["time_losses"]),
        p95_trip_time_loss_s=percentile(tripinfo["time_losses"], 0.95),
        avg_depart_delay_s=safe_mean(tripinfo["depart_delays"]),
        p95_depart_delay_s=percentile(tripinfo["depart_delays"], 0.95),
        avg_waiting_count_per_trip=safe_mean(tripinfo["waiting_counts"]),
        avg_stop_count_per_trip=safe_mean(tripinfo["stop_counts"]),
        throughput_veh_per_hour=throughput_veh_per_hour,
        spillback_proxy_peak_halting=(
            int(max(summary["step_halting"])) if summary["step_halting"] else None
        ),
        spillback_proxy_avg_halting=safe_mean(summary["step_halting"]),
        notes=[],
    )

    if stats.get("teleports_total") not in (None, "0"):
        metrics.notes.append(f"Teleports detected: {stats['teleports_total']}")
    if stats.get("collisions") not in (None, "0"):
        metrics.notes.append(f"Collisions detected: {stats['collisions']}")
    if stats.get("emergency_stops") not in (None, "0"):
        metrics.notes.append(f"Emergency stops detected: {stats['emergency_stops']}")

    return metrics


__all__ = [
    "BaselineMetrics",
    "compute_baseline_metrics",
    "parse_statistics",
    "parse_summary",
    "parse_tripinfo",
    "percentile",
    "safe_mean",
    "safe_median",
]
