"""Tests for the refactored baseline-metrics module."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from madrid_twin.eval.baseline import (
    BaselineMetrics,
    compute_baseline_metrics,
    parse_statistics,
    parse_summary,
    parse_tripinfo,
    percentile,
    safe_mean,
    safe_median,
)

# ---------------------------------------------------------------------------
# Numeric helpers.
# ---------------------------------------------------------------------------


class TestPercentile:
    def test_empty_returns_none(self) -> None:
        assert percentile([], 0.5) is None

    def test_single_value(self) -> None:
        assert percentile([42.0], 0.95) == 42.0

    def test_known_quantiles(self) -> None:
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert percentile(data, 0.0) == 1.0
        assert percentile(data, 0.5) == 3.0
        assert percentile(data, 1.0) == 5.0

    def test_linear_interpolation_at_p95(self) -> None:
        # pos = (n-1) * q = 4 * 0.95 = 3.8 -> interpolate between idx 3 and 4
        # values[3]=4, values[4]=5, frac=0.8 -> 4*0.2 + 5*0.8 = 4.8
        assert percentile([1.0, 2.0, 3.0, 4.0, 5.0], 0.95) == pytest.approx(4.8)

    def test_unsorted_input_is_handled(self) -> None:
        assert percentile([5.0, 1.0, 4.0, 2.0, 3.0], 0.5) == 3.0

    @pytest.mark.parametrize("q", [-0.1, 1.1, 2.0])
    def test_invalid_q_raises(self, q: float) -> None:
        with pytest.raises(ValueError):
            percentile([1.0, 2.0], q)


class TestSafeMean:
    def test_empty(self) -> None:
        assert safe_mean([]) is None

    def test_simple(self) -> None:
        assert safe_mean([1.0, 2.0, 3.0]) == pytest.approx(2.0)


class TestSafeMedian:
    def test_empty(self) -> None:
        assert safe_median([]) is None

    def test_odd_length(self) -> None:
        assert safe_median([3.0, 1.0, 2.0]) == 2.0

    def test_even_length(self) -> None:
        assert safe_median([1.0, 2.0, 3.0, 4.0]) == 2.5


# ---------------------------------------------------------------------------
# Parsers.
# ---------------------------------------------------------------------------


class TestParseSummary:
    def test_parses_step_traces(self, sumo_outputs: dict[str, Path]) -> None:
        parsed = parse_summary(sumo_outputs["summary"])
        assert parsed["simulation_duration_s"] == 3.0
        assert parsed["total_inserted"] == 3
        assert parsed["total_ended"] == 3
        assert len(parsed["step_mean_speeds"]) == 4
        assert parsed["step_halting"] == [0.0, 0.0, 1.0, 0.0]


class TestParseTripinfo:
    def test_parses_all_trips(self, sumo_outputs: dict[str, Path]) -> None:
        parsed = parse_tripinfo(sumo_outputs["tripinfo"])
        assert parsed["durations"] == [100.0, 200.0, 300.0]
        assert parsed["route_lengths"] == [1500.0, 2500.0, 3500.0]
        assert parsed["waiting_times"] == [10.0, 40.0, 80.0]


class TestParseStatistics:
    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        assert parse_statistics(tmp_path / "does-not-exist.xml") == {}

    def test_parses_present_file(self, sumo_outputs: dict[str, Path]) -> None:
        parsed = parse_statistics(sumo_outputs["statistics"])
        assert parsed["teleports_total"] == "0"
        assert parsed["collisions"] == "0"


# ---------------------------------------------------------------------------
# End-to-end orchestrator.
# ---------------------------------------------------------------------------


class TestComputeBaselineMetrics:
    def test_smoke(self, sumo_outputs: dict[str, Path]) -> None:
        metrics = compute_baseline_metrics(
            summary_path=sumo_outputs["summary"],
            tripinfo_path=sumo_outputs["tripinfo"],
            statistics_path=sumo_outputs["statistics"],
        )
        assert isinstance(metrics, BaselineMetrics)
        # All 3 vehicles end -> completion ratio 1.0
        assert metrics.completion_ratio == pytest.approx(1.0)
        # Throughput = 3 vehicles ended in 3 s -> 3600 veh/h
        assert metrics.throughput_veh_per_hour == pytest.approx(3600.0)
        # Spillback proxy uses peak halting (= 1 in our fixture)
        assert metrics.spillback_proxy_peak_halting == 1
        # No teleports/collisions in the fixture -> notes stay empty
        assert metrics.notes == []

    def test_missing_summary_raises(self, sumo_outputs: dict[str, Path]) -> None:
        with pytest.raises(FileNotFoundError):
            compute_baseline_metrics(
                summary_path=sumo_outputs["summary"].parent / "nope.xml",
                tripinfo_path=sumo_outputs["tripinfo"],
            )

    def test_missing_tripinfo_raises(self, sumo_outputs: dict[str, Path]) -> None:
        with pytest.raises(FileNotFoundError):
            compute_baseline_metrics(
                summary_path=sumo_outputs["summary"],
                tripinfo_path=sumo_outputs["tripinfo"].parent / "nope.xml",
            )

    def test_to_json_roundtrip(self, sumo_outputs: dict[str, Path]) -> None:
        import json

        metrics = compute_baseline_metrics(
            summary_path=sumo_outputs["summary"],
            tripinfo_path=sumo_outputs["tripinfo"],
        )
        data = json.loads(metrics.to_json())
        assert data["total_ended_vehicles"] == 3
        assert math.isclose(data["completion_ratio"], 1.0)
