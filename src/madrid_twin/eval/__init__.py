"""Evaluation layer — baselines, ablations, robustness, statistics."""

from __future__ import annotations

from madrid_twin.eval.baseline import (
    BaselineMetrics,
    compute_baseline_metrics,
    parse_statistics,
    parse_summary,
    parse_tripinfo,
)
from madrid_twin.eval.validation import (
    DEFAULT_GEH_PASS_RATE,
    DEFAULT_GEH_THRESHOLD,
    DetectorPair,
    GEHReport,
    geh,
    geh_batch,
    geh_pass_rate,
    travel_time_rmse,
)

__all__ = [
    "DEFAULT_GEH_PASS_RATE",
    "DEFAULT_GEH_THRESHOLD",
    "BaselineMetrics",
    "DetectorPair",
    "GEHReport",
    "compute_baseline_metrics",
    "geh",
    "geh_batch",
    "geh_pass_rate",
    "parse_statistics",
    "parse_summary",
    "parse_tripinfo",
    "travel_time_rmse",
]
