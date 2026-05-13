"""Evaluation layer — baselines, ablations, robustness, statistics.

Responsibilities
----------------

Compute headline metrics from SUMO outputs (delay, throughput, p95
travel time, spillback proxy, emissions, equity Gini), run the
ablation programme, drive robustness stress tests (demand shift,
capacity-drop misestimation, sensor dropout), and produce
publication-ready statistics with paired bootstrap CIs, Wilcoxon
signed-rank tests, and Sobol total-order sensitivity indices.

Submodules
----------

- ``baseline``   Parse a SUMO run bundle into a structured metrics object
                 (refactored from the original ``scripts/analyze_baseline.py``).
- (planned) ``equity``, ``robustness``, ``stats``, ``sensitivity``.
"""

from __future__ import annotations

from madrid_twin.eval.baseline import (
    BaselineMetrics,
    compute_baseline_metrics,
    parse_statistics,
    parse_summary,
    parse_tripinfo,
)

__all__ = [
    "BaselineMetrics",
    "compute_baseline_metrics",
    "parse_statistics",
    "parse_summary",
    "parse_tripinfo",
]
