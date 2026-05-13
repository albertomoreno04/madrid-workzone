"""Thin CLI wrapper around :mod:`madrid_twin.eval.baseline`.

Run from the repo root::

    python -m scripts.analyze_baseline

or via the ``baseline-metrics`` Makefile target. All real logic lives in
``madrid_twin.eval.baseline``; this script is kept for backwards
compatibility with the original repo layout.
"""

from __future__ import annotations

from madrid_twin.config import (
    BASELINE_METRICS_JSON,
    BASELINE_STATISTICS_XML,
    BASELINE_SUMMARY_XML,
    BASELINE_TRIPINFO_XML,
)
from madrid_twin.eval.baseline import compute_baseline_metrics


def main() -> None:
    metrics = compute_baseline_metrics(
        summary_path=BASELINE_SUMMARY_XML,
        tripinfo_path=BASELINE_TRIPINFO_XML,
        statistics_path=BASELINE_STATISTICS_XML,
    )

    BASELINE_METRICS_JSON.parent.mkdir(parents=True, exist_ok=True)
    BASELINE_METRICS_JSON.write_text(metrics.to_json(), encoding="utf-8")

    print("Baseline metrics written to:", BASELINE_METRICS_JSON)
    print(metrics.to_json())


if __name__ == "__main__":
    main()
