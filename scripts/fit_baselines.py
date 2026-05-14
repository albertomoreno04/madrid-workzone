"""Fit the classical Phase 2 forecaster baselines on synthetic data.

This is a placeholder entry point that demonstrates the predictive-layer
API end to end on a deterministic synthetic series — useful as a smoke
test and as a reference call site for the real Phase 2 calibration loop
that will land once Phase 1 emits trajectories.

Output: ``data/outputs/predict/baselines_report.json`` — per-model MAE
on a held-out validation split, plus the conformal interval coverage.
"""

from __future__ import annotations

import json

import numpy as np

from madrid_twin.config import DATA_OUTPUTS
from madrid_twin.predict.baselines import (
    AR1Forecaster,
    HistoricalAverage,
    NaiveLastValue,
)
from madrid_twin.predict.conformal import SplitConformalPredictor


def _synthetic_series(n_steps: int = 1500, n_nodes: int = 4, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = np.sin(np.linspace(0.0, 6.0 * np.pi, n_steps))[:, None]
    nodes = rng.normal(0.0, 1.0, size=(n_steps, n_nodes))
    return base + 0.3 * nodes


def _mae(true: np.ndarray, pred: np.ndarray) -> float:
    return float(np.abs(true - pred).mean())


def main() -> None:
    series = _synthetic_series()
    T = series.shape[0]
    train, cal, test = series[: T // 2], series[T // 2 : 3 * T // 4], series[3 * T // 4 :]

    ctx_len = 12
    horizons = [1]

    report: dict[str, object] = {"models": {}}

    for name, model in (
        ("HistoricalAverage", HistoricalAverage()),
        ("NaiveLastValue", NaiveLastValue()),
        ("AR1", AR1Forecaster()),
    ):
        model.fit(train)
        # Build calibration pairs from the cal split.
        cal_ctx = [cal[i : i + ctx_len] for i in range(len(cal) - ctx_len - 1)]
        cal_tgt = [cal[i + ctx_len : i + ctx_len + 1] for i in range(len(cal) - ctx_len - 1)]
        cp = SplitConformalPredictor(base=model, alpha=0.1, per_horizon_per_node=False)
        cp.calibrate(contexts=cal_ctx, targets=cal_tgt, horizons_s=horizons)

        # Evaluate on the test split.
        errors: list[float] = []
        covered = 0
        n_eval = 0
        for i in range(len(test) - ctx_len - 1):
            ctx = test[i : i + ctx_len]
            true_next = test[i + ctx_len, :]
            out = cp.predict(ctx, horizons_s=horizons)
            errors.append(_mae(true_next, out.point[0]))
            lo = out.lower[0]
            hi = out.upper[0]
            if np.all((lo <= true_next) & (true_next <= hi)):
                covered += 1
            n_eval += 1

        report["models"][name] = {
            "mae": float(np.mean(errors)),
            "conformal_coverage_90pct": covered / max(n_eval, 1),
        }

    out_dir = DATA_OUTPUTS / "predict"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "baselines_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nWritten: {out_path}")


if __name__ == "__main__":
    main()
