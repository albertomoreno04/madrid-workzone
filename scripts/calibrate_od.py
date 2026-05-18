"""W-SPSA OD calibration demo on synthetic data.

Demonstrates the Phase 4 calibration loop in isolation from SUMO. The
"observed" detector counts are produced by feeding a known OD into a
linear detector model; the calibrator's job is to recover the OD from
the counts alone, starting from a noisy initial estimate.

When this works on synthetic data with known ground truth, it gives us
confidence that the algorithm itself is correct — the only remaining
unknown for the real SUMO version is whether the simulator's
count-matching surface is well-behaved enough for W-SPSA to descend.
"""

from __future__ import annotations

import json

import numpy as np

from madrid_twin.config import DATA_OUTPUTS
from madrid_twin.sim.calibrate import (
    WSPSAConfig,
    count_matching_loss,
    wspsa,
)


def main() -> None:
    rng = np.random.default_rng(0)
    n_od = 16
    n_detectors = 8

    # Synthetic mapping: counts = A @ od_true (linear detector model).
    A = rng.uniform(0.0, 1.0, size=(n_detectors, n_od))
    od_true = rng.uniform(50.0, 500.0, size=n_od)
    observed = A @ od_true

    def loss(theta: np.ndarray) -> float:
        simulated = A @ theta
        return count_matching_loss(simulated, observed)

    # Initial estimate: noisy multiplier of the truth.
    od_init = od_true * rng.uniform(0.5, 1.5, size=n_od)

    cfg = WSPSAConfig(a=0.5, c=0.05, max_iter=500, seed=42)
    od_star, report = wspsa(od_init, loss, cfg)

    err_init = float(np.linalg.norm(od_init - od_true))
    err_final = float(np.linalg.norm(od_star - od_true))
    relative_improvement = (err_init - err_final) / err_init

    report_dict = {
        "n_od_entries": n_od,
        "n_detectors": n_detectors,
        "initial_loss": report.initial_loss,
        "final_loss": report.final_loss,
        "initial_od_error_l2": err_init,
        "final_od_error_l2": err_final,
        "relative_l2_improvement": relative_improvement,
        "n_iter": report.n_iter,
    }

    out_dir = DATA_OUTPUTS / "sim"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "calibrate_od_report.json"
    out_path.write_text(json.dumps(report_dict, indent=2), encoding="utf-8")
    print(json.dumps(report_dict, indent=2))
    print(f"\nWritten: {out_path}")


if __name__ == "__main__":
    main()
