"""MADTwin — Madrid workzone digital twin.

Top-level package for the MADTwin research codebase. The project is organized
into four subpackages that map one-to-one to the layers of the architecture:

- ``madrid_twin.data``    Open-data ingestion, OD seeds, workzone descriptors,
                          census/zone shapes. DVC-tracked outputs.
- ``madrid_twin.sim``     SUMO-based digital twin: network build, workzone
                          parametric module, demand calibration (W-SPSA),
                          TraCI/libsumo bridge, scenario runner.
- ``madrid_twin.predict`` Spatiotemporal predictive layer: ST-GNN baselines,
                          physics-informed variant, conformal uncertainty.
- ``madrid_twin.control`` Heterogeneous multi-agent RL controller: signal,
                          lane, and VMS agents trained under domain
                          randomization and adversarial demand perturbation.
- ``madrid_twin.eval``    Evaluation harness: baseline metrics, ablations,
                          robustness stress tests, statistical analysis.

See PROJECT_PLAN.md at the repo root for the full research plan.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
