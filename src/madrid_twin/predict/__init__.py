"""Predictive layer — short-horizon spatiotemporal forecasting.

Responsibilities
----------------

Given a stream of network state (detector counts, queues, link travel
times) plus the workzone descriptor, forecast queue length, spillback
probability, and link travel time at 5/15/30-minute horizons. Provide
distribution-free uncertainty intervals via split conformal prediction.

Contract
--------

A predictor is a Python protocol with two methods, ``fit(train_data)``
and ``predict(state) -> ForecastBundle``. ``ForecastBundle`` carries the
point forecast plus calibrated lower/upper conformal bounds. Predictors
are persisted as artefacts and tracked in MLflow.

Submodules (planned)
--------------------

- ``baselines``  DCRNN, GraphWaveNet, AGCRN, STAEformer wrappers.
- ``pignn``      Physics-informed GNN with LWR conservation residual.
- ``conformal``  Split-conformal prediction calibration utilities.
- ``features``   Graph construction from the SUMO network + detector map.
"""

from __future__ import annotations

__all__: list[str] = []
