"""Predictive layer — short-horizon spatiotemporal forecasting."""

from __future__ import annotations

from madrid_twin.predict.baselines import (
    AR1Forecaster,
    ForecastBundle,
    Forecaster,
    HistoricalAverage,
    NaiveLastValue,
)
from madrid_twin.predict.conformal import SplitConformalPredictor
from madrid_twin.predict.features import (
    AdjacencyKind,
    RoadEdge,
    RoadGraph,
    adjacency_matrix,
    detector_to_node_indices,
    parse_sumo_network,
)
from madrid_twin.predict.physics import lwr_residual, lwr_residual_loss
from madrid_twin.predict.stgnn import (
    STGNNConfig,
    STGNNModel,
    STGNNNotInstalledError,
    load_stgnn,
)

__all__ = [
    "AR1Forecaster",
    "AdjacencyKind",
    "ForecastBundle",
    "Forecaster",
    "HistoricalAverage",
    "NaiveLastValue",
    "RoadEdge",
    "RoadGraph",
    "STGNNConfig",
    "STGNNModel",
    "STGNNNotInstalledError",
    "SplitConformalPredictor",
    "adjacency_matrix",
    "detector_to_node_indices",
    "load_stgnn",
    "lwr_residual",
    "lwr_residual_loss",
    "parse_sumo_network",
]
