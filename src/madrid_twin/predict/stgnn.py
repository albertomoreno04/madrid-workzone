"""ST-GNN interface skeleton.

This module pins the API that the spatiotemporal graph neural network
forecasters in MADTwin must satisfy. Concrete model classes (DCRNN,
GraphWaveNet, AGCRN, STAEformer and our physics-informed variant) will
land as Phase 3 work and live as submodules here; this file documents
the contract and provides a lazy ``load_stgnn`` factory that can be
called from the experiment harness without forcing a torch import at
package load time.

The torch / torch-geometric dependencies are intentionally kept behind
the ``[predict]`` extras (see ``pyproject.toml``). Importing this
module is cheap and torch-free; calling :func:`load_stgnn` triggers
the heavy imports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

# Type aliases used in docstrings — defined here so static checkers see them
# without us importing torch at module load time.
TorchTensor = Any  # actually torch.Tensor when the model is loaded
NumpyArray = Any  # actually np.ndarray


STGNNModel = Literal[
    "dcrnn",
    "graphwavenet",
    "agcrn",
    "staeformer",
    "pi_graphwavenet",  # our physics-informed contribution
]


@dataclass(frozen=True)
class STGNNConfig:
    """Hyperparameters shared across ST-GNN variants.

    These follow the conventions of the citation-standard ST-GNN
    literature (DCRNN, GraphWaveNet, etc.) so the same config can be
    swapped between models in ablations.
    """

    input_steps: int = 12  # 1 hour of 5-minute observations
    output_steps: int = 12
    hidden_dim: int = 64
    n_layers: int = 2
    dropout: float = 0.1
    learning_rate: float = 1e-3
    batch_size: int = 64
    max_epochs: int = 100
    # Physics-informed loss weight. 0 disables the LWR term.
    lwr_weight: float = 0.0
    # Random seed for reproducibility.
    seed: int = 0


class STGNNNotInstalledError(RuntimeError):
    """Raised when an ST-GNN is requested but the [predict] extras are absent."""


def load_stgnn(
    model: STGNNModel,
    config: STGNNConfig,
    *,
    adjacency: NumpyArray,
) -> Any:
    """Construct an ST-GNN model instance, lazy-importing torch.

    This is the only entry point to torch-backed predictors. The
    function raises :class:`STGNNNotInstalledError` with an actionable
    message when torch / torch-geometric-temporal aren't installed.

    The actual model classes will be added as submodules in Phase 3
    (e.g. ``madrid_twin.predict.stgnn_dcrnn``). For now this raises
    :class:`NotImplementedError` to make the unimplemented edge
    explicit to callers; downstream code paths can therefore branch
    cleanly on availability.
    """
    try:
        import torch  # noqa: F401
    except ImportError as exc:  # pragma: no cover - guard, easy to verify by hand
        raise STGNNNotInstalledError(
            "ST-GNN models require the [predict] extras. Install with:\n"
            '    pip install -e ".[predict]"\n'
            "(this pulls torch, torch-geometric and torch-geometric-temporal.)"
        ) from exc

    raise NotImplementedError(
        f"ST-GNN model {model!r} is not yet implemented. "
        f"The interface is locked in (STGNNConfig + load_stgnn factory); "
        f"the model classes themselves land as Phase 3 work."
    )


__all__ = [
    "STGNNConfig",
    "STGNNModel",
    "STGNNNotInstalledError",
    "load_stgnn",
]
