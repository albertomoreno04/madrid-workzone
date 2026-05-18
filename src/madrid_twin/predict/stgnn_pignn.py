"""PI-GraphWaveNet — physics-informed GraphWaveNet for traffic forecasting.

The MADTwin methodological contribution at the predictive layer.
Builds on GraphWaveNet (Wu et al., 2019) — dilated temporal
convolutions + diffusion graph convolution — and adds an LWR
conservation-law residual to the training loss. The result is a
spatiotemporal forecaster that softly enforces mass balance on the
road graph: vehicles produced upstream must reappear downstream,
modulo segment density change.

torch is lazy-imported so this module is inspectable without the
``[predict]`` extras. The model classes themselves are nn.Modules
defined inside ``build_pignn_model`` to keep the heavy import inside
the function body.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from madrid_twin.predict.physics import lwr_residual_loss


@dataclass(frozen=True)
class PIGraphWaveNetConfig:
    """Hyperparameters for PI-GraphWaveNet.

    Defaults match the citation-standard GraphWaveNet config on
    METR-LA / PEMS-BAY, augmented with the physics-loss weight.
    """

    n_nodes: int
    input_steps: int = 12
    output_steps: int = 12
    hidden_channels: int = 32
    n_blocks: int = 4
    dilation_factor: int = 2
    dropout: float = 0.3

    # Physics term: weight on the LWR conservation residual loss.
    lwr_weight: float = 0.1

    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    max_epochs: int = 100
    seed: int = 0

    def __post_init__(self) -> None:
        if self.n_nodes <= 0:
            raise ValueError("n_nodes must be positive")
        if self.input_steps <= 0 or self.output_steps <= 0:
            raise ValueError("input_steps and output_steps must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.lwr_weight < 0:
            raise ValueError("lwr_weight must be non-negative")


def build_pignn_model(
    config: PIGraphWaveNetConfig,
    adjacency: np.ndarray,
) -> Any:
    """Construct a PI-GraphWaveNet model. Lazy-imports torch.

    Returns a torch ``nn.Module`` ready to be trained on (input, target)
    tensors of shape ``(B, input_steps, N)`` / ``(B, output_steps, N)``.
    The module exposes a ``physics_loss(predictions, inflow, outflow,
    segment_length_m, dt_s)`` method that returns the LWR-weighted loss
    term to combine with the task loss.
    """
    try:
        import torch  # noqa: PLC0415
        from torch import nn  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "PI-GraphWaveNet requires the [predict] extras. Install with:\n"
            '    pip install -e ".[predict]"'
        ) from exc

    if adjacency.shape != (config.n_nodes, config.n_nodes):
        raise ValueError(
            f"adjacency shape {adjacency.shape} must be ({config.n_nodes}, {config.n_nodes})"
        )

    # Build a fixed adjacency tensor for the diffusion convolution.
    A_tilde = adjacency + np.eye(config.n_nodes)
    d = A_tilde.sum(axis=1)
    d_inv = np.where(d > 1e-9, 1.0 / d, 0.0)
    A_norm = (A_tilde.T * d_inv).T  # row-normalised
    A_tensor = torch.from_numpy(A_norm.astype(np.float32))

    class DiffusionGraphConv(nn.Module):
        """One step of diffusion graph convolution (Li et al., 2018).

        Approximates the heat kernel on the graph by ``K`` powers of
        the row-normalised adjacency.
        """

        def __init__(self, in_ch: int, out_ch: int, K: int = 2) -> None:
            super().__init__()
            self.K = K
            self.linear = nn.Linear(in_ch * (K + 1), out_ch)
            self.register_buffer("A", A_tensor)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (B, T, N, C) -> apply along N.
            B, T, N, C = x.shape
            terms = [x]
            for _ in range(self.K):
                terms.append(torch.einsum("ij,btjc->btic", self.A, terms[-1]))
            stacked = torch.cat(terms, dim=-1)
            return self.linear(stacked)

    class TemporalConv(nn.Module):
        """Dilated causal 1-D convolution along the time dimension."""

        def __init__(self, ch: int, dilation: int) -> None:
            super().__init__()
            kernel = 3
            pad = (kernel - 1) * dilation
            self.conv_filter = nn.Conv2d(
                ch, ch, (kernel, 1), padding=(pad, 0), dilation=(dilation, 1)
            )
            self.conv_gate = nn.Conv2d(
                ch, ch, (kernel, 1), padding=(pad, 0), dilation=(dilation, 1)
            )
            self.pad = pad

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (B, C, T, N)
            f = torch.tanh(self.conv_filter(x))
            g = torch.sigmoid(self.conv_gate(x))
            out = f * g
            # Trim the right pad to keep causality.
            return out[..., : -self.pad if self.pad > 0 else None, :]

    class PIGraphWaveNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_proj = nn.Linear(1, config.hidden_channels)
            self.temporal_blocks = nn.ModuleList(
                [
                    TemporalConv(config.hidden_channels, dilation=config.dilation_factor**i)
                    for i in range(config.n_blocks)
                ]
            )
            self.graph_blocks = nn.ModuleList(
                [
                    DiffusionGraphConv(config.hidden_channels, config.hidden_channels)
                    for _ in range(config.n_blocks)
                ]
            )
            self.dropout = nn.Dropout(config.dropout)
            self.output_head = nn.Linear(config.hidden_channels, config.output_steps)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (B, T_in, N) -> add channel dim -> (B, T_in, N, 1)
            x = x.unsqueeze(-1)
            x = self.input_proj(x)  # (B, T_in, N, C)
            for tconv, gconv in zip(self.temporal_blocks, self.graph_blocks, strict=True):
                x_t = tconv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
                x_g = gconv(x)
                x = self.dropout(x_t + x_g)
            # Reduce across time -> (B, N, C)
            x = x.mean(dim=1)
            # Project to horizon -> (B, N, output_steps) -> (B, output_steps, N)
            out = self.output_head(x).permute(0, 2, 1)
            return out

        def physics_loss(
            self,
            density: np.ndarray,
            inflow: np.ndarray,
            outflow: np.ndarray,
            *,
            segment_length_m: np.ndarray | float,
            dt_s: float,
        ) -> float:
            """LWR conservation residual — same formula as the numpy reference."""
            return lwr_residual_loss(
                density,
                inflow,
                outflow,
                segment_length_m=segment_length_m,
                dt_s=dt_s,
            )

    return PIGraphWaveNet()


__all__ = [
    "PIGraphWaveNetConfig",
    "build_pignn_model",
]
