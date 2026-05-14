"""LWR conservation residual.

The Lighthill–Whitham–Richards (LWR) model — the foundational
hyperbolic conservation law of macroscopic traffic flow — says that on
a road segment, mass is conserved:

    dρ/dt + dq/dx = 0

In discrete time over a directed graph the equivalent statement is
that, for each node (road segment), the change in density between two
timesteps equals the net flow (inflow minus outflow) times the
timestep, divided by segment length:

    ρ(t+Δt) - ρ(t)  ≈  Δt / L  *  (Σ inflows - Σ outflows)

A predictor that respects this constraint produces internally
consistent forecasts. We turn this into a *soft* regulariser: the
absolute violation of the conservation equation, averaged across nodes
and timesteps. Training a GNN with this term added to the MSE loss
nudges the model toward physically plausible forecasts without
hard-constraining it (the LWR model is itself an approximation, so a
strict equality would be too aggressive).

This module is pure numpy. The Phase 3 torch implementation will reuse
the same algebra inside a differentiable loss; the numpy version here
is the reference implementation against which the torch loss is
unit-tested.
"""

from __future__ import annotations

import numpy as np


def lwr_residual(
    density: np.ndarray,
    inflow: np.ndarray,
    outflow: np.ndarray,
    *,
    segment_length_m: np.ndarray | float,
    dt_s: float,
) -> np.ndarray:
    """Per-node, per-timestep LWR conservation residual.

    Parameters
    ----------
    density
        Shape ``(T+1, N)``. Vehicle density (veh/m) on each node at
        each timestep. The "+1" carries the post-prediction state.
    inflow, outflow
        Shape ``(T, N)``. Flows (veh/s) into and out of each node
        during each timestep.
    segment_length_m
        Either a scalar (uniform segments) or shape ``(N,)`` array of
        per-node segment lengths in metres. Must be strictly positive.
    dt_s
        Timestep size in seconds. Must be strictly positive.

    Returns
    -------
    Shape ``(T, N)`` array of conservation residuals
    ``rho_next - rho_prev - dt/L * (inflow - outflow)``. A perfectly
    conservation-respecting predictor returns zeros.
    """
    if dt_s <= 0:
        raise ValueError(f"dt_s must be positive, got {dt_s}")
    density = np.asarray(density, dtype=np.float64)
    inflow = np.asarray(inflow, dtype=np.float64)
    outflow = np.asarray(outflow, dtype=np.float64)
    if density.ndim != 2:
        raise ValueError(f"density must be 2-D (T+1, N), got shape {density.shape}")
    if inflow.shape != outflow.shape:
        raise ValueError(f"inflow {inflow.shape} and outflow {outflow.shape} must share a shape")
    if density.shape[0] != inflow.shape[0] + 1:
        raise ValueError(
            f"density has {density.shape[0]} timesteps; expected {inflow.shape[0] + 1}"
        )
    if density.shape[1] != inflow.shape[1]:
        raise ValueError(f"density has {density.shape[1]} nodes but flows have {inflow.shape[1]}")

    seg_len = np.asarray(segment_length_m, dtype=np.float64)
    if seg_len.ndim == 0:
        seg_len = np.full(density.shape[1], float(seg_len))
    if (seg_len <= 0).any():
        raise ValueError("segment_length_m must be strictly positive everywhere")

    delta_rho = density[1:, :] - density[:-1, :]
    expected = (dt_s / seg_len) * (inflow - outflow)
    return delta_rho - expected


def lwr_residual_loss(
    density: np.ndarray,
    inflow: np.ndarray,
    outflow: np.ndarray,
    *,
    segment_length_m: np.ndarray | float,
    dt_s: float,
    reduction: str = "mean_abs",
) -> float:
    """Scalar reduction of the residual — the form that goes into a loss.

    ``reduction`` is one of:
        - ``"mean_abs"`` (default): mean of absolute residuals.
        - ``"mean_sq"``: mean of squared residuals.
        - ``"max_abs"``: maximum absolute residual.
    """
    r = lwr_residual(
        density,
        inflow,
        outflow,
        segment_length_m=segment_length_m,
        dt_s=dt_s,
    )
    if reduction == "mean_abs":
        return float(np.abs(r).mean())
    if reduction == "mean_sq":
        return float((r**2).mean())
    if reduction == "max_abs":
        return float(np.abs(r).max())
    raise ValueError(f"unknown reduction: {reduction!r}")


__all__ = ["lwr_residual", "lwr_residual_loss"]
