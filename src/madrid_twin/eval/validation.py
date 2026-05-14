"""Calibration validation — GEH statistic and travel-time RMSE.

The GEH statistic (Geoffrey E. Havers, 1970) is the de-facto industry test
for whether modelled traffic counts match observed counts. The formula

::

    GEH = sqrt( 2 * (m - c)^2 / (m + c) )

penalises absolute differences when both counts are low and relative
differences when both counts are high, which is the right behaviour for
hourly traffic flows. The Wisconsin DOT calibration manual — widely
adopted as the de-facto standard in microsimulation literature — calls a
detector "calibrated" when ``GEH < 5`` and requires at least 85% of
detectors to meet that threshold for the model as a whole.

This module exposes the primitive (``geh``), the batch version
(``geh_batch``), and a one-shot pass/fail report
(``geh_pass_rate`` / ``GEHReport``) that the rest of the validation
harness consumes.

A complementary travel-time RMSE helper is included because most
calibration reports pair GEH (flows) with RMSE (travel time).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import NamedTuple

# Wisconsin DOT / FHWA microsimulation calibration default thresholds.
DEFAULT_GEH_THRESHOLD: float = 5.0
DEFAULT_GEH_PASS_RATE: float = 0.85


class DetectorPair(NamedTuple):
    """One detector's modelled-vs-observed hourly flow."""

    detector_id: str
    modelled: float
    observed: float


@dataclass(frozen=True)
class GEHReport:
    """Result of running GEH validation across a set of detectors."""

    n_detectors: int
    n_pass: int
    pass_rate: float
    threshold: float
    target_pass_rate: float
    per_detector: tuple[tuple[str, float], ...] = field(default_factory=tuple)

    @property
    def calibrated(self) -> bool:
        """True when ``pass_rate`` meets the target threshold."""
        return self.pass_rate >= self.target_pass_rate

    def summary(self) -> str:
        verdict = "PASS" if self.calibrated else "FAIL"
        return (
            f"GEH<{self.threshold:g}: {self.n_pass}/{self.n_detectors} "
            f"({self.pass_rate:.1%}) — target {self.target_pass_rate:.0%} — {verdict}"
        )


# ---------------------------------------------------------------------------
# Primitives.
# ---------------------------------------------------------------------------


def geh(modelled: float, observed: float) -> float:
    """Compute the GEH statistic for a single (modelled, observed) pair.

    Both arguments must be non-negative hourly counts. The edge case
    ``modelled == observed == 0`` returns 0 by convention (the detector
    is "calibrated" — both sides agree there is no flow).
    """
    if modelled < 0 or observed < 0:
        raise ValueError(
            f"GEH expects non-negative counts; got modelled={modelled!r}, observed={observed!r}"
        )
    total = modelled + observed
    if total == 0:
        return 0.0
    diff_sq = (modelled - observed) ** 2
    return math.sqrt(2.0 * diff_sq / total)


def geh_batch(pairs: Iterable[tuple[float, float]]) -> list[float]:
    """Vectorless batch GEH (stdlib-only)."""
    return [geh(m, c) for m, c in pairs]


def geh_pass_rate(
    pairs: Sequence[DetectorPair] | Sequence[tuple[str, float, float]],
    *,
    threshold: float = DEFAULT_GEH_THRESHOLD,
    target_pass_rate: float = DEFAULT_GEH_PASS_RATE,
) -> GEHReport:
    """Run GEH validation across a set of detectors and return a report.

    ``pairs`` may be either a sequence of :class:`DetectorPair` (preferred)
    or plain 3-tuples ``(detector_id, modelled, observed)`` — the latter
    is accepted for ergonomic call sites that build the data inline.
    """
    if not pairs:
        raise ValueError("geh_pass_rate requires at least one detector pair")

    # Normalize to DetectorPair so the rest of the function is uniform.
    detectors: list[DetectorPair] = []
    for p in pairs:
        if isinstance(p, DetectorPair):
            detectors.append(p)
        else:
            det_id, modelled, observed = p
            detectors.append(DetectorPair(str(det_id), float(modelled), float(observed)))

    per_detector: list[tuple[str, float]] = []
    n_pass = 0
    for d in detectors:
        g = geh(d.modelled, d.observed)
        per_detector.append((d.detector_id, g))
        if g < threshold:
            n_pass += 1

    return GEHReport(
        n_detectors=len(detectors),
        n_pass=n_pass,
        pass_rate=n_pass / len(detectors),
        threshold=threshold,
        target_pass_rate=target_pass_rate,
        per_detector=tuple(per_detector),
    )


# ---------------------------------------------------------------------------
# Travel-time RMSE — companion metric to GEH.
# ---------------------------------------------------------------------------


def travel_time_rmse(modelled: Sequence[float], observed: Sequence[float]) -> float:
    """Root-mean-square error between modelled and observed travel times.

    Sequences must be the same length and correspond to the same OD pairs
    or corridor segments in the same order. Empty input raises.
    """
    if len(modelled) != len(observed):
        raise ValueError(f"travel_time_rmse: mismatched lengths {len(modelled)} vs {len(observed)}")
    if not modelled:
        raise ValueError("travel_time_rmse requires at least one sample")

    squared = sum((m - c) ** 2 for m, c in zip(modelled, observed, strict=True))
    return math.sqrt(squared / len(modelled))


__all__ = [
    "DEFAULT_GEH_PASS_RATE",
    "DEFAULT_GEH_THRESHOLD",
    "DetectorPair",
    "GEHReport",
    "geh",
    "geh_batch",
    "geh_pass_rate",
    "travel_time_rmse",
]
