"""Tests for fixed-time and max-pressure controllers."""

from __future__ import annotations

import numpy as np
import pytest

from madrid_twin.control.baselines import (
    FixedTimeController,
    IntersectionState,
    MaxPressureController,
)


def _state(intersection_id: str, n_phases: int, incoming, outgoing) -> IntersectionState:
    return IntersectionState(
        intersection_id=intersection_id,
        n_phases=n_phases,
        incoming_queues=np.array(incoming, dtype=float),
        outgoing_queues=np.array(outgoing, dtype=float),
    )


class TestIntersectionStateValidation:
    def test_non_positive_n_phases_raises(self) -> None:
        with pytest.raises(ValueError):
            IntersectionState("x", 0, np.zeros(0), np.zeros(0))

    def test_wrong_incoming_shape_raises(self) -> None:
        with pytest.raises(ValueError):
            IntersectionState("x", 3, np.zeros(2), np.zeros(3))


class TestFixedTimeController:
    def test_round_robin(self) -> None:
        ctrl = FixedTimeController(phase_durations={"int_0": (2, 2)})
        s = _state("int_0", 2, [0.0, 0.0], [0.0, 0.0])
        # t=0, 1 -> phase 0; t=2, 3 -> phase 1; t=4 wraps to phase 0.
        assert ctrl.act([s])["int_0"] == 0
        assert ctrl.act([s])["int_0"] == 0
        assert ctrl.act([s])["int_0"] == 1
        assert ctrl.act([s])["int_0"] == 1
        assert ctrl.act([s])["int_0"] == 0

    def test_unknown_intersection_returns_phase_zero(self) -> None:
        ctrl = FixedTimeController(phase_durations={})
        s = _state("unknown", 4, [1, 2, 3, 4], [0, 0, 0, 0])
        assert ctrl.act([s])["unknown"] == 0

    def test_reset_restarts_cycle(self) -> None:
        ctrl = FixedTimeController(phase_durations={"int_0": (1, 1)})
        s = _state("int_0", 2, [0.0, 0.0], [0.0, 0.0])
        ctrl.act([s])
        ctrl.act([s])
        ctrl.reset()
        assert ctrl.act([s])["int_0"] == 0


class TestMaxPressureController:
    def test_chooses_phase_with_highest_pressure(self) -> None:
        ctrl = MaxPressureController()
        s = _state("int_0", 4, [10, 50, 5, 20], [5, 10, 0, 5])
        # Pressures: 5, 40, 5, 15 -> argmax = 1.
        assert ctrl.act([s])["int_0"] == 1

    def test_all_zero_pressure_holds_phase_zero(self) -> None:
        ctrl = MaxPressureController()
        s = _state("int_0", 3, [0, 0, 0], [0, 0, 0])
        assert ctrl.act([s])["int_0"] == 0

    def test_negative_differential_clipped(self) -> None:
        # outgoing > incoming should never produce negative pressure that
        # beats a zero pressure elsewhere.
        ctrl = MaxPressureController()
        s = _state("int_0", 2, [5, 0], [100, 0])
        # Pressures: max(0, 5-100)=0, max(0, 0-0)=0 -> argmax ties at 0.
        assert ctrl.act([s])["int_0"] == 0
