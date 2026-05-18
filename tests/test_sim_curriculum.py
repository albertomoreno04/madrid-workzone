"""Tests for the curriculum scheduler."""

from __future__ import annotations

import pytest

from madrid_twin.sim.curriculum import (
    CurriculumSchedule,
    PiecewiseSchedule,
    derive_workzone_params,
)


class TestCurriculumSchedule:
    def test_start_and_end_values(self) -> None:
        s = CurriculumSchedule(start_severity=0.0, end_severity=1.0, total_steps=1000)
        assert s.severity_at(-1) == 0.0
        assert s.severity_at(0) == 0.0
        assert s.severity_at(10_000) == 1.0

    def test_linear_ramp_midway(self) -> None:
        s = CurriculumSchedule(
            start_severity=0.0,
            end_severity=1.0,
            ramp_end_fraction=0.5,
            total_steps=1000,
        )
        # Ramp ends at step 500; midpoint (step 250) -> severity 0.5.
        assert s.severity_at(250) == pytest.approx(0.5, abs=1e-6)

    def test_plateau_after_ramp(self) -> None:
        s = CurriculumSchedule(
            start_severity=0.1,
            end_severity=0.9,
            ramp_end_fraction=0.3,
            total_steps=1000,
        )
        # After step 300 we're on the plateau.
        assert s.severity_at(500) == pytest.approx(0.9)
        assert s.severity_at(999) == pytest.approx(0.9)

    @pytest.mark.parametrize("bad_severity", [-0.1, 1.1])
    def test_invalid_severity_raises(self, bad_severity: float) -> None:
        with pytest.raises(ValueError):
            CurriculumSchedule(start_severity=bad_severity)


class TestPiecewiseSchedule:
    def test_default_stage_lookup(self) -> None:
        s = PiecewiseSchedule()
        assert s.severity_at(0) == 0.0
        assert s.severity_at(200_000) == 0.3
        assert s.severity_at(499_999) == 0.3
        assert s.severity_at(500_000) == 0.7
        assert s.severity_at(800_000) == 1.0
        assert s.severity_at(10_000_000) == 1.0

    def test_unsorted_raises(self) -> None:
        with pytest.raises(ValueError):
            PiecewiseSchedule(stages=((1000, 0.5), (500, 0.1)))

    def test_invalid_severity_raises(self) -> None:
        with pytest.raises(ValueError):
            PiecewiseSchedule(stages=((0, 1.5),))


class TestDeriveWorkzoneParams:
    def test_endpoints(self) -> None:
        cap_lo, dem_lo = derive_workzone_params(0.0)
        cap_hi, dem_hi = derive_workzone_params(1.0)
        assert cap_lo == pytest.approx(0.20)
        assert dem_lo == pytest.approx(1.0)
        assert cap_hi == pytest.approx(0.45)
        assert dem_hi == pytest.approx(1.3)

    def test_midpoint(self) -> None:
        cap, dem = derive_workzone_params(0.5)
        assert cap == pytest.approx(0.325)
        assert dem == pytest.approx(1.15)

    def test_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError):
            derive_workzone_params(1.5)
