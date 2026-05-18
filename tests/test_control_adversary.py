"""Tests for the RARL-style adversary."""

from __future__ import annotations

import numpy as np
import pytest

from madrid_twin.control.adversary import (
    AdversaryConfig,
    apply_perturbation,
    project_perturbation,
    random_adversary_action,
)


class TestAdversaryConfig:
    def test_out_of_range_max_rel_raises(self) -> None:
        with pytest.raises(ValueError):
            AdversaryConfig(max_relative_perturbation=1.5)

    def test_negative_total_budget_raises(self) -> None:
        with pytest.raises(ValueError):
            AdversaryConfig(max_total_relative_budget=-0.1)


class TestProjectPerturbation:
    def test_within_bounds_passes_through(self) -> None:
        baseline = np.array([100.0, 100.0])
        cfg = AdversaryConfig(max_relative_perturbation=0.2, max_total_relative_budget=None)
        # Delta is within ±20 per element; no global cap.
        raw = np.array([10.0, -15.0])
        out = project_perturbation(raw, baseline, cfg)
        np.testing.assert_allclose(out, [10.0, -15.0])

    def test_per_element_clip(self) -> None:
        baseline = np.array([100.0, 100.0])
        cfg = AdversaryConfig(max_relative_perturbation=0.1, max_total_relative_budget=None)
        raw = np.array([50.0, -50.0])
        out = project_perturbation(raw, baseline, cfg)
        # Clipped to ±10 each.
        np.testing.assert_allclose(out, [10.0, -10.0])

    def test_global_l1_rescale(self) -> None:
        baseline = np.array([100.0, 100.0])
        cfg = AdversaryConfig(
            max_relative_perturbation=1.0,  # disables per-element clip
            max_total_relative_budget=0.1,  # total cap = 0.1 * 200 = 20
        )
        raw = np.array([30.0, -30.0])  # |L1| = 60 -> must scale to 20
        out = project_perturbation(raw, baseline, cfg)
        assert float(np.abs(out).sum()) == pytest.approx(20.0)
        np.testing.assert_allclose(out, [10.0, -10.0])

    def test_zero_total_budget_zeroes_perturbation(self) -> None:
        baseline = np.array([100.0])
        cfg = AdversaryConfig(max_relative_perturbation=1.0, max_total_relative_budget=0.0)
        out = project_perturbation(np.array([5.0]), baseline, cfg)
        np.testing.assert_allclose(out, [0.0])

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            project_perturbation(np.zeros(3), np.zeros(2), AdversaryConfig())


class TestApplyPerturbation:
    def test_floors_at_zero(self) -> None:
        baseline = np.array([10.0, 5.0])
        perturbation = np.array([-50.0, 0.0])
        out = apply_perturbation(baseline, perturbation)
        np.testing.assert_array_equal(out, [0.0, 5.0])


class TestRandomAdversary:
    def test_respects_bounds(self) -> None:
        rng = np.random.default_rng(0)
        baseline = np.full(8, 1000.0)
        cfg = AdversaryConfig(max_relative_perturbation=0.1, max_total_relative_budget=0.2)
        for _ in range(50):
            delta = random_adversary_action(baseline, cfg, rng)
            assert np.all(np.abs(delta) <= 0.1 * baseline + 1e-9)
            assert float(np.abs(delta).sum()) <= 0.2 * baseline.sum() + 1e-9

    def test_reproducible_with_seed(self) -> None:
        baseline = np.full(4, 100.0)
        cfg = AdversaryConfig()
        a = random_adversary_action(baseline, cfg, np.random.default_rng(3))
        b = random_adversary_action(baseline, cfg, np.random.default_rng(3))
        np.testing.assert_allclose(a, b)
