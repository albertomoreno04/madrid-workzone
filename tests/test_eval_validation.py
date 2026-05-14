"""Tests for the GEH validator and travel-time RMSE."""

from __future__ import annotations

import math

import pytest

from madrid_twin.eval.validation import (
    DetectorPair,
    GEHReport,
    geh,
    geh_batch,
    geh_pass_rate,
    travel_time_rmse,
)

# ---------------------------------------------------------------------------
# geh()
# ---------------------------------------------------------------------------


class TestGEH:
    def test_perfect_match_is_zero(self) -> None:
        assert geh(500.0, 500.0) == 0.0

    def test_both_zero_is_zero(self) -> None:
        # Convention: a detector with zero modelled and zero observed flow is
        # trivially calibrated — both sides agree there is no flow.
        assert geh(0.0, 0.0) == 0.0

    def test_known_value(self) -> None:
        # Worked example: m=700, c=500 -> diff^2=40000, sum=1200,
        # GEH = sqrt(2*40000/1200) = sqrt(66.666...) ≈ 8.165.
        assert geh(700.0, 500.0) == pytest.approx(8.16496580927726)

    def test_symmetric(self) -> None:
        # GEH(m, c) == GEH(c, m) by construction.
        assert geh(300.0, 450.0) == pytest.approx(geh(450.0, 300.0))

    def test_only_one_zero(self) -> None:
        # m=100, c=0 -> GEH = sqrt(2 * 10000 / 100) = sqrt(200) ≈ 14.14.
        assert geh(100.0, 0.0) == pytest.approx(math.sqrt(200.0))

    @pytest.mark.parametrize(("m", "c"), [(-1.0, 0.0), (0.0, -1.0), (-5.0, -5.0)])
    def test_negative_raises(self, m: float, c: float) -> None:
        with pytest.raises(ValueError):
            geh(m, c)


# ---------------------------------------------------------------------------
# geh_batch()
# ---------------------------------------------------------------------------


class TestGEHBatch:
    def test_empty_returns_empty(self) -> None:
        assert geh_batch([]) == []

    def test_preserves_order_and_values(self) -> None:
        pairs = [(500.0, 500.0), (700.0, 500.0), (0.0, 0.0)]
        result = geh_batch(pairs)
        assert len(result) == 3
        assert result[0] == 0.0
        assert result[1] == pytest.approx(8.16496580927726)
        assert result[2] == 0.0


# ---------------------------------------------------------------------------
# geh_pass_rate()
# ---------------------------------------------------------------------------


class TestGEHPassRate:
    def test_all_perfect_passes(self) -> None:
        pairs = [
            DetectorPair("d1", 500.0, 500.0),
            DetectorPair("d2", 300.0, 300.0),
            DetectorPair("d3", 100.0, 100.0),
        ]
        report = geh_pass_rate(pairs)
        assert isinstance(report, GEHReport)
        assert report.n_detectors == 3
        assert report.n_pass == 3
        assert report.pass_rate == 1.0
        assert report.calibrated is True

    def test_mixed(self) -> None:
        # d1, d2 calibrated (GEH=0); d3 not (GEH≈14.14 > 5).
        pairs = [
            DetectorPair("d1", 500.0, 500.0),
            DetectorPair("d2", 300.0, 300.0),
            DetectorPair("d3", 100.0, 0.0),
        ]
        report = geh_pass_rate(pairs)
        assert report.n_pass == 2
        assert report.pass_rate == pytest.approx(2.0 / 3.0)
        assert report.calibrated is False  # below 85%

    def test_accepts_plain_tuples(self) -> None:
        report = geh_pass_rate(
            [("d1", 500.0, 500.0), ("d2", 700.0, 700.0)],
        )
        assert report.n_detectors == 2
        assert report.calibrated is True

    def test_custom_thresholds(self) -> None:
        pairs = [
            DetectorPair("d1", 500.0, 500.0),
            DetectorPair("d2", 700.0, 500.0),  # GEH ~8.16
        ]
        # With threshold=10 both pass.
        loose = geh_pass_rate(pairs, threshold=10.0)
        assert loose.n_pass == 2
        # With threshold=5 only one passes.
        tight = geh_pass_rate(pairs, threshold=5.0)
        assert tight.n_pass == 1

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            geh_pass_rate([])

    def test_summary_string(self) -> None:
        pairs = [DetectorPair("d1", 500.0, 500.0)]
        report = geh_pass_rate(pairs)
        s = report.summary()
        assert "PASS" in s
        assert "1/1" in s


# ---------------------------------------------------------------------------
# travel_time_rmse()
# ---------------------------------------------------------------------------


class TestTravelTimeRMSE:
    def test_perfect_match_is_zero(self) -> None:
        assert travel_time_rmse([100.0, 200.0, 300.0], [100.0, 200.0, 300.0]) == 0.0

    def test_known_value(self) -> None:
        # diffs: 10, -10 -> mean square = 100 -> RMSE = 10
        assert travel_time_rmse([110.0, 190.0], [100.0, 200.0]) == pytest.approx(10.0)

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            travel_time_rmse([1.0, 2.0], [1.0])

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            travel_time_rmse([], [])
