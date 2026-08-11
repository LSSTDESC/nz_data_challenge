"""Unit tests for nz_data_challenge.utils."""

import numpy as np
import pytest

from nz_data_challenge import utils


class TestHistogramStats:
    def test_uniform_histogram(self) -> None:
        edges = np.array([0.0, 1.0, 2.0, 3.0])
        values = np.array([1.0, 1.0, 1.0])
        result = utils.histogram_stats(values, edges)
        assert result["mean"] == pytest.approx(1.5)
        assert result["std"] == pytest.approx(np.sqrt(2.0 / 3.0))

    def test_single_bin_nonzero(self) -> None:
        edges = np.array([0.0, 1.0, 2.0, 3.0])
        values = np.array([0.0, 5.0, 0.0])
        result = utils.histogram_stats(values, edges)
        assert result["mean"] == pytest.approx(1.5)
        assert result["std"] == pytest.approx(0.0)

    def test_weighted_mean(self) -> None:
        edges = np.array([0.0, 1.0, 2.0])
        values = np.array([3.0, 1.0])
        result = utils.histogram_stats(values, edges)
        expected_mean = (3.0 * 0.5 + 1.0 * 1.5) / 4.0
        assert result["mean"] == pytest.approx(expected_mean)

    def test_mismatched_lengths_raises(self) -> None:
        edges = np.array([0.0, 1.0, 2.0])
        values = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="bin_edges must have length"):
            utils.histogram_stats(values, edges)

    def test_zero_weight_raises(self) -> None:
        edges = np.array([0.0, 1.0, 2.0])
        values = np.array([0.0, 0.0])
        with pytest.raises(ValueError, match="zero"):
            utils.histogram_stats(values, edges)


class TestHistogramStats2d:
    def test_matches_1d(self) -> None:
        edges = np.array([0.0, 1.0, 2.0, 3.0])
        row = np.array([2.0, 3.0, 5.0])
        result_1d = utils.histogram_stats(row, edges)
        result_2d = utils.histogram_stats_2d(row.reshape(1, -1), edges)
        assert result_2d["mean"][0] == pytest.approx(result_1d["mean"])
        assert result_2d["std"][0] == pytest.approx(result_1d["std"])

    def test_multiple_rows(self) -> None:
        edges = np.array([0.0, 1.0, 2.0, 3.0])
        values = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        result = utils.histogram_stats_2d(values, edges)
        assert result["mean"][0] == pytest.approx(0.5)
        assert result["mean"][1] == pytest.approx(2.5)
        assert len(result["mean"]) == 2
        assert len(result["std"]) == 2

    def test_not_2d_raises(self) -> None:
        edges = np.array([0.0, 1.0, 2.0])
        values = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="2D"):
            utils.histogram_stats_2d(values, edges)

    def test_mismatched_columns_raises(self) -> None:
        edges = np.array([0.0, 1.0, 2.0])
        values = np.array([[1.0, 2.0, 3.0]])
        with pytest.raises(ValueError, match="bin_edges must have length"):
            utils.histogram_stats_2d(values, edges)

    def test_zero_row_raises(self) -> None:
        edges = np.array([0.0, 1.0, 2.0])
        values = np.array([[1.0, 2.0], [0.0, 0.0]])
        with pytest.raises(ValueError, match="zero total weight"):
            utils.histogram_stats_2d(values, edges)


class TestGetTrueNzDistributions:
    def test_basic(self) -> None:
        true_redshifts = np.array([0.5, 0.5, 1.5, 1.5, 1.5])
        bin_assignments = np.array([0, 0, 1, 1, 1])
        grid_edges = np.array([0.0, 1.0, 2.0])
        result = utils.get_true_nz_distributions(
            true_redshifts, bin_assignments, grid_edges, n_bins=2
        )
        assert result.shape == (2, 2)
        assert result[0, 0] == 2
        assert result[0, 1] == 0
        assert result[1, 0] == 0
        assert result[1, 1] == 3

    def test_empty_bin(self) -> None:
        true_redshifts = np.array([0.5, 0.5])
        bin_assignments = np.array([0, 0])
        grid_edges = np.array([0.0, 1.0, 2.0])
        result = utils.get_true_nz_distributions(
            true_redshifts, bin_assignments, grid_edges, n_bins=2
        )
        assert result[1].sum() == 0


class TestGetTrueBinAssignments:
    def test_basic_binning(self) -> None:
        bin_edges = np.array([0.0, 1.0, 2.0, 3.0])
        redshifts = np.array([0.5, 1.5, 2.5])
        assignments = utils.get_true_bin_assignments(redshifts, bin_edges)
        np.testing.assert_array_equal(assignments, [0, 1, 2])

    def test_on_edge(self) -> None:
        bin_edges = np.array([0.0, 1.0, 2.0, 3.0])
        redshifts = np.array([1.0, 2.0])
        assignments = utils.get_true_bin_assignments(redshifts, bin_edges)
        assert assignments[0] == 1
        assert assignments[1] == 2

    def test_below_range(self) -> None:
        bin_edges = np.array([0.0, 1.0, 2.0, 3.0])
        redshifts = np.array([-0.5])
        assignments = utils.get_true_bin_assignments(redshifts, bin_edges)
        assert assignments[0] == 0

    def test_above_range(self) -> None:
        bin_edges = np.array([0.0, 1.0, 2.0, 3.0])
        redshifts = np.array([3.5])
        assignments = utils.get_true_bin_assignments(redshifts, bin_edges)
        assert assignments[0] == 2
