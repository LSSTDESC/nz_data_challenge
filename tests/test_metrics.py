"""Unit tests for nz_data_challenge.metrics."""

import numpy as np
import pytest

from nz_data_challenge import metrics


class TestLogLossFromLabels:
    def test_perfect_predictions(self) -> None:
        y_true = np.array([0, 1, 2, 3, 4])
        y_pred = np.array([0, 1, 2, 3, 4])
        loss = metrics.log_loss_from_labels(y_true, y_pred)
        assert loss == pytest.approx(0.0, abs=1e-10)

    def test_worst_case(self) -> None:
        y_true = np.array([0, 0, 0])
        y_pred = np.array([1, 1, 1])
        loss = metrics.log_loss_from_labels(y_true, y_pred)
        assert loss > 0.0

    def test_symmetry(self) -> None:
        y_true = np.array([0, 1, 2])
        y_pred = np.array([1, 2, 0])
        loss = metrics.log_loss_from_labels(y_true, y_pred)
        assert loss == pytest.approx(
            metrics.log_loss_from_labels([1, 2, 0], [2, 0, 1])
        )

    def test_custom_num_classes(self) -> None:
        y_true = np.array([0, 1, 2])
        y_pred = np.array([0, 1, 2])
        loss = metrics.log_loss_from_labels(y_true, y_pred, num_classes=3)
        assert loss == pytest.approx(0.0, abs=1e-10)


class TestCohensKappa:
    def test_perfect_agreement(self) -> None:
        y = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
        assert metrics.cohens_kappa(y, y) == pytest.approx(1.0)

    def test_random_agreement(self) -> None:
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 5, size=1000)
        y_pred = rng.integers(0, 5, size=1000)
        kappa = metrics.cohens_kappa(y_true, y_pred)
        assert -0.1 < kappa < 0.1

    def test_known_value(self) -> None:
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 0, 1, 1, 2, 2])
        assert metrics.cohens_kappa(y_true, y_pred, num_classes=3) == pytest.approx(1.0)

    def test_partial_agreement(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 0])
        kappa = metrics.cohens_kappa(y_true, y_pred, num_classes=2)
        assert kappa == pytest.approx(0.0)


class TestBalancedAccuracy:
    def test_perfect(self) -> None:
        y = np.array([0, 1, 2, 3, 4])
        assert metrics.balanced_accuracy(y, y) == pytest.approx(1.0)

    def test_all_wrong(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])
        assert metrics.balanced_accuracy(y_true, y_pred, num_classes=2) == pytest.approx(0.0)

    def test_imbalanced_classes(self) -> None:
        y_true = np.array([0, 0, 0, 0, 1])
        y_pred = np.array([0, 0, 0, 0, 0])
        ba = metrics.balanced_accuracy(y_true, y_pred, num_classes=2)
        assert ba == pytest.approx(0.5)

    def test_skips_absent_classes(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        ba = metrics.balanced_accuracy(y_true, y_pred, num_classes=5)
        assert ba == pytest.approx(1.0)


class TestKLDivergence:
    def test_identical_distributions(self) -> None:
        p = np.array([0.2, 0.3, 0.5])
        assert metrics.kl_divergence(p, p) == pytest.approx(0.0, abs=1e-8)

    def test_positive(self) -> None:
        p = np.array([0.9, 0.1])
        q = np.array([0.1, 0.9])
        assert metrics.kl_divergence(p, q) > 0.0

    def test_asymmetric(self) -> None:
        p = np.array([0.9, 0.1])
        q = np.array([0.5, 0.5])
        kl_pq = metrics.kl_divergence(p, q)
        kl_qp = metrics.kl_divergence(q, p)
        assert kl_pq != pytest.approx(kl_qp)

    def test_unnormalized_input(self) -> None:
        p = np.array([2.0, 3.0, 5.0])
        q = np.array([1.0, 1.0, 1.0])
        kl = metrics.kl_divergence(p, q)
        assert kl > 0.0


class TestTotalInformationLoss:
    def test_identical_returns_zero(self) -> None:
        dists = [np.array([0.2, 0.3, 0.5]), np.array([0.4, 0.4, 0.2])]
        total, per_class = metrics.total_information_loss(dists, dists)
        assert total == pytest.approx(0.0, abs=1e-8)
        assert all(v == pytest.approx(0.0, abs=1e-8) for v in per_class)

    def test_with_weights(self) -> None:
        true_dists = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
        pred_dists = [np.array([0.5, 0.5]), np.array([0.5, 0.5])]
        total_equal, _ = metrics.total_information_loss(true_dists, pred_dists)
        total_weighted, _ = metrics.total_information_loss(
            true_dists, pred_dists, weights=np.array([10.0, 1.0])
        )
        assert total_equal > 0
        assert total_weighted > 0

    def test_per_class_length(self) -> None:
        dists = [np.array([0.5, 0.5])] * 4
        _, per_class = metrics.total_information_loss(dists, dists)
        assert len(per_class) == 4


class TestWassersteinDist:
    def test_identical(self) -> None:
        x = np.array([1.0, 2.0, 3.0])
        p = np.array([0.2, 0.5, 0.3])
        assert metrics.wasserstein_dist(x, p, p) == pytest.approx(0.0)

    def test_shifted(self) -> None:
        x = np.array([0.0, 1.0, 2.0])
        p = np.array([1.0, 0.0, 0.0])
        q = np.array([0.0, 0.0, 1.0])
        assert metrics.wasserstein_dist(x, p, q) == pytest.approx(2.0)

    def test_symmetric(self) -> None:
        x = np.array([1.0, 2.0, 3.0, 4.0])
        p = np.array([0.4, 0.3, 0.2, 0.1])
        q = np.array([0.1, 0.2, 0.3, 0.4])
        assert metrics.wasserstein_dist(x, p, q) == pytest.approx(
            metrics.wasserstein_dist(x, q, p)
        )


class TestMutualInfo:
    def test_perfect_separation(self) -> None:
        true_redshifts = np.concatenate([
            np.full(100, 0.5), np.full(100, 1.5)
        ])
        bin_assignments = np.concatenate([np.zeros(100), np.ones(100)]).astype(int)
        mi = metrics.mutual_info(true_redshifts, bin_assignments)
        assert mi > 0.5

    def test_random_assignment_low_mi(self) -> None:
        rng = np.random.default_rng(42)
        true_redshifts = rng.uniform(0, 2, size=500)
        bin_assignments = rng.integers(0, 3, size=500)
        mi = metrics.mutual_info(true_redshifts, bin_assignments)
        assert mi < 0.1


class TestRms0DeltaSummaryStats:
    def test_identical_returns_zero(self) -> None:
        edges = np.array([0.0, 1.0, 2.0, 3.0])
        dists = np.array([[1.0, 2.0, 1.0], [0.5, 1.0, 0.5]])
        result = metrics.rms0_delta_summary_stats(dists, dists, edges)
        assert result["mean"] == pytest.approx(0.0, abs=1e-10)
        assert result["std"] == pytest.approx(0.0, abs=1e-10)

    def test_shifted_distributions(self) -> None:
        edges = np.array([0.0, 1.0, 2.0, 3.0])
        true_dists = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        pred_dists = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
        result = metrics.rms0_delta_summary_stats(pred_dists, true_dists, edges)
        assert result["mean"] > 0.0

    def test_returns_dict_keys(self) -> None:
        edges = np.array([0.0, 1.0, 2.0])
        dists = np.array([[1.0, 1.0]])
        result = metrics.rms0_delta_summary_stats(dists, dists, edges)
        assert set(result.keys()) == {"mean", "std"}
