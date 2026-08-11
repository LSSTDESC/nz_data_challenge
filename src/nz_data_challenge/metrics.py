"""Metrics for evaluating photo-z and tomographic bin assignments."""

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import wasserstein_distance
from sklearn.feature_selection import mutual_info_classif

from . import utils


def log_loss_from_labels(
    y_true: ArrayLike, y_pred: ArrayLike, num_classes: int = 5, eps: float = 1e-15
) -> float:
    """Compute log loss from hard predicted labels.

    Constructs a one-hot probability matrix from the predicted labels
    and evaluates the negative log-likelihood of the true labels.

    Parameters
    ----------
    y_true
        Array of true integer labels (0 to num_classes-1).
    y_pred
        Array of predicted integer labels (0 to num_classes-1).
    num_classes
        Number of classes.
    eps
        Smoothing value to prevent log(0).

    Returns
    -------
    float
        Mean log loss.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n_samples = len(y_true)

    probs = np.full((n_samples, num_classes), eps)
    probs[np.arange(n_samples), y_pred] = 1.0 - eps

    true_class_probs = probs[np.arange(n_samples), y_true]

    return -np.mean(np.log(true_class_probs))


def cohens_kappa(
    y_true: ArrayLike, y_pred: ArrayLike, num_classes: int = 5
) -> float:
    """Compute Cohen's Kappa statistic.

    Measures inter-rater agreement for categorical items, correcting
    for agreement occurring by chance.

    Parameters
    ----------
    y_true
        Array of true integer labels (0 to num_classes-1).
    y_pred
        Array of predicted integer labels (0 to num_classes-1).
    num_classes
        Number of classes.

    Returns
    -------
    float
        Cohen's Kappa coefficient in [-1, 1].
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n_samples = len(y_true)

    conf = np.zeros((num_classes, num_classes), dtype=float)
    for t, p in zip(y_true, y_pred):
        conf[t, p] += 1

    p_o = np.trace(conf) / n_samples

    true_marginals = conf.sum(axis=1) / n_samples
    pred_marginals = conf.sum(axis=0) / n_samples
    p_e = np.sum(true_marginals * pred_marginals)

    if p_e == 1.0:
        return 1.0

    return (p_o - p_e) / (1 - p_e)


def balanced_accuracy(
    y_true: ArrayLike, y_pred: ArrayLike, num_classes: int = 5
) -> float:
    """Compute balanced accuracy as the mean of per-class recall.

    Parameters
    ----------
    y_true
        Array of true integer labels (0 to num_classes-1).
    y_pred
        Array of predicted integer labels (0 to num_classes-1).
    num_classes
        Number of classes.

    Returns
    -------
    float
        Balanced accuracy in [0, 1].
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    recalls = []
    for k in range(num_classes):
        actual_k = y_true == k
        n_actual = np.sum(actual_k)

        if n_actual == 0:
            continue

        true_positives = np.sum(actual_k & (y_pred == k))
        recalls.append(true_positives / n_actual)

    return np.mean(recalls)


def kl_divergence(p: ArrayLike, q: ArrayLike, eps: float = 1e-12) -> float:
    """Compute KL divergence KL(P || Q) in bits.

    Both distributions are normalized internally. A small epsilon is
    added for numerical stability.

    Parameters
    ----------
    p
        First distribution (will be normalized).
    q
        Second distribution (will be normalized).
    eps
        Smoothing constant to prevent division by zero or log(0).

    Returns
    -------
    float
        KL divergence in bits.
    """
    p = np.asarray(p, dtype=float) + eps
    q = np.asarray(q, dtype=float) + eps
    p /= p.sum()
    q /= q.sum()
    return np.sum(p * np.log2(p / q))


def total_information_loss(
    true_dists: list[ArrayLike],
    pred_dists: list[ArrayLike],
    weights: ArrayLike | None = None,
) -> tuple[float, NDArray[np.floating]]:
    """Compute the weighted sum of per-class KL divergences.

    Parameters
    ----------
    true_dists
        List of per-class true distributions (each a probability vector).
    pred_dists
        List of per-class predicted distributions (each a probability vector).
    weights
        Per-class weights (e.g., object counts). If None, equal weighting
        is used.

    Returns
    -------
    total
        Weighted average KL divergence (bits per object).
    per_class
        Array of per-class KL divergences.
    """
    n_classes = len(true_dists)
    per_class = np.array([
        kl_divergence(true_dists[k], pred_dists[k]) for k in range(n_classes)
    ])

    if weights is None:
        weights = np.ones(n_classes)
    weights = np.asarray(weights, dtype=float)
    weights /= weights.sum()

    total = np.sum(weights * per_class)
    return total, per_class


def wasserstein_dist(
    x: ArrayLike, p: ArrayLike, q: ArrayLike
) -> float:
    """Compute the Wasserstein distance between two distributions on ordered bins.

    Parameters
    ----------
    x
        Bin centers (shared support for both distributions).
    p
        Weights of the first distribution.
    q
        Weights of the second distribution.

    Returns
    -------
    float
        First Wasserstein distance.
    """
    return wasserstein_distance(x, x, u_weights=p, v_weights=q)


def mutual_info(
    true_redshifts: ArrayLike,
    bin_assignments: ArrayLike,
) -> float:
    """Compute mutual information between redshifts and bin assignments.

    Uses scikit-learn's mutual information estimator for continuous
    features and converts the result from nats to bits.

    Parameters
    ----------
    true_redshifts
        Array of true redshift values (continuous).
    bin_assignments
        Array of discrete bin assignment labels.

    Returns
    -------
    float
        Mutual information in bits.
    """
    mi_nats = mutual_info_classif(
        np.asarray(true_redshifts).reshape(-1, 1),
        np.asarray(bin_assignments),
        discrete_features=False,
    )[0]
    mi_bits = mi_nats / np.log(2)
    return mi_bits


def rms0_delta_summary_stats(
    nz_distributions: ArrayLike,
    true_distributions: ArrayLike,
    grid_edges: ArrayLike,
) -> dict[str, float]:
    """Compute RMS of the per-bin differences in summary statistics.

    For each tomographic bin, computes the mean and std of both the
    estimated and true n(z) distributions, then returns the
    root-mean-square of the differences across bins.

    Parameters
    ----------
    nz_distributions
        2D array of estimated n(z) distributions, shape (n_bins, N).
    true_distributions
        2D array of true n(z) distributions, shape (n_bins, N).
    grid_edges
        Shared bin edges (monotonically increasing), length N+1.

    Returns
    -------
    dict
        Dictionary with 'mean' (RMS of per-bin mean differences) and
        'std' (RMS of per-bin std differences).
    """
    nz_stats = utils.histogram_stats_2d(nz_distributions, grid_edges)
    nz_true_stats = utils.histogram_stats_2d(true_distributions, grid_edges)
    del_mean = nz_stats['mean'] - nz_true_stats['mean']
    del_std = nz_stats['std'] - nz_true_stats['std']
    rms0_mean = np.sqrt((del_mean * del_mean).mean())
    rms0_std = np.sqrt((del_std * del_std).mean())
    return {"mean": rms0_mean, "std": rms0_std}
