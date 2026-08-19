"""Utility functions for n(z) distribution extraction and binning."""

import qp
import numpy as np
from numpy.typing import ArrayLike, NDArray

# Tasksets defined in the challenge so far
TASKSETS: list[str] = ['taskset_1', 'taskset_2']
# Types of simulations used
SIMS: list[str] = ['cardinal', 'flagship']
# Scenarios considered
SCENARIOS: list[str] = ['1yr', '4yr']

# Tomographic bins we want people to use, by taskset
TOMO_BIN_EDGES: dict[str, np.ndarray] = dict(
    taskset_1 = np.array([0.0, 0.32, 0.47, 0.61, 0.78, 2.5 ]),
    taskset_2 = np.array([0.0, 0.42, 0.64, 0.87, 1.20, 2.5 ]),
)   

# Binning for z estimation plots, by taskset
Z_MIN_TASKSET_1 = 0.
Z_MAX_TASKSET_1 = 1.5
NZ_BINS_TASKSET_1 = 150
Z_BIN_EDGES_TASKSET_1 = np.linspace(Z_MIN_TASKSET_1, Z_MAX_TASKSET_1, NZ_BINS_TASKSET_1 + 1)

Z_MIN_TASKSET_2 = 0.
Z_MAX_TASKSET_2 = 3.0
NZ_BINS_TASKSET_2 = 150
Z_BIN_EDGES_TASKSET_2 = np.linspace(Z_MIN_TASKSET_2, Z_MAX_TASKSET_2, NZ_BINS_TASKSET_2 + 1)

Z_BIN_EDGES: dict[str, np.ndarray] = dict(
    taskset_1=Z_BIN_EDGES_TASKSET_1,
    taskset_2=Z_BIN_EDGES_TASKSET_2,
)    



def histogram_stats(
    bin_values: ArrayLike, bin_edges: ArrayLike
) -> dict[str, float]:
    """Compute statistics of a histogram.

    Parameters
    ----------
    bin_values
        The value (count or weight) in each bin, length N.
    bin_edges
        The edges of the bins (monotonically increasing), length N+1.

    Returns
    -------
    dict
        Dictionary with keys 'mean' (weighted mean using bin centers)
        and 'std' (standard deviation about the mean).

    Raises
    ------
    ValueError
        If ``bin_edges`` does not have length ``len(bin_values) + 1``
        or if the sum of bin values is zero.
    """
    bin_values = np.asarray(bin_values, dtype=float)
    bin_edges = np.asarray(bin_edges, dtype=float)

    if len(bin_edges) != len(bin_values) + 1:
        raise ValueError(
            f"bin_edges must have length len(bin_values)+1 "
            f"({len(bin_values)+1}), got {len(bin_edges)}"
        )

    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    total_weight = bin_values.sum()
    if total_weight == 0:
        raise ValueError("Sum of bin values is zero; statistics undefined.")

    mean = np.sum(bin_values * centers) / total_weight

    variance = np.sum(bin_values * (centers - mean) ** 2) / total_weight
    std = np.sqrt(variance)

    return {"mean": mean, "std": std}


def histogram_stats_2d(
    bin_values: ArrayLike, bin_edges: ArrayLike
) -> dict[str, NDArray[np.floating]]:
    """Compute statistics for each row of a 2D histogram array.

    Each row is treated as a separate histogram sharing the same bin
    edges.

    Parameters
    ----------
    bin_values
        2D array of shape (n_rows, N) where each row contains the
        values (counts or weights) for one histogram.
    bin_edges
        The shared bin edges (monotonically increasing), length N+1.

    Returns
    -------
    dict
        Dictionary with keys 'mean' and 'std', each an array of length
        n_rows.

    Raises
    ------
    ValueError
        If ``bin_edges`` length does not match the number of columns
        in ``bin_values``, or if any row sums to zero.
    """
    bin_values = np.asarray(bin_values, dtype=float)
    bin_edges = np.asarray(bin_edges, dtype=float)

    if bin_values.ndim != 2:
        raise ValueError(
            f"bin_values must be 2D, got {bin_values.ndim}D"
        )

    if len(bin_edges) != bin_values.shape[1] + 1:
        raise ValueError(
            f"bin_edges must have length bin_values.shape[1]+1 "
            f"({bin_values.shape[1]+1}), got {len(bin_edges)}"
        )

    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    total_weights = bin_values.sum(axis=1)
    if np.any(total_weights == 0):
        raise ValueError("At least one row has zero total weight; statistics undefined.")

    means = (bin_values @ centers) / total_weights

    variance = (bin_values * (centers[np.newaxis, :] - means[:, np.newaxis]) ** 2).sum(axis=1)
    stds = np.sqrt(variance / total_weights)

    return {"mean": means, "std": stds}


def get_nz_distributions(
    nz_estimates: qp.Ensemble, grid_centers: NDArray, n_bins: int
) -> NDArray[np.floating]:
    """Extract normalized n(z) distributions from a qp Ensemble.

    For each tomographic bin, the PDF is evaluated at the given grid
    centers and scaled by the number of objects in that bin.

    Parameters
    ----------
    nz_estimates
        A qp Ensemble containing the n(z) estimates with an ancillary
        table that includes 'n_objects'.
    grid_centers
        Redshift grid centers at which to evaluate the PDFs.
    n_bins
        Number of tomographic bins to extract.

    Returns
    -------
    ndarray
        Array of shape (n_bins, len(grid_centers)) with normalized
        distributions.
    """
    pdfs = nz_estimates.pdf(grid_centers)
    norms = nz_estimates.ancil['n_objects']
    out_list = []
    for i in range(n_bins):
        binx = pdfs[i]
        binx_normed = norms[i] * binx / binx.sum()
        out_list.append(binx_normed)
    return np.array(out_list)


def get_true_nz_distributions(
    true_redshifts: NDArray,
    bin_assignments: NDArray,
    grid_edges: NDArray,
    n_bins: int,
) -> NDArray[np.floating]:
    """Build true n(z) histograms from redshifts and bin assignments.

    Parameters
    ----------
    true_redshifts
        Array of true redshift values.
    bin_assignments
        Array of integer bin assignment labels for each object.
    grid_edges
        Bin edges for the redshift histogram.
    n_bins
        Number of tomographic bins.

    Returns
    -------
    ndarray
        Array of shape (n_bins, len(grid_edges)-1) with histogram
        counts per bin.
    """
    out_list = []
    for i in range(n_bins):
        hist = np.histogram(true_redshifts[bin_assignments == i], grid_edges)[0]
        out_list.append(hist)
    return np.array(out_list)


def get_true_bin_assignments(
    true_redshifts: NDArray,
    bin_edges: NDArray,
) -> NDArray[np.intp]:
    """Assign objects to tomographic bins based on true redshifts.

    Uses ``np.digitize`` with the interior bin edges so that the
    returned labels are 0-indexed.

    Parameters
    ----------
    true_redshifts
        Array of true redshift values.
    bin_edges
        Array of bin edges (length n_bins + 1). Only the interior
        edges are used for digitization.

    Returns
    -------
    ndarray
        Integer array of bin assignments (0-indexed).
    """
    return np.digitize(true_redshifts, bin_edges[1:-1])
