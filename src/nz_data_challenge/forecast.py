"""Fisher matrix forecasting for n(z) challenge submissions."""

import numpy as np

from fisherA2Z.fisher_flex import FisherFlex, FisherFlexResult

import qp
from . import utils

# The survey numbers and the effective-density conversion live in utils, so
# that code which does not need fisherA2Z can use them; they are re-exported
# here for backwards compatibility.
from .utils import (  # noqa: F401
    N_OBJECTS,
    N_SAMPLED_OBJ,
    SAMPLING_RATIO,
    SIM_SKY_AREA,
    SIM_SKY_AREA_MIN2,
    TOTAL_EFFECTIVE_DENSITY,
    tomo_bins_effective_density,
)

FORECAST_Z_MIN = 0.02
FORECAST_Z_MAX = 3.0
FORECAST_N_ZBINS = 150
FORECAST_Z_GRID = np.linspace(FORECAST_Z_MIN, FORECAST_Z_MAX, FORECAST_N_ZBINS)
FORECAST_Z_GRID_FULL = np.linspace(0, FORECAST_Z_MAX, FORECAST_N_ZBINS + 1)

# Forecast params
FORECAST_FSKY = 0.5  # fraction of sky observed
FORECAST_SIGMA_E = 0.26  # per-component ellipticity dispersion


def fisher_forecast(
    qp_nz_central: qp.Ensemble,
    qp_nz_samples: qp.Ensemble,
    bhat_table: dict[str, np.ndarray],
    neff: np.ndarray,
    mode: str,  # '3x2pt' | '2x2pt' | 'cosmic_shear'
) -> FisherFlexResult:
    """Run a Fisher matrix forecast using n(z) estimates and their uncertainties.

    Parameters
    ----------
    qp_nz_central
        qp Ensemble containing the central n(z) estimates per tomographic bin.
    qp_nz_samples
        qp Ensemble containing bootstrap/sample realizations of the n(z),
        with ancillary columns 'bin_idx' and 'i_realization'.
    bhat_table
        Dictionary with at least a 'tomo_bin_index' key mapping to an array
        of bin assignments per object.
    neff
        Effective number density per tomographic bin (arcmin^-2).
    mode
        Analysis mode: '3x2pt', '2x2pt', or 'cosmic_shear'.

    Returns
    -------
    FisherFlexResult
        Fisher forecast result object from fisherA2Z.
    """
    n_bins = qp_nz_samples.ancil["bin_idx"].max() + 1
    n_samples = qp_nz_samples.ancil["i_realization"].max() + 1

    nz_realizations = qp_nz_samples.pdf(FORECAST_Z_GRID).reshape(
        n_bins, n_samples, FORECAST_N_ZBINS
    )
    nz_central = qp_nz_central.pdf(FORECAST_Z_GRID)

    counts = np.squeeze(qp_nz_central.ancil["n_objects"])

    print(f"effective number density in bins = {neff}")
    print(f"number counts in bins = {counts}")

    # Define the parameters for the forecast
    forecast_params = dict(ell_max_cs=1800, ell_min_cs=300)

    flex = FisherFlex(
        # -- the n(z) and its uncertainty ---------------------------------
        nz_source=nz_central,
        nz_realizations=nz_realizations,
        z_grid=FORECAST_Z_GRID,
        # -- the survey ---------------------------------------------------
        neff_source=neff,  # arcmin^-2, per tomographic bin
        fsky=FORECAST_FSKY,
        sigma_e=FORECAST_SIGMA_E,
        # -- the analysis -------------------------------------------------
        mode=mode,
        nz_model="shift_stretch",  # the default; see below for the alternative
    )

    flex.compute(parallel=True)
    res = flex.forecast(**forecast_params)

    return res


def fisher_bias_forecast(
    qp_nz_central: qp.Ensemble,
    bhat_table: dict[str, np.ndarray],
    neff: np.ndarray,
    mode: str,  # '3x2pt' | '2x2pt' | 'cosmic_shear'
    truth: dict[str, np.ndarray],
) -> FisherFlexResult:
    """Run a Fisher matrix forecast using n(z) estimates and their uncertainties.

    Parameters
    ----------
    qp_nz_central
        qp Ensemble containing the central n(z) estimates per tomographic bin.
    bhat_table
        Dictionary with at least a 'tomo_bin_index' key mapping to an array
        of bin assignments per object.
    neff
        Effective number density per tomographic bin (arcmin^-2).
    mode
        Analysis mode: '3x2pt', '2x2pt', or 'cosmic_shear'.
    truth
        If provided, dictionary with at least a 'redshift' key for
        computing true n(z) distributions. If None, truth is not used.

    Returns
    -------
    FisherFlexResult
        Fisher forecast result object from fisherA2Z.
    """
    nz_central = qp_nz_central.pdf(FORECAST_Z_GRID)
    n_bins = qp_nz_central.npdf

    if truth is not None:
        true_redshifts = truth["redshift"]
        nz_true = utils.get_true_nz_distributions(
            true_redshifts,
            np.squeeze(bhat_table["tomo_bin_index"]),
            FORECAST_Z_GRID_FULL,
            n_bins,
        )

    counts = np.squeeze(qp_nz_central.ancil["n_objects"])

    print(f"effective number density in bins = {neff}")
    print(f"number counts in bins = {counts}")

    # Define the parameters for the forecast
    forecast_params = dict(ell_max_cs=1800, ell_min_cs=300)

    flex = FisherFlex(
        # -- the n(z) and its uncertainty ---------------------------------
        nz_source=nz_central,
        z_grid=FORECAST_Z_GRID,
        # -- the survey ---------------------------------------------------
        neff_source=neff,  # arcmin^-2, per tomographic bin
        fsky=FORECAST_FSKY,
        sigma_e=FORECAST_SIGMA_E,
        # -- the analysis -------------------------------------------------
        mode=mode,
        nz_model="no_uncertainty",
    )

    flex.compute(parallel=True)
    res = flex.forecast(**forecast_params)
    bias = flex.forecast_bias(nz_truth=nz_true, forecast_params=forecast_params)
    return res, bias
