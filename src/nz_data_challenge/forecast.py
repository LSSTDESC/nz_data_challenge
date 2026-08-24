"""Fisher matrix forecasting for n(z) challenge submissions."""

import numpy as np

from fisherA2Z.fisher_flex import FisherFlex, FisherFlexResult

import qp
from . import utils

# These are the numbers of objects that pass selection cuts in the different catalogs
N_OBJECTS = dict(
    taskset_1=dict(
        flagship_1yr=3625282,
        flagship_4yr=3617704,
        cardinal_1yr=3505864,
        cardinal_4yr=3500333,
    ),
    taskset_2=dict(
        flagship_1yr=21227337,
        flagship_4yr=22302511,
        cardinal_1yr=17424236,
        cardinal_4yr=18170490,
    ),
)

# For the WFD files we selection 1M objects from the catalogs
N_SAMPLED_OBJ = 1000000

# This is the total area of the sims (32 HEALPix nside=32 pixels)
# That corresponds to 32 / (12 * 32 * 32) = 0.002604166667 of the sky
SIM_SKY_AREA = 107.4295
SIM_SKY_AREA_MIN2 = 107.4295 * 3600


# These are the sampling factors, i.e., number of catalog objects / number sampled
SAMPLING_RATIO = dict(
    taskset_1={
        key: n_obj / N_SAMPLED_OBJ for key, n_obj in N_OBJECTS["taskset_1"].items()
    },
    taskset_2={
        key: n_obj / N_SAMPLED_OBJ for key, n_obj in N_OBJECTS["taskset_2"].items()
    },
)

# This is the effective density for total number of sampled objects
TOTAL_EFFECTIVE_DENSITY = dict(
    taskset_1={
        key: n_obj / SIM_SKY_AREA_MIN2 for key, n_obj in N_OBJECTS["taskset_1"].items()
    },
    taskset_2={
        key: n_obj / SIM_SKY_AREA_MIN2 for key, n_obj in N_OBJECTS["taskset_2"].items()
    },
)

FORECAST_Z_MIN = 0.02
FORECAST_Z_MAX = 3.0
FORECAST_N_ZBINS = 150
FORECAST_Z_GRID = np.linspace(FORECAST_Z_MIN, FORECAST_Z_MAX, FORECAST_N_ZBINS)
FORECAST_Z_GRID_FULL = np.linspace(0, FORECAST_Z_MAX, FORECAST_N_ZBINS + 1)

# Forecast params
FORECAST_FSKY = 0.5  # fraction of sky observed
FORECAST_SIGMA_E = 0.26  # per-component ellipticity dispersion


def tomo_bins_effective_density(
    n_objects: np.ndarray,
    taskset: str,
    sim: str,
    scenario: str,
) -> np.ndarray:
    """Compute the effective number density for the tomographic bins.

    Parameters
    ----------
    n_objects
        Number of objects in each bin
    taskset
        Name of the taskset
    sim
        Name of the simulation ('cardinal' or 'flagship')
    scenario
        Name of the scenario ('1yr' or '4yr')

    Returns
    -------
    Effective number density (in gal / min^2)
    """
    conversion = TOTAL_EFFECTIVE_DENSITY[taskset][f"{sim}_{scenario}"]
    return n_objects * conversion / N_SAMPLED_OBJ


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
