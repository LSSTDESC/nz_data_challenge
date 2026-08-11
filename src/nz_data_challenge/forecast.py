import numpy as np

# These are the numbers of objects that pass selection cuts in the different catalogs
N_OBJECTS = dict(
    taskset_1 = dict(
        flagship_1yr = 3625282,
        flagship_4yr = 3617704,
        cardinal_1yr = 3505864,
        cardinal_4yr = 3500333,
    ),
    taskset_2 = dict(
        flagship_1yr = 21227337,
        flagship_4yr = 22302511,
        cardinal_1yr = 17424236,
        cardinal_4yr = 18170490,
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
    taskset_1 = {key: n_obj / N_SAMPLED_OBJ for key, n_obj in N_OBJECTS['taskset_1'].items()},
    taskset_2 = {key: n_obj / N_SAMPLED_OBJ for key, n_obj in N_OBJECTS['taskset_2'].items()},                 
)

# This is the effective density for total number of sampled objects
TOTAL_EFFECTIVE_DENSITY = dict(
    taskset_1 = {key: n_obj / SIM_SKY_AREA_MIN2 for key, n_obj in N_OBJECTS['taskset_1'].items()},
    taskset_2 = {key: n_obj / SIM_SKY_AREA_MIN2 for key, n_obj in N_OBJECTS['taskset_2'].items()},                 
)


def tomo_bins_effective_density(
    n_objects: np.ndarray,
    taskset: str,
    sim: str,
    scenario: str,
) -> np.ndarray:
    """Compute the effective number density for the tomographic bins

    Parameter
    ---------
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
    return n_objects*conversion/N_SAMPLED_OBJ
    
    

