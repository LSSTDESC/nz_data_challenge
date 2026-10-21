import tables_io, qp
import numpy as np
import os
from pathlib import Path

try:
    import tables_io

    from rail.estimation.algos.tpz_lite import TPZliteEstimator, TPZliteInformer
    from rail.estimation.algos.naive_stack import NaiveStackMaskedSummarizer
    from rail.estimation.algos.uniform_binning import UniformBinningClassifier
    
    from rail.core.data import TableHandle, QPHandle
    from rail.utils import catalog_utils

    # RAIL setup
    catalog_utils.clear()
    catalog_utils.load_yaml("tests/catalogs.yaml")
    CATALOG_TAG = "cardinal_roman_rubin"
    catalog_utils.apply(CATALOG_TAG)
    
    IMPORTS_OK = True
except ImportError:
    IMPORTS_OK = False
    

from .utils import TOMO_BIN_EDGES

# Change these to match the name of the submission
# and a URL to download the sumission data files
# and needed model files
SUBMISSION_NAME: str = "tpz_test"
SUBMISSION_URL: str = (
    "https://portal.nersc.gov/cfs/lsst/schmidt9/tpz_s3dffiles_tarstuff.tgz"
)
MODEL_URL: str = (
    "https://portal.nersc.gov/cfs/lsst/schmidt9/tpz_s3dffiles_tarstuff.tgz"
)


mag_limits_4yr = {
    'mag_u_lsst': 27.2926,
    'mag_g_lsst': 28.5426,
    'mag_r_lsst': 28.5626,
    'mag_i_lsst': 28.1226,
    'mag_z_lsst': 27.4826,
    'mag_y_lsst': 26.5526,
    'mag_Y_roman': 26.4,
    'mag_J_roman': 26.4,
    'mag_H_roman': 26.4,
    'mag_F_roman': 26.4,
}
mag_limits_1yr = {
    'mag_u_lsst': 26.54,
    'mag_g_lsst': 27.79,
    'mag_r_lsst': 27.81,
    'mag_i_lsst': 27.37,
    'mag_z_lsst': 26.73,
    'mag_y_lsst': 25.8,
    'mag_Y_roman': 26.4,
    'mag_J_roman': 26.4,
    'mag_H_roman': 26.4,
    'mag_F_roman': 26.4,
}

feature_limits_4yr = {
    "mag_i_lsst": 28.1226,
    'mag_u_lsstmag_g_lsst': 0.0,
    'mag_g_lsstmag_r_lsst': 0.0,
    'mag_r_lsstmag_i_lsst': 0.0,
    'mag_i_lsstmag_z_lsst': 0.0,
    'mag_z_lsstmag_y_lsst': 0.0,
    'mag_y_lsstmag_Y_roman': 0.0,
    'mag_Y_romanmag_J_roman': 0.0,
    'mag_J_romanmag_H_roman': 0.0,
    'curve_mag_u_lsstmag_g_lsstmag_r_lsst': 0.0,
    'curve_mag_g_lsstmag_r_lsstmag_i_lsst': 0.0,
    'curve_mag_r_lsstmag_i_lsstmag_z_lsst': 0.0,
    'curve_mag_i_lsstmag_z_lsstmag_y_lsst': 0.0,
    'curve_mag_z_lsstmag_y_lsstmag_Y_roman': 0.0,
    'curve_mag_y_lsstmag_Y_romanmag_J_roman': 0.0,
    'curve_mag_Y_romanmag_J_romanmag_H_roman': 0.0,    
}


lsstbands = ['u','g','r','i','z','y']
romanbands = ['Y', 'J', 'H']
errbands = []
bands = []
for band in lsstbands:
    bands.append(f"mag_{band}_lsst")
    errbands.append(f"mag_{band}_lsst_err")
for band in romanbands:
    bands.append(f"mag_{band}_roman")
    errbands.append(f"mag_{band}_roman_err")
    
# don't change these
SUBMIT_DIR: str = f"submissions/{SUBMISSION_NAME}"
PUBLIC_AREA: str = "tests/public"

def prepare_data(infile):
    xdata = tables_io.read(infile, tType=3)
    # copy the COSMOS redshifts to redshift *only* for the NaN redshifts
    if "ddf" in infile:
        zmask = ~np.isfinite(xdata['redshift'])
        xdata.loc[zmask, "redshift"] = xdata["redshift_manyband"][zmask]

    for band, errb in zip(bands, errbands):
        mask = ~(np.logical_and(np.isfinite(xdata[band]), (np.isfinite(xdata[errb]))))
        xdata.loc[mask, band] = mag_limits_4yr[band]
        xdata.loc[mask, errb] = 1.25

    allbands = []
    allerrs = []
    features = ['mag_i_lsst']
    featureerrs = ['mag_i_lsst_err']
    
    for band in lsstbands:
        allbands.append(f"mag_{band}_lsst")
        allerrs.append(f"mag_{band}_lsst_err")
    for band in romanbands:
        allbands.append(f"mag_{band}_roman")
        allerrs.append(f"mag_{band}_roman_err")
    nbands = len(allbands)
    for ii in range(nbands - 1):
        featurename = f"{allbands[ii]}{allbands[ii+1]}"
        xdata[featurename] = xdata[allbands[ii]] - xdata[allbands[ii+1]]
        features.append(featurename)
        featureerrname = f"{allbands[ii]}{allbands[ii+1]}_err"
        featureerr = np.sqrt((xdata[allerrs[ii]]**2) + (xdata[allerrs[ii+1]]**2))
        xdata[featureerrname] = featureerr
        featureerrs.append(featureerrname)

    for ii in range(nbands - 2):
        featurename = f"curve_{allbands[ii]}{allbands[ii+1]}{allbands[ii+2]}"
        xdata[f"curve_{allbands[ii]}{allbands[ii+1]}{allbands[ii+2]}"] = xdata[allbands[ii]] - 2.0 * xdata[allbands[ii+1]] + xdata[allbands[ii+2]]
        features.append(featurename)
        featureerrname = featurename + "_err"
        featurerr = np.sqrt((xdata[allerrs[ii]]**2) + 2. * (xdata[allerrs[ii+1]]**2) + (xdata[allerrs[ii+2]]**2))
        xdata[featureerrname] = featureerr
        featureerrs.append(featureerrname)
        basename = infile[:-5]
        outname = basename + "_transform.hdf5"
        tables_io.write(xdata, outname)
        #data = tables_io.convert(xdata, tables_io.types.NUMPY_DICT)
        return outname, features, featureerrs

def get_tomo_bin_edges(
    key: str,
) -> np.ndarray:
    taskset = key[0:9]
    tomo_bin_edges = TOMO_BIN_EDGES[taskset]
    return tomo_bin_edges


def run_training(
    key: str,
    wfd_file: str | Path,
    ddf_files: list[str | Path],
    models_dir: str | Path,
) -> None:

    model_file = f"{models_dir}/nz_challenge_{key}_pz_model.pkl"
    ddf_file = ddf_files[0]

    train_datax, tfeatures, tfeatureerrs = prepare_data(ddf_file)
    train_data = TableHandle("train", path=train_datax)

    ## Replace the np.nan redshifts with manyband redshfits and write a 'cleaned' file
    #ddf_file_cleaned = ddf_file.replace(".hdf5", "_cleaned.hdf5")
    #ddf_data = tables_io.read(ddf_file)
    #ddf_data["redshift"] = np.where(
    #    np.isfinite(ddf_data["redshift"]),
    #    ddf_data["redshift"],
    #    ddf_data["redshift_manyband"],
    #)
    #tables_io.write(ddf_data, ddf_file_cleaned)

    # Make a TPZ estiamtor and run it
    tpz_informer = TPZliteInformer.make_stage(
        name=f"inform_{key}",
        hdf5_groupname="",
        bands=tfeatures,
        err_bands=tfeatureerrs,
        nondetect_val=np.nan,
        mag_limits=feature_limits_4yr,
        seed=1994,
        redshift_col="redshift",
        n_random=3,
        n_trees=3,
        min_leaf=5,
        n_att=3,
        model=model_file
    )

    # test_handle = TableHandle(f"input_{key}", path=ddf_file_cleaned)
    inform = tpz_informer.inform(train_data)


def run_estimation(
    key: str,
    wfd_file: str | Path,
    models_dir: str | Path,
    output_nz_estimate_file: str | Path,
    output_bhat_file: str | Path,
    output_nz_samples_file: str | Path,
) -> None:

    tomo_bin_edges = get_tomo_bin_edges(key)
    tomo_bin_centers = 0.5 * (tomo_bin_edges[0:-1] + tomo_bin_edges[1:])
    n_tomo_bins = len(tomo_bin_edges) - 1

    model_file = f"{models_dir}/nz_challenge_{key}_pz_model.pkl"
    pz_file = f"{models_dir}/nz_challenge_{key}_pz_estimates.hdf5"

    test_datax, features, featureerrs = prepare_data(wfd_file)
    test_data =	TableHandle("test", path=test_datax)

    # Make a TPZ estiamtor and run it
    tpz_estimate = TPZliteEstimator.make_stage(
        name=f"estimate_{key}",
        model=model_file,
        hdf5_groupname="",
        bands=features,
        redshift_col="redshift",
        err_bands=featureerrs,
        nondetect_val=np.nan,
        mag_limits=feature_limits_4yr,
    )

    # Bin the objects by mode of the p(z) distribution
    bin_classifier = UniformBinningClassifier.make_stage(
        name=f"classify_{key}",
        zbin_edges=tomo_bin_edges,
        no_assign=-1,
        object_id_col="object_id",
    )

    # Using naive pdf stacking to summarize the n(z) distritubions
    summarizer = NaiveStackMaskedSummarizer.make_stage(
        name=f"summarize_{key}",
        selected_bin=0,
        n_tomo_bins=n_tomo_bins,
        n_samples=100,
        chunk_size=10000,
    )

    do_pz_estimate = True
    do_nz_bin_assignment = True
    do_nz_estimate = True

    test_handle = TableHandle(f"test_{key}", path=str(wfd_file))

    if do_pz_estimate:
        pz_estimates = tpz_estimate.estimate(test_data)
        # This is so that the next stage reads the whole file, not just the
        # current chunk
        pz_estimates.data = None
        os.system(f"cp {pz_estimates.path} {pz_file}")
    else:
        pz_estimates = QPHandle(f"output_{key}", path=pz_file)

    if do_nz_bin_assignment:
        bin_assignments = bin_classifier.classify(pz_estimates)
        # This is so that the next stage reads the whole file, not just the
        # current chunk
        bin_assignments.data = None
        os.system(f"cp {bin_assignments.path} {output_bhat_file}")
    else:
        bin_assignments = TableHandle(f"bhat_{key}", path=output_bhat_file)

    if do_nz_estimate:
        samples_nz = summarizer.summarize(pz_estimates, bin_assignments)
        single_nz = summarizer.get_handle("single_NZ")
        os.system(f"cp {samples_nz.path} {output_nz_samples_file}")
        os.system(f"cp {single_nz.path} {output_nz_estimate_file}")
    else:
        samples = QPHandle(f"output_summarize_{key}", path=output_nz_samples_file)
        single_nz = QPHandle(f"single_NZ_summarize_{key}", path=output_nz_estimate_file)


def copy_samples_files(
    input_nz_samples_file: str | Path,
    output_nz_samples_file: str | Path,
) -> None:
    os.system(f"\\mv {input_nz_samples_file} {output_nz_samples_file}")


def run_taskset_1_estimation_only(
    key: str,
    wfd_file: str | Path,
    models_dir: str | Path,
    output_nz_estimate_file: str | Path,
    output_bhat_file: str | Path,
    output_nz_samples_file: str | Path | None,
) -> None:
    run_estimation(
        key,
        wfd_file,
        models_dir,
        output_nz_estimate_file,
        output_bhat_file,
        output_nz_samples_file,
    )


def run_taskset_2_estimation_only(
    key: str,
    wfd_file: str | Path,
    models_dir: str | Path,
    output_nz_estimate_file: str | Path,
    output_bhat_file: str | Path,
    output_nz_samples_file: str | Path | None,
) -> None:
    run_estimation(
        key,
        wfd_file,
        models_dir,
        output_nz_estimate_file,
        output_bhat_file,
        output_nz_samples_file,
    )


# def run_taskset_3_estimation_only(
#    key: str,
#    wfd_file: str | Path,
#    models_dir: str | Path,
#    output_nz_estimate_file: str | Path,
#    output_bhat_file: str | Path,
#    output_nz_samples_file: str | Path,
# ) -> None:
#    input_nz_samples_file = Path(
#        output_nz_samples_file.replace("taskset_3", "taskset_2")
#    )
#    if not input_nz_samples_file.exists():
#        run_taskset_2_estimation_only(
#            key.replace("taskset_3", "taskset_2"),
#            wfd_file.replace("taskset_3", "taskset_2"),
#            models_dir,
#            output_nz_estimate_file.replace("taskset_3", "taskset_2"),
#            output_bhat_file.replace("taskset_3", "taskset_2"),
#            output_nz_samples_file.replace("taskset_3", "taskset_2"),
#        )
#    copy_samples_files(input_nz_samples_file, output_nz_samples_file)


def run_taskset_1_training_and_estimation(
    key: str,
    wfd_file: str | Path,
    models_dir: str | Path,
    ddf_files: list[str | Path],
    output_nz_estimate_file: str | Path,
    output_bhat_file: str | Path,
    output_nz_samples_file: str | Path | None,
) -> None:
    run_training(key, wfd_file, ddf_files, models_dir)
    run_estimation(
        key,
        wfd_file,
        models_dir,
        output_nz_estimate_file,
        output_bhat_file,
        output_nz_samples_file,
    )


def run_taskset_2_training_and_estimation(
    key: str,
    wfd_file: str | Path,
    models_dir: str | Path,
    ddf_files: list[str | Path],
    output_nz_estimate_file: str | Path,
    output_bhat_file: str | Path,
    output_nz_samples_file: str | Path | None,
) -> None:
    run_training(key, wfd_file, ddf_files, models_dir)
    run_estimation(
        key,
        wfd_file,
        models_dir,
        output_nz_estimate_file,
        output_bhat_file,
        output_nz_samples_file,
    )


# def run_taskset_3_training_and_estimation(
#    key: str,
#    wfd_file: str | Path,
#    models_dir: str | Path,
#    ddf_files: list[str | Path],
#    output_nz_estimate_file: str | Path,
#    output_bhat_file: str | Path,
#    output_nz_samples_file: str | Path,
# ) -> None:
#
#    input_nz_samples_file = Path(
#        output_nz_samples_file.replace("taskset_3", "taskset_2")
#    )
#    if not input_nz_samples_file.exists():
#        run_taskset_3_training_and_estimation(
#            key.replace("taskset_3", "taskset_2"),
#            wfd_file.replace("taskset_3", "taskset_2"),
#            models_dir,
#            [val.replace("taskset_3", "taskset_2") for val in ddf_files],
#            output_nz_estimate_file.replace("taskset_3", "taskset_2"),
#            output_bhat_file.replace("taskset_3", "taskset_2"),
#            output_nz_samples_file.replace("taskset_3", "taskset_2"),
#        )
#    copy_samples_files(input_nz_samples_file, output_nz_samples_file)
