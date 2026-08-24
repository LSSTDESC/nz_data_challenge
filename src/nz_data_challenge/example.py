import tables_io, qp
import numpy as np
import os
from pathlib import Path

from rail.estimation.algos.k_nearneigh import KNearNeighEstimator, KNearNeighInformer
from rail.estimation.algos.naive_stack import NaiveStackMaskedSummarizer
from rail.estimation.algos.uniform_binning import UniformBinningClassifier

from rail.core.data import TableHandle, QPHandle
from rail.utils import catalog_utils
import tables_io

from .utils import TOMO_BIN_EDGES, TASKSETS, SIMS, SCENARIOS

# Change these to match the name of the submission
# and a URL to download the sumission data files
# and needed model files
SUBMISSION_NAME: str = "example"
SUBMISSION_URL: str = (
    "http://s3df.slac.stanford.edu/people/echarles/data_challenge/nz_estimates_example.tgz"
)
MODEL_URL: str = (
    "http://s3df.slac.stanford.edu/people/echarles/data_challenge/nz_models_example.tgz"
)


# RAIL setup
catalog_utils.clear()
catalog_utils.load_yaml("tests/catalogs.yaml")
CATALOG_TAG = "cardinal_roman_rubin"
catalog_utils.apply(CATALOG_TAG)


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

    # Replace the np.nan redshifts with manyband redshfits and write a 'cleaned' file
    ddf_file_cleaned = ddf_file.replace(".hdf5", "_cleaned.hdf5")
    ddf_data = tables_io.read(ddf_file)
    ddf_data["redshift"] = np.where(
        np.isfinite(ddf_data["redshift"]),
        ddf_data["redshift"],
        ddf_data["redshift_manyband"],
    )
    tables_io.write(ddf_data, ddf_file_cleaned)

    # Make a KNN estiamtor and run it
    knn_informer = KNearNeighInformer.make_stage(
        name=f"estimate_{key}",
        model=model_file,
        hdf5_groupname="",
    )

    test_handle = TableHandle(f"input_{key}", path=ddf_file_cleaned)
    inform = knn_informer.inform(test_handle)


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

    # Make a KNN estiamtor and run it
    knn_estimate = KNearNeighEstimator.make_stage(
        name=f"estimate_{key}",
        model=model_file,
        hdf5_groupname="",
        id_col="object_id",
        nzbins=301,
        zmax=3.0,
        chunk_size=10000,
        nondetect_val=np.nan,
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
        pz_estimates = knn_estimate.estimate(test_handle)
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


def run_taskset_3_estimation_only(
    key: str,
    wfd_file: str | Path,
    models_dir: str | Path,
    output_nz_estimate_file: str | Path,
    output_bhat_file: str | Path,
    output_nz_samples_file: str | Path,
) -> None:
    input_nz_samples_file = Path(
        output_nz_samples_file.replace("taskset_3", "taskset_2")
    )
    if not input_nz_samples_file.exists():
        run_taskset_2_estimation_only(
            key.replace("taskset_3", "taskset_2"),
            wfd_file.replace("taskset_3", "taskset_2"),
            models_dir,
            output_nz_estimate_file.replace("taskset_3", "taskset_2"),
            output_bhat_file.replace("taskset_3", "taskset_2"),
            output_nz_samples_file.replace("taskset_3", "taskset_2"),
        )
    copy_samples_files(output_nz_samples_file)


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


def run_taskset_3_training_and_estimation(
    key: str,
    wfd_file: str | Path,
    models_dir: str | Path,
    ddf_files: list[str | Path],
    output_nz_estimate_file: str | Path,
    output_bhat_file: str | Path,
    output_nz_samples_file: str | Path,
) -> None:

    input_nz_samples_file = Path(
        output_nz_samples_file.replace("taskset_3", "taskset_2")
    )
    if not input_nz_samples_file.exists():
        run_taskset_3_training_and_estimation(
            key.replace("taskset_3", "taskset_2"),
            wfd_file.replace("taskset_3", "taskset_2"),
            models_dir,
            [val.replace("taskset_3", "taskset_2") for val in ddf_files],
            output_nz_estimate_file.replace("taskset_3", "taskset_2"),
            output_bhat_file.replace("taskset_3", "taskset_2"),
            output_nz_samples_file.replace("taskset_3", "taskset_2"),
        )
    copy_samples_files(output_nz_samples_file)
