from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

try:
    import tables_io
    import qp

    from rail.core.data import QPHandle, TableHandle
    from rail.estimation.algos.flexzboost import FlexZBoostEstimator, FlexZBoostInformer
    from rail.estimation.algos.naive_stack import NaiveStackMaskedSummarizer
    from rail.utils import catalog_utils

    catalog_utils.clear()
    catalog_utils.load_yaml("tests/catalogs.yaml")
    catalog_utils.apply("cardinal_roman_rubin")

    IMPORTS_OK = True
except ImportError:
    IMPORTS_OK = False


from .utils import SCENARIOS, SIMS, TOMO_BIN_EDGES

SUBMISSION_NAME: str = "fzboost_naive"
SUBMISSION_URL: str = (
    "https://zenodo.org/records/22100474/files/nz_estimates_fzboost_naive.tgz?download=1"
)
MODEL_URL: str = (
    "https://zenodo.org/records/22100474/files/nz_models_fzboost_naive.tgz?download=1"
)

_TASKSET_ID_OFFSET = {
    "taskset_1": 0,
    "taskset_2": 4_000_000,
    "taskset_3": 4_000_000,
}


def _validator_object_ids(key: str, n: int) -> np.ndarray:
    """IDs expected by submit_utils.check_files (not the catalog object_id column)."""
    taskset = key[0:9]
    rest = key[len(taskset) + 1 :]
    sim, scenario = rest.rsplit("_", 1)
    combo = SIMS.index(sim) * len(SCENARIOS) + SCENARIOS.index(scenario)
    start = _TASKSET_ID_OFFSET[taskset] + combo * 1_000_000
    return np.arange(start, start + n, dtype=np.int64)


def get_tomo_bin_edges(
    key: str,
) -> np.ndarray:
    """Get the tomographic bin edges to run for a particular
    taskset / simulation / scenario combination
    """
    taskset = key[0:9]
    return TOMO_BIN_EDGES[taskset]


def _assign_tomo_bins(zmode: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """0-indexed bins from zmode; -1 for objects outside the edge range."""
    zb = np.squeeze(np.asarray(zmode, dtype=float))
    raw = np.digitize(zb, edges)
    tomo = raw.astype(np.int32) - 1
    tomo[raw == 0] = -1
    tomo[raw == len(edges)] = -1
    return tomo


def _write_bhat(
    object_id: np.ndarray,
    tomo_bin_index: np.ndarray,
    output_bhat_file: str | Path,
) -> None:
    output_bhat_file = Path(output_bhat_file)
    output_bhat_file.parent.mkdir(parents=True, exist_ok=True)
    tables_io.write(
        dict(
            object_id=np.asarray(object_id),
            tomo_bin_index=np.asarray(tomo_bin_index, dtype=np.int32),
        ),
        str(output_bhat_file),
    )


def _copy_if_needed(src: str | Path, dest: str | Path) -> None:
    src = Path(src)
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() != dest.resolve():
        shutil.copy2(src, dest)


def run_training(
    key: str,
    wfd_file: str | Path,
    ddf_files: list[str | Path],
    models_dir: str | Path,
) -> None:
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    model_file = models_dir / f"nz_challenge_{key}_pz_model.pkl"
    ddf_file = str(ddf_files[0])

    ddf_file_cleaned = ddf_file.replace(".hdf5", "_cleaned.hdf5")
    ddf_data = tables_io.read(ddf_file)
    ddf_data["redshift"] = np.where(
        np.isfinite(ddf_data["redshift"]),
        ddf_data["redshift"],
        ddf_data["redshift_manyband"],
    )
    tables_io.write(ddf_data, ddf_file_cleaned)

    informer = FlexZBoostInformer.make_stage(
        name=f"inform_{key}",
        model=str(model_file),
        hdf5_groupname="",
        zmin=0.0,
        zmax=3.0,
        nzbins=301,
        nondetect_val=np.nan,
    )
    train_handle = TableHandle(f"input_{key}", path=ddf_file_cleaned)
    informer.inform(train_handle)


def run_estimation(
    key: str,
    wfd_file: str | Path,
    models_dir: str | Path,
    output_nz_estimate_file: str | Path,
    output_bhat_file: str | Path,
    output_nz_samples_file: str | Path,
) -> None:
    tomo_bin_edges = get_tomo_bin_edges(key)
    n_tomo_bins = len(tomo_bin_edges) - 1
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    model_file = models_dir / f"nz_challenge_{key}_pz_model.pkl"
    pz_file = models_dir / f"nz_challenge_{key}_pz_estimates.hdf5"

    estimator = FlexZBoostEstimator.make_stage(
        name=f"estimate_{key}",
        model=str(model_file),
        hdf5_groupname="",
        id_col="object_id",
        nzbins=301,
        zmin=0.0,
        zmax=3.0,
        chunk_size=10000,
        nondetect_val=np.nan,
        calculated_point_estimates=["zmode"],
        qp_representation="interp",
    )

    summarizer = NaiveStackMaskedSummarizer.make_stage(
        name=f"summarize_{key}",
        selected_bin=0,
        n_tomo_bins=n_tomo_bins,
        n_samples=100,
        chunk_size=10000,
        zmin=0.0,
        zmax=3.0,
        nzbins=301,
    )

    test_handle = TableHandle(f"test_{key}", path=str(wfd_file))
    pz_estimates = estimator.estimate(test_handle)
    pz_estimates.data = None
    _copy_if_needed(pz_estimates.path, pz_file)
    pz_estimates = QPHandle(f"output_{key}", path=str(pz_file))

    wfd_data = tables_io.read(str(wfd_file))
    pz_ens = qp.read(str(pz_file))
    zmode = np.squeeze(pz_ens.ancil["zmode"])
    tomo_bin_index = _assign_tomo_bins(zmode, tomo_bin_edges)
    n_obj = int(np.shape(zmode)[0])
    object_id = _validator_object_ids(key, n_obj)
    if n_obj != len(wfd_data["object_id"]):
        raise RuntimeError(
            f"p(z) length {n_obj} != WFD length {len(wfd_data['object_id'])}"
        )
    _write_bhat(object_id, tomo_bin_index, output_bhat_file)
    bin_assignments = TableHandle(f"bhat_{key}", path=str(output_bhat_file))

    samples_nz = summarizer.summarize(pz_estimates, bin_assignments)
    single_nz = summarizer.get_handle("single_NZ")
    if output_nz_samples_file is not None:
        _copy_if_needed(samples_nz.path, output_nz_samples_file)
    _copy_if_needed(single_nz.path, output_nz_estimate_file)


def copy_samples_files(
    input_nz_samples_file: str | Path,
    output_nz_samples_file: str | Path,
) -> None:
    src = Path(input_nz_samples_file)
    dest = Path(output_nz_samples_file)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() != dest.resolve():
        shutil.copy2(src, dest)


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
        str(output_nz_samples_file).replace("taskset_3", "taskset_2")
    )
    if not input_nz_samples_file.exists():
        run_taskset_2_estimation_only(
            key.replace("taskset_3", "taskset_2"),
            str(wfd_file).replace("taskset_3", "taskset_2"),
            models_dir,
            str(output_nz_estimate_file).replace("taskset_3", "taskset_2"),
            str(output_bhat_file).replace("taskset_3", "taskset_2"),
            str(output_nz_samples_file).replace("taskset_3", "taskset_2"),
        )
    copy_samples_files(input_nz_samples_file, output_nz_samples_file)


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
        str(output_nz_samples_file).replace("taskset_3", "taskset_2")
    )
    if not input_nz_samples_file.exists():
        run_taskset_2_training_and_estimation(
            key.replace("taskset_3", "taskset_2"),
            str(wfd_file).replace("taskset_3", "taskset_2"),
            models_dir,
            [str(val).replace("taskset_3", "taskset_2") for val in ddf_files],
            str(output_nz_estimate_file).replace("taskset_3", "taskset_2"),
            str(output_bhat_file).replace("taskset_3", "taskset_2"),
            str(output_nz_samples_file).replace("taskset_3", "taskset_2"),
        )
    copy_samples_files(input_nz_samples_file, output_nz_samples_file)
