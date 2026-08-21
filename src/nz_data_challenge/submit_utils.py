"""Utilities for downloading and validating photo-z submissions."""

import os
import tarfile
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import qp
import tables_io

from .utils import TASKSETS, SIMS, SCENARIOS, TOMO_BIN_EDGES

# This is here in case we can't update the data at NERSC
NERSC_IS_OUTDATED = True

# don't change this
if NERSC_IS_OUTDATED:
    PUBLIC_URL: str = "https://s3df.slac.stanford.edu/people/echarles/data_challenge/public_nz.tgz"
    BACKUP_URL: str | None = None
else:
    PUBLIC_URL: str = "https://portal.nersc.gov/cfs/lsst/PZ/data_challenge/public_nz.tgz"
    BACKUP_URL: str | None = "https://s3df.slac.stanford.edu/people/echarles/data_challenge/public_nz.tgz"


def download_and_extract_tar(
    url: str,
    extract_to: str | Path = ".",
    backup_url: str | None = None,
) -> None:
    """Download a tar file from a URL and extract its contents.

    Parameters
    ----------
    url
        URL of the tar file to download. Supports .tar, .tar.gz, .tgz,
        .tar.bz2, and .tar.xz formats.
    extract_to
        Directory path where the contents will be extracted.
        Default is the current directory ('.').
    backup_url
        if provided, a backup URL of the tar file to download. Supports .tar, .tar.gz, .tgz,
        .tar.bz2, and .tar.xz formats.

    Raises
    ------
    urllib.error.URLError
        If the download fails due to network issues or invalid URL.
    tarfile.TarError
        If the file is not a valid tar archive or extraction fails.
    PermissionError
        If there are insufficient permissions to write to the extraction
        directory or create temporary files.

    Notes
    -----
    The function automatically detects the compression format of the tar
    file. The downloaded tar file is stored in a temporary location and
    automatically deleted after extraction.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tar") as tmp_file:
        tmp_path: str = tmp_file.name
        try:
            urllib.request.urlretrieve(url, tmp_path)
        except Exception:
            if backup_url is None:
                raise
            urllib.request.urlretrieve(url, tmp_path)

    try:
        with tarfile.open(tmp_path, "r:*") as tar:
            tar.extractall(path=extract_to, filter="data")
    finally:
        os.unlink(tmp_path)


def setup_public_area() -> None:
    """
    A function download the public data
    """
    print(f"copying data from {PUBLIC_URL} (or backup: {BACKUP_URL})\n")

    if not os.path.exists("public"):
        # Note that the tar file has "public" as top level directory
        # so we if we extract to "tests" the files actually end
        # up in "tests/public"
        download_and_extract_tar(PUBLIC_URL, ".", BACKUP_URL)

        
def check_files(
    nz_file: str | Path,
    bhat_file: str | Path,
    test_ids: set,
    n_tomo_bins: int | None = None,
) -> None:
    """Validate photo-z submission files against test data requirements.

    Checks that a submission file exists, is in valid qp format, contains
    required ancillary data, and that the object IDs match those in the
    test file.

    Parameters
    ----------
    nz_file
        Path to the n(z) submission file to validate. Must be in
        qp-readable format.
    bhat_file
        Path to the bhat (bin assignment) submission file to validate.
        Must be in tables_io readable format.
    test_ids
        Set of object_ids we expect to find.
    n_tomo_bins
        If set, requires that nz_file have this many tomographic bins.

    Raises
    ------
    RuntimeError
        If validation fails.

    Notes
    -----
    The function performs the following checks in order:

    1. n(z) file existence
    2. Valid qp ensemble format
    3. Presence of ancillary dictionary and 'n_objects' in ancillary data
    4. Correct number of tomographic bins
    5. bhat file existence
    6. bhat file is tables_io readable and has 'bhat_for_wide_data' and
       'object_id' columns
    7. Matching number of objects
    8. Matching object_id columns in bhat_file against test_file
    """
    if not Path(nz_file).exists():
        raise RuntimeError(f"n(z) file {nz_file} does not exist")

    try:
        qp_ens = qp.read(nz_file)
    except Exception as exc:
        raise RuntimeError(
            f"n(z) file {nz_file} could not be read by qp because {exc}"
        ) from exc

    try:
        n_objects = qp_ens.ancil['n_objects']
    except Exception as exc:
        raise RuntimeError(
            f"n(z) file {nz_file} does not have 'n_objects' in its ancil table"
        ) from exc

    if n_tomo_bins is not None:
        if qp_ens.npdf != n_tomo_bins:
            raise RuntimeError(
                f"n(z) file {nz_file} has {qp_ens.npdf} tomo bins"
                f" with {n_tomo_bins} expected"
            )

    if not Path(bhat_file).exists():
        raise RuntimeError(
            f"bhat (bin assignment) file {bhat_file} does not exist"
        )

    try:
        bhat = tables_io.read(bhat_file)
    except Exception as exc:
        raise RuntimeError(
            f"bhat (bin assignment) file {bhat_file} could not be read"
            f" by tables_io because {exc}"
        ) from exc

    if 'tomo_bin_index' not in bhat:
        raise RuntimeError(
            f"bhat (bin assignment) file {bhat_file} does not contain"
            " 'tomo_bin_index'"
        )
    if 'object_id' not in bhat:
        raise RuntimeError(
            f"bhat (bin assignment) file {bhat_file} does not contain"
            " 'object_id'"
        )

    submit_ids = set(bhat['object_id'])

    n_assigned = (bhat['tomo_bin_index'] >= 0).sum()
    if n_assigned != n_objects.sum():
        raise RuntimeError(
            f"Number of assigned objects in {bhat_file} ({n_assigned})"
            f" != sum of n_objects in n(z) file ({n_objects.sum()})"
        )

    if submit_ids != test_ids:
        diff_set = submit_ids - test_ids
        raise RuntimeError(
            f"Object ids in bhat (bin assignment) file {bhat_file}"
            f" do not match expected test ids: {diff_set}"
        )


def check_submission(
    submit_dir: str | Path,
    tasksets: list[str]=TASKSETS,
) -> None:
    """Validate all files in a submission against the public test data.

    Iterates over all taskset, simulation, and scenario combinations,
    checking that n(z) and bin assignment files pass validation.

    Parameters
    ----------
    submit_dir
        Path to the submission directory containing n(z) and bhat files.
    tasksets:
        Which tasksets to check?

    Raises
    ------
    RuntimeError
        If any file fails validation checks performed by check_files.
    """
    OFFSETS = dict(
        taskset_1=0,
        taskset_2=4_000_000,        
    )
    for taskset in tasksets:
        id_offset = OFFSETS[taskset]
        tomo_bin_edges = TOMO_BIN_EDGES[taskset]
        n_tomo_bins = len(tomo_bin_edges) - 1
        for sim in SIMS:
            for scenario in SCENARIOS:                
                nz_file = f"{submit_dir}/nz_challenge_{taskset}_{sim}_{scenario}_nz_estimate_wfd.hdf5"
                bhat_file = f"{submit_dir}/nz_challenge_{taskset}_{sim}_{scenario}_bhat_wfd.hdf5"
                test_ids = set(np.arange(id_offset, id_offset+1_000_000).astype(int))
                check_files(nz_file, bhat_file, test_ids, n_tomo_bins)
                id_offset += 1_000_000


def estimate_only(
    the_function,
    public_dir: str | Path,
    submit_dir: str | Path,
    models_dir: str | Path,
    taskset: str | Path,
) -> None:

    for sim in SIMS:
        for scenario in SCENARIOS:
            key = f"{taskset}_{sim}_{scenario}"
            wfd_file = f"{public_dir}/nz_challenge_{taskset}_{sim}_{scenario}_wfd.hdf5"
            output_nz_estimate_file = f"{submit_dir}/nz_challenge_{taskset}_{sim}_{scenario}_nz_estimate_wfd.hdf5"
            output_nz_samples_file = f"{submit_dir}/nz_challenge_{taskset}_{sim}_{scenario}_nz_samples_wfd.hdf5"            
            output_bhat_file = f"{submit_dir}/nz_challenge_{taskset}_{sim}_{scenario}_bhat_wfd.hdf5"
            the_function(key, wfd_file, models_dir, output_nz_estimate_file, output_bhat_file, output_nz_samples_file)


def train_and_estimate(
    the_function,
    public_dir: str | Path,
    submit_dir: str | Path,
    models_dir: str | Path,
    taskset: str | Path,
) -> None:

    for sim in SIMS:
        for scenario in SCENARIOS:
            key = f"{taskset}_{sim}_{scenario}"
            wfd_file = f"{public_dir}/nz_challenge_{taskset}_{sim}_{scenario}_wfd.hdf5"
            ddf_files = [f"{public_dir}/nz_challenge_{taskset}_{sim}_{scenario}_ddf_{iddf:02}.hdf5" for iddf in range(5)]
            output_nz_estimate_file = f"{submit_dir}/nz_challenge_{taskset}_{sim}_{scenario}_nz_estimate_wfd.hdf5"
            output_nz_samples_file = f"{submit_dir}/nz_challenge_{taskset}_{sim}_{scenario}_nz_samples_wfd.hdf5"            
            output_bhat_file = f"{submit_dir}/nz_challenge_{taskset}_{sim}_{scenario}_bhat_wfd.hdf5"
            the_function(key, wfd_file, models_dir, ddf_files, output_nz_estimate_file, output_bhat_file, output_nz_samples_file)
                
        
