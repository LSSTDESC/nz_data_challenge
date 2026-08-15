"""Utilities for downloading and validating photo-z submissions."""

import os
import tarfile
import tempfile
import urllib.request
from pathlib import Path

import qp
import tables_io

from .utils import TASKSETS, SIMS, SCENARIOS


def download_and_extract_tar(url: str, extract_to: str | Path = ".") -> None:
    """Download a tar file from a URL and extract its contents.

    Parameters
    ----------
    url
        URL of the tar file to download. Supports .tar, .tar.gz, .tgz,
        .tar.bz2, and .tar.xz formats.
    extract_to
        Directory path where the contents will be extracted.
        Default is the current directory ('.').

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
        urllib.request.urlretrieve(url, tmp_path)

    try:
        with tarfile.open(tmp_path, "r:*") as tar:
            tar.extractall(path=extract_to, filter="data")
    finally:
        os.unlink(tmp_path)


def check_files(
    nz_file: str | Path,
    bhat_file: str | Path,
    test_file: str | Path,
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
    test_file
        Path to the test file containing reference object IDs. Must be
        readable by tables_io.
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

    try:
        test_data = tables_io.read(test_file)
        test_ids = set(test_data["object_id"])
    except Exception as exc:
        raise RuntimeError(
            f"Failed to read test file {test_file}: {exc}"
        ) from exc

    if submit_ids != test_ids:
        diff_set = submit_ids - test_ids
        raise RuntimeError(
            f"Object ids in bhat (bin assignment) file {bhat_file}"
            f" do not match {test_file}: {diff_set}"
        )


def check_submission(
    public_dir: str | Path,
    submit_dir: str | Path,
) -> None:
    """Validate all files in a submission against the public test data.

    Iterates over all taskset, simulation, and scenario combinations,
    checking that n(z) and bin assignment files pass validation.

    Parameters
    ----------
    public_dir
        Path to the public data directory containing test WFD files.
    submit_dir
        Path to the submission directory containing n(z) and bhat files.

    Raises
    ------
    RuntimeError
        If any file fails validation checks performed by check_files.
    """
    for taskset in TASKSETS:
        for sim in SIMS:
            for scenario in SCENARIOS:
                wfd_file = f"{public_dir}/nz_challenge_{taskset}_{sim}_{scenario}_wfd.hdf5"
                nz_file = f"{submit_dir}/nz_challenge_{taskset}_{sim}_{scenario}_nz_estimate_wfd.hdf5"
                bhat_file = f"{submit_dir}/nz_challenge_{taskset}_{sim}_{scenario}_bhat_wfd.hdf5"
                check_files(nz_file, bhat_file, wfd_file, 5)
