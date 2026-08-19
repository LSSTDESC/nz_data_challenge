import os
from pathlib import Path
import pytest

# Put needed import here

# These are used by test scripts
from nz_data_challenge.taskset_1 import run_taskset_1
from nz_data_challenge.taskset_2 import run_taskset_2

from nz_data_challenge import submit_utils

# Change these to match the name of the submission
# and a URL to download the sumission data files
# and needed model files
SUBMISSION_NAME: str = "__SUBMISSION_NAME__"
SUBMISSION_URL: str = ""

# don't change these
SUBMIT_DIR: str = f"submissions/{SUBMISSION_NAME}"
PUBLIC_AREA: str = "tests/public"


@pytest.fixture(name="setup_submit_area", scope="module")
def setup_submit_area(request: pytest.FixtureRequest) -> int:
    """
    A pytest fixture to download the submission data

    If all the submission data are in a tar file with the
    proper structure you should not need to change this function.
    """
    
    if not os.path.exists(SUBMIT_DIR):
        if not SUBMISSION_URL:
            raise ValueError(f"SUBMISSION_URL in tests/test_{SUBMISSION_NAME}.py has not been set")
        submit_utils.download_and_extract_tar(SUBMISSION_URL, SUBMIT_DIR)

    def teardown_submit_area() -> None:
        if not os.environ.get("NO_TEARDOWN"):
            os.system(f"\\rm -rf {SUBMIT_DIR}")

    request.addfinalizer(teardown_submit_area)

    return 0


def run_taskset_1_estimation_only(
    key: str,
    wfd_file: str | Path,
    models_dir: str | Path,
    output_nz_estimate_file: str | Path,
    output_bhat_file: str | Path,
) -> None:
    return


def run_taskset_2_estimation_only(
    key: str,
    wfd_file: str | Path,
    models_dir: str | Path,
    output_nz_estimate_file: str | Path,
    output_bhat_file: str | Path,
) -> None:
    return


def run_taskset_3_estimation_only(
    key: str,
    wfd_file: str | Path,
    models_dir: str | Path,
    output_nz_estimate_file: str | Path,
    output_nz_samples_file: str | Path,
    output_bhat_file: str | Path,
) -> None:
    return


def run_taskset_1_training_and_estimation(
    key: str,
    wfd_file: str | Path,
    models_dir: str | Path,
    ddf_files: list[str | Path],
    output_nz_estimate_file: str | Path,
    output_bhat_file: str | Path,
) -> None:
    return


def run_taskset_2_training_and_estimation(
    key: str,
    wfd_file: str | Path,
    models_dir: str | Path,
    ddf_files: list[str | Path],    
    output_nz_estimate_file: str | Path,
    output_bhat_file: str | Path,
) -> None:
    return


def run_taskset_3_training_and_estimation(
    key: str,
    wfd_file: str | Path,
    models_dir: str | Path,
    ddf_files: list[str | Path],    
    output_nz_estimate_file: str | Path,
    output_nz_samples_file: str | Path,
    output_bhat_file: str | Path,
) -> None:
    return


def test___SUBMISSION_NAME___submit_taskset_1(
    setup_submit_area: int,
) -> None:
    """
    Test fuction to validate a submisson
    """    
    assert setup_submit_area == 0
    submit_utils.check_submission(SUBMIT_DIR, 'taskset_1')
    

def test___SUBMISSION_NAME___submit_taskset_2(
    setup_submit_area: int,
) -> None:
    """
    Test fuction to validate a submisson
    """    
    assert setup_submit_area == 0
    submit_utils.check_submission(SUBMIT_DIR, 'taskset_2')


def test___SUBMISSION_NAME___submit_taskset_3(
    setup_submit_area: int,
) -> None:
    """
    Test fuction to validate a submisson
    """    
    assert setup_submit_area == 0
    submit_utils.check_submission(SUBMIT_DIR, 'taskset_3')


    

    
