import os
from pathlib import Path
import pytest

# These are used by test scripts
from nz_data_challenge import submit_utils

from nz_data_challenge.example import (
    SUBMISSION_NAME,
    SUBMISSION_URL,
)

try:
    from nz_data_challenge.example import (
        run_taskset_1_estimation_only,
        run_taskset_2_estimation_only,
        run_taskset_3_estimation_only,
        run_taskset_1_training_and_estimation,
        run_taskset_2_training_and_estimation,
        run_taskset_3_training_and_estimation,
        MODEL_URL
    )
    FUNCTION_IMPORTS = True
except ImportError:
    FUNCTION_IMPORTS = False


# don't change these
SUBMIT_DIR: str = f"submission/{SUBMISSION_NAME}"
MODEL_DIR: str = f"models/{SUBMISSION_NAME}"
PUBLIC_AREA: str = "public"


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


@pytest.fixture(name="setup_model_area", scope="module")
def setup_model_area(request: pytest.FixtureRequest) -> int:
    """
    A pytest fixture to download the model data

    If all the model data are in a tar file with the
    proper structure you should not need to change this function.
    """

    if not os.path.exists(MODEL_DIR):
        if not MODEL_URL:
            raise ValueError(f"MODEL_URL in tests/test_{SUBMISSION_NAME}.py has not been set")
        submit_utils.download_and_extract_tar(MODEL_URL, MODEL_DIR)

    def teardown_model_area() -> None:
        if not os.environ.get("NO_TEARDOWN"):
            os.system(f"\\rm -rf {MODEL_DIR}")

    request.addfinalizer(teardown_model_area)

    return 0


def test_example_submit_taskset_1(
    setup_submit_area: int,
) -> None:
    """
    Test fuction to validate a submisson
    """
    assert setup_submit_area == 0
    submit_utils.check_submission(SUBMIT_DIR, ['taskset_1'])


def test_example_submit_taskset_2(
    setup_submit_area: int,
) -> None:
    """
    Test fuction to validate a submisson
    """
    assert setup_submit_area == 0
    submit_utils.check_submission(SUBMIT_DIR, ['taskset_2'])


def test_example_submit_taskset_3(
    setup_submit_area: int,
) -> None:
    """
    Test fuction to validate a submisson
    """
    assert setup_submit_area == 0
    submit_utils.check_submission(SUBMIT_DIR, ['taskset_3'])


def test_example_estimate_only_taskset_1(
    setup_model_area: int,
) -> None:
    """
    Test fuction to validate a submisson
    """
    assert setup_model_area == 0
    if not FUNCTION_IMPORTS:
        raise RuntimeError(f"Test functions not imported for {SUBMISSION_NAME}")
    test_submit_area = f"submission_test/{SUBMISSION_NAME}"
    submit_utils.estimate_only(
        run_taskset_1_estimation_only,
        PUBLIC_AREA,
        test_submit_area,
        MODEL_DIR,
        'taskset_1',
    )
    submit_utils.check_submission(test_submit_area, ['taskset_1'])


def test_example_estimate_only_taskset_2(
    setup_model_area: int,
) -> None:
    """
    Test fuction to validate a submisson
    """
    assert setup_model_area == 0
    if not FUNCTION_IMPORTS:
        raise RuntimeError(f"Test functions not imported for {SUBMISSION_NAME}")
    test_submit_area = f"submission_test/{SUBMISSION_NAME}"
    submit_utils.estimate_only(
        run_taskset_2_estimation_only,
        PUBLIC_AREA,
        test_submit_area,
        MODEL_DIR,
        'taskset_2',
    )
    submit_utils.check_submission(test_submit_area, ['taskset_2'])


def test_example_estimate_only_taskset_3(
    setup_model_area: int,
) -> None:
    """
    Test fuction to validate a submisson
    """
    assert setup_model_area == 0
    if not FUNCTION_IMPORTS:
        raise RuntimeError(f"Test functions not imported for {SUBMISSION_NAME}")
    test_submit_area = f"submission_test/{SUBMISSION_NAME}"
    submit_utils.estimate_only(
        run_taskset_3_estimation_only,
        PUBLIC_AREA,
        test_submit_area,
        MODEL_DIR,
        'taskset_3',
    )
    submit_utils.check_submission(test_submit_area, ['taskset_3'])


def test_example_train_and_estimate_taskset_1(
) -> None:
    """
    Test fuction to validate a submisson
    """
    if not FUNCTION_IMPORTS:
        raise RuntimeError(f"Test functions not imported for {SUBMISSION_NAME}")
    test_submit_area = f"submission_test2/{SUBMISSION_NAME}"
    test_model_area = f"models_test2/{SUBMISSION_NAME}"
    submit_utils.train_and_estimate(
        run_taskset_1_training_and_estimation,
        PUBLIC_AREA,
        test_submit_area,
        test_model_area,
        'taskset_1',
    )
    submit_utils.check_submission(test_submit_area, ['taskset_1'])


def test_example_train_and_estimate_taskset_2(
    setup_model_area: int,
) -> None:
    """
    Test fuction to validate a submisson
    """
    if not FUNCTION_IMPORTS:
        raise RuntimeError(f"Test functions not imported for {SUBMISSION_NAME}")
    test_submit_area = f"submission_test2/{SUBMISSION_NAME}"
    test_model_area = f"models_test2/{SUBMISSION_NAME}"
    submit_utils.train_and_estimate(
        run_taskset_2_training_and_estimation,
        PUBLIC_AREA,
        test_submit_area,
        test_model_area,
        'taskset_2',
    )
    submit_utils.check_submission(test_submit_area, ['taskset_2'])


def test_example_train_and_estimate_taskset_3(
    setup_model_area: int,
) -> None:
    """
    Test fuction to validate a submisson
    """
    if not FUNCTION_IMPORTS:
        raise RuntimeError(f"Test functions not imported for {SUBMISSION_NAME}")
    test_submit_area = f"submission_test2/{SUBMISSION_NAME}"
    test_model_area = f"models_test2/{SUBMISSION_NAME}"
    submit_utils.train_and_estimate(
        run_taskset_3_training_and_estimation,
        PUBLIC_AREA,
        test_submit_area,
        test_model_area,
        'taskset_3',
    )
    submit_utils.check_submission(test_submit_area, ['taskset_3'])
