import os
import subprocess
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from typing import Union
import qp
import tables_io

# Set these to be correct for your submission
SUBMISSION_NAME: str="example"
SUBMISSION_URL: str="https://portal.nersc.gov/cfs/lsst/PZ/data_challenge/example_submission.tgz"

# don't change these
PUBLIC_URL: str="https://portal.nersc.gov/cfs/lsst/PZ/data_challenge/public.tgz"
SIMS = ['cardinal', 'flagship']
SCENARIOS = ['1yr', '10yr']


def run_taskset_1_estimation_only(
    model_file: str|Path,
    test_file: str|Path,
    output_file: str|Path,
) -> bool:
    """
    User supplied function to run estimation for task set 1
    
    This function should use a model stored in model_file, which
    is downloaded as part of the submission tar file.

    This function should write output data to output_file in qp
    format.  

    Parameters
    ----------
    model_file:
        Path to the model.  This should be part of the submission
        tar file.  
    test_file:
        Path to the test file contains the photometric test data on
        which the PZ estimation will be run
    output_file:
        Path to write the output data to.  The output data should
        be written in qp format.
    """
    return
    

def run_taskset_1_training_and_estimation(
    test_file: str|Path,
    output_file: str|Path,
) -> None:
    """
    User supplied function to run training and estimation for task set 1
    
    This function should train a model and use it.

    This function should write output data to output_file in qp
    format.  

    Parameters
    ----------
    test_file:
        Path to the test file contains the photometric test data on
        which the PZ estimation will be run
    output_file:
        Path to write the output data to.  The output data should
        be written in qp format.
    """    
    return
    
    
def run_taskset_2_estimation_only(
    model_file: str|Path,
    test_file: str|Path,
    output_file: str|Path,
) -> None:
    """
    User supplied function to run estimation for task set 1
    
    This function should use a model stored in model_file, which
    is downloaded as part of the submission tar file.

    This function should write output data to output_file in qp
    format.  

    Parameters
    ----------
    model_file:
        Path to the model.  This should be part of the submission
        tar file.  
    test_file:
        Path to the test file contains the photometric test data on
        which the PZ estimation will be run
    output_file:
        Path to write the output data to.  The output data should
        be written in qp format.
    """    
    return


def run_taskset_2_training_and_estimation(
    test_file: str|Path,
    output_file: str|Path,
) -> bool:
    """
    User supplied function to run training and estimation for task set 1
    
    This function should train a model and use it.

    This function should write output data to output_file in qp
    format.  

    Parameters
    ----------
    test_file:
        Path to the test file contains the photometric test data on
        which the PZ estimation will be run
    output_file:
        Path to write the output data to.  The output data should
        be written in qp format.
    """    
    return



def download_and_extract_tar(
    url: str,
    extract_to: Union[str, Path] = '.'
) -> None:
    """
    Download a tar file from a URL and extract its contents.

    Parameters
    ----------
    url : str
        URL of the tar file to download. Supports .tar, .tar.gz, .tgz,
        .tar.bz2, and .tar.xz formats.
    extract_to : str or Path, optional
        Directory path where the contents will be extracted.
        Default is the current directory ('.').

    Returns
    -------
    None

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
    # Download to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.tar') as tmp_file:
        tmp_path: str = tmp_file.name
        urllib.request.urlretrieve(url, tmp_path)
    
    try:
        # Extract with automatic format detection
        with tarfile.open(tmp_path, 'r:*') as tar:
            tar.extractall(path=extract_to, filter="data")
    finally:
        # Clean up temporary file
        os.unlink(tmp_path)


def check_pz_submission_file(
    submit_file: Union[str, Path],
    test_file: Union[str, Path]
) -> None:
    """
    Validate a photo-z submission file against test data requirements.

    This function checks that a submission file exists, is in valid qp format,
    contains required ancillary data (z_mode and object_id), and that the
    object IDs match those in the test file.

    Parameters
    ----------
    submit_file : str or Path
        Path to the submission file to validate. Must be in qp-readable format.
    test_file : str or Path
        Path to the test file containing reference object IDs. Must be readable
        by tables_io.

    Raises
    ------
    FileNotFoundError
        If submit_file does not exist.
    ValueError
        If the submission file is missing required ancillary data (z_mode or
        object_id), or if object IDs don't match the test file.
    Exception
        If the file cannot be read as a valid qp ensemble format.

    Notes
    -----
    The function performs the following checks in order:
    1. File existence
    2. Valid qp ensemble format
    3. Presence of ancillary dictionary
    4. Presence of 'z_mode' in ancillary data
    5. Presence of 'object_id' in ancillary data
    6. Matching object IDs between submission and test files
    
    """
    # Convert to Path objects for easier handling
    submit_path: Path = Path(submit_file)
    test_path: Path = Path(test_file)
    
    # Check that submit_file exists
    if not submit_path.exists():
        raise FileNotFoundError(f"Submission file not found: {submit_file}")
    
    # Open and validate qp format
    try:
        ensemble = qp.read(submit_file)
    except Exception as e:
        raise Exception(f"Failed to read submission file as qp ensemble: {e}")
    
    # Check that ancillary dict exists
    try:
        ancil = ensemble.ancil
    except AttributeError:
        raise ValueError("Submission file does not contain ancillary data")
    
    if ancil is None:
        raise ValueError("Ancillary dictionary is None")
    
    # Check for z_mode entry
    if 'z_mode' not in ancil:
        raise ValueError("Missing required ancillary entry: z_mode")
    
    # Check for object_id entry
    if 'object_id' not in ancil:
        raise ValueError("Missing required ancillary entry: object_id")
    
    # Get object IDs from submission file
    submit_ids = set(ancil['object_id'])
    
    # Get object IDs from test file
    try:
        test_data = tables_io.read(test_file)
        test_ids = set(test_data['object_id'])
    except Exception as e:
        raise Exception(f"Failed to read test file: {e}")
    
    # Check that object IDs match
    if submit_ids != test_ids:
        missing_in_submit = test_ids - submit_ids
        extra_in_submit = submit_ids - test_ids
        
        error_msg = "Object ID mismatch between submission and test files"
        if missing_in_submit:
            error_msg += f"\n  Missing {len(missing_in_submit)} IDs in submission"
        if extra_in_submit:
            error_msg += f"\n  Extra {len(extra_in_submit)} IDs in submission"
        
        raise ValueError(error_msg)

    
if __name__ == '__main__':

    submit_dir: str = f"submissions/{SUBMISSION_NAME}"

    if not os.path.exists('.pz_challenge_check_file'):
        raise RuntimeError("You must run this script from a top level pz challenge directory")

    if not os.path.exists('public'):
        download_and_extract_tar(PUBLIC_URL, '.')

    if not os.path.exists(submit_dir):
        download_and_extract_tar(SUBMISSION_URL, submit_dir)

    try:
        os.makedirs('outputs_1')
    except Exception:
        pass
        
    # Task Set 1
    for sim in SIMS:
        for scenario in SCENARIOS:
            submit_file = os.path.join(submit_dir, f"pz_challenge_taskset_1_{sim}_pz_estimate_{scenario}.hdf5")
            model_file = os.path.join(submit, f"pz_challenge_taskset_1_{sim}_pz_model_{scenario}.hdf5")        
            training_file = os.path.join('public', f"pz_challenge_taskset_1_{sim}_training_{scenario}.hdf5")        
            test_file = os.path.join('public', f"pz_challenge_taskset_1_{sim}_test_{scenario}.hdf5")
            output_file_1 = os.path.join('outputs_1', f"pz_challenge_taskset_1_{sim}_pz_estimate_{scenario}.hdf5")
            output_file_2 = os.path.join('outputs_2', f"pz_challenge_taskset_1_{sim}_pz_estimate_{scenario}.hdf5")

            check_pz_submission_file(submit_file, test_file)
            if run_taskset_1_estimation_only(model_file, test_file, output_file_1):
                check_pz_submission_file(output_file_1, test_file)

            if run_taskset_1_training_and_estimation(test_file, output_file_2):
                check_pz_submission_file(output_file_2, test_file)


    # Task Set 2
    for sim in SIMS:
        for scenario in SCENARIOS:
            submit_file = os.path.join('submit', f"pz_challenge_taskset_2_{sim}_pz_estimate_{scenario}.hdf5")
            model_file = os.path.join('submit', f"pz_challenge_taskset_2_{sim}_pz_model_{scenario}.hdf5")        
            training_file = os.path.join('public', f"pz_challenge_taskset_2_{sim}_training_{scenario}.hdf5")        
            test_file = os.path.join('public', f"pz_challenge_taskset_2_{sim}_test_{scenario}.hdf5")
            output_file_1 = os.path.join('outputs_1', f"pz_challenge_taskset_2_{sim}_pz_estimate_{scenario}.hdf5")
            output_file_2 = os.path.join('outputs_2', f"pz_challenge_taskset_2_{sim}_pz_estimate_{scenario}.hdf5")

            check_pz_submission_file(submit_file, test_file)
            if run_taskset_2_estimation_only(model_file, test_file, output_file_1):
                check_pz_submission_file(output_file_1, test_file)

            if run_taskset_2_training_and_estimation(test_file, output_file_1)
                check_pz_submission_file(output_file_1, test_file)
