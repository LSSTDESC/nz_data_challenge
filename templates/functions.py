from pathlib import Path

try:
    import tables_io
    # Put other imports here, that way if they fail
    # the SUBMISSION_NAME, SUBMISSION_URL and MODEL_URL
    # still get set
    IMPORTS_OK = True
except ImportError:
    IMPORTS_OK = False
   

from .utils import TOMO_BIN_EDGES

# Change these to match the name of the submission
# and a URL to download the sumission data files
# and needed model files
SUBMISSION_NAME: str = ""
SUBMISSION_URL: str = ""
MODEL_URL: str = ""


def get_tomo_bin_edges(
    key: str,
) -> np.ndarray:
    """Get the tomographic bin edges to run for a particular 
    taskset / simulation / scenario combination
    """
    taskset = key[0:9]    
    tomo_bin_edges = TOMO_BIN_EDGES[taskset]
    return tomo_bin_edges
    

def run_taskset_1_estimation_only(
    key: str,
    wfd_file: str | Path,
    models_dir: str | Path,
    output_nz_estimate_file: str | Path,
    output_bhat_file: str | Path,
    output_nz_samples_file: str | Path | None,    
) -> None:    
    

def run_taskset_2_estimation_only(
    key: str,
    wfd_file: str | Path,
    models_dir: str | Path,
    output_nz_estimate_file: str | Path,
    output_bhat_file: str | Path,
    output_nz_samples_file: str | Path | None,    
) -> None:


def run_taskset_3_estimation_only(
    key: str,
    wfd_file: str | Path,
    models_dir: str | Path,
    output_nz_estimate_file: str | Path,
    output_bhat_file: str | Path,
    output_nz_samples_file: str | Path,    
) -> None:


def run_taskset_1_training_and_estimation(
    key: str,
    wfd_file: str | Path,
    models_dir: str | Path,
    ddf_files: list[str | Path],
    output_nz_estimate_file: str | Path,
    output_bhat_file: str | Path,
    output_nz_samples_file: str | Path | None,     
) -> None:


def run_taskset_2_training_and_estimation(
    key: str,
    wfd_file: str | Path,
    models_dir: str | Path,
    ddf_files: list[str | Path],
    output_nz_estimate_file: str | Path,
    output_bhat_file: str | Path,
    output_nz_samples_file: str | Path | None,    
    
) -> None:


def run_taskset_3_training_and_estimation(
    key: str,
    wfd_file: str | Path,
    models_dir: str | Path,
    ddf_files: list[str | Path],
    output_nz_estimate_file: str | Path,
    output_bhat_file: str | Path,
    output_nz_samples_file: str | Path,    
) -> None:
