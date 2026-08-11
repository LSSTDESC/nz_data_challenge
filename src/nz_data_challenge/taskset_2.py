"""Validation runner for task set 2 submissions."""

from . import submit_utils

SIMS = ["cardinal", "flagship"]
SCENARIOS = ["1yr", "4yr"]
N_TOMO_BINS = 5


def run_taskset_2(
    public_area: str,
    submission: str,
) -> None:
    """Run validation checks for task set 2 submissions.

    Iterates over all simulation and scenario combinations, checking
    that n(z) and bin assignment files conform to requirements.

    Parameters
    ----------
    public_area
        Path to the public data area containing test WFD files.
    submission
        Name of the submission directory under ``submissions/``.

    Raises
    ------
    RuntimeError
        If any submission file fails validation. The exception message
        contains all collected errors.
    """
    submit_dir: str = f"submissions/{submission}"
    taskset: str = 'taskset_1'
    except_list: list[Exception] = []

    for sim in SIMS:
        for scenario in SCENARIOS:

            nz_file = f"{submit_dir}/nz_challenge_{taskset}_{sim}_{scenario}_nz_estimate_wfd.hdf5"
            bhat_file = f"{submit_dir}/nz_challenge_{taskset}_{sim}_{scenario}_bhat_wfd.hdf5"
            wfd_file = f"{public_area}/nz_challenge_{taskset}_{sim}_{scenario}_wfd.hdf5"

            try:
                submit_utils.check_files(
                    nz_file, bhat_file, wfd_file, N_TOMO_BINS,
                )
            except Exception as exc:
                except_list.append(exc)

    if except_list:
        raise RuntimeError(f"Caught Exceptions {except_list}")
