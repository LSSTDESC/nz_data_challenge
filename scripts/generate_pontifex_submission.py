"""Generate full submission files for Pontifex across all tasksets, sims, and scenarios."""

import os
import time

from nz_data_challenge import submit_utils
from nz_data_challenge import pontifex

PUBLIC_DIR = "public"
SUBMIT_DIR = "submission/pontifex"
MODELS_DIR = "models/pontifex"

SIMS = ["cardinal", "flagship"]
SCENARIOS = ["1yr", "4yr"]


def generate_all():
    start_time = time.time()
    os.makedirs(SUBMIT_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("================================================================================")
    print("Generating Pontifex Submission Files")
    print("================================================================================")

    # 1. TASKSET 1
    print("\n--- Processing Taskset 1 ---")
    for sim in SIMS:
        for scenario in SCENARIOS:
            key = f"taskset_1_{sim}_{scenario}"
            t0 = time.time()
            print(f"Generating {key}...")
            wfd_file = f"{PUBLIC_DIR}/nz_challenge_taskset_1_{sim}_{scenario}_wfd.hdf5"
            ddf_files = [
                f"{PUBLIC_DIR}/nz_challenge_taskset_1_{sim}_{scenario}_ddf_{i:02d}.hdf5"
                for i in range(5)
            ]
            nz_est_file = f"{SUBMIT_DIR}/nz_challenge_taskset_1_{sim}_{scenario}_nz_estimate_wfd.hdf5"
            bhat_file = f"{SUBMIT_DIR}/nz_challenge_taskset_1_{sim}_{scenario}_bhat_wfd.hdf5"
            samples_file = f"{SUBMIT_DIR}/nz_challenge_taskset_1_{sim}_{scenario}_nz_samples_wfd.hdf5"

            pontifex.run_taskset_1_training_and_estimation(
                key=key,
                wfd_file=wfd_file,
                models_dir=MODELS_DIR,
                ddf_files=ddf_files,
                output_nz_estimate_file=nz_est_file,
                output_bhat_file=bhat_file,
                output_nz_samples_file=samples_file,
            )
            print(f"Completed {key} in {time.time() - t0:.1f}s")

    # 2. TASKSET 2
    print("\n--- Processing Taskset 2 ---")
    for sim in SIMS:
        for scenario in SCENARIOS:
            key = f"taskset_2_{sim}_{scenario}"
            t0 = time.time()
            print(f"Generating {key}...")
            wfd_file = f"{PUBLIC_DIR}/nz_challenge_taskset_2_{sim}_{scenario}_wfd.hdf5"
            ddf_files = [
                f"{PUBLIC_DIR}/nz_challenge_taskset_2_{sim}_{scenario}_ddf_{i:02d}.hdf5"
                for i in range(5)
            ]
            nz_est_file = f"{SUBMIT_DIR}/nz_challenge_taskset_2_{sim}_{scenario}_nz_estimate_wfd.hdf5"
            bhat_file = f"{SUBMIT_DIR}/nz_challenge_taskset_2_{sim}_{scenario}_bhat_wfd.hdf5"
            samples_file = f"{SUBMIT_DIR}/nz_challenge_taskset_2_{sim}_{scenario}_nz_samples_wfd.hdf5"

            pontifex.run_taskset_2_training_and_estimation(
                key=key,
                wfd_file=wfd_file,
                models_dir=MODELS_DIR,
                ddf_files=ddf_files,
                output_nz_estimate_file=nz_est_file,
                output_bhat_file=bhat_file,
                output_nz_samples_file=samples_file,
            )
            print(f"Completed {key} in {time.time() - t0:.1f}s")

    # 3. TASKSET 3
    print("\n--- Processing Taskset 3 ---")
    for sim in SIMS:
        for scenario in SCENARIOS:
            key = f"taskset_3_{sim}_{scenario}"
            t0 = time.time()
            print(f"Generating {key}...")
            # Reuses taskset 2 inputs
            wfd_file = f"{PUBLIC_DIR}/nz_challenge_taskset_2_{sim}_{scenario}_wfd.hdf5"
            samples_file = f"{SUBMIT_DIR}/nz_challenge_taskset_3_{sim}_{scenario}_nz_samples_wfd.hdf5"
            nz_est_file = f"{SUBMIT_DIR}/nz_challenge_taskset_3_{sim}_{scenario}_nz_estimate_wfd.hdf5"
            bhat_file = f"{SUBMIT_DIR}/nz_challenge_taskset_3_{sim}_{scenario}_bhat_wfd.hdf5"

            # Use estimation_only which loads the trained taskset 2 model
            pontifex.run_taskset_3_estimation_only(
                key=key,
                wfd_file=wfd_file,
                models_dir=MODELS_DIR,
                output_nz_estimate_file=nz_est_file,
                output_bhat_file=bhat_file,
                output_nz_samples_file=samples_file,
            )
            print(f"Completed {key} in {time.time() - t0:.1f}s")

    print(f"\nAll files generated in {time.time() - start_time:.1f}s!")
    
    print("\n--- Validating Full Submission via submit_utils.check_submission ---")
    submit_utils.check_submission(SUBMIT_DIR, tasksets=["taskset_1", "taskset_2", "taskset_3"])
    print("\n>>> ALL VALIDATION CHECKS PASSED FOR TASKSETS 1, 2, AND 3! <<<\n")


if __name__ == "__main__":
    generate_all()
