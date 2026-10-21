"""Aionplus (Pontifex v6.0.0) submission module for the NZ Data Challenge.

Exposes all submission functions, metadata, and runner logic for the aionplus
submission entry.
"""

from .pontifex import (
    IMPORTS_OK,
    KEY_OFFSETS,
    PHYSICAL_LIMITS,
    compute_som_density_weights,
    extract_features,
    predict_and_generate_submission,
    run_taskset_1_estimation_only,
    run_taskset_1_training_and_estimation,
    run_taskset_2_estimation_only,
    run_taskset_2_training_and_estimation,
    run_taskset_3_estimation_only,
    run_taskset_3_training_and_estimation,
    train_pontifex_pipeline,
)

SUBMISSION_NAME: str = "aionplus"
SUBMISSION_URL: str = (
    "https://github.com/mardom/nz_data_challenge/releases/download/v6.0.0-aionplus/submit_aionplus.tgz"
)
MODEL_URL: str = (
    "https://github.com/mardom/nz_data_challenge/releases/download/v6.0.0-aionplus/submit_aionplus_models.tgz"
)

__all__ = [
    "SUBMISSION_NAME",
    "SUBMISSION_URL",
    "MODEL_URL",
    "IMPORTS_OK",
    "PHYSICAL_LIMITS",
    "KEY_OFFSETS",
    "extract_features",
    "compute_som_density_weights",
    "train_pontifex_pipeline",
    "predict_and_generate_submission",
    "run_taskset_1_estimation_only",
    "run_taskset_2_estimation_only",
    "run_taskset_3_estimation_only",
    "run_taskset_1_training_and_estimation",
    "run_taskset_2_training_and_estimation",
    "run_taskset_3_training_and_estimation",
]
