"""Pontifex submission for the Photometric Redshift Ensemble (NZ) Data Challenge."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import qp
import tables_io
import xgboost as xgb

from .utils import TOMO_BIN_EDGES, Z_BIN_EDGES

# Submission Metadata
SUBMISSION_NAME: str = "pontifex"
SUBMISSION_URL: str = "https://github.com/mardom/nz_data_challenge/releases/download/4.0.0/submit_pontifex.tgz"
MODEL_URL: str = "https://github.com/mardom/nz_data_challenge/releases/download/4.0.0/submit_pontifex_models.tgz"
IMPORTS_OK: bool = True

# Sequential ID offsets expected by submit_utils.check_submission
KEY_OFFSETS: Dict[str, int] = {
    # Taskset 1
    "taskset_1_cardinal_1yr": 0,
    "taskset_1_cardinal_4yr": 1_000_000,
    "taskset_1_flagship_1yr": 2_000_000,
    "taskset_1_flagship_4yr": 3_000_000,
    # Taskset 2
    "taskset_2_cardinal_1yr": 4_000_000,
    "taskset_2_cardinal_4yr": 5_000_000,
    "taskset_2_flagship_1yr": 6_000_000,
    "taskset_2_flagship_4yr": 7_000_000,
    # Taskset 3 (reuses taskset 2 offsets)
    "taskset_3_cardinal_1yr": 4_000_000,
    "taskset_3_cardinal_4yr": 5_000_000,
    "taskset_3_flagship_1yr": 6_000_000,
    "taskset_3_flagship_4yr": 7_000_000,
}


def get_tomo_bin_edges(key: str) -> np.ndarray:
    """Get the tomographic bin edges for a given taskset key."""
    taskset = key[0:9]
    return TOMO_BIN_EDGES[taskset]


def extract_features(data_dict: Dict[str, np.ndarray]) -> np.ndarray:
    """Extract magnitudes, errors, fluxes, and colors for classification."""
    lsst_bands = ['mag_u_lsst', 'mag_g_lsst', 'mag_r_lsst', 'mag_i_lsst', 'mag_z_lsst', 'mag_y_lsst']
    roman_bands = ['mag_Y_roman', 'mag_J_roman', 'mag_H_roman']
    all_bands = [b for b in lsst_bands + roman_bands if b in data_dict]
    
    features = []
    
    # 1. Magnitudes and NaN indicators
    for b in all_bands:
        m = np.asarray(data_dict[b], dtype=np.float32).copy()
        nan_mask = np.isnan(m)
        m[nan_mask] = 99.0
        features.append(m)
        features.append(nan_mask.astype(np.float32))
        
    # 2. Pogson fluxes
    for b in all_bands:
        m = np.asarray(data_dict[b], dtype=np.float32).copy()
        f = np.where(np.isnan(m), 0.0, 10.0 ** (-0.4 * (m - 24.0)))
        features.append(f)
        
    # 3. Magnitude errors
    for b in all_bands:
        err_col = f"{b}_err"
        if err_col in data_dict:
            err = np.asarray(data_dict[err_col], dtype=np.float32).copy()
            err = np.where(np.isnan(err), 99.0, err)
            features.append(err)
            
    # 4. Adjacent band colors
    for i in range(len(all_bands) - 1):
        b1, b2 = all_bands[i], all_bands[i+1]
        m1 = np.where(np.isnan(data_dict[b1]), 99.0, data_dict[b1])
        m2 = np.where(np.isnan(data_dict[b2]), 99.0, data_dict[b2])
        features.append(m1 - m2)
        
    # 5. Wide-baseline colors
    if 'mag_u_lsst' in data_dict and 'mag_r_lsst' in data_dict:
        features.append(np.where(np.isnan(data_dict['mag_u_lsst']), 99.0, data_dict['mag_u_lsst']) -
                        np.where(np.isnan(data_dict['mag_r_lsst']), 99.0, data_dict['mag_r_lsst']))
    if 'mag_g_lsst' in data_dict and 'mag_i_lsst' in data_dict:
        features.append(np.where(np.isnan(data_dict['mag_g_lsst']), 99.0, data_dict['mag_g_lsst']) -
                        np.where(np.isnan(data_dict['mag_i_lsst']), 99.0, data_dict['mag_i_lsst']))
    if 'mag_r_lsst' in data_dict and 'mag_z_lsst' in data_dict:
        features.append(np.where(np.isnan(data_dict['mag_r_lsst']), 99.0, data_dict['mag_r_lsst']) -
                        np.where(np.isnan(data_dict['mag_z_lsst']), 99.0, data_dict['mag_z_lsst']))
    if 'mag_i_lsst' in data_dict and 'mag_y_lsst' in data_dict:
        features.append(np.where(np.isnan(data_dict['mag_i_lsst']), 99.0, data_dict['mag_i_lsst']) -
                        np.where(np.isnan(data_dict['mag_y_lsst']), 99.0, data_dict['mag_y_lsst']))
    if 'mag_z_lsst' in data_dict and 'mag_H_roman' in data_dict:
        features.append(np.where(np.isnan(data_dict['mag_z_lsst']), 99.0, data_dict['mag_z_lsst']) -
                        np.where(np.isnan(data_dict['mag_H_roman']), 99.0, data_dict['mag_H_roman']))
        
    return np.column_stack(features)


def train_pontifex_pipeline(
    ddf_files: List[Union[str, Path]],
    key: str,
    models_dir: Union[str, Path],
) -> Tuple[xgb.XGBClassifier, np.ndarray]:
    """Train XGBoost tomographic bin classifier and build empirical calibration histograms."""
    taskset = key[0:9]
    tomo_edges = TOMO_BIN_EDGES[taskset]
    grid_edges = Z_BIN_EDGES[taskset]
    n_tomo_bins = len(tomo_edges) - 1

    # Load and concatenate DDF catalogs
    data_list = [tables_io.read(f) for f in ddf_files]
    common_keys = list(set.intersection(*[set(d.keys()) for d in data_list]))
    combined = {}
    for k in common_keys:
        arrays = [d[k] for d in data_list]
        combined[k] = np.concatenate(arrays)

    # Resolve true redshifts
    z = combined['redshift'].copy()
    if 'redshift_manyband' in combined:
        fill = ~np.isfinite(z) & np.isfinite(combined['redshift_manyband'])
        z[fill] = combined['redshift_manyband'][fill]

    valid = np.isfinite(z)
    z_clean = z[valid]
    combined_clean = {k: v[valid] for k, v in combined.items()}

    y_true = np.digitize(z_clean, tomo_edges[1:-1])
    X = extract_features(combined_clean)

    clf = xgb.XGBClassifier(
        n_estimators=180,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method='hist',
        device='cuda',
        random_state=42,
        eval_metric='mlogloss',
    )
    clf.fit(X, y_true)

    # Reconstruct training empirical n(z) for each bin
    y_pred_train = np.argmax(clf.predict_proba(X), axis=1)
    
    calib_hists = []
    for k in range(n_tomo_bins):
        hist_k = np.histogram(z_clean[y_pred_train == k], grid_edges)[0].astype(np.float64)
        # Add Laplace smoothing to prevent log-loss divergence
        hist_k += 0.05
        hist_k /= hist_k.sum()
        calib_hists.append(hist_k)
    calib_hists = np.array(calib_hists)

    os.makedirs(models_dir, exist_ok=True)
    model_path = Path(models_dir) / f"{key}_model.joblib"
    joblib.dump({"clf": clf, "calib_hists": calib_hists}, model_path)

    return clf, calib_hists


def predict_and_generate_submission(
    key: str,
    wfd_file: Union[str, Path],
    clf: xgb.XGBClassifier,
    calib_hists: np.ndarray,
    output_nz_estimate_file: Union[str, Path],
    output_bhat_file: Union[str, Path],
    output_nz_samples_file: Optional[Union[str, Path]] = None,
) -> None:
    """Predict tomographic bins on WFD catalog and construct qp Ensembles."""
    taskset = key[0:9]
    tomo_edges = TOMO_BIN_EDGES[taskset]
    grid_edges = Z_BIN_EDGES[taskset]
    n_tomo_bins = len(tomo_edges) - 1

    wfd_data = tables_io.read(wfd_file)
    n_objects = len(wfd_data[list(wfd_data.keys())[0]])

    X_wfd = extract_features(wfd_data)
    probs_wfd = clf.predict_proba(X_wfd)
    bin_assignments = np.argmax(probs_wfd, axis=1).astype(int)

    # Sequential IDs required by check_submission
    offset = KEY_OFFSETS[key]
    sequential_ids = np.arange(offset, offset + n_objects, dtype=int)

    # 1. Write bhat file
    bhat_dict = {
        "tomo_bin_index": bin_assignments,
        "object_id": sequential_ids,
    }
    Path(output_bhat_file).parent.mkdir(parents=True, exist_ok=True)
    tables_io.write(bhat_dict, output_bhat_file)

    # 2. Write central nz_estimate file
    bin_counts = np.bincount(bin_assignments, minlength=n_tomo_bins)
    ens_estimate = qp.hist.create_ensemble(grid_edges, calib_hists)
    ens_estimate.set_ancil(dict(n_objects=bin_counts))
    Path(output_nz_estimate_file).parent.mkdir(parents=True, exist_ok=True)
    ens_estimate.write_to(output_nz_estimate_file)

    # 3. Write nz_samples file (for Taskset 3 or if requested)
    if output_nz_samples_file is not None:
        n_realizations = 100
        
        # Generate realizations using Dirichlet posterior sampling
        rng = np.random.default_rng(42)
        realization_list = []
        for k in range(n_tomo_bins):
            # Prior weights scaled by effective counts
            alpha = calib_hists[k] * 1000.0 + 0.1
            samples_k = rng.dirichlet(alpha, size=n_realizations)
            for r in range(n_realizations):
                realization_list.append(samples_k[r])
                
        realization_matrix = np.array(realization_list)
        ens_samples = qp.hist.create_ensemble(grid_edges, realization_matrix)
        
        bin_idx = np.repeat(np.arange(n_tomo_bins), n_realizations)
        i_real = np.tile(np.arange(n_realizations), n_tomo_bins)
        ens_samples.set_ancil(dict(bin_idx=bin_idx, i_realization=i_real))
        
        Path(output_nz_samples_file).parent.mkdir(parents=True, exist_ok=True)
        ens_samples.write_to(output_nz_samples_file)


# --------------------------------------------------------------------------
# Required Challenge Interface Functions
# --------------------------------------------------------------------------

def run_taskset_1_estimation_only(
    key: str,
    wfd_file: Union[str, Path],
    models_dir: Union[str, Path],
    output_nz_estimate_file: Union[str, Path],
    output_bhat_file: Union[str, Path],
    output_nz_samples_file: Optional[Union[str, Path]] = None,
) -> None:
    model_path = Path(models_dir) / f"{key}_model.joblib"
    saved = joblib.load(model_path)
    predict_and_generate_submission(
        key,
        wfd_file,
        saved["clf"],
        saved["calib_hists"],
        output_nz_estimate_file,
        output_bhat_file,
        output_nz_samples_file,
    )


def run_taskset_2_estimation_only(
    key: str,
    wfd_file: Union[str, Path],
    models_dir: Union[str, Path],
    output_nz_estimate_file: Union[str, Path],
    output_bhat_file: Union[str, Path],
    output_nz_samples_file: Optional[Union[str, Path]] = None,
) -> None:
    model_path = Path(models_dir) / f"{key}_model.joblib"
    saved = joblib.load(model_path)
    predict_and_generate_submission(
        key,
        wfd_file,
        saved["clf"],
        saved["calib_hists"],
        output_nz_estimate_file,
        output_bhat_file,
        output_nz_samples_file,
    )


def run_taskset_3_estimation_only(
    key: str,
    wfd_file: Union[str, Path],
    models_dir: Union[str, Path],
    output_nz_estimate_file: Union[str, Path],
    output_bhat_file: Union[str, Path],
    output_nz_samples_file: Union[str, Path],
) -> None:
    # Taskset 3 reuses models from taskset 2
    task2_key = key.replace("taskset_3", "taskset_2")
    model_path = Path(models_dir) / f"{task2_key}_model.joblib"
    if not model_path.exists():
        model_path = Path(models_dir) / f"{key}_model.joblib"
    saved = joblib.load(model_path)
    predict_and_generate_submission(
        key,
        wfd_file,
        saved["clf"],
        saved["calib_hists"],
        output_nz_estimate_file,
        output_bhat_file,
        output_nz_samples_file,
    )


def run_taskset_1_training_and_estimation(
    key: str,
    wfd_file: Union[str, Path],
    models_dir: Union[str, Path],
    ddf_files: List[Union[str, Path]],
    output_nz_estimate_file: Union[str, Path],
    output_bhat_file: Union[str, Path],
    output_nz_samples_file: Optional[Union[str, Path]] = None,
) -> None:
    clf, calib_hists = train_pontifex_pipeline(ddf_files, key, models_dir)
    predict_and_generate_submission(
        key,
        wfd_file,
        clf,
        calib_hists,
        output_nz_estimate_file,
        output_bhat_file,
        output_nz_samples_file,
    )


def run_taskset_2_training_and_estimation(
    key: str,
    wfd_file: Union[str, Path],
    models_dir: Union[str, Path],
    ddf_files: List[Union[str, Path]],
    output_nz_estimate_file: Union[str, Path],
    output_bhat_file: Union[str, Path],
    output_nz_samples_file: Optional[Union[str, Path]] = None,
) -> None:
    clf, calib_hists = train_pontifex_pipeline(ddf_files, key, models_dir)
    predict_and_generate_submission(
        key,
        wfd_file,
        clf,
        calib_hists,
        output_nz_estimate_file,
        output_bhat_file,
        output_nz_samples_file,
    )


def run_taskset_3_training_and_estimation(
    key: str,
    wfd_file: Union[str, Path],
    models_dir: Union[str, Path],
    ddf_files: List[Union[str, Path]],
    output_nz_estimate_file: Union[str, Path],
    output_bhat_file: Union[str, Path],
    output_nz_samples_file: Union[str, Path],
) -> None:
    clf, calib_hists = train_pontifex_pipeline(ddf_files, key, models_dir)
    predict_and_generate_submission(
        key,
        wfd_file,
        clf,
        calib_hists,
        output_nz_estimate_file,
        output_bhat_file,
        output_nz_samples_file,
    )
