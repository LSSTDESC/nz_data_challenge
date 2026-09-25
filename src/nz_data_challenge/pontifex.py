"""Pontifex submission for the Photometric Redshift Ensemble (NZ) Data Challenge."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import qp
import tables_io
import xgboost as xgb
from minisom import MiniSom
from scipy.spatial.distance import cdist

from .utils import TOMO_BIN_EDGES, Z_BIN_EDGES

# Submission Metadata
SUBMISSION_NAME: str = "pontifex"
SUBMISSION_URL: str = "https://github.com/mardom/nz_data_challenge/releases/download/5.0.0-ascention/submit_pontifex.tgz"
MODEL_URL: str = "https://github.com/mardom/nz_data_challenge/releases/download/5.0.0-ascention/submit_pontifex_models.tgz"
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


def compute_som_density_weights(
    X_train: np.ndarray,
    X_target: np.ndarray,
    n_neurons: int = 16,
    subsample_target: int = 40000,
) -> np.ndarray:
    """Compute transfer weights w(x) = P_target(x) / P_train(x) via Self-Organizing Map."""
    feat_dim = min(8, X_train.shape[1])
    mu = np.mean(X_train[:, :feat_dim], axis=0)
    std = np.std(X_train[:, :feat_dim], axis=0) + 1e-5
    
    Z_tr = (X_train[:, :feat_dim] - mu) / std
    Z_te = (X_target[:, :feat_dim] - mu) / std
    
    if len(Z_te) > subsample_target:
        rng = np.random.default_rng(42)
        Z_te_sub = Z_te[rng.choice(len(Z_te), size=subsample_target, replace=False)]
    else:
        Z_te_sub = Z_te
        
    som = MiniSom(n_neurons, n_neurons, feat_dim, sigma=1.2, learning_rate=0.4, random_seed=42)
    som.train(Z_tr[: min(len(Z_tr), 15000)], num_iteration=600, verbose=False)
    
    win_tr = np.array([som.winner(x) for x in Z_tr])
    win_te = np.array([som.winner(x) for x in Z_te_sub])
    
    idx_tr = win_tr[:, 0] * n_neurons + win_tr[:, 1]
    idx_te = win_te[:, 0] * n_neurons + win_te[:, 1]
    
    n_cells = n_neurons * n_neurons
    counts_tr = np.bincount(idx_tr, minlength=n_cells).astype(float)
    counts_te = np.bincount(idx_te, minlength=n_cells).astype(float)
    
    norm_tr = counts_tr / (counts_tr.sum() + 1e-8)
    norm_te = counts_te / (counts_te.sum() + 1e-8)
    
    eps = 1e-4
    cell_weights = (norm_te + eps) / (norm_tr + eps)
    cell_weights = np.clip(cell_weights, 0.1, 5.0)
    cell_weights /= np.mean(cell_weights)
    
    weights = cell_weights[idx_tr]
    return weights


def train_pontifex_pipeline(
    ddf_files: List[Union[str, Path]],
    key: str,
    models_dir: Union[str, Path],
    wfd_file: Optional[Union[str, Path]] = None,
) -> Tuple[xgb.XGBClassifier, np.ndarray]:
    """Train XGBoost tomographic bin classifier with SOM transfer reweighting and build empirical calibration histograms."""
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

    # Compute SOM transfer density ratio weights if WFD target data is available
    sample_weights = None
    if wfd_file is not None and Path(wfd_file).exists():
        try:
            wfd_sample = tables_io.read(wfd_file)
            X_wfd = extract_features(wfd_sample)
            sample_weights = compute_som_density_weights(X, X_wfd)
        except Exception:
            sample_weights = None

    # 1. Stratified 5-Fold Cross-Validation for Out-Of-Fold (OOF) Calibration
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred_oof = np.zeros(len(y_true), dtype=int)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_true)):
        sw_fold = sample_weights[train_idx] if sample_weights is not None else None
        clf_fold = xgb.XGBClassifier(
            n_estimators=180,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method='hist',
            device='cuda',
            random_state=42 + fold,
            eval_metric='mlogloss',
        )
        clf_fold.fit(X[train_idx], y_true[train_idx], sample_weight=sw_fold)
        probs_val = clf_fold.predict_proba(X[val_idx])
        y_pred_oof[val_idx] = np.argmax(probs_val, axis=1)

    # 2. Reconstruct empirical n(z) for each bin using genuine OUT-OF-FOLD predictions
    # This prevents in-sample memorization and models true adjacent-bin spillover
    calib_hists = []
    for k in range(n_tomo_bins):
        mask_k = (y_pred_oof == k)
        if sample_weights is not None:
            w_k = sample_weights[mask_k]
            hist_k = np.histogram(z_clean[mask_k], grid_edges, weights=w_k)[0].astype(np.float64)
        else:
            hist_k = np.histogram(z_clean[mask_k], grid_edges)[0].astype(np.float64)
        # Add Laplace smoothing to prevent log-loss divergence
        hist_k += 0.05
        hist_k /= hist_k.sum()
        calib_hists.append(hist_k)
    calib_hists = np.array(calib_hists)

    # 3. Train final production classifier on all training data
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
    clf.fit(X, y_true, sample_weight=sample_weights)

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
    """Predict tomographic bins on WFD catalog with Hybrid MoE entropy regularization and construct qp Ensembles."""
    taskset = key[0:9]
    tomo_edges = TOMO_BIN_EDGES[taskset]
    grid_edges = Z_BIN_EDGES[taskset]
    n_tomo_bins = len(tomo_edges) - 1

    wfd_data = tables_io.read(wfd_file)
    n_objects = len(wfd_data[list(wfd_data.keys())[0]])

    X_wfd = extract_features(wfd_data)
    probs_wfd = clf.predict_proba(X_wfd)

    # Hybrid MoE entropy regularization for high-uncertainty boundary objects
    entropy = -np.sum(probs_wfd * np.log(np.maximum(probs_wfd, 1e-12)), axis=1)
    uncertain = entropy > 1.25
    probs_wfd[uncertain] = 0.88 * probs_wfd[uncertain] + 0.12 * (1.0 / n_tomo_bins)

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

    # 3. Write nz_samples file (for Taskset 3 or if requested) with correlated covariance
    if output_nz_samples_file is not None:
        n_realizations = 100
        n_grid = len(grid_edges) - 1
        z_mid = 0.5 * (grid_edges[:-1] + grid_edges[1:])
        
        # Spatial correlation kernel across redshift bins
        dist_mat = cdist(z_mid[:, None], z_mid[:, None])
        cov_mat = 0.04 * np.exp(-0.5 * (dist_mat / 0.15) ** 2) + 1e-6 * np.eye(n_grid)
        gp_chol = np.linalg.cholesky(cov_mat)
        
        rng = np.random.default_rng(42)
        realization_list = []
        for k in range(n_tomo_bins):
            alpha = calib_hists[k] * 1000.0 + 0.1
            dirichlet_draws = rng.dirichlet(alpha, size=n_realizations)
            
            # Correlated Gaussian process mode modulation
            gp_noise = rng.standard_normal(size=(n_realizations, n_grid))
            gp_modes = gp_noise @ gp_chol.T
            correlated_draws = dirichlet_draws * np.exp(gp_modes)
            correlated_draws /= np.sum(correlated_draws, axis=1, keepdims=True)
            
            for r in range(n_realizations):
                realization_list.append(correlated_draws[r])
                
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
    clf, calib_hists = train_pontifex_pipeline(ddf_files, key, models_dir, wfd_file=wfd_file)
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
    clf, calib_hists = train_pontifex_pipeline(ddf_files, key, models_dir, wfd_file=wfd_file)
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
    clf, calib_hists = train_pontifex_pipeline(ddf_files, key, models_dir, wfd_file=wfd_file)
    predict_and_generate_submission(
        key,
        wfd_file,
        clf,
        calib_hists,
        output_nz_estimate_file,
        output_bhat_file,
        output_nz_samples_file,
    )
