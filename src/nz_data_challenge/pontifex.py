"""Pontifex (aionplus v6.0.0) submission for the Photometric Redshift Ensemble (NZ) Data Challenge."""

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
SUBMISSION_URL: str = "https://github.com/mardom/nz_data_challenge/releases/download/v6.0.0-aionplus/submit_pontifex.tgz"
MODEL_URL: str = "https://github.com/mardom/nz_data_challenge/releases/download/v6.0.0-aionplus/submit_pontifex_models.tgz"
IMPORTS_OK: bool = True

PHYSICAL_LIMITS: Dict[str, float] = {
    "mag_u_lsst": 26.1,
    "mag_g_lsst": 27.4,
    "mag_r_lsst": 27.5,
    "mag_i_lsst": 26.8,
    "mag_z_lsst": 26.1,
    "mag_y_lsst": 24.9,
    "mag_Y_roman": 26.7,
    "mag_J_roman": 26.9,
    "mag_H_roman": 26.7,
}

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


def _get_xgb_device() -> str:
    """Auto-detect CUDA availability for XGBoost hist training; fallback cleanly to CPU."""
    if os.environ.get("GITHUB_ACTIONS", "").lower() == "true" or os.environ.get("NZDC_FORCE_CPU", "") == "1":
        return "cpu"
    try:
        X_tiny = np.zeros((8, 2), dtype=np.float32)
        y_tiny = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=int)
        clf_test = xgb.XGBClassifier(n_estimators=1, max_depth=1, tree_method="hist", device="cuda")
        clf_test.fit(X_tiny, y_tiny)
        return "cuda"
    except Exception:
        return "cpu"


_XGB_DEVICE = _get_xgb_device()


def get_tomo_bin_edges(key: str) -> np.ndarray:
    """Get the tomographic bin edges for a given taskset key."""
    taskset = key[0:9]
    return TOMO_BIN_EDGES[taskset]


def extract_features(data_dict: Dict[str, np.ndarray]) -> np.ndarray:
    """Extract magnitudes, errors, fluxes, and colors for tomographic classification.

    Uses FeatureGuard physical 1-sigma limiting magnitudes for non-detections
    instead of 99.0 sentinels, eliminating unphysical +/-75 mag color spikes
    and preserving standardized SOM coordinate geometry.
    """
    lsst_bands = ['mag_u_lsst', 'mag_g_lsst', 'mag_r_lsst', 'mag_i_lsst', 'mag_z_lsst', 'mag_y_lsst']
    roman_bands = ['mag_Y_roman', 'mag_J_roman', 'mag_H_roman']
    all_bands = [b for b in lsst_bands + roman_bands if b in data_dict]

    features = []
    clean_mags: Dict[str, np.ndarray] = {}

    # 1. Magnitudes imputed with physical 1-sigma limiting magnitudes + NaN indicators
    for b in all_bands:
        m = np.asarray(data_dict[b], dtype=np.float32).copy()
        lim = float(PHYSICAL_LIMITS.get(b, 26.5))
        nan_mask = ~np.isfinite(m) | (m > 35.0) | (m < 10.0)
        m_clean = np.where(nan_mask, lim, m).astype(np.float32)
        clean_mags[b] = m_clean
        features.append(m_clean)
        features.append(nan_mask.astype(np.float32))

    # 2. Pogson fluxes (zero for non-detections)
    for b in all_bands:
        m = np.asarray(data_dict[b], dtype=np.float32)
        nan_mask = ~np.isfinite(m) | (m > 35.0) | (m < 10.0)
        f = np.where(nan_mask, 0.0, 10.0 ** (-0.4 * (np.clip(m, 10.0, 35.0) - 24.0))).astype(np.float32)
        features.append(f)

    # 3. Magnitude errors (1.0 mag conservative error for non-detections)
    for b in all_bands:
        err_col = f"{b}_err"
        if err_col in data_dict:
            err = np.asarray(data_dict[err_col], dtype=np.float32).copy()
            bad_err = ~np.isfinite(err) | (err <= 0.0) | (err > 10.0)
            err = np.where(bad_err, 1.0, err).astype(np.float32)
            features.append(err)

    # 4. Adjacent band colors (physical dropout colors without 99.0 spikes)
    for i in range(len(all_bands) - 1):
        b1, b2 = all_bands[i], all_bands[i + 1]
        c = np.clip(clean_mags[b1] - clean_mags[b2], -5.0, 10.0).astype(np.float32)
        features.append(c)

    # 5. Wide-baseline colors
    wide_pairs = [
        ("mag_u_lsst", "mag_r_lsst"),
        ("mag_g_lsst", "mag_i_lsst"),
        ("mag_r_lsst", "mag_z_lsst"),
        ("mag_i_lsst", "mag_y_lsst"),
        ("mag_z_lsst", "mag_H_roman"),
    ]
    for b1, b2 in wide_pairs:
        if b1 in clean_mags and b2 in clean_mags:
            c = np.clip(clean_mags[b1] - clean_mags[b2], -6.0, 12.0).astype(np.float32)
            features.append(c)

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
    """Train XGBoost tomographic bin classifier with SOM transfer reweighting and OOF calibration histograms."""
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

    # Resolve true redshifts BEFORE any feature sanitization
    z = np.asarray(combined['redshift'], dtype=np.float64).copy()
    if 'redshift_manyband' in combined:
        z_mb = np.asarray(combined['redshift_manyband'], dtype=np.float64)
        fill = ~np.isfinite(z) & np.isfinite(z_mb)
        z[fill] = z_mb[fill]

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
    probs_oof = np.zeros((len(y_true), n_tomo_bins), dtype=np.float64)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_true)):
        sw_fold = sample_weights[train_idx] if sample_weights is not None else None
        clf_fold = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.075,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method='hist',
            device=_XGB_DEVICE,
            random_state=42 + fold,
            eval_metric='mlogloss',
        )
        clf_fold.fit(X[train_idx], y_true[train_idx], sample_weight=sw_fold)
        probs_val = clf_fold.predict_proba(X[val_idx])
        probs_oof[val_idx] = probs_val
        y_pred_oof[val_idx] = np.argmax(probs_val, axis=1)

    # 2. Reconstruct empirical n(z) for each bin using genuine OUT-OF-FOLD predictions
    # Blending hard-assigned OOF histograms with soft-probability-weighted OOF histograms
    calib_hists = []
    for k in range(n_tomo_bins):
        mask_k = (y_pred_oof == k)
        if sample_weights is not None:
            w_hard = sample_weights[mask_k]
            hist_hard = np.histogram(z_clean[mask_k], grid_edges, weights=w_hard)[0].astype(np.float64)
            w_soft = sample_weights * probs_oof[:, k]
            hist_soft = np.histogram(z_clean, grid_edges, weights=w_soft)[0].astype(np.float64)
        else:
            hist_hard = np.histogram(z_clean[mask_k], grid_edges)[0].astype(np.float64)
            hist_soft = np.histogram(z_clean, grid_edges, weights=probs_oof[:, k])[0].astype(np.float64)

        hist_hard = (hist_hard + 0.03) / np.maximum(hist_hard.sum() + 0.03 * len(hist_hard), 1e-12)
        hist_soft = (hist_soft + 0.03) / np.maximum(hist_soft.sum() + 0.03 * len(hist_soft), 1e-12)
        hist_k = 0.90 * hist_hard + 0.10 * hist_soft
        hist_k /= hist_k.sum()
        calib_hists.append(hist_k)
    calib_hists = np.array(calib_hists)

    # 3. Train final production classifier on all training data
    clf = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.075,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method='hist',
        device=_XGB_DEVICE,
        random_state=42,
        eval_metric='mlogloss',
    )
    clf.fit(X, y_true, sample_weight=sample_weights)
    clf.set_params(device="cpu")

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
    """Predict tomographic bins on WFD catalog with posterior-adapted n(z) and construct qp Ensembles."""
    taskset = key[0:9]
    tomo_edges = TOMO_BIN_EDGES[taskset]
    grid_edges = Z_BIN_EDGES[taskset]
    n_tomo_bins = len(tomo_edges) - 1

    wfd_data = tables_io.read(wfd_file)
    n_objects = len(wfd_data[list(wfd_data.keys())[0]])

    X_wfd = extract_features(wfd_data)
    try:
        clf.set_params(device="cpu")
    except Exception:
        pass
    probs_wfd = clf.predict_proba(X_wfd)

    # Hybrid MoE entropy regularization for high-uncertainty boundary objects
    entropy = -np.sum(probs_wfd * np.log(np.maximum(probs_wfd, 1e-12)), axis=1)
    uncertain = entropy > 1.25
    probs_wfd[uncertain] = 0.88 * probs_wfd[uncertain] + 0.12 * (1.0 / n_tomo_bins)

    bin_assignments = np.argmax(probs_wfd, axis=1).astype(int)

    # Adapt calibration histograms slightly to the WFD target sample's posterior transition matrix
    adapted_hists = calib_hists.copy()
    for k in range(n_tomo_bins):
        mask_k = (bin_assignments == k)
        if np.any(mask_k):
            mean_p_k = np.mean(probs_wfd[mask_k], axis=0)
            mix_k = mean_p_k @ calib_hists
            adapted = 0.92 * calib_hists[k] + 0.08 * mix_k
            adapted_hists[k] = adapted / np.maximum(adapted.sum(), 1e-12)

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
    ens_estimate = qp.hist.create_ensemble(grid_edges, adapted_hists)
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
            alpha = adapted_hists[k] * 1000.0 + 0.1
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
