"""Bula Enhanced Pontifex Pipeline: 5-Fold Cross-Validation Comparison."""

import numpy as np
import tables_io
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from scipy.spatial import cKDTree

from nz_data_challenge import utils, metrics


def extract_features_bula(data_dict: dict, apply_noise_filter: bool = True) -> np.ndarray:
    """Extract features with Pontifex S/N < 2.0 noise-floor filtering."""
    lsst_bands = ['mag_u_lsst', 'mag_g_lsst', 'mag_r_lsst', 'mag_i_lsst', 'mag_z_lsst', 'mag_y_lsst']
    roman_bands = ['mag_Y_roman', 'mag_J_roman', 'mag_H_roman']
    all_bands = [b for b in lsst_bands + roman_bands if b in data_dict]
    
    features = []
    
    # 1. Magnitudes and noise floor
    for b in all_bands:
        m = np.asarray(data_dict[b], dtype=np.float32).copy()
        err_col = f"{b}_err"
        
        nan_mask = np.isnan(m) | ~np.isfinite(m)
        if apply_noise_filter and err_col in data_dict:
            err = np.asarray(data_dict[err_col], dtype=np.float32)
            # S/N < 2.0 corresponds to err > 0.54 mag
            noisy_mask = (err > 0.54) | np.isnan(err) | (err <= 0)
            m[noisy_mask] = 25.0  # Pontifex neutral noise-floor imputation
        
        m[nan_mask] = 26.5
        features.append(m)
        features.append(nan_mask.astype(np.float32))
        
    # 2. Pogson fluxes
    for b in all_bands:
        m = np.asarray(data_dict[b], dtype=np.float32).copy()
        f = np.where(np.isnan(m) | (m > 28.0), 0.0, 10.0 ** (-0.4 * (m - 24.0)))
        features.append(f)
        
    # 3. Magnitude errors
    for b in all_bands:
        err_col = f"{b}_err"
        if err_col in data_dict:
            err = np.asarray(data_dict[err_col], dtype=np.float32).copy()
            err = np.where(np.isnan(err) | (err > 10.0), 10.0, err)
            features.append(err)
            
    # 4. Adjacent band colors with noise clipping
    for i in range(len(all_bands) - 1):
        b1, b2 = all_bands[i], all_bands[i+1]
        m1 = np.where(np.isnan(data_dict[b1]), 25.0, data_dict[b1])
        m2 = np.where(np.isnan(data_dict[b2]), 25.0, data_dict[b2])
        col = np.clip(m1 - m2, -5.0, 5.0)
        features.append(col)
        
    # 5. Wide-baseline break tracers
    pairs = [
        ('mag_u_lsst', 'mag_r_lsst'),
        ('mag_g_lsst', 'mag_i_lsst'),
        ('mag_r_lsst', 'mag_z_lsst'),
        ('mag_i_lsst', 'mag_y_lsst'),
    ]
    for b1, b2 in pairs:
        if b1 in data_dict and b2 in data_dict:
            m1 = np.where(np.isnan(data_dict[b1]), 25.0, data_dict[b1])
            m2 = np.where(np.isnan(data_dict[b2]), 25.0, data_dict[b2])
            features.append(np.clip(m1 - m2, -5.0, 5.0))
            
    return np.column_stack(features)


def compute_density_weights(
    train_feats: np.ndarray,
    target_feats: np.ndarray,
    n_neighbors: int = 30,
) -> np.ndarray:
    """Compute transfer weights w(x) = P_target(x) / P_train(x) using color-magnitude density estimation."""
    mean = np.mean(train_feats[:, :8], axis=0)
    std = np.std(train_feats[:, :8], axis=0) + 1e-5
    
    X_tr = (train_feats[:, :8] - mean) / std
    X_te = (target_feats[:, :8] - mean) / std
    
    # Subsample target for speed if necessary
    if len(X_te) > 50000:
        idx = np.random.default_rng(42).choice(len(X_te), size=50000, replace=False)
        X_te_sub = X_te[idx]
    else:
        X_te_sub = X_te

    tree_tr = cKDTree(X_tr)
    tree_te = cKDTree(X_te_sub)
    
    d_tr, _ = tree_tr.query(X_tr, k=n_neighbors)
    d_te, _ = tree_te.query(X_tr, k=n_neighbors)
    
    r_tr = d_tr[:, -1] + 1e-6
    r_te = d_te[:, -1] + 1e-6
    
    d_dim = min(8, X_tr.shape[1])
    weights = (r_tr / r_te) ** (d_dim * 0.45)
    weights = np.clip(weights, 0.1, 6.0)
    weights /= np.mean(weights)
    return weights


def load_training_data(taskset: str = "taskset_2", sim: str = "cardinal", scenario: str = "1yr"):
    files = [f"public/nz_challenge_{taskset}_{sim}_{scenario}_ddf_{i:02d}.hdf5" for i in range(5)]
    data_list = [tables_io.read(f) for f in files]
    common_keys = list(set.intersection(*[set(d.keys()) for d in data_list]))
    combined = {}
    for k in common_keys:
        arrays = [d[k] for d in data_list]
        combined[k] = np.concatenate(arrays)
        
    z = combined['redshift'].copy()
    if 'redshift_manyband' in combined:
        fill = ~np.isfinite(z) & np.isfinite(combined['redshift_manyband'])
        z[fill] = combined['redshift_manyband'][fill]
        
    valid = np.isfinite(z)
    z_clean = z[valid]
    combined_clean = {k: v[valid] for k, v in combined.items()}
    return combined_clean, z_clean


def evaluate_single_pipeline(X, y_true, z_true, tomo_edges, grid_edges, sample_weights=None, use_em=False):
    n_tomo_bins = len(tomo_edges) - 1
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    results = {
        'acc': [], 'bal_acc': [], 'kappa': [],
        'rms_dmu': [], 'rms_dsigma': []
    }
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_true)):
        clf = xgb.XGBClassifier(
            n_estimators=160, max_depth=6, learning_rate=0.08,
            subsample=0.8, colsample_bytree=0.8, tree_method='hist',
            device='cuda', random_state=42, eval_metric='mlogloss'
        )
        
        sw = sample_weights[train_idx] if sample_weights is not None else None
        clf.fit(X[train_idx], y_true[train_idx], sample_weight=sw)
        
        probs_val = clf.predict_proba(X[val_idx])
        y_pred = np.argmax(probs_val, axis=1)
        
        # Classification metrics
        from sklearn.metrics import balanced_accuracy_score
        acc = float(np.mean(y_true[val_idx] == y_pred))
        bal_acc = float(balanced_accuracy_score(y_true[val_idx], y_pred))
        kappa = float(metrics.cohens_kappa(y_true[val_idx], y_pred, num_classes=n_tomo_bins))
        
        # Calibration
        probs_tr = clf.predict_proba(X[train_idx])
        y_pred_tr = np.argmax(probs_tr, axis=1)
        
        true_nz_list = []
        est_nz_list = []
        for k in range(n_tomo_bins):
            hist_val = np.histogram(z_true[val_idx][y_pred == k], grid_edges)[0].astype(float)
            true_nz_list.append(hist_val)
            
            n_val_k = (y_pred == k).sum()
            mask_tr = (y_pred_tr == k)
            if sample_weights is not None:
                w_k = sample_weights[train_idx][mask_tr]
                hist_tr = np.histogram(z_true[train_idx][mask_tr], bins=grid_edges, weights=w_k)[0].astype(float)
            else:
                hist_tr = np.histogram(z_true[train_idx][mask_tr], bins=grid_edges)[0].astype(float)
                
            hist_tr += 0.05
            hist_norm = hist_tr / hist_tr.sum()
            est_nz_list.append(hist_norm * n_val_k)
            
        true_nz = np.array(true_nz_list)
        est_nz = np.array(est_nz_list)
        
        if use_em:
            z_centers = 0.5 * (grid_edges[:-1] + grid_edges[1:])
            cosmic_prior = (z_centers ** 1.8) * np.exp(-(z_centers / 0.7) ** 1.2)
            cosmic_prior /= np.sum(cosmic_prior)
            # EM adjustment
            tot_est = np.sum(est_nz, axis=0) + 1e-8
            ratio = (cosmic_prior / (tot_est / np.sum(tot_est))) ** 0.15
            est_nz = est_nz * ratio
            
        rms_stats = metrics.rms0_delta_summary_stats(est_nz, true_nz, grid_edges)
        
        results['acc'].append(acc)
        results['bal_acc'].append(bal_acc)
        results['kappa'].append(kappa)
        results['rms_dmu'].append(rms_stats['mean'])
        results['rms_dsigma'].append(rms_stats['std'])
        
    return {k: (np.mean(v), np.std(v)) for k, v in results.items()}


def run_benchmark(taskset: str = "taskset_2", sim: str = "cardinal", scenario: str = "1yr"):
    print("\n================================================================================")
    print(f"RUNNING BULA BENCHMARK COMPARISON ON: {taskset} | {sim} | {scenario}")
    print("================================================================================")
    
    tomo_edges = utils.TOMO_BIN_EDGES[taskset]
    grid_edges = utils.Z_BIN_EDGES[taskset]
    
    data_dict, z_true = load_training_data(taskset, sim, scenario)
    y_true = utils.get_true_bin_assignments(z_true, tomo_edges)
    
    wfd_file = f"public/nz_challenge_{taskset}_{sim}_{scenario}_wfd.hdf5"
    wfd_data = tables_io.read(wfd_file)
    
    print("1. Extracting baseline features...")
    X_base = extract_features_bula(data_dict, apply_noise_filter=False)
    
    print("2. Extracting Bula features (noise floor imputation)...")
    X_bula = extract_features_bula(data_dict, apply_noise_filter=True)
    X_wfd_bula = extract_features_bula(wfd_data, apply_noise_filter=True)
    
    print("3. Computing transfer density weights w(x)...")
    weights = compute_density_weights(X_bula, X_wfd_bula)
    print(f"   Transfer weights: min={weights.min():.2f}, mean={weights.mean():.2f}, max={weights.max():.2f}")
    
    print("\n--- Running Baseline Pipeline 5-Fold CV ---")
    base_res = evaluate_single_pipeline(X_base, y_true, z_true, tomo_edges, grid_edges, sample_weights=None, use_em=False)
    
    print("\n--- Running Bula Enhanced Pipeline 5-Fold CV ---")
    bula_res = evaluate_single_pipeline(X_bula, y_true, z_true, tomo_edges, grid_edges, sample_weights=weights, use_em=True)
    
    print("\n================================================================================")
    print("RESULTS COMPARISON: BASELINE vs BULA ENHANCED PIPELINE")
    print("================================================================================")
    print(f"{'Metric':<20} | {'Baseline (Release 4.0.0)':<25} | {'Bula Enhanced':<25} | {'Gain / Improvement'}")
    print("-" * 88)
    for m in ['acc', 'bal_acc', 'kappa', 'rms_dmu', 'rms_dsigma']:
        b_mean, b_std = base_res[m]
        u_mean, u_std = bula_res[m]
        if 'rms' in m:
            gain = (b_mean - u_mean) / b_mean * 100.0
            gain_str = f"-{gain:.1f}% error reduction"
        else:
            gain = (u_mean - b_mean) * 100.0
            gain_str = f"+{gain:.2f}% gain"
        print(f"{m:<20} | {b_mean:.5f} ± {b_std:.5f}       | {u_mean:.5f} ± {u_std:.5f}       | {gain_str}")


if __name__ == '__main__':
    run_benchmark("taskset_2", "cardinal", "1yr")
