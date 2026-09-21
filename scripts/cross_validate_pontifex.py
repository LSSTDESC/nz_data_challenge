"""5-Fold Cross Validation for Pontifex on nz_data_challenge."""

import argparse
import numpy as np
import tables_io
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold

from nz_data_challenge import utils, metrics


def extract_features(data_dict: dict) -> np.ndarray:
    """Extract magnitudes, colors, errors, and fluxes from galaxy catalog."""
    lsst_bands = ['mag_u_lsst', 'mag_g_lsst', 'mag_r_lsst', 'mag_i_lsst', 'mag_z_lsst', 'mag_y_lsst']
    roman_bands = ['mag_Y_roman', 'mag_J_roman', 'mag_H_roman']
    all_bands = [b for b in lsst_bands + roman_bands if b in data_dict]
    
    features = []
    
    # 1. Clean magnitudes and missing indicators
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
        
    # 5. Broad baseline colors
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


def load_training_data(taskset: str, sim: str, scenario: str, public_dir: str = "public") -> tuple[dict, np.ndarray]:
    """Load DDF catalogs and extract clean photometry and reference redshifts."""
    data_list = []
    # Load DDF files
    for iddf in range(5):
        path = f"{public_dir}/nz_challenge_{taskset}_{sim}_{scenario}_ddf_{iddf:02d}.hdf5"
        data_list.append(tables_io.read(path))
        
    # Concatenate only common columns across DDF files
    common_keys = list(set.intersection(*[set(d.keys()) for d in data_list]))
    combined = {}
    for k in common_keys:
        arrays = [d[k] for d in data_list]
        combined[k] = np.concatenate(arrays)
        
    # Resolve true redshift using spec-z supplemented by manyband photo-z
    z = combined['redshift'].copy()
    if 'redshift_manyband' in combined:
        fill = ~np.isfinite(z) & np.isfinite(combined['redshift_manyband'])
        z[fill] = combined['redshift_manyband'][fill]
        
    valid = np.isfinite(z)
    z_clean = z[valid]
    combined_clean = {k: v[valid] for k, v in combined.items()}
    
    return combined_clean, z_clean


def run_cross_validation(taskset: str = "taskset_1", sim: str = "cardinal", scenario: str = "1yr", n_splits: int = 5):
    print("\n================================================================================")
    print(f"Running 5-Fold Cross Validation: {taskset} | {sim} | {scenario}")
    print("================================================================================")
    
    tomo_edges = utils.TOMO_BIN_EDGES[taskset]
    grid_edges = utils.Z_BIN_EDGES[taskset]
    n_tomo_bins = len(tomo_edges) - 1
    
    print("Loading datasets...")
    data_dict, z_true = load_training_data(taskset, sim, scenario)
    y_true = utils.get_true_bin_assignments(z_true, tomo_edges)
    
    print("Extracting features...")
    X = extract_features(data_dict)
    n_samples = len(y_true)
    print(f"Total samples: {n_samples}, Features: {X.shape[1]}")
    print(f"Tomographic Bin Population: {np.bincount(y_true, minlength=n_tomo_bins)}")
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    fold_metrics = {
        'accuracy': [],
        'balanced_accuracy': [],
        'cohens_kappa': [],
        'mutual_info': [],
        'log_loss': [],
        'total_information_loss': [],
        'rms0_delta_mean': [],
        'rms0_delta_std': [],
    }
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_true)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y_true[train_idx], y_true[val_idx]
        z_train, z_val = z_true[train_idx], z_true[val_idx]
        
        # Train Pontifex classifier (XGBoost)
        clf = xgb.XGBClassifier(
            n_estimators=180,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method='hist',
            device='cuda',
            random_state=42 + fold,
            eval_metric='mlogloss'
        )
        clf.fit(X_train, y_train)
        
        probs_val = clf.predict_proba(X_val)
        y_pred = np.argmax(probs_val, axis=1)
        
        # Evaluate bin assignment metrics
        acc = float((y_pred == y_val).sum() / len(y_val))
        bal_acc = metrics.balanced_accuracy(y_val, y_pred, num_classes=n_tomo_bins)
        kappa = metrics.cohens_kappa(y_val, y_pred, num_classes=n_tomo_bins)
        log_l = metrics.log_loss_from_labels(y_val, y_pred, num_classes=n_tomo_bins)
        mi = metrics.mutual_info(z_val, y_pred)
        
        # Predict on train to obtain unbiased empirical n_k(z) calibration
        probs_train = clf.predict_proba(X_train)
        y_pred_train = np.argmax(probs_train, axis=1)
        
        # Reconstruct n_k(z)
        est_nz_list = []
        true_nz_list = []
        val_counts = []
        for k in range(n_tomo_bins):
            hist_val = np.histogram(z_val[y_pred == k], grid_edges)[0].astype(float)
            true_nz_list.append(hist_val)
            
            n_val_k = (y_pred == k).sum()
            val_counts.append(n_val_k)
            
            hist_train = np.histogram(z_train[y_pred_train == k], grid_edges)[0].astype(float)
            if hist_train.sum() > 0:
                hist_norm = hist_train / hist_train.sum()
                hist_scaled = hist_norm * n_val_k
            else:
                hist_scaled = np.ones_like(hist_val) * (n_val_k / len(hist_val))
            est_nz_list.append(hist_scaled)
            
        true_nz = np.array(true_nz_list)
        est_nz = np.array(est_nz_list)
        counts_arr = np.array(val_counts)
        
        tot_kl, _ = metrics.total_information_loss(true_nz, est_nz, counts_arr)
        rms_stats = metrics.rms0_delta_summary_stats(est_nz, true_nz, grid_edges)
        
        fold_metrics['accuracy'].append(acc)
        fold_metrics['balanced_accuracy'].append(bal_acc)
        fold_metrics['cohens_kappa'].append(kappa)
        fold_metrics['mutual_info'].append(mi)
        fold_metrics['log_loss'].append(log_l)
        fold_metrics['total_information_loss'].append(tot_kl)
        fold_metrics['rms0_delta_mean'].append(rms_stats['mean'])
        fold_metrics['rms0_delta_std'].append(rms_stats['std'])
        
        print(f"Fold {fold+1}/{n_splits} -> Acc: {acc:.4f} | BalAcc: {bal_acc:.4f} | Kappa: {kappa:.4f} | LogLoss: {log_l:.4f} | KL: {tot_kl:.6f} | d_mu: {rms_stats['mean']:.6f} | d_std: {rms_stats['std']:.6f}")
        
    print(f"\n--- 5-Fold Cross Validation Summary for {taskset} {sim} {scenario} ---")
    for k, v in fold_metrics.items():
        mean_val = np.mean(v)
        std_val = np.std(v)
        print(f"{k:25s}: {mean_val:.6f} +/- {std_val:.6f}")
        
    return fold_metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--taskset', type=str, default='taskset_1', choices=['taskset_1', 'taskset_2'])
    parser.add_argument('--sim', type=str, default='cardinal', choices=['cardinal', 'flagship'])
    parser.add_argument('--scenario', type=str, default='1yr', choices=['1yr', '4yr'])
    args = parser.parse_args()
    
    run_cross_validation(taskset=args.taskset, sim=args.sim, scenario=args.scenario)
