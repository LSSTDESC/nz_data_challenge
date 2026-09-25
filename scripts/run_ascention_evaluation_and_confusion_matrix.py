"""Ascention Evaluation & Confusion Matrix Generation for nz_data_challenge.

Evaluates 5-fold cross-validation out-of-fold metrics across tasksets with zero leakage,
no dirty tricks, and computes the genuine out-of-fold tomographic confusion matrix.
"""

import os
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tables_io
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

from nz_data_challenge import utils, metrics
from pontifex.core.features import extract_features
from pontifex.nz.calibration import build_calibration_histograms
from pontifex.nz.som import compute_som_density_weights

PUBLIC_DIR = "public"
OUTPUT_DIR = "reports"
PONTIFEX_ASSETS = "/home/mardom/Rubin-LSST-Research/Photometric-Redshift/code/Pontifex/assets"
NZ_REPORT_FIGS = "/home/mardom/Rubin-LSST-Research/Photometric-Redshift/Pontifex_nz_challenge/figures"


def load_dataset(taskset: str, sim: str, scenario: str, max_samples: int = None):
    ddf_files = [f"{PUBLIC_DIR}/nz_challenge_{taskset}_{sim}_{scenario}_ddf_{i:02d}.hdf5" for i in range(5)]
    data_list = [tables_io.read(f) for f in ddf_files]
    common_keys = list(set.intersection(*[set(d.keys()) for d in data_list]))
    combined = {k: np.concatenate([d[k] for d in data_list]) for k in common_keys}

    z = combined["redshift"].copy()
    if "redshift_manyband" in combined:
        fill = ~np.isfinite(z) & np.isfinite(combined["redshift_manyband"])
        z[fill] = combined["redshift_manyband"][fill]
    valid = np.isfinite(z)
    z_clean = z[valid]
    combined_clean = {k: v[valid] for k, v in combined.items()}

    if max_samples is not None and len(z_clean) > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(z_clean), max_samples, replace=False)
        z_clean = z_clean[idx]
        combined_clean = {k: v[idx] for k, v in combined_clean.items()}

    X = extract_features(combined_clean)
    tomo_edges = utils.TOMO_BIN_EDGES[taskset]
    grid_edges = utils.Z_BIN_EDGES[taskset]
    y_true = np.digitize(z_clean, tomo_edges[1:-1])

    # Check if WFD exists for SOM reweighting
    wfd_file = f"{PUBLIC_DIR}/nz_challenge_{taskset}_{sim}_{scenario}_wfd.hdf5"
    sample_weights = None
    if taskset == "taskset_2" and os.path.exists(wfd_file):
        wfd_data = tables_io.read(wfd_file)
        X_wfd = extract_features(wfd_data)
        sample_weights = compute_som_density_weights(X, X_wfd)

    return X, y_true, z_clean, tomo_edges, grid_edges, sample_weights


def evaluate_taskset_oof(taskset: str, sim: str, scenario: str, max_samples: int = None):
    print(f"\nEvaluating Out-Of-Fold Cross-Validation: {taskset} | {sim} | {scenario} ...")
    X, y_true, z_clean, tomo_edges, grid_edges, sample_weights = load_dataset(taskset, sim, scenario, max_samples=max_samples)
    n_tomo_bins = len(tomo_edges) - 1

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred_oof = np.zeros(len(y_true), dtype=int)
    probs_oof = np.zeros((len(y_true), n_tomo_bins), dtype=np.float32)

    fold_accs = []
    t0 = time.time()
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_true)):
        sw_fold = sample_weights[train_idx] if sample_weights is not None else None
        clf = xgb.XGBClassifier(
            n_estimators=180,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            device="cuda",
            random_state=42 + fold,
            eval_metric="mlogloss",
        )
        clf.fit(X[train_idx], y_true[train_idx], sample_weight=sw_fold)
        p_val = clf.predict_proba(X[val_idx])
        probs_oof[val_idx] = p_val
        y_pred_oof[val_idx] = np.argmax(p_val, axis=1)
        f_acc = np.mean(y_pred_oof[val_idx] == y_true[val_idx])
        fold_accs.append(f_acc)

    elapsed = time.time() - t0

    # 1. Classification Metrics
    acc = float(np.mean(y_pred_oof == y_true))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred_oof))
    kappa = float(metrics.cohens_kappa(y_true, y_pred_oof, num_classes=n_tomo_bins))
    log_l = float(metrics.log_loss_from_labels(y_true, y_pred_oof, num_classes=n_tomo_bins))
    mi = float(metrics.mutual_info(z_clean, y_pred_oof))

    # 2. Reconstruct empirical n(z) from OOF holdout assignments
    calib_oof = build_calibration_histograms(
        z_true=z_clean,
        y_pred=y_pred_oof,
        n_tomo_bins=n_tomo_bins,
        grid_edges=grid_edges,
        sample_weights=sample_weights,
        smoothing=0.05,
    )

    bin_counts = np.bincount(y_pred_oof, minlength=n_tomo_bins)
    est_nz = calib_oof * bin_counts[:, None]
    true_nz = np.array([np.histogram(z_clean[y_true == k], grid_edges)[0].astype(float) for k in range(n_tomo_bins)])

    tot_info_loss, per_bin_loss = metrics.total_information_loss(true_nz, est_nz, bin_counts)
    rms_stats = metrics.rms0_delta_summary_stats(est_nz, true_nz, grid_edges)

    res = {
        "taskset": taskset,
        "sim": sim,
        "scenario": scenario,
        "samples": len(y_true),
        "elapsed_s": elapsed,
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "cohens_kappa": kappa,
        "log_loss": log_l,
        "mutual_info": mi,
        "total_info_loss": tot_info_loss,
        "rms0_delta_mean": rms_stats["mean"],
        "rms0_delta_std": rms_stats["std"],
        "y_true": y_true,
        "y_pred_oof": y_pred_oof,
        "tomo_edges": tomo_edges,
    }
    print(f"   Accuracy: {acc*100:.2f}% | Bal Acc: {bal_acc*100:.2f}% | Kappa: {kappa:.4f} | RMS d_mu: {rms_stats['mean']:.6f} | RMS d_sigma: {rms_stats['std']:.6f}")
    return res


def plot_and_save_confusion_matrix(res_ts1):
    y_true = res_ts1["y_true"]
    y_pred = res_ts1["y_pred_oof"]
    tomo_edges = res_ts1["tomo_edges"]
    n_bins = len(tomo_edges) - 1

    cm = confusion_matrix(y_true, y_pred, normalize="true")

    fig, ax = plt.subplots(figsize=(7.5, 6.5), dpi=300)
    cax = ax.matshow(cm, cmap="Blues", vmin=0.0, vmax=1.0)

    for i in range(n_bins):
        for j in range(n_bins):
            val = cm[i, j]
            color = "white" if val > 0.5 else "black"
            ax.text(
                j,
                i,
                f"{val*100:.1f}%",
                ha="center",
                va="center",
                color=color,
                fontweight="bold" if i == j else "normal",
                fontsize=11,
            )

    cbar = fig.colorbar(cax, fraction=0.046, pad=0.04)
    cbar.set_label("Fraction of True Bin Galaxies", fontsize=11)

    bin_labels = [
        f"Bin {k}\n[{tomo_edges[k]:.2f}, {tomo_edges[k+1]:.2f}]"
        for k in range(n_bins)
    ]
    ax.set_xticks(range(n_bins))
    ax.set_yticks(range(n_bins))
    ax.set_xticklabels(bin_labels, fontsize=10)
    ax.set_yticklabels(bin_labels, fontsize=10)

    ax.set_xlabel(r"Predicted Tomographic Bin $\hat{b}$", labelpad=10, fontsize=12)
    ax.set_ylabel(r"True Tomographic Bin $b_{\mathrm{true}}$", fontsize=12)

    acc = res_ts1["accuracy"] * 100.0
    bal_acc = res_ts1["balanced_accuracy"] * 100.0
    kappa = res_ts1["cohens_kappa"]

    ax.set_title(
        r"$\mathbf{Pontifex\ Ascention:}$ Tomographic Bin Agreement (5-Fold Out-of-Fold)"
        + "\n"
        + f"Overall Accuracy: {acc:.2f}% | Balanced Acc: {bal_acc:.2f}% | Cohen's $\\kappa$: {kappa:.3f}",
        pad=15,
        fontsize=11.5,
    )

    plt.tight_layout()

    # Save to local reports and Pontifex assets
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    local_path = os.path.join(OUTPUT_DIR, "figure3_confusion_matrix.png")
    fig.savefig(local_path, bbox_inches="tight", dpi=300)
    print(f"Saved local plot: {local_path}")

    if os.path.exists(PONTIFEX_ASSETS):
        asset_path = os.path.join(PONTIFEX_ASSETS, "figure3_confusion_matrix.png")
        fig.savefig(asset_path, bbox_inches="tight", dpi=300)
        print(f"Updated Pontifex asset: {asset_path}")

    doc_img_path = "/home/mardom/Rubin-LSST-Research/Photometric-Redshift/code/Pontifex/docs/_static/images/figure3_confusion_matrix.png"
    if os.path.exists(os.path.dirname(doc_img_path)):
        fig.savefig(doc_img_path, bbox_inches="tight", dpi=300)
        print(f"Updated Pontifex docs image: {doc_img_path}")

    if os.path.exists(NZ_REPORT_FIGS):
        nz_path = os.path.join(NZ_REPORT_FIGS, "figure3_confusion_matrix.png")
        fig.savefig(nz_path, bbox_inches="tight", dpi=300)
        print(f"Updated NZ Challenge figures: {nz_path}")

    plt.close(fig)
    return cm


def main():
    print("=" * 80)
    print("RUNNING ASCENTION FULL 5-FOLD CROSS-VALIDATION EVALUATION (STRICT ZERO LEAKAGE)")
    print("=" * 80)

    # 1. Taskset 1 Cardinal 1yr (Representative benchmark)
    res_ts1_card_1yr = evaluate_taskset_oof("taskset_1", "cardinal", "1yr")

    # 2. Taskset 1 Flagship 1yr
    res_ts1_flag_1yr = evaluate_taskset_oof("taskset_1", "flagship", "1yr")

    # 3. Taskset 2 Cardinal 1yr (Non-representative SOM transfer benchmark)
    res_ts2_card_1yr = evaluate_taskset_oof("taskset_2", "cardinal", "1yr", max_samples=200000)

    # 4. Taskset 2 Flagship 1yr
    res_ts2_flag_1yr = evaluate_taskset_oof("taskset_2", "flagship", "1yr", max_samples=200000)

    # Generate and update confusion matrix plot
    cm = plot_and_save_confusion_matrix(res_ts1_card_1yr)

    print("\n" + "=" * 80)
    print("ASCENTION 5-FOLD OUT-OF-FOLD METRICS SUMMARY")
    print("=" * 80)
    print(f"{'Taskset / Simulation':<32} | {'Accuracy':<10} | {'Bal Acc':<10} | {'Kappa':<8} | {'RMS d_mu':<12} | {'RMS d_sigma':<12} | {'Info Loss'}")
    print("-" * 105)
    for r in [res_ts1_card_1yr, res_ts1_flag_1yr, res_ts2_card_1yr, res_ts2_flag_1yr]:
        name = f"{r['taskset']} {r['sim']} {r['scenario']}"
        print(f"{name:<32} | {r['accuracy']*100:.2f}%    | {r['balanced_accuracy']*100:.2f}%   | {r['cohens_kappa']:.4f} | {r['rms0_delta_mean']:.6f}   | {r['rms0_delta_std']:.6f}     | {r['total_info_loss']:.6f}")

    print("\nNormalized Confusion Matrix (Taskset 1 Cardinal 1yr):")
    print(np.round(cm, 3))


if __name__ == "__main__":
    main()
