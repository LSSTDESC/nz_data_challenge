"""Evaluation pipeline tools for n(z) challenge submissions."""

import glob
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pandas as pd
import qp
import tables_io
import yaml
from jinja2 import Template

from . import metrics, utils, forecast
from .utils import TASKSETS, SIMS, SCENARIOS

# Colors for n(z) plots
TOMO_BIN_COLORS = [
    'violet', 'indigo', 'magenta', 'blue', 'cyan',
    'green', 'yellow', 'orange', 'red', 'gray',
]

# Performance merics, with plotting info and scoring criteria
METRICS: dict[str, dict[str, Any]] = dict(
    accuracy=dict(
        label='Bin Assignment Accuracy',
        limits=[0, 1],
        ranges=[[0.8, 1.0], [0.6, 1.0], [0.4, 1.0]],
    ),
    balanced_accuracy=dict(
        label='Balanced Bin Assignment Accuracy',
        limits=[0, 1],
        ranges=[[0.8, 1.0], [0.6, 1.0], [0.4, 1.0]],
    ),
    cohens_kappa=dict(
        label="Cohen's Kappa",
        limits=[0, 1],
        ranges=[[0.8, 1.0], [0.6, 1.0], [0.4, 1.0]],
    ),
    log_loss=dict(
        label='Log information loss [bits]',
        limits=[0, 20],
        ranges=[[0., 5.0], [0., 10.0], [0., 15.0]],
    ),
    mutual_info=dict(
        label='Mutual information',
        limits=[0, 10],
        ranges=[[0., 2.],[0., 4.],[0., 6.]],
    ),
    rms0_delta_mean=dict(
        label=r'$RMS \delta \mu_{i}$',
        limits=[0, 0.1],
        ranges=[[0, 0.01],[0., 0.02],[0., 0.03]],
    ),
    rms0_delta_std=dict(
        label=r'$RMS \delta \sigma_{i}$',
        limits=[0, 0.1],
        ranges=[[0, 0.01],[0., 0.02],[0., 0.03]],
    ),
    total_information_loss=dict(
        label='Total information loss [bits]',
        limits=[0, 0.1],
        ranges=[[0, 0.01],[0., 0.02],[0., 0.03]],
    ),
)


def get_submissions(accept_dir: str = "../accepted") -> list[str]:
    """
    Retrieve list of submission names from test files in the accepted directory.

    Scans the specified directory for Python test files matching the pattern
    'test_*.py' and extracts submission identifiers.

    Parameters
    ----------
    accept_dir
        Path to directory containing accepted submission test files.
        Default is "../accepted".

    Returns
    -------
    submissions
        List of submission identifier strings extracted from filenames.

    Examples
    --------
    >>> submissions = get_submissions("./accepted")
    >>> print(submissions)
    ['baseline', 'improved_algo', 'fast_estimator']
    """
    test_string = f"{accept_dir}/test_"
    submissions = [
        f.replace(test_string, "").replace(".py", "")
        for f in glob.glob(f"{test_string}*.py")
    ]
    return submissions


def evaluate_bin_assignments(
    true_assignments: np.ndarray,
    bin_assignments: np.ndarray,
) -> dict[str, float]:
    """Evaluate tomographic bin assignment quality using multiple metrics.

    Parameters
    ----------
    true_assignments
        Array of true bin labels for each object.
    bin_assignments
        Array of predicted bin labels for each object.

    Returns
    -------
    dict
        Dictionary with keys 'log_loss', 'accuracy', 'balanced_accuracy',
        'cohens_kappa', and 'mutual_info'.
    """
    the_dict: dict[str, float] = dict(
        log_loss = metrics.log_loss_from_labels(true_assignments, bin_assignments),
        accuracy = float((bin_assignments == true_assignments).sum() / true_assignments.size),
        balanced_accuracy = metrics.balanced_accuracy(true_assignments, bin_assignments),
        cohens_kappa = metrics.cohens_kappa(true_assignments, bin_assignments),
        mutual_info = metrics.mutual_info(true_assignments, bin_assignments),
    )
    return the_dict


def evaluate_distributions(
    true_distributions: np.ndarray,
    nz_distributions: np.ndarray,
    grid_edges: np.ndarray,
    n_objects: np.ndarray,
) -> dict[str, Any]:
    """Evaluate n(z) distribution estimates against truth.

    Parameters
    ----------
    true_distributions
        List of true n(z) distributions per tomographic bin.
    nz_distributions
        List of estimated n(z) distributions per tomographic bin.
    grid_edges
        Redshift bin edges used for summary statistics.
    n_objects
        Number of objects per tomographic bin, used as weights for
        information loss.

    Returns
    -------
    dict
        Dictionary with keys 'total_information_loss',
        'per_bin_information_lost', 'rms0_delta_mean', and
        'rms0_delta_std'.
    """
    total_information_loss, per_bin_loss = metrics.total_information_loss(
        true_distributions, nz_distributions, n_objects,
    )
    rms0_delta_summary_stats = metrics.rms0_delta_summary_stats(
        nz_distributions,
        true_distributions,
        grid_edges,
    )

    the_dict: dict[str, Any] = dict(
        total_information_loss=total_information_loss,
        per_bin_information_lost=per_bin_loss,
        rms0_delta_mean=rms0_delta_summary_stats['mean'],
        rms0_delta_std=rms0_delta_summary_stats['std'],
    )
    return the_dict


def plot_confusion_matrix(
    true_assignments: np.ndarray,
    bin_assignments: np.ndarray,
    n_bins: int,
) -> Figure:
    """Plot a confusion matrix of true vs. assigned tomographic bins.

    Parameters
    ----------
    true_assignments
        Array of true bin labels.
    bin_assignments
        Array of predicted bin labels.
    n_bins
        Number of tomographic bins.

    Returns
    -------
    Figure
        Matplotlib Figure with a 2D histogram (log-scaled).
    """
    bin_sides = np.linspace(-0.5,n_bins-0.5,n_bins+1)
    fig = plt.figure(figsize=(6,6))
    axes = fig.subplots(1, 1)

    the_hist = axes.hist2d(true_assignments, bin_assignments, bins=(bin_sides, bin_sides), norm='log')
    axes.set_xlabel('True bin')
    axes.set_ylabel('Assigned bin')
    fig.colorbar(the_hist[3], ax=axes, label='Counts')
    fig.tight_layout()
    return fig


def plot_nz_data(
    true_distributions: np.ndarray,
    nz_distributions: np.ndarray,
    grid_edges: np.ndarray,
) -> Figure:
    """Plot estimated and true n(z) distributions for all tomographic bins.

    Parameters
    ----------
    true_distributions
        Array of true n(z) distributions per bin.
    nz_distributions
        Array of estimated n(z) distributions per bin.
    grid_edges
        Redshift bin edges for the staircase plot.

    Returns
    -------
    Figure
        Matplotlib Figure showing true (solid) and estimated (dashed)
        distributions.
    """
    fig = plt.figure(figsize=(6,6))
    axes = fig.subplots(1, 1)

    for i, (nz_dist, true_dist) in enumerate(zip(nz_distributions, true_distributions)):
        axes.stairs(nz_dist, grid_edges, ls='--', color=TOMO_BIN_COLORS[i])
        axes.stairs(true_dist, grid_edges, color=TOMO_BIN_COLORS[i])
        axes.set_xlabel('z')
        axes.set_ylabel('Objects / [0.01]')

    fig.tight_layout()
    return fig


def plot_nz_mean_and_rms(
    true_distributions: np.ndarray,
    nz_distributions: np.ndarray,
    grid_edges: np.ndarray,
) -> Figure:
    """Plot estimated and true n(z) distributions for all tomographic bins.

    Parameters
    ----------
    true_distributions
        Array of true n(z) distributions per bin.
    nz_distributions
        Array of estimated n(z) distributions per bin.
    grid_edges
        Redshift bin edges for stats evaluation

    Returns
    -------
    Figure
        Matplotlib Figure mean and rms
    """
    fig = plt.figure(figsize=(6,6))
    axes = fig.subplots(1, 1)

    estimate_stats = utils.histogram_stats_2d(nz_distributions, grid_edges)
    true_stats = utils.histogram_stats_2d(true_distributions, grid_edges)

    axes.scatter(true_stats['mean'], estimate_stats['mean'] - true_stats['mean'], label='Delta mean')
    axes.scatter(true_stats['mean'], estimate_stats['std'] - true_stats['std'], label='Delta rms')
    
    axes.set_xlabel('z')
    axes.set_ylabel(r'$\Delta$ Statistic')

    axes.set_ylim(-0.25, 0.25)
    axes.set_xlim()

    axes.legend()
    fig.tight_layout()
    return fig


def evaluate_submission(
    submit_dir: str | Path,
    public_dir: str | Path,
    truth_dir: str | Path,
    results_dir: str | Path,
    suffix: str = "wfd",
) -> None:
    """Run full evaluation of a submission across all tasksets, sims, and scenarios.

    Computes bin assignment and distribution metrics, generates confusion
    matrix and n(z) comparison plots, and writes results to YAML.

    Parameters
    ----------
    submit_dir
        Path to the submission directory containing n(z) and bhat files.
    public_dir
        Path to the public data directory with test WFD files.
    truth_dir
        Path to the truth directory containing true redshifts.
    results_dir
        Path to the output directory for plots and results YAML.
    suffix
        File suffix identifier (default 'wfd').
    """
    full_output: dict[str, dict[str, Any]] = {}

    Path(results_dir).mkdir(parents=True, exist_ok=True)

    for taskset in TASKSETS:

        tomo_bin_edges = utils.TOMO_BIN_EDGES[taskset]
        grid_edges = utils.Z_BIN_EDGES[taskset]
        grid_centers = 0.5*(grid_edges[0:-1]+grid_edges[1:])
        n_tomo_bins = len(tomo_bin_edges) - 1
        
        for sim in SIMS:
            for scenario in SCENARIOS:
                key = f"{taskset}_{sim}_{scenario}"

                nz_file = f"{submit_dir}/nz_challenge_{taskset}_{sim}_{scenario}_nz_estimate_{suffix}.hdf5"
                bhat_file = f"{submit_dir}/nz_challenge_{taskset}_{sim}_{scenario}_bhat_{suffix}.hdf5"
                truth_file = f'{truth_dir}/nz_challenge_{taskset}_{sim}_{scenario}_{suffix}.hdf5'

                nz_estimates = qp.read(nz_file)
                bhat_data = tables_io.read(bhat_file)
                truth = tables_io.read(truth_file)
                true_redshifts = truth['redshift']

                bin_assignments = np.squeeze(bhat_data['tomo_bin_index'])
                true_assignments = utils.get_true_bin_assignments(true_redshifts, tomo_bin_edges)

                nz_distributions = utils.get_nz_distributions(nz_estimates, grid_centers, n_tomo_bins)
                true_distributions = utils.get_true_nz_distributions(true_redshifts, bin_assignments, grid_edges, n_tomo_bins)

                counts = np.squeeze(nz_estimates.ancil['n_objects'])
                neff = forecast.tomo_bins_effective_density(counts, taskset, sim, scenario)

                res_cs, bias_res_cs = forecast.fisher_bias_forecast(
                    nz_estimates,
                    bhat_data,
                    neff,
                    mode='cosmic_shear',
                    truth=truth,
                )

                res_3x2pt, bias_res_3x2pt = forecast.fisher_bias_forecast(
                    nz_estimates,
                    bhat_data,
                    neff,
                    mode='3x2pt',
                    truth=truth,
                )
                
                full_output[key] = evaluate_bin_assignments(true_assignments, bin_assignments)
                full_output[key].update(
                    **evaluate_distributions(true_distributions, nz_distributions, grid_edges, nz_estimates.ancil['n_objects'])
                )
                full_output[key].update(
                    wowa_fom_cs=res_cs.fom('w_0', 'w_a'),
                    wowa_fom_3x2pt=res_3x2pt.fom('w_0', 'w_a'),                    
                )

                fig_confusion = plot_confusion_matrix(true_assignments, bin_assignments, n_tomo_bins)
                fig_nz = plot_nz_data(true_distributions, nz_distributions, grid_edges)
                fig_mean_and_rms = plot_nz_mean_and_rms(true_distributions, nz_distributions, grid_edges)

                corner_params = ["omega_m", "sigma_8", "w_0", "w_a"]

                fig_cs = res_cs.corner(corner_params, color="C3", label="Cosmic Shear")
                fig_cs.legend(loc="upper right", fontsize=10)
                bias_res_cs.corner_arrows(corner_params, fig=fig_cs, shifted_contour=True, color="C1")

                fig_3x2pt = res_cs.corner(corner_params, color="C3", label="3x2pt")
                fig_3x2pt.legend(loc="upper right", fontsize=10)
                bias_res_3x2pt.corner_arrows(corner_params, fig=fig_3x2pt, shifted_contour=True, color="C1")

                fig_confusion.savefig(f"{results_dir}/{key}_confusion_matrix.png")
                fig_nz.savefig(f"{results_dir}/{key}_nz_distributions.png")
                fig_mean_and_rms.savefig(f"{results_dir}/{key}_nz_mean_and_rms.png")

                fig_cs.savefig(f"{results_dir}/{key}_forecast_cosmic_shear.png")
                fig_3x2pt.savefig(f"{results_dir}/{key}_forecast_3x2pt.png")
                
                
    with open(f"{results_dir}/full_results.yaml", "w") as fout:
        yaml.dump(full_output, fout)        


def summarize_submissions(
    submissions: list[str],
    results_dir: str | Path,
) -> None:

    dataframe = build_summary_stats_dataframe(
        submissions,
        results_dir,
    )

    scores_df = build_scores_dataframe(dataframe)
    all_scores = get_all_scores(scores_df, submissions)
    
    # first do the per-submission plots
    for submission in submissions:
        score_matrix = extract_score_matrix(scores_df, submission)
        fig_score_matrix = plot_score_matrix(score_matrix)
        fig_score_matrix.savefig(f"{results_dir}/{submission}/score_matrix.png")

        for strip_plot in METRICS:
            the_fig = make_strip_plot(dataframe, [submission], strip_plot)
            the_fig.savefig(f"{results_dir}/{submission}/{strip_plot}.png")

    # Now do the top-level version
    for strip_plot in METRICS:
        the_fig = make_strip_plot(dataframe, submissions, strip_plot)
        the_fig.savefig(f"{results_dir}/{strip_plot}.png")

    all_scores.to_csv(f"{results_dir}/all_scores.csv", index=False)
        
        
RUN_LABELS: list[str] = []
RUN_LABEL_DICT: dict[str, int] = {}

for taskset in TASKSETS:
    for sim in SIMS:
        for scenario in SCENARIOS:
            run_label = f'{taskset}_{sim}_{scenario}'
            idx = len(RUN_LABELS)
            RUN_LABELS.append(run_label)
            RUN_LABEL_DICT[run_label] = idx


def get_tuple_from_key(
    key: str,
) -> tuple[int, int, int, int]:
    """Convert a run label key to its index components.

    Parameters
    ----------
    key
        Run label string of the form '{taskset}_{sim}_{scenario}'.

    Returns
    -------
    tuple
        Tuple of (flat_index, taskset_index, sim_index, scenario_index).
    """
    idx = RUN_LABEL_DICT[key]
    taskset = int(idx // 4)
    sim = int((idx % 4) // 2)
    scenario = idx % 2
    return (idx, taskset, sim, scenario)


def build_summary_stats_dataframe(
    submissions: list[str],
    results_top_dir: str | Path,
) -> pd.DataFrame:
    """Build a DataFrame of all metric values across submissions and runs.

    Parameters
    ----------
    submissions
        List of submission names (subdirectories of results_top_dir).
    results_top_dir
        Top-level directory containing per-submission result directories.

    Returns
    -------
    DataFrame
        DataFrame with columns for submission, run, taskset, sim,
        scenario, and each metric value.
    """
    out_dict: dict[str, list] = {}

    first = True
    for submission in submissions:
        results_dir = Path(results_top_dir) / submission
        results_file = results_dir / "full_results.yaml"

        with open(results_file, 'rb') as fin:
            submission_data = yaml.safe_load(fin)

        for key, data in submission_data.items():
            run, taskset, sim, scenario = get_tuple_from_key(key)

            if first:
                out_dict['submission'] = [submission]
                out_dict['run'] = [run]
                out_dict['taskset'] = [taskset]
                out_dict['sim'] = [sim]
                out_dict['scenario'] = [scenario]
                for metric in METRICS:
                    out_dict[metric] = [data[metric]]
                first = False
            else:
                out_dict['submission'].append(submission)
                out_dict['run'].append(run)
                out_dict['taskset'].append(taskset)
                out_dict['sim'].append(sim)
                out_dict['scenario'].append(scenario)
                for metric in METRICS:
                    out_dict[metric].append(data[metric])

    out_data = pd.DataFrame(out_dict)
    return out_data


def build_scores_dataframe(
    metrics_df: pd.DataFrame,
) -> pd.DataFrame:
    """Convert metric values to integer scores based on predefined ranges.

    Each metric is scored by counting how many predefined acceptable
    ranges it falls within (0 to 3).

    Parameters
    ----------
    metrics_df
        DataFrame of metric values (as produced by
        build_summary_stats_dataframe).

    Returns
    -------
    DataFrame
        DataFrame with metric columns replaced by integer scores.
    """
    out_data = metrics_df.to_dict()
    n_data = len(metrics_df)
    
    for metric, metric_info in METRICS.items():
        metric_data = metrics_df[metric]
        metric_ranges = metric_info['ranges']
        out_vector = np.zeros((n_data), dtype=int)
        for metric_range in metric_ranges:
            out_vector += np.bitwise_and(
                metric_data >= metric_range[0],
                metric_data <= metric_range[1],
            ).astype(int)

        out_data[metric] = out_vector
    out_df = pd.DataFrame(out_data)
    return out_df


def make_strip_plot(
    data: pd.DataFrame,
    submissions: list[str],
    metric_name: str,
) -> Figure:
    """Create a strip plot comparing a metric across submissions and runs.

    Displays scatter points for each submission across different
    taskset/sim/scenario combinations with shaded acceptable ranges.

    Parameters
    ----------
    data
        DataFrame containing columns 'submission', 'run', and the
        metric specified by metric_name.
    submissions
        List of submission names to include in the plot.
    metric_name
        Name of the metric column to plot (must be a key in METRICS).

    Returns
    -------
    Figure
        Matplotlib Figure containing the strip plot.
    """
    fig, ax = plt.subplots(figsize=(11, 5))  # wider, and grab ax explicitly

    y_label_strings = RUN_LABELS
    metric_info = METRICS[metric_name]
    metric_label = metric_info['label']
    metric_limits = metric_info['limits']
    metric_ranges = metric_info['ranges']
    
    n_y_labels = len(y_label_strings)
    y_min, y_max = -0.5, n_y_labels - 0.5

    n_methods = len(submissions)
    cmap = plt.get_cmap('tab20')
    colors = cmap(np.linspace(0, 1, n_methods))[:n_methods]

    # shaded bands first, so they sit under the points
    for metric_range in metric_ranges:
        ax.fill_between(metric_range, y_min, y_max, color="gray", alpha=0.1, zorder=0)

    metric_vals = data[metric_name]
    run_vals = data['run']
    handles = {}
    
    for i, submission in enumerate(submissions):
        mask = data['submission'] == submission
        if n_methods == 1:
            color='black'
        else:
            color=colors[i]
        handles[submission] = ax.scatter(
            metric_vals[mask],
            run_vals[mask],
            color=color,
            marker='o',
            alpha=0.7,
            label=submission,
            zorder=3,
            linewidths=1.2,
        )

    ax.set_yticks(np.arange(n_y_labels))
    ax.set_yticklabels(y_label_strings)
    ax.set_xlabel(metric_label)
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(metric_limits)

    all_handles = []
    for k in submissions:
        all_handles.append(handles[k])
    ax.legend(
        all_handles,
        submissions,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=8,
        handletextpad=0.3,
    )
    fig.tight_layout()
    return fig


def extract_score_matrix(
    scores_df: pd.DataFrame,
    submisison: str,
) -> np.ndarray:
    """Extract a score matrix for a single submission.

    Parameters
    ----------
    scores_df
        DataFrame of integer scores (as produced by build_scores_dataframe).
    submisison
        Name of the submission to extract.

    Returns
    -------
    ndarray
        2D array of shape (n_metrics, n_runs) with integer scores.
    """
    mask = scores_df['submission'] == submisison
    sub_data = scores_df[mask]

    score_matrix = np.zeros((len(sub_data), len(METRICS)), dtype=int)
    
    for i in range(len(sub_data)):
        for j, metric in enumerate(METRICS):
            score_matrix[j, i] = sub_data[metric][i]

    return score_matrix


def plot_score_matrix(
    score_matrix: np.ndarray,
) -> Figure:
    """Plot a score matrix as a color-coded image.

    Parameters
    ----------
    score_matrix
        2D array of integer scores, shape (n_metrics, n_runs).

    Returns
    -------
    Figure
        Matplotlib Figure with the score heatmap.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    y_label_strings = list(METRICS.keys())
    n_y_labels = len(y_label_strings)
    
    the_image = ax.imshow(score_matrix, cmap='rainbow_r')
    
    ax.set_xlabel('run')

    ax.set_yticks(np.arange(n_y_labels))
    ax.set_yticklabels(y_label_strings)
    
    fig.colorbar(the_image, ax=ax, label='Score')
    
    fig.tight_layout()
    return fig


def get_all_scores(
    scores_df: pd.DataFrame,
    submisisons: list[str],
) -> pd.DataFrame:
    """Compute aggregate scores per taskset for each submission.

    Parameters
    ----------
    scores_df
        DataFrame of integer scores (as produced by build_scores_dataframe).
    submisisons
        List of submission names to score.

    Returns
    -------
    DataFrame
        DataFrame with columns 'submission' and one column per taskset
        containing normalized aggregate scores in [0, 1].
    """
    n_tasksets = 2
    n_submissions = len(submisisons)

    submissions_list: list[str] = []
    taskset_scores: dict[str, np.ndarray] = {}
    for i in range(n_tasksets):
        taskset_scores[f'taskset_{i}'] = np.zeros((n_submissions))

    for i, submission in enumerate(submisisons):
        score_matrix = extract_score_matrix(scores_df, submission)

        for j in range(n_tasksets):
            the_slice = score_matrix[:,4*j:4*(j+1)]
            score = the_slice.sum() / (3.*the_slice.size)
            taskset_scores[f'taskset_{j}'][i] = score
        submissions_list.append(submission)

    out_dict: dict[str, Any] = {'submission': submissions_list}
    out_dict.update(taskset_scores)
    return pd.DataFrame(out_dict)


def make_submission_summary_rst(
    submissions: list[str],
    results_dir: str | Path,
    template_file: str | Path,
) -> None:
    """Render RST summary pages for each submission from a Jinja2 template.

    Parameters
    ----------
    results_dir : str
        Base directory for output RST files.
    submissions : list[str]
        List of submission identifiers.
    template_file : str
        Path to the Jinja2 RST template file.
    """
    with open(template_file, "r") as f:
        template = Template(f.read())

    base_path = Path(results_dir)

    # Generate RST file for each directory
    for submission_name in submissions:
        # Render template
        content = template.render(dir_name=submission_name)

        # Write to file
        output_path = base_path / Path(submission_name) / "index.rst"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            f.write(content)

            print(f"Generated: {output_path}")


def evaluate_all_submissions(
    submissions: list[str],
    submit_top_dir: str | Path,
    public_dir: str | Path,
    truth_dir: str | Path | None,    
    results_top_dir: str | Path,
    template_jinja_file: str | Path,
    suffix: str="wfd",
) -> None:

    Path(submit_top_dir).mkdir(parents=True, exist_ok=True)
    Path(results_top_dir).mkdir(parents=True, exist_ok=True)

    if truth_dir is None:
        raise ValueError("truth_dir must not be None")

    for submission in submissions:
        submit_dir = f'{submit_top_dir}/{submission}'
        results_dir = f'{results_top_dir}/{submission}'
        evaluate_submission(
            submit_dir,
            public_dir,
            truth_dir,
            results_dir,
            suffix=suffix,
        )
        
    summarize_submissions(
        submissions,
        results_top_dir,
    )

    make_submission_summary_rst(
        submissions,
        results_top_dir,
        template_file=template_jinja_file,
    )


def run_submission(
    submission_name: str, results_dir: str, *, force: bool = False
) -> None:
    """Run the code for a submission.

    Parameters
    ----------
    submission_name : str
        Name identifier for the submission.
    results_dir : str
        Path to the directory where results will be stored.
    """

    if os.environ.get("SKIP_RUN"):
        return

    if os.path.exists(os.path.join(results_dir, "pytest.log")) and not force:
        return

    if not os.environ.get("SKIP_INSTALL"):
        try:
            subprocess.run(
                ["pip", "install", "-r", f"requirements_{submission_name}.txt"],
                check=True,
            )
        except Exception:
            pass

    os.environ["NO_TEARDOWN"] = "1"

    try:
        os.makedirs(results_dir)
    except Exception:
        pass

    if not os.environ.get("SKIP_PYTEST"):
        output = subprocess.run(
            ["py.test", f"tests/test_{submission_name}.py"],
            check=True,
            capture_output=True,
        )

        with open(
            os.path.join(results_dir, "pytest.log"), "w", encoding="utf-8"
        ) as fout:
            fout.write(output.stdout.decode())


def run_submissions(
    submissions: list[str] | None,
    results_top_dir: str,
    accepeted_dir: str,
    *,
    force: bool = False,
) -> None:

    if submissions is None:
        submissions = get_submissions(accepeted_dir)

    for submission in submissions:
        run_submission(submission, f"{results_top_dir}/{submission}", force=force)

    
