from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pandas as pd
import qp
import tables_io
import yaml

from . import metrics, utils
from .utils import TASKSETS, SIMS, SCENARIOS

TOMO_BIN_COLORS = [
    'violet', 'indigo', 'magenta', 'blue', 'cyan',
    'green', 'yellow', 'orange', 'red', 'gray',
]

METRICS = dict(
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

z_min = 0
z_max = 1.5
n_grid_points = 151
grid_edges = np.linspace(z_min, z_max, n_grid_points)
grid_centers = 0.5*(grid_edges[0:-1]+grid_edges[1:])


def evaluate_bin_assignments(
    true_assignments,
    bin_assignments,
) -> dict[str, float]:

    the_dict: dict[str, float] = dict(
        log_loss = metrics.log_loss_from_labels(true_assignments, bin_assignments),
        accuracy = float((bin_assignments == true_assignments).sum() / true_assignments.size),
        balanced_accuracy = metrics.balanced_accuracy(true_assignments, bin_assignments),
        cohens_kappa = metrics.cohens_kappa(true_assignments, bin_assignments),
        mutual_info = metrics.mutual_info(true_assignments, bin_assignments),
    )
    return the_dict


def evaluate_distributions(
    true_distributions,
    nz_distributions,
    grid_edges,
    n_objects,
) -> dict[str, float]:


    total_information_loss, per_bin_loss = metrics.total_information_loss(
        true_distributions, nz_distributions, n_objects,
    )
    rms0_delta_summary_stats = metrics.rms0_delta_summary_stats(
        nz_distributions,
        true_distributions,
        grid_edges,
    )

    the_dict: dict[str, float] = dict(
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
    true_distributions,
    nz_distributions,
    grid_edges,
) -> Figure:

    fig = plt.figure(figsize=(6,6))
    axes = fig.subplots(1, 1)

    for i, (nz_dist, true_dist) in enumerate(zip(nz_distributions, true_distributions)):
        axes.stairs(nz_dist, grid_edges, ls='--', color=TOMO_BIN_COLORS[i])
        axes.stairs(true_dist, grid_edges, color=TOMO_BIN_COLORS[i])
        axes.set_xlabel('z')
        axes.set_ylabel('Objects / [0.01]')

    fig.tight_layout()
    return fig


def evaluate_submission(
    submit_dir: str | Path,
    public_dir: str | Path,
    truth_dir: str | Path,
    results_dir: str | Path,
    tomo_bin_edges: np.ndarray,
    suffix: str = "wfd",
) -> None:

    full_output = {}

    for taskset in TASKSETS:
        for sim in SIMS:
            for scenario in SCENARIOS:
                key = f"{taskset}_{sim}_{scenario}"

                wfd_file = f"{public_dir}/nz_challenge_{taskset}_{sim}_{scenario}_{suffix}.hdf5"
                nz_file = f"{submit_dir}/nz_challenge_{taskset}_{sim}_{scenario}_nz_estimate_{suffix}.hdf5"
                bhat_file = f"{submit_dir}/nz_challenge_{taskset}_{sim}_{scenario}_bhat_{suffix}.hdf5"
                truth_file = f'{truth_dir}/nz_challenge_{taskset}_{sim}_{scenario}_{suffix}.hdf5'

                nz_estimates = qp.read(nz_file)
                test_data = tables_io.read(wfd_file)
                bhat_data = tables_io.read(bhat_file)
                truth = tables_io.read(truth_file)
                true_redshifts = truth['redshift']

                bin_assignments = np.squeeze(bhat_data['tomo_bin_index'])
                true_assignments = utils.get_true_bin_assignments(true_redshifts, tomo_bin_edges)

                nz_distributions = utils.get_nz_distributions(nz_estimates, grid_centers, 5)
                true_distributions = utils.get_true_nz_distributions(true_redshifts, bin_assignments, grid_edges, 5)

                full_output[key] = evaluate_bin_assignments(true_assignments, bin_assignments)
                full_output[key].update(
                    **evaluate_distributions(true_distributions, nz_distributions, grid_edges, nz_estimates.ancil['n_objects'])
                )

                fig_confusion = plot_confusion_matrix(true_assignments, bin_assignments, 5)
                fig_nz = plot_nz_data(true_distributions, nz_distributions, grid_edges)

                fig_confusion.savefig(f"{results_dir}/{key}_confusion_matrix.png")
                fig_nz.savefig(f"{results_dir}/{key}_nz_distributions.png")

    with open(f"{results_dir}/full_results.yaml", "w") as fout:
        yaml.dump(full_output, fout)


RUN_LABELS = []
RUN_LABEL_DICT = {}

TASKSETS = ['taskset_1', 'taskset_2']
SIMS = ['cardinal', 'flagship']
SCENARIOS = ['1yr', '4yr']

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
    idx = RUN_LABEL_DICT[key]
    taskset = int(idx // 4)
    sim = int((idx % 4) // 2)
    scenario = idx % 2
    return (idx, taskset, sim, scenario)


def build_summary_stats_dataframe(
    submissions: list[str],
    results_top_dir: str | Path,
) -> pd.DataFrame:

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
    """
    Create a generic strip plot for comparing metrics across configurations.

    Displays scatter points for each submission across different taskset/scenario
    combinations with highlighted metric ranges.

    Parameters
    ----------
    data
        Dictionary mapping submission names to (values, run_ids) tuples.
    metric_name
        Name of the metric in the DataFrame

    Returns
    -------
    fig
        Matplotlib Figure object containing the strip plot.

    Examples
    --------
    >>> fig = make_strip_plot(metric_data, "Bias", [-0.1, 0.1],
    ...                       [[-0.02, 0.02], [-0.05, 0.05]])
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
    colors = plt.cm.tab20(np.linspace(0, 1, n_methods))[: n_methods]

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

    n_tasksets = 2
    n_submissions = len(submisisons)
    
    out_dict = dict(
        submission=[],
    )
    for i in range(n_tasksets):
        out_dict[f'taskset_{i}'] = np.zeros((n_submissions))

    for i, submission in enumerate(submisisons):
        score_matrix = extract_score_matrix(scores_df, submission)

        for j in range(n_tasksets):
            the_slice = score_matrix[:,4*j:4*(j+1)]
            score = the_slice.sum() / (3.*the_slice.size)
            out_dict[f'taskset_{j}'][i] = score
        out_dict['submission'].append(submission)

    return pd.DataFrame(out_dict)
