
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from . import metrics

TOMO_BIN_COLORS = [
    'violet', 'indigo', 'magenta', 'blue', 'cyan',
    'green', 'yellow', 'orange', 'red', 'gray',
]


def evaluate_bin_assignments(
    true_assignments,
    bin_assignments,
) -> dict[str, float]:

    the_dict: dict[str, float] = dict(
        log_loss = metrics.log_loss_from_labels(true_assignments, bin_assignments),
        accuracy = (bin_assignments == true_assignments).sum() / true_assignments.size,
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


    the_dict: dict[str, float] = dict(
        total_information_loss=metrics.total_information_loss(
            true_distributions, nz_distributions, n_objects,
        ),
    )
    the_dict.update(
        **metrics.rms0_delta_summary_stats(
            nz_distributions, 
            true_distributions, 
            grid_edges,
        )
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
