"""Visualization utilities for inspecting public challenge data."""

from pathlib import Path
import numpy as np
import tables_io
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


DDF_COLORS = [
    'violet', 'indigo', 'magenta', 'blue', 'cyan',
    'green', 'yellow', 'orange', 'red', 'gray',
]

RUBIN_BAND_COLORS = [
    'violet', 'indigo', 'blue',
    'green', 'orange', 'red',
]

ROMAN_BAND_COLORS = [
    'violet', 'indigo', 'blue',
]

MIN_MAG = 13.
MAX_MAG = 27.
MAG_N_BINS = 140
MAG_BINS = np.linspace(MIN_MAG, MAX_MAG, MAG_N_BINS+1) 

def get_data(
    public_data: str|Path,
    taskset: str,
    sim: str,
    scenario: str,
) -> tuple[dict, list[dict]]:
    """Load WFD and DDF data files for a given configuration.

    Parameters
    ----------
    public_data
        Path to the public data directory.
    taskset
        Name of the taskset (e.g., 'taskset_1').
    sim
        Name of the simulation ('cardinal' or 'flagship').
    scenario
        Name of the scenario ('1yr' or '4yr').

    Returns
    -------
    wfd
        Dictionary of WFD catalog data.
    ddf_list
        List of 10 DDF catalog dictionaries.
    """
    wfd_path = Path(public_data) / f"nz_challenge_{taskset}_{sim}_{scenario}_wfd.hdf5"
    wfd = tables_io.read(wfd_path)
    ddf_path_list = [
        Path(public_data) / f"nz_challenge_{taskset}_{sim}_{scenario}_ddf_{i:02}.hdf5" for i in range(5)
    ]
    ddf_list = [tables_io.read(ddf_path) for ddf_path in ddf_path_list]
    return wfd, ddf_list
    

def draw_footprint(wfd: dict, ddf_list: list[dict]) -> Figure:
    """Plot the sky footprint of WFD and DDF fields.

    Parameters
    ----------
    wfd
        WFD catalog dictionary with 'ra' and 'dec' keys.
    ddf_list
        List of DDF catalog dictionaries, each with 'ra' and 'dec' keys.

    Returns
    -------
    Figure
        Matplotlib Figure showing RA/Dec positions of WFD and DDF objects.
    """
    fig = plt.figure(figsize=(8, 12))
    axes = fig.subplots(1,1)
    axes.scatter(wfd['ra'][::50], wfd['dec'][::50], s=1, color='black')
    for i, ddf in enumerate(ddf_list):
        axes.scatter(ddf['ra'][::50], ddf['dec'][::50], s=1, color=DDF_COLORS[i])
    axes.set_xlabel('RA [degrees]')
    axes.set_ylabel('DEC [degrees]')
    fig.tight_layout()
    return fig


def draw_rubin_mags(wfd: dict, ddf: dict) -> Figure:
    """Plot Rubin LSST magnitude distributions for WFD and a DDF field.

    Parameters
    ----------
    wfd
        WFD catalog dictionary with 'mag_{band}_lsst' keys.
    ddf
        Single DDF catalog dictionary with 'mag_{band}_lsst' keys.

    Returns
    -------
    Figure
        Matplotlib Figure with histograms for each ugrizy band.
    """
    fig = plt.figure(figsize=(8, 12))
    axes = fig.subplots(6,2)
    for i, band in enumerate('ugrizy'):
        axes[i][0].hist(ddf[f'mag_{band}_lsst'], bins=MAG_BINS, color=RUBIN_BAND_COLORS[i])
        axes[i][0].set_xlabel(f'{band} [mag]')
        axes[i][0].set_ylabel("Objects / [0.1 mag]")
        axes[i][1].hist(wfd[f'mag_{band}_lsst'], bins=MAG_BINS, color=RUBIN_BAND_COLORS[i])
        axes[i][1].set_xlabel(f'{band} [mag]')
        axes[i][1].set_ylabel("Objects / [0.1 mag]")
    fig.tight_layout()
    return fig


def draw_roman_mags(wfd: dict, ddf: dict) -> Figure:
    """Plot Roman magnitude distributions for WFD and a DDF field.

    Parameters
    ----------
    wfd
        WFD catalog dictionary with 'mag_{band}_roman' keys.
    ddf
        Single DDF catalog dictionary with 'mag_{band}_roman' keys.

    Returns
    -------
    Figure
        Matplotlib Figure with histograms for each YJH band.
    """
    fig = plt.figure(figsize=(8, 6))
    axes = fig.subplots(3,2)
    for i, band in enumerate('YJH'):
        axes[i][0].hist(ddf[f'mag_{band}_roman'], bins=MAG_BINS, color=ROMAN_BAND_COLORS[i])
        axes[i][0].set_xlabel(f'{band} [mag]')
        axes[i][0].set_ylabel("Objects / [0.1 mag]")
        axes[i][1].hist(wfd[f'mag_{band}_roman'], bins=MAG_BINS, color=ROMAN_BAND_COLORS[i])
        axes[i][1].set_xlabel(f'{band} [mag]')
        axes[i][1].set_ylabel("Objects / [0.1 mag]")
    fig.tight_layout()
    return fig


