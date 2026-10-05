"""Fisher matrix forecasting for n(z) challenge submissions, using Augur.

The Augur pathway next to :mod:`nz_data_challenge.forecast` (fisherA2Z's
FisherFlex).  The analysis settings live in ``data/augur_forecast.yaml``: a base
Augur configuration, plus the few settings Augur has no place for.  For every
forecast this module writes the source n(z) to a file, fills in the
per-forecast parts of the configuration, and runs :func:`augur.generate.generate`
(firecrown theory, TJPCov covariance) and :class:`augur.analyze.Analyze`
(derivatives).

The Fisher matrix and the bias are assembled here from Augur's derivatives and
covariance rather than by ``Analyze.get_fisher_matrix`` / ``get_fisher_bias``:
Augur has no lower multipole cut, ``get_fisher_bias`` inverts the Fisher matrix
without priors, and its ``Omega_c -> Omega_m`` transform cannot be combined with
a bias.

Everything is written to the output directory: the n(z) files, the Augur
configuration, the sacc files, the Fisher ingredients (``forecast.npz``) and a
summary (``summary.yaml``).  A ``truth`` sub-directory holds the same for the
true n(z).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import qp
import yaml

from augur.analyze import Analyze
from augur.generate import generate
from augur.utils.config_io import parse_array

from . import utils

CONFIG_FILE = Path(__file__).parent / "data" / "augur_forecast.yaml"

# Redshift grid the n(z) are tabulated on.  The n(z) estimates are evaluated at
# the centres of the same bins the true redshifts are histogrammed in, so the
# two are directly comparable.
FORECAST_Z_MIN = 0.0
FORECAST_Z_MAX = 3.0
FORECAST_N_ZBINS = 150
FORECAST_Z_EDGES = np.linspace(FORECAST_Z_MIN, FORECAST_Z_MAX, FORECAST_N_ZBINS + 1)
FORECAST_Z_GRID = 0.5 * (FORECAST_Z_EDGES[1:] + FORECAST_Z_EDGES[:-1])

# Analysis modes, as in FisherFlex
MODE_PROBES: dict[str, tuple[str, ...]] = {
    "3x2pt": ("cs", "ggl", "gc"),
    "2x2pt": ("ggl", "gc"),
    "cosmic_shear": ("cs",),
}

# Sacc data type of each probe
PROBE_DATA_TYPES = dict(
    cs="galaxy_shear_cl_ee",
    ggl="galaxy_shearDensity_cl_e",
    gc="galaxy_density_cl",
)


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------


def load_config(path: str | Path = CONFIG_FILE) -> dict[str, Any]:
    """Read a forecast configuration (see ``data/augur_forecast.yaml``).

    The lens n(z) file is resolved relative to the configuration file.
    """
    path = Path(path)
    with open(path) as fin:
        config = yaml.safe_load(fin)
    lenses = config["augur"].get("lenses")
    if lenses is not None:
        nz_kwargs = lenses["Nz_kwargs"]
        nz_kwargs["input_file"] = str((path.parent / nz_kwargs["input_file"]).resolve())
    return config


def write_nz_file(path: str | Path, z: np.ndarray, nz: np.ndarray, neff: np.ndarray) -> float:
    """Write n(z) in the ascii layout read by Augur's ``ZDistFromFile``.

    Augur splits a sample's total number density across its bins in
    proportion to the sum of each bin's tabulated n(z).  Each row is therefore
    scaled to sum to its own ``neff``, which makes that split reproduce
    ``neff`` exactly; CCL normalizes n(z) internally, so the theory is
    unaffected.

    Returns
    -------
    float
        The total number density, to be passed as the sample's ``ndens``.
    """
    nz = np.clip(np.atleast_2d(np.asarray(nz, dtype=float)), 0.0, None)
    neff = np.atleast_1d(np.asarray(neff, dtype=float))
    sums = nz.sum(axis=1)
    if np.any(sums <= 0):
        raise ValueError("every n(z) bin must have a positive integral")
    scaled = nz / sums[:, None] * neff[:, None]
    header = "z " + " ".join(f"bin{i}" for i in range(nz.shape[0]))
    np.savetxt(path, np.column_stack([z, scaled.T]), header=header)
    return float(neff.sum())


def nz_mean(z: np.ndarray, nz: np.ndarray) -> np.ndarray:
    """Mean redshift of each row of ``nz``."""
    nz = np.atleast_2d(nz)
    return np.trapezoid(nz * z, z, axis=-1) / np.trapezoid(nz, z, axis=-1)


def ggl_pairs(
    z_mean_source: np.ndarray,
    z_eff_lens: list[float],
    offset: float,
) -> list[tuple[int, int]]:
    """Lens-source pairs ``(lens, source)`` with ``<z>_source > z_lens + offset``."""
    return [
        (l, s)
        for l in range(len(z_eff_lens))
        for s in range(len(z_mean_source))
        if z_mean_source[s] > z_eff_lens[l] + offset
    ]


def derivative_steps(
    fiducials: dict[str, float],
    rel_step: float,
    zero_fid_scale: dict[str, float],
) -> dict[str, float]:
    """Absolute derivative step of every parameter in ``fiducials``.

    ``rel_step`` times the fiducial value or, where that is zero, times the
    parameter's scale in ``zero_fid_scale``.
    """
    steps = {}
    for p, fid in fiducials.items():
        if fid != 0:
            steps[p] = rel_step * abs(float(fid))
        elif p in zero_fid_scale:
            steps[p] = rel_step * zero_fid_scale[p]
        else:
            raise ValueError(
                f"{p!r} has a fiducial value of 0 and no entry in "
                "derivative_zero_fid_scale to set its derivative step"
            )
    return steps


# --------------------------------------------------------------------------
# Augur configuration
# --------------------------------------------------------------------------


def _ell_bins(
    ell_edges_expr: str, ell_max: float, windows: bool
) -> tuple[list[float], np.ndarray]:
    """Bin edges handed to Augur, and all bin centres.

    The binning is cut after the last bin that passes ``ell_max``: TJPCov
    evaluates the covariance at every integer multipole up to the last edge, so
    carrying unused bins makes it far slower.  With bandpower windows one more
    bin is kept; see :func:`_augur_ell_max`.
    """
    edges = np.asarray(parse_array(ell_edges_expr), dtype=float)
    centres = np.sqrt(edges[1:] * edges[:-1])
    n_used = int(np.sum(centres <= ell_max))
    return [float(e) for e in edges[: n_used + 1 + int(windows)]], centres


def _augur_ell_max(ell_max: float, centres: np.ndarray, windows: bool) -> float:
    """The ``lmax`` to give Augur for a scale cut of ``ell_max``.

    Augur builds top-hat windows only up to the largest bin centre in the data
    vector, which would cut the last bin's window in half.  With windows every
    tracer pair therefore carries one extra bin past its scale cut, which
    :func:`_scale_cut_mask` drops again.
    """
    if not windows:
        return float(ell_max)
    return float(centres[centres > ell_max][0])


def build_augur_config(
    config: dict[str, Any],
    output_dir: str | Path,
    nz_source_file: str | Path,
    n_source: int,
    ndens_source: float,
    mode: str,
    pairs: list[tuple[int, int]],
    cov_type: str | None = None,
) -> dict[str, Any]:
    """Fill in the per-forecast parts of the base Augur configuration.

    Parameters
    ----------
    config
        Forecast configuration, as returned by :func:`load_config`.
    output_dir
        Directory the sacc file and Fisher outputs are written to.
    nz_source_file, n_source, ndens_source
        Source n(z) file (see :func:`write_nz_file`), number of bins and total
        number density.
    mode
        '3x2pt', '2x2pt' or 'cosmic_shear'.
    pairs
        Accepted ``(lens, source)`` GGL pairs.
    cov_type
        Override the configuration's covariance type, e.g. 'gaus_internal'
        when the covariance is not needed.

    Returns
    -------
    dict
        Configuration for :func:`augur.generate.generate` and
        :class:`augur.analyze.Analyze`.
    """
    if mode not in MODE_PROBES:
        raise ValueError(f"unknown mode {mode!r}; use one of {sorted(MODE_PROBES)}")
    analysis = config["analysis"]
    augur = copy.deepcopy(config["augur"])
    output_dir = Path(output_dir)
    probes = MODE_PROBES[mode]
    use_lenses = "ggl" in probes or "gc" in probes
    windows = augur["general"]["bandpower_windows"] == "TopHat"

    cs_min, cs_max = analysis["cosmic_shear_ell_range"]
    lens_lmax = analysis["lens_ell_max"]
    ell_max = max([cs_max] * ("cs" in probes) + list(lens_lmax) * use_lenses)
    ell_edges, centres = _ell_bins(analysis["ell_edges"], ell_max, windows)

    augur["sources"].update(nbins=int(n_source), ndens=float(ndens_source))
    augur["sources"]["Nz_kwargs"]["input_file"] = str(nz_source_file)
    var_pars = list(analysis["var_pars"])

    stats: dict[str, Any] = {}
    if "cs" in probes:
        stats[PROBE_DATA_TYPES["cs"]] = dict(
            tracer_combs=[[i, j] for i in range(n_source) for j in range(i, n_source)],
            ell_edges=ell_edges,
            lmax=_augur_ell_max(cs_max, centres, windows),
        )
    if use_lenses:
        var_pars += [f"lens{i}_bias" for i in range(augur["lenses"]["nbins"])]
        if "ggl" in probes and pairs:
            stats[PROBE_DATA_TYPES["ggl"]] = dict(
                tracer_combs=[[int(l), int(s)] for l, s in pairs],
                ell_edges=ell_edges,
                lmax=[_augur_ell_max(lens_lmax[l], centres, windows) for l, _ in pairs],
            )
        if "gc" in probes:
            stats[PROBE_DATA_TYPES["gc"]] = dict(
                tracer_combs=[[i, i] for i in range(len(lens_lmax))],
                ell_edges=ell_edges,
                lmax=[_augur_ell_max(v, centres, windows) for v in lens_lmax],
            )
    else:
        augur.pop("lenses", None)
        augur["systematics"] = {
            k: v for k, v in augur["systematics"].items() if not k.startswith("lens")
        }
    augur["statistics"] = stats
    augur["fiducial_sacc_path"] = str(output_dir / "datavector.sacc")

    cov_options = augur["cov_options"]
    if cov_type is not None:
        cov_options = dict(cov_type=cov_type, fsky=cov_options["fsky"])
    if cov_options["cov_type"] == "tjpcov":
        cov_options["binning_info"] = dict(ell_edges=ell_edges)
    augur["cov_options"] = cov_options

    # Augur's own fisher.step is a single absolute step for every parameter, so
    # the per-parameter steps are passed to Analyze.get_derivatives instead.
    fiducials = {**augur["cosmo"], **augur["systematics"]}
    augur["fisher"].update(
        var_pars=var_pars,
        derivative_steps=derivative_steps(
            {p: fiducials[p] for p in var_pars},
            analysis["derivative_rel_step"],
            analysis["derivative_zero_fid_scale"],
        ),
        output=str(output_dir / "fisher_no_priors.dat"),
        fid_output=str(output_dir / "fiducials.dat"),
    )
    return augur


def _write_config(config: dict[str, Any], path: Path) -> None:
    with open(path, "w") as fout:
        yaml.safe_dump(config, fout, sort_keys=False)


def _run_generate(config: dict[str, Any]):
    """Call Augur's ``generate``, protecting ``config`` from its in-place edits."""
    return generate(copy.deepcopy(config), return_all_outputs=True)


def _data_layout(lk, S) -> tuple[np.ndarray, np.ndarray, list]:
    """Sacc indices of the likelihood's data vector, its multipoles, and labels."""
    idx = np.concatenate([np.asarray(st.statistic.sacc_indices) for st in lk.statistics])
    ells = np.array([S.data[i]["ell"] for i in idx], dtype=float)
    labels = [(S.data[i].data_type, *S.data[i].tracers, S.data[i]["ell"]) for i in idx]
    return idx, ells, labels


def _scale_cut_mask(labels: list, analysis: dict[str, Any]) -> np.ndarray:
    """Which data points pass the scale cuts.

    Augur applies only the upper cuts, and with bandpower windows deliberately
    keeps one bin past them.
    """
    cs_min, cs_max = analysis["cosmic_shear_ell_range"]
    keep = []
    for data_type, tr1, tr2, ell in labels:
        if data_type == PROBE_DATA_TYPES["cs"]:
            keep.append(cs_min <= ell <= cs_max)
        else:
            lens = tr1 if tr1.startswith("lens") else tr2
            keep.append(ell <= analysis["lens_ell_max"][int(lens.removeprefix("lens"))])
    return np.array(keep)


# --------------------------------------------------------------------------
# results
# --------------------------------------------------------------------------


@dataclass
class AugurFisherResult:
    """Fisher matrix of one Augur forecast, priors included.

    Parameters are named as in Augur, except that ``Omega_m`` replaces
    ``Omega_c``.
    """

    fisher: np.ndarray
    param_order: list[str]
    param_fid: dict[str, float]
    prior_sigma: np.ndarray
    mask: np.ndarray
    n_data_used: int
    output_dir: str | None = None

    def __post_init__(self):
        self.cov = np.linalg.inv(self.fisher)

    def index(self, param: str) -> int:
        if param not in self.param_order:
            raise KeyError(f"unknown parameter {param!r}; available: {self.param_order}")
        return self.param_order.index(param)

    def sigma(self, param: str) -> float:
        """Marginalized 1-sigma uncertainty."""
        i = self.index(param)
        return float(np.sqrt(self.cov[i, i]))

    def sigmas(self) -> dict[str, float]:
        return {p: self.sigma(p) for p in self.param_order}

    def marginalize(self, p1: str, p2: str) -> np.ndarray:
        """2x2 marginal Fisher block (the inverse of the 2x2 covariance)."""
        idx = [self.index(p1), self.index(p2)]
        return np.linalg.inv(self.cov[np.ix_(idx, idx)])

    def fom(self, p1: str, p2: str) -> float:
        """Figure of merit ``1 / sqrt(det(Cov_2x2))``."""
        return float(np.sqrt(np.linalg.det(self.marginalize(p1, p2))))

    def s8(self) -> tuple[float, float]:
        """``S_8 = sigma8 sqrt(Omega_m / 0.3)`` and its propagated error."""
        om, s8v = self.param_fid["Omega_m"], self.param_fid["sigma8"]
        i, j = self.index("Omega_m"), self.index("sigma8")
        grad = np.array([0.5 * s8v / np.sqrt(0.3 * om), np.sqrt(om / 0.3)])
        c = self.cov[np.ix_([i, j], [i, j])]
        return float(s8v * np.sqrt(om / 0.3)), float(np.sqrt(grad @ c @ grad))

    def summary(self, params: list[str] | None = None) -> str:
        """One line per parameter: fiducial, sigma and prior width."""
        params = params or self.param_order
        lines = [f"{'parameter':<16}{'fiducial':>12}{'sigma':>12}{'prior':>12}"]
        for p in params:
            lines.append(
                f"{p:<16}{self.param_fid[p]:>12.5g}{self.sigma(p):>12.5g}"
                f"{self.prior_sigma[self.index(p)]:>12.5g}"
            )
        return "\n".join(lines)


@dataclass
class AugurBias2D:
    """Parameter bias in the plane of two parameters."""

    params: tuple[str, str]
    delta: np.ndarray
    cov: np.ndarray
    chi2: float

    @property
    def n_sigma(self) -> float:
        """Mahalanobis distance ``sqrt(chi2)`` of the shift."""
        return float(np.sqrt(self.chi2))

    @property
    def confidence(self) -> float:
        """Probability enclosed by the contour through the biased point."""
        return float(1.0 - np.exp(-0.5 * self.chi2))


@dataclass
class AugurFisherBias:
    """Parameter bias from analysing the truth data vector with the inferred n(z).

    ``delta_p = F^-1 . (dd/dp . Cov^-1 . (d_truth - d_inferred))``, with ``F``
    the Fisher matrix of :attr:`result`, priors included.
    """

    delta_p: dict[str, float]
    result: AugurFisherResult
    delta_dv: np.ndarray
    data_vector: np.ndarray

    def shift(self, param: str) -> float:
        return float(self.delta_p[param])

    def n_sigma(self, param: str) -> float:
        """Bias in units of the marginalized 1-sigma uncertainty."""
        return self.shift(param) / self.result.sigma(param)

    def n_sigmas(self) -> dict[str, float]:
        return {p: self.n_sigma(p) for p in self.result.param_order}

    def shifted_fid(self, param: str) -> float:
        return self.result.param_fid[param] + self.shift(param)

    def s8(self) -> tuple[float, float, float, float]:
        """``(S8_fid, S8_shifted, delta_S8, delta_S8 / sigma_S8)``.

        The shifted S8 is evaluated at the shifted ``(Omega_m, sigma8)``, as in
        FisherFlex.
        """
        s8_fid, s8_err = self.result.s8()
        s8_shift = self.shifted_fid("sigma8") * np.sqrt(self.shifted_fid("Omega_m") / 0.3)
        d = s8_shift - s8_fid
        return s8_fid, float(s8_shift), float(d), float(d / s8_err)

    def bias_2d(self, p1: str, p2: str) -> AugurBias2D:
        """Shift in the ``(p1, p2)`` plane against the marginal 2x2 covariance."""
        i, j = self.result.index(p1), self.result.index(p2)
        delta = np.array([self.shift(p1), self.shift(p2)])
        return AugurBias2D(
            params=(p1, p2),
            delta=delta,
            cov=self.result.cov[np.ix_([i, j], [i, j])],
            chi2=float(delta @ self.result.marginalize(p1, p2) @ delta),
        )

    def mafe(self) -> float:
        """Mean absolute fractional data-vector residual over the used elements."""
        m = self.result.mask
        return float(np.mean(np.abs(self.delta_dv[m] / self.data_vector[m])))

    def summary(self, params: list[str] | None = None) -> str:
        """One line per parameter: fiducial, bias, bias/sigma, shifted value."""
        params = params or self.result.param_order
        lines = [f"{'parameter':<16}{'fiducial':>12}{'bias':>12}{'bias/sigma':>12}{'shifted':>12}"]
        for p in params:
            lines.append(
                f"{p:<16}{self.result.param_fid[p]:>12.5g}{self.shift(p):>12.5g}"
                f"{self.n_sigma(p):>12.3f}{self.shifted_fid(p):>12.5g}"
            )
        return "\n".join(lines)


# Axis labels of the parameters, as FisherFlex writes them
PARAM_LABELS = dict(
    Omega_m=r"$\Omega_m$",
    sigma8=r"$\sigma_8$",
    n_s=r"$n_s$",
    w0=r"$w_0$",
    wa=r"$w_a$",
    Omega_b=r"$\Omega_b$",
    h=r"$h$",
    ia_bias=r"$A_{\rm IA}$",
    alphaz=r"$\alpha_z$",
)


def _param_label(param: str) -> str:
    if param.startswith("lens") and param.endswith("_bias"):
        # FisherFlex numbers the galaxy biases from 1
        return rf"$b_g^{{{int(param[4:-5]) + 1}}}$"
    return PARAM_LABELS.get(param, param)


def as_fisher_flex(result: AugurFisherResult, bias: AugurFisherBias):
    """Wrap a forecast and its bias in fisherA2Z's result classes.

    Lets the Augur results be plotted with ``FisherFlexResult.corner`` and
    ``FisherFlexBias.corner_arrows``, exactly like the fisherA2Z backend's.
    Only what the plotting reads is filled in; the n(z) fields of the bias
    are left empty.

    Returns
    -------
    FisherFlexResult, FisherFlexBias
    """
    from fisherA2Z.fisher_flex import FisherFlexBias, FisherFlexResult

    flex_result = FisherFlexResult(
        fisher=result.fisher,
        param_order=list(result.param_order),
        param_labels=[_param_label(p) for p in result.param_order],
        param_fid=dict(result.param_fid),
        mask=result.mask,
        n_data_used=result.n_data_used,
        prior_sigma=result.prior_sigma,
    )
    flex_bias = FisherFlexBias(
        delta_p=dict(bias.delta_p),
        bias_vec=None,
        result=flex_result,
        delta_cl=bias.delta_dv,
        data_vector=bias.data_vector,
        nz_truth=None,
        nz_fid=None,
        z=None,
        n_data_used=result.n_data_used,
    )
    return flex_result, flex_bias


# --------------------------------------------------------------------------
# the forecast
# --------------------------------------------------------------------------


def _to_omega_m_basis(names: list[str], fid: dict[str, float], deriv: np.ndarray):
    """Trade ``Omega_c`` for ``Omega_m`` at fixed ``Omega_b``.

    With ``Omega_c = Omega_m - Omega_b``, ``d/dOmega_m = d/dOmega_c`` and
    ``d/dOmega_b|Omega_m = d/dOmega_b|Omega_c - d/dOmega_c``.
    """
    names = list(names)
    deriv = deriv.copy()
    fid = dict(fid)
    i_c, i_b = names.index("Omega_c"), names.index("Omega_b")
    deriv[i_b] = deriv[i_b] - deriv[i_c]
    names[i_c] = "Omega_m"
    fid["Omega_m"] = fid.pop("Omega_c") + fid["Omega_b"]
    return names, fid, deriv


def _prior_fisher(names: list[str], priors: dict[str, float]):
    """Prior Fisher matrix and the 1-sigma prior width of every parameter.

    ``priors['lens_bias']`` applies to every ``lensN_bias``; parameters without
    a prior get an infinite width.
    """
    sigma = np.array([
        priors.get("lens_bias" if p.startswith("lens") and p.endswith("_bias") else p, np.inf)
        for p in names
    ], dtype=float)
    return np.diag(1.0 / sigma**2), sigma


def augur_forecast(
    nz_source: np.ndarray,
    neff: np.ndarray,
    mode: str,
    output_dir: str | Path,
    nz_truth: np.ndarray | None = None,
    z_grid: np.ndarray = FORECAST_Z_GRID,
    config: dict[str, Any] | None = None,
) -> tuple[AugurFisherResult, AugurFisherBias | None]:
    """Run an Augur Fisher forecast with the source n(z) taken as known.

    Parameters
    ----------
    nz_source
        Source n(z) per tomographic bin, shape ``(n_tomo, n_z)``; any
        normalization.
    neff
        Effective number density per tomographic bin (arcmin^-2).
    mode
        Analysis mode: '3x2pt', '2x2pt', or 'cosmic_shear'.
    output_dir
        Where the n(z) files, Augur configuration, sacc files and results are
        written.
    nz_truth
        True source n(z), shape ``(n_tomo, n_z)``.  If given, the parameter
        bias from analysing the truth data vector with ``nz_source`` is
        computed too.
    z_grid
        Redshift grid the n(z) are tabulated on.
    config
        Forecast configuration; :func:`load_config` of ``CONFIG_FILE`` by
        default.

    Returns
    -------
    result, bias
        The forecast, and the bias (``None`` without ``nz_truth``).
    """
    config = load_config() if config is None else config
    analysis = config["analysis"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    nz_source = np.atleast_2d(np.asarray(nz_source, dtype=float))
    neff = np.atleast_1d(np.asarray(neff, dtype=float))
    n_source = nz_source.shape[0]

    nz_source_file = output_dir / "nz_source.txt"
    ndens_source = write_nz_file(nz_source_file, z_grid, nz_source, neff)
    pairs = ggl_pairs(nz_mean(z_grid, nz_source), analysis["lens_z_eff"], analysis["ggl_z_offset"])

    augur_config = build_augur_config(
        config, output_dir, nz_source_file, n_source, ndens_source, mode, pairs
    )
    _write_config(augur_config, output_dir / "config.yaml")

    # -- data vector, covariance and derivatives ----------------------
    lk, S, tools, sys_params = _run_generate(augur_config)
    analyze = Analyze(augur_config, likelihood=lk, tools=tools, req_params=sys_params)
    steps = augur_config["fisher"]["derivative_steps"]
    deriv = np.asarray(
        analyze.get_derivatives(step=np.array([steps[p] for p in analyze.var_pars])),
        dtype=float,
    )
    fid = dict(zip(analyze.var_pars, analyze.x.astype(float)))
    names, fid, deriv = _to_omega_m_basis(analyze.var_pars, fid, deriv)

    idx, ells, labels = _data_layout(lk, S)
    cov = S.covariance.dense[np.ix_(idx, idx)]
    data_vector = np.asarray(lk.get_data_vector(), dtype=float)
    mask = _scale_cut_mask(labels, analysis)

    # -- Fisher matrix, with priors -----------------------------------
    prior, prior_sigma = _prior_fisher(names, analysis["priors"])
    d_m = deriv[:, mask]
    cov_m = cov[np.ix_(mask, mask)]
    fisher = d_m @ np.linalg.solve(cov_m, d_m.T)
    fisher = 0.5 * (fisher + fisher.T) + prior

    result = AugurFisherResult(
        fisher=fisher,
        param_order=names,
        param_fid=fid,
        prior_sigma=prior_sigma,
        mask=mask,
        n_data_used=int(mask.sum()),
        output_dir=str(output_dir),
    )

    # -- bias against the truth data vector ---------------------------
    bias = None
    delta = np.zeros_like(data_vector)
    if nz_truth is not None:
        truth_dir = output_dir / "truth"
        truth_dir.mkdir(exist_ok=True)
        nz_truth_file = truth_dir / "nz_source.txt"
        write_nz_file(nz_truth_file, z_grid, nz_truth, neff)
        # Same analysis as the inferred forecast -- including the GGL pairs --
        # with only the source n(z) replaced.  Its covariance is never used,
        # so skip TJPCov where Augur allows it: its internal Gaussian
        # covariance does not support bandpower windows.
        windows = augur_config["general"]["bandpower_windows"] == "TopHat"
        truth_config = build_augur_config(
            config, truth_dir, nz_truth_file, n_source, ndens_source, mode, pairs,
            cov_type="tjpcov" if windows else "gaus_internal",
        )
        _write_config(truth_config, truth_dir / "config.yaml")
        lk_t, S_t, _, _ = _run_generate(truth_config)
        if _data_layout(lk_t, S_t)[2] != labels:
            raise RuntimeError("truth and inferred data vectors are not aligned")
        delta = np.asarray(lk_t.get_data_vector(), dtype=float) - data_vector

        b_vec = d_m @ np.linalg.solve(cov_m, delta[mask])
        delta_p = result.cov @ b_vec
        bias = AugurFisherBias(
            delta_p=dict(zip(names, map(float, delta_p))),
            result=result,
            delta_dv=delta,
            data_vector=data_vector,
        )

    _save_products(output_dir, result, bias, deriv, cov, data_vector, delta, ells, labels)
    return result, bias


def _save_products(output_dir, result, bias, deriv, cov, data_vector, delta, ells,
                   labels) -> None:
    """Write the Fisher ingredients (npz) and a human-readable summary (YAML)."""
    np.savez_compressed(
        output_dir / "forecast.npz",
        param_names=np.array(result.param_order),
        param_fid=np.array([result.param_fid[p] for p in result.param_order]),
        prior_sigma=result.prior_sigma,
        fisher=result.fisher,
        derivatives=deriv,
        covariance=cov,
        data_vector=data_vector,
        delta_data_vector=delta,
        ell=ells,
        data_type=np.array([lab[0] for lab in labels]),
        tracer1=np.array([lab[1] for lab in labels]),
        tracer2=np.array([lab[2] for lab in labels]),
        mask=result.mask,
    )
    s8, s8_err = result.s8()
    summary: dict[str, Any] = dict(
        n_data_used=result.n_data_used,
        S8=s8,
        S8_err=s8_err,
        fom_w0_wa=result.fom("w0", "wa"),
        fiducial={p: float(v) for p, v in result.param_fid.items()},
        sigma=result.sigmas(),
    )
    if bias is not None:
        summary.update(
            bias=dict(bias.delta_p),
            bias_n_sigma=bias.n_sigmas(),
            S8_bias_n_sigma=bias.s8()[3],
            w0_wa_bias_n_sigma=bias.bias_2d("w0", "wa").n_sigma,
            mafe=bias.mafe(),
        )
    with open(output_dir / "summary.yaml", "w") as fout:
        yaml.safe_dump(summary, fout, sort_keys=False)


def fisher_bias_forecast(
    qp_nz_central: qp.Ensemble,
    bhat_table: dict[str, np.ndarray],
    neff: np.ndarray,
    mode: str,  # '3x2pt' | '2x2pt' | 'cosmic_shear'
    truth: dict[str, np.ndarray],
    output_dir: str | Path,
    config: dict[str, Any] | None = None,
) -> tuple[AugurFisherResult, AugurFisherBias]:
    """Run an Augur Fisher forecast and the bias from the true n(z).

    Augur counterpart of :func:`nz_data_challenge.forecast.fisher_bias_forecast`:
    the n(z) is taken as exactly known, and the bias is that of analysing the
    data vector of the true n(z) of the assigned bins with the estimated one.

    Parameters
    ----------
    qp_nz_central
        qp Ensemble containing the central n(z) estimates per tomographic bin.
    bhat_table
        Dictionary with at least a 'tomo_bin_index' key mapping to an array
        of bin assignments per object.
    neff
        Effective number density per tomographic bin (arcmin^-2).
    mode
        Analysis mode: '3x2pt', '2x2pt', or 'cosmic_shear'.
    truth
        Dictionary with at least a 'redshift' key, for computing the true
        n(z) distributions.
    output_dir
        Directory for the Augur products.
    config
        Forecast configuration; the packaged one by default.

    Returns
    -------
    AugurFisherResult, AugurFisherBias
    """
    nz_central = qp_nz_central.pdf(FORECAST_Z_GRID)
    nz_true = utils.get_true_nz_distributions(
        truth["redshift"],
        np.squeeze(bhat_table["tomo_bin_index"]),
        FORECAST_Z_EDGES,
        qp_nz_central.npdf,
    )

    print(f"effective number density in bins = {neff}")
    print(f"number counts in bins = {np.squeeze(qp_nz_central.ancil['n_objects'])}")

    return augur_forecast(
        nz_central, neff, mode, output_dir, nz_truth=nz_true, config=config
    )
