"""Utilities to validate posterior traces from simulated Gaia data.

The functions in this module do not fit any model. They only consume an
``arviz.InferenceData`` object and arrays with known simulation truth. This is
intended for the validation stage: run the model in a notebook or Colab, save the
trace, then call these helpers to compute diagnostics and simulation metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import arviz as az
import numpy as np


@dataclass(frozen=True)
class DiagnosticsResult:
    """Summary of MCMC diagnostics for a posterior variable."""

    variable: str
    max_rhat: float | None
    min_ess_bulk: float | None
    min_ess_tail: float | None
    n_divergences: int | None
    passed: bool
    messages: tuple[str, ...]


@dataclass(frozen=True)
class SimulationMetrics:
    """Distance-posterior metrics against known simulated truth."""

    variable: str
    n: int
    rmse: float
    mean_bias: float
    mean_relative_bias: float
    coverage_68: float
    coverage_95: float
    mean_width_68: float
    mean_width_95: float
    median_width_95: float


def _posterior_array(idata: az.InferenceData, variable: str) -> np.ndarray:
    if not hasattr(idata, "posterior"):
        raise ValueError("InferenceData must contain a posterior group.")
    if variable not in idata.posterior:
        available = ", ".join(str(name) for name in idata.posterior.data_vars)
        raise KeyError(f"Variable {variable!r} not found. Available: {available}")
    values = idata.posterior[variable].values
    if values.ndim < 3:
        raise ValueError(
            f"Posterior variable {variable!r} must have chain, draw and item dimensions."
        )
    return np.asarray(values)


def _flatten_chain_draw(values: np.ndarray) -> np.ndarray:
    """Return samples with shape (sample, ...)."""

    chain, draw = values.shape[:2]
    return values.reshape(chain * draw, *values.shape[2:])


def posterior_interval(
    idata: az.InferenceData,
    variable: str = "distance_pc",
    prob: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute equal-tailed posterior intervals for a variable.

    Parameters
    ----------
    idata:
        ArviZ object with a posterior group.
    variable:
        Posterior variable to evaluate. For this project the default is
        ``distance_pc`` with shape ``(chain, draw, star)``.
    prob:
        Interval probability, e.g. 0.68 or 0.95.
    """

    if not 0.0 < prob < 1.0:
        raise ValueError("prob must be between 0 and 1.")

    samples = _flatten_chain_draw(_posterior_array(idata, variable))
    alpha = 1.0 - prob
    lower = np.quantile(samples, alpha / 2.0, axis=0)
    upper = np.quantile(samples, 1.0 - alpha / 2.0, axis=0)
    return lower, upper


def coverage(
    idata: az.InferenceData,
    truth: Iterable[float],
    variable: str = "distance_pc",
    prob: float = 0.95,
) -> float:
    """Empirical coverage of posterior intervals against known truth."""

    truth_array = np.asarray(truth, dtype=float)
    lower, upper = posterior_interval(idata, variable=variable, prob=prob)
    if truth_array.shape != lower.shape:
        raise ValueError(
            f"truth shape {truth_array.shape} does not match posterior item shape {lower.shape}."
        )
    return float(np.mean((truth_array >= lower) & (truth_array <= upper)))


def evaluate_distance_posterior(
    idata: az.InferenceData,
    true_distance_pc: Iterable[float],
    variable: str = "distance_pc",
    point_estimator: str = "mean",
) -> SimulationMetrics:
    """Evaluate distance posterior quality on simulations with known truth.

    Use this after fitting a model to GUMS/GOG simulated observations. The
    posterior variable must contain the distance for the same stars and in the
    same order as ``true_distance_pc``.
    """

    truth = np.asarray(true_distance_pc, dtype=float)
    samples = _flatten_chain_draw(_posterior_array(idata, variable))
    if point_estimator == "mean":
        posterior_point = np.mean(samples, axis=0)
    elif point_estimator == "median":
        posterior_point = np.median(samples, axis=0)
    else:
        raise ValueError(f"Unknown point_estimator: {point_estimator!r}; use 'mean' or 'median'.")
    if truth.shape != posterior_point.shape:
        raise ValueError(
            f"truth shape {truth.shape} does not match posterior item shape {posterior_point.shape}."
        )

    error = posterior_point - truth
    lower_68, upper_68 = posterior_interval(idata, variable=variable, prob=0.68)
    lower_95, upper_95 = posterior_interval(idata, variable=variable, prob=0.95)
    width_68 = upper_68 - lower_68
    width_95 = upper_95 - lower_95

    return SimulationMetrics(
        variable=variable,
        n=int(truth.size),
        rmse=float(np.sqrt(np.mean(error**2))),
        mean_bias=float(np.mean(error)),
        mean_relative_bias=float(np.mean(error / truth)),
        coverage_68=float(np.mean((truth >= lower_68) & (truth <= upper_68))),
        coverage_95=float(np.mean((truth >= lower_95) & (truth <= upper_95))),
        mean_width_68=float(np.mean(width_68)),
        mean_width_95=float(np.mean(width_95)),
        median_width_95=float(np.median(width_95)),
    )


def check_mcmc_diagnostics(
    idata: az.InferenceData,
    variable: str = "distance_pc",
    *,
    max_rhat: float = 1.01,
    min_ess_bulk: float = 400.0,
    min_ess_tail: float = 400.0,
) -> DiagnosticsResult:
    """Check standard MCMC diagnostics for a posterior variable.

    For VI traces this function is usually not meaningful because VI samples are
    often stored as one artificial chain. Use simulation coverage and ELBO
    stability for VI instead.
    """

    messages: list[str] = []
    rhat_max: float | None = None
    ess_bulk_min: float | None = None
    ess_tail_min: float | None = None
    n_divergences: int | None = None

    try:
        rhat = az.rhat(idata, var_names=[variable])[variable].values
        rhat_max = float(np.nanmax(rhat))
        if rhat_max > max_rhat:
            messages.append(f"max R-hat {rhat_max:.4f} > {max_rhat:.2f}")
    except Exception as exc:  # noqa: BLE001 - diagnostic function should report all issues.
        messages.append(f"could not compute R-hat: {exc}")

    try:
        ess_bulk = az.ess(idata, var_names=[variable], method="bulk")[variable].values
        ess_tail = az.ess(idata, var_names=[variable], method="tail")[variable].values
        ess_bulk_min = float(np.nanmin(ess_bulk))
        ess_tail_min = float(np.nanmin(ess_tail))
        if ess_bulk_min < min_ess_bulk:
            messages.append(f"min bulk ESS {ess_bulk_min:.1f} < {min_ess_bulk:.0f}")
        if ess_tail_min < min_ess_tail:
            messages.append(f"min tail ESS {ess_tail_min:.1f} < {min_ess_tail:.0f}")
    except Exception as exc:  # noqa: BLE001
        messages.append(f"could not compute ESS: {exc}")

    sample_stats = getattr(idata, "sample_stats", None)
    if sample_stats is not None and "diverging" in sample_stats:
        n_divergences = int(np.asarray(sample_stats["diverging"].values).sum())
        if n_divergences > 0:
            messages.append(f"NUTS reported {n_divergences} divergences")

    return DiagnosticsResult(
        variable=variable,
        max_rhat=rhat_max,
        min_ess_bulk=ess_bulk_min,
        min_ess_tail=ess_tail_min,
        n_divergences=n_divergences,
        passed=not messages,
        messages=tuple(messages),
    )


def summarize_elbo(losses: Iterable[float], tail_fraction: float = 0.1) -> dict[str, float]:
    """Summarize VI loss/ELBO stability from an SVI loss history.

    NumPyro's SVI reports a loss to minimize (negative ELBO). A stable tail is a
    necessary but not sufficient condition for a useful VI approximation.
    """

    values = np.asarray(list(losses), dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("losses must be a non-empty one-dimensional sequence.")
    if not 0.0 < tail_fraction <= 1.0:
        raise ValueError("tail_fraction must be in (0, 1].")

    tail_n = max(1, int(np.ceil(values.size * tail_fraction)))
    tail = values[-tail_n:]
    return {
        "n_iterations": float(values.size),
        "initial_loss": float(values[0]),
        "final_loss": float(values[-1]),
        "best_loss": float(np.min(values)),
        "tail_mean": float(np.mean(tail)),
        "tail_sd": float(np.std(tail, ddof=0)),
        "tail_relative_sd": float(np.std(tail, ddof=0) / max(abs(np.mean(tail)), 1e-12)),
    }
