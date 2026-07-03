"""Validation helpers for posterior traces and Gaia simulations."""

from .arviz_checks import (
    DiagnosticsResult,
    SimulationMetrics,
    check_mcmc_diagnostics,
    coverage,
    evaluate_distance_posterior,
    posterior_interval,
    summarize_elbo,
)

__all__ = [
    "DiagnosticsResult",
    "SimulationMetrics",
    "check_mcmc_diagnostics",
    "coverage",
    "evaluate_distance_posterior",
    "posterior_interval",
    "summarize_elbo",
]
