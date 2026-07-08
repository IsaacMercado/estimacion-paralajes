# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 3.0 Experimento principal
#
# Este notebook es el esqueleto del experimento final. La idea es ejecutarlo en una
# máquina con más recursos, por ejemplo Google Colab, sin mezclarlo con los notebooks
# exploratorios anteriores.
#
# Ruta final:
#
# 1. cargar muestra simulada GUMS/GOG con verdad conocida;
# 2. preparar variables observadas y cortes de calidad;
# 3. construir grilla fija del CMD independiente de la muestra de evaluación;
# 4. ajustar Modelo 0 (solo paralaje, prior uniforme truncado);
# 5. ajustar Modelo 1 (CMD marginalizado, pesos flexibles);
# 6. ajustar Modelo 2 (CMD marginalizado + escala global lambda);
# 7. validar VI con métricas contra verdad simulada y NUTS en submuestra;
# 8. guardar trazas ArviZ y tablas listas para el Capítulo IV.

# %% [markdown]
# ## Instalación en Colab
#
# Si se ejecuta en Colab, correr una celda equivalente a:
#
# ```python
# !pip install -q "jax[cuda12]" numpyro arviz blackjax pandas pyarrow astropy h5netcdf
# ```
#
# Ajustar `jax[cuda12]` según la versión de CUDA disponible. Si no hay GPU, usar
# `jax[cpu]`.

# %%
import jax
print('Devices:', jax.devices())
print('Backend:', jax.default_backend())

# %%
# !pip install "project[colab] @ git+https://github.com/IsaacMercado/estimacion-paralajes.git"

# %%
from pathlib import Path

import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
import polars as pl

from jax import random
from jax.scipy.special import logsumexp
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoLowRankMultivariateNormal
from numpyro.optim import Adam

from project.validation import evaluate_distance_posterior, summarize_elbo
from project.utils import DATA_RAW_DIR
from project.utils.load import load_with_polars

jax.config.update("jax_enable_x64", True)

# %%
SIMULATION_FILE = DATA_RAW_DIR / "simulation_data.ecsv"
SEED = 2026

# %% [markdown]
# ## Preparación de datos
#
# Reemplazar esta función con la lectura final de la muestra simulada. Es importante
# guardar y conservar `source_id` para verificar que la traza y la tabla estén en el
# mismo orden.

# %%

# %%
# def flux_to_mag_sigma(flux, flux_error):
#     flux = np.asarray(flux, dtype=float)
#     flux_error = np.asarray(flux_error, dtype=float)
#     return 2.5 / np.log(10.0) * flux_error / flux


def flux_to_mag_sigma(flux_col: str, flux_error_col: str) -> pl.Expr:
    snr = pl.col(flux_col) / pl.col(flux_error_col)
    return (2.5 / np.log(10.0) / snr).cast(pl.Float32)


def load_simulated_catalog(path: Path) -> pl.DataFrame:
    df, columns = load_with_polars(path)
    return (
        df.select(
            "source_id",
            "barycentric_distance",
            "parallax",
            "parallax_error",
            "phot_g_mean_mag",
            "phot_bp_mean_mag",
            "phot_rp_mean_mag",
            "phot_g_mean_flux",
            "phot_g_mean_flux_error",
            "phot_bp_mean_flux",
            "phot_bp_mean_flux_error",
            "phot_rp_mean_flux",
            "phot_rp_mean_flux_error",
        )
        .drop_nans()
        .drop_nulls()
        .filter(
            (pl.col("barycentric_distance") > 0)
            & (pl.col("parallax_error") > 0)
            & (pl.col("phot_g_mean_flux") > 0)
            & (pl.col("phot_bp_mean_flux") > 0)
            & (pl.col("phot_rp_mean_flux") > 0)
            & (pl.col("phot_g_mean_flux_error") > 0)
            & (pl.col("phot_bp_mean_flux_error") > 0)
            & (pl.col("phot_rp_mean_flux_error") > 0)
        )
        .with_columns(
            (pl.col("parallax") / pl.col("parallax_error")).alias("parallax_snr"),
            (pl.col("phot_bp_mean_mag") - pl.col("phot_rp_mean_mag")).alias("color_obs"),
        )
        .with_columns(
            flux_to_mag_sigma("phot_g_mean_flux", "phot_g_mean_flux_error").alias("sigma_g_mag"),
            flux_to_mag_sigma("phot_bp_mean_flux", "phot_bp_mean_flux_error").alias("sigma_bp_mag"),
            flux_to_mag_sigma("phot_rp_mean_flux", "phot_rp_mean_flux_error").alias("sigma_rp_mag"),
        )
        .with_columns(
            (
                (pl.col("sigma_bp_mag").pow(2) + pl.col("sigma_rp_mag").pow(2))
                .sqrt()
                .alias("sigma_color")
            ),
        )
    )


def select_simulation_sample(df: pl.DataFrame, n_stars: int, seed: int) -> pd.DataFrame:
    pool = df.filter(
        (pl.col("parallax_snr") > 1.0)
        & ((pl.col("phot_g_mean_flux") / pl.col("phot_g_mean_flux_error")) > 50.0)
        & ((pl.col("phot_bp_mean_flux") / pl.col("phot_bp_mean_flux_error")) > 20.0)
        & ((pl.col("phot_rp_mean_flux") / pl.col("phot_rp_mean_flux_error")) > 20.0)
    )
    if pool.height > n_stars:
        pool = pool.sample(n=n_stars, seed=seed)
    return pool.sort("source_id")


# %% [markdown]
# ## Grilla fija del CMD
#
# No construir esta grilla con cuantiles de la misma muestra evaluada. Aquí se usa un
# rango fijo amplio para Gaia. Ajustar resolución según recursos disponibles.

# %%
def build_fixed_cmd_grid(
    color_range=(-0.5, 4.0),
    abs_mag_range=(-2.0, 12.0),
    n_color=25,
    n_abs_mag=35,
):
    color_edges = np.linspace(color_range[0], color_range[1], n_color + 1)
    mag_edges = np.linspace(abs_mag_range[0], abs_mag_range[1], n_abs_mag + 1)
    color_step = color_edges[1] - color_edges[0]
    mag_step = mag_edges[1] - mag_edges[0]
    color_centers = 0.5 * (color_edges[:-1] + color_edges[1:])
    mag_centers = 0.5 * (mag_edges[:-1] + mag_edges[1:])
    grid_color, grid_mag = np.meshgrid(color_centers, mag_centers, indexing="xy")
    return {
        "mu_color": grid_color.ravel(),
        "mu_abs_mag": grid_mag.ravel(),
        "sigma_color_kernel": np.full(grid_color.size, color_step),
        "sigma_abs_mag_kernel": np.full(grid_mag.size, mag_step),
        "color_edges": color_edges,
        "abs_edges": mag_edges,
        "shape": grid_mag.shape,
    }


def prepare_model_data(df: pd.DataFrame, grid: dict, r_min=10.0, r_max=20_000.0) -> dict:
    return {
        "parallax": df["parallax"].to_jax(),
        "parallax_error": df["parallax_error"].to_jax(),
        "m_obs": df["phot_g_mean_mag"].to_jax(),
        "sigma_m": df["sigma_g_mag"].to_jax(),
        "color_obs": df["color_obs"].to_jax(),
        "sigma_color": df["sigma_color"].to_jax(),
        "r_min": float(r_min),
        "r_max": float(r_max),
        **grid,
    }


# %% [markdown]
# ## Modelos NumPyro

# %%
def model_0_parallax_only(parallax, parallax_error, r_min, r_max, **_):
    n_stars = parallax.shape[0]
    with numpyro.plate("star", n_stars):
        distance_pc = numpyro.sample("distance_pc", dist.Uniform(r_min, r_max))
        numpyro.sample("parallax_like", dist.Normal(1000.0 / distance_pc, parallax_error), obs=parallax)


def model_1_cmd_marginalized(
    parallax,
    parallax_error,
    m_obs,
    sigma_m,
    color_obs,
    sigma_color,
    r_min,
    r_max,
    mu_abs_mag,
    mu_color,
    sigma_abs_mag_kernel,
    sigma_color_kernel,
    **_,
):
    n_stars = parallax.shape[0]
    n_kernels = mu_abs_mag.shape[0]
    weights = numpyro.sample("weights", dist.Dirichlet(jnp.ones(n_kernels)))

    with numpyro.plate("star", n_stars):
        distance_pc = numpyro.sample("distance_pc", dist.Uniform(r_min, r_max))

    log_parallax = dist.Normal(1000.0 / distance_pc, parallax_error).log_prob(parallax)
    mu_m = mu_abs_mag[None, :] + 5.0 * jnp.log10(distance_pc[:, None]) - 5.0
    sigma_m_total = jnp.sqrt(sigma_m[:, None] ** 2 + sigma_abs_mag_kernel[None, :] ** 2)
    sigma_c_total = jnp.sqrt(sigma_color[:, None] ** 2 + sigma_color_kernel[None, :] ** 2)
    log_cmd = logsumexp(
        jnp.log(weights + 1e-12)[None, :]
        + dist.Normal(mu_m, sigma_m_total).log_prob(m_obs[:, None])
        + dist.Normal(mu_color[None, :], sigma_c_total).log_prob(color_obs[:, None]),
        axis=-1,
    )
    numpyro.factor("obs_loglike", jnp.sum(log_parallax + log_cmd))


def model_2_cmd_lambda(**kwargs):
    lambda_cmd = numpyro.sample("lambda_cmd", dist.LogNormal(0.0, 0.25))
    modified = dict(kwargs)
    modified["sigma_abs_mag_kernel"] = kwargs["sigma_abs_mag_kernel"] * lambda_cmd
    modified["sigma_color_kernel"] = kwargs["sigma_color_kernel"] * lambda_cmd
    return model_1_cmd_marginalized(**modified)


# %% [markdown]
# ## Ajuste VI

# %%
def run_vi(model, model_data, *, seed, steps=20_000, lr=0.01, rank=25, samples=4_000):
    guide = AutoLowRankMultivariateNormal(model, rank=rank)
    svi = SVI(model, guide, Adam(lr), loss=Trace_ELBO())
    rng_key = random.PRNGKey(seed)
    result = svi.run(rng_key, steps, progress_bar=True, **model_data)
    sample_key = random.PRNGKey(seed + 1)
    posterior = guide.sample_posterior(sample_key, result.params, sample_shape=(samples,), **model_data)
    idata = az.from_dict(
        posterior={name: np.asarray(value)[None, ...] for name, value in posterior.items()},
        coords={"star": np.arange(model_data["parallax"].shape[0])},
        dims={"distance_pc": ["star"]},
    )
    return result, idata


# %% [markdown]
# ## Ejecución final
#
# Descomentar y ajustar `N_STARS` cuando se ejecute en Colab.

# %%
N_STARS = 10_000
catalog = load_simulated_catalog(SIMULATION_FILE)
sample = select_simulation_sample(catalog, n_stars=N_STARS, seed=SEED)
grid = build_fixed_cmd_grid(n_color=25, n_abs_mag=35)
model_data_np = prepare_model_data(sample, grid, r_min=10.0, r_max=20_000.0)
model_data = {key: jnp.asarray(value) if isinstance(value, np.ndarray) else value for key, value in model_data_np.items()}

# %%
vi0, idata0 = run_vi(model_0_parallax_only, model_data, seed=SEED)

# %%
# vi1, idata1 = run_vi(model_1_cmd_marginalized, model_data, seed=SEED + 10)
# vi2, idata2 = run_vi(model_2_cmd_lambda, model_data, seed=SEED + 20)
# idata1.to_netcdf(MODELS_DIR / "modelo_1_vi.nc")
# idata2.to_netcdf(MODELS_DIR / "modelo_2_vi.nc")

# %% [markdown]
# ## Validación contra verdad simulada

# %%
# true_distance = sample["barycentric_distance"].to_numpy(dtype=float)
# metrics_m1 = evaluate_distance_posterior(idata1, true_distance, variable="distance_pc")
# metrics_m2 = evaluate_distance_posterior(idata2, true_distance, variable="distance_pc")
# elbo_m1 = summarize_elbo(vi1.losses)
# elbo_m2 = summarize_elbo(vi2.losses)
# metrics_m1, metrics_m2, elbo_m1, elbo_m2

# %% [markdown]
# ## Pendientes para la corrida final
#
# - Agregar Modelo 0 y sus métricas a la tabla.
# - Agregar NUTS en submuestra de 500-1000 estrellas.
# - Guardar `sample_source_id` junto a cada resultado para verificar orden.
# - Estratificar métricas por S/N de paralaje.
# - Generar figuras del Capítulo IV.
