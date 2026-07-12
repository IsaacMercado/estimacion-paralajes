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
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown] id="30f3e3bb"
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

# %% [markdown] id="3b5f1a2f"
# ## Instalación en Colab

# %% colab={"base_uri": "https://localhost:8080/"} id="4082be2a-3a85-43ce-9dea-a9ce44c6a514" outputId="c86b2956-8aad-4a7c-c6f1-be0cb366b367"
# !pip install -q "project[colab] @ git+https://github.com/IsaacMercado/estimacion-paralajes.git"

# %% colab={"base_uri": "https://localhost:8080/"} id="e07b2c7a" outputId="67d792c5-3fdb-4c3b-f7c7-0f6e11ac1ed6"
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
from numpyro.infer import MCMC, NUTS
from numpyro.infer.autoguide import AutoLowRankMultivariateNormal
from numpyro.optim import Adam

from project.validation import evaluate_distance_posterior, summarize_elbo
from project.utils import DATA_RAW_DIR, MODELS_DIR
from project.utils.load import load_with_polars

jax.config.update("jax_enable_x64", True)

print('Devices:', jax.devices())
print('Backend:', jax.default_backend())

# %% id="zW9KtaaPNMcN"
SIMULATION_FILENAME = "simulation_data.ecsv"
SIMULATION_FILE = DATA_RAW_DIR / SIMULATION_FILENAME
SEED = 2026

# %% colab={"base_uri": "https://localhost:8080/"} id="WQxMtNlNL3Vs" outputId="3a9258a9-112b-41f5-ded6-8a5428dfd4bb"
try:
  from google.colab import drive
  import shutil

  path_drive = Path("/content/drive")
  drive.mount(path_drive.as_posix())

  path_simulation_file = path_drive / "MyDrive" / SIMULATION_FILENAME
  if not path_simulation_file.exists():
      raise FileNotFoundError(f"File {path_simulation_file} not found.")

  shutil.copyfile(path_simulation_file, SIMULATION_FILE)
except ModuleNotFoundError:
  print("Not running in Colab.")

# %% [markdown] id="067aa602"
# ## Preparación de datos
#
# Reemplazar esta función con la lectura final de la muestra simulada. Es importante
# guardar y conservar `source_id` para verificar que la traza y la tabla estén en el
# mismo orden.

# %% id="92d216df"
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


# %% [markdown] id="c0cb20de"
# ## Grilla fija del CMD
#
# No construir esta grilla con cuantiles de la misma muestra evaluada. Aquí se usa un
# rango fijo amplio para Gaia. Ajustar resolución según recursos disponibles.

# %% id="15fb9830"
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


# %% [markdown] id="cb2752a1"
# ## Modelos NumPyro

# %% id="cdc33cef"
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


# %% [markdown] id="1d6e7eeb"
# ## Ajuste VI

# %% id="155942cd"
def run_vi(model, model_data, *, seed, steps=20_000, lr=0.01, rank=25, samples=4_000):
    guide = AutoLowRankMultivariateNormal(model, rank=rank)
    svi = SVI(model, guide, Adam(lr), loss=Trace_ELBO())
    rng_key = random.PRNGKey(seed)
    result = svi.run(rng_key, steps, progress_bar=True, **model_data)
    sample_key = random.PRNGKey(seed + 1)
    with jax.default_device(jax.devices("cpu")[0]):
        posterior = guide.sample_posterior(sample_key, result.params, sample_shape=(samples,), **model_data)
    idata = az.from_dict(
        posterior={name: np.asarray(value)[None, ...] for name, value in posterior.items()},
        coords={"star": np.arange(model_data["parallax"].shape[0])},
        dims={"distance_pc": ["star"]},
    )
    return result, idata


# %% [markdown] id="fd45c398"
# ## Ejecución final
#
# Descomentar y ajustar `N_STARS` cuando se ejecute en Colab.

# %% id="2781bbd3"
N_STARS = 50_000
catalog = load_simulated_catalog(SIMULATION_FILE)
sample = select_simulation_sample(catalog, n_stars=N_STARS, seed=SEED)
grid = build_fixed_cmd_grid(n_color=25, n_abs_mag=35)
model_data_np = prepare_model_data(sample, grid, r_min=10.0, r_max=20_000.0)
model_data = {key: jnp.asarray(value) if isinstance(value, np.ndarray) else value for key, value in model_data_np.items()}

# %% id="tEZqYIrMpt2b"
numpyro.render_model(
    model_0_parallax_only,
    model_kwargs=model_data_np,
    render_distributions=True,
)

# %% id="5VYGZON8pto_"
numpyro.render_model(
    model_1_cmd_marginalized,
    model_kwargs=model_data_np,
    render_distributions=True,
)

# %% id="jhJ332EgnYJ4"
numpyro.render_model(
    model_2_cmd_lambda,
    model_kwargs=model_data_np,
    render_distributions=True,
)

# %% colab={"base_uri": "https://localhost:8080/"} id="c12e978e-541f-43ee-8b06-bd9118052289" outputId="1f91bf9b-3c39-4bdc-bc4d-013ecaea1ca0"
vi0, idata0 = run_vi(model_0_parallax_only, model_data, seed=SEED)

# %% colab={"base_uri": "https://localhost:8080/"} id="8eazoRPePzoy" outputId="b5d9f5ad-6bd9-4351-e410-02cab845f3de"
vi1, idata1 = run_vi(model_1_cmd_marginalized, model_data, seed=SEED + 10)

# %% colab={"base_uri": "https://localhost:8080/"} id="FhgpEsKZP2vI" outputId="415bd9d9-88fa-4398-ea9c-5c76c3308565"
vi2, idata2 = run_vi(model_2_cmd_lambda, model_data, seed=SEED + 20)

# %% colab={"base_uri": "https://localhost:8080/"} id="bed0a057" outputId="dabb2612-76d2-443c-8fa6-281e5f928cd5"
idata0.to_netcdf(MODELS_DIR / "modelo_0_vi.nc")
idata1.to_netcdf(MODELS_DIR / "modelo_1_vi.nc")
idata2.to_netcdf(MODELS_DIR / "modelo_2_vi.nc")

# %% [markdown] id="ed18b0e5"
# ## Validación contra verdad simulada

# %% colab={"base_uri": "https://localhost:8080/"} id="48507d8b" outputId="737ba147-5ab0-427f-f022-272985342648"
true_distance = sample["barycentric_distance"].to_numpy()
metrics_m0 = evaluate_distance_posterior(idata0, true_distance, variable="distance_pc")
metrics_m1 = evaluate_distance_posterior(idata1, true_distance, variable="distance_pc")
metrics_m2 = evaluate_distance_posterior(idata2, true_distance, variable="distance_pc")

elbo_m0 = summarize_elbo(vi0.losses)
elbo_m1 = summarize_elbo(vi1.losses)
elbo_m2 = summarize_elbo(vi2.losses)

metrics_m0, metrics_m1, metrics_m2, elbo_m0, elbo_m1, elbo_m2


# %% [markdown] id="be6b3be9"
# ## Pendientes para la corrida final
#
# - Agregar Modelo 0 y sus métricas a la tabla.
# - Agregar NUTS en submuestra de 500-1000 estrellas.
# - Guardar `sample_source_id` junto a cada resultado para verificar orden.
# - Estratificar métricas por S/N de paralaje.
# - Generar figuras del Capítulo IV.

# %% [markdown] id="c1fe7577"
# ### Muestreo MCMC (NUTS) en submuestra
# Ejecutamos NUTS para obtener la posterior exacta en un subconjunto de los datos y validar los resultados de VI.

# %% id="bzOoQ2Z7ZGvJ"
def run_nuts(model, model_data, *, seed, num_samples=1000, num_warmup=500):
    nuts_kernel = NUTS(model)
    mcmc = MCMC(
        nuts_kernel,
        num_samples=num_samples,
        num_warmup=num_warmup,
        num_chains=4,
        progress_bar=True,
    )
    rng_key = random.PRNGKey(seed)
    mcmc.run(rng_key, **model_data)

    posterior_samples = mcmc.get_samples()
    idata = az.from_dict(
        posterior={name: np.asarray(value)[None, ...] for name, value in posterior_samples.items()},
        coords={"star": np.arange(model_data["parallax"].shape[0])},
        dims={"distance_pc": ["star"]}
    )
    return mcmc, idata


# %% id="DYBn5wLvZWxy"
N_NUTS = 3000
sub_sample = sample.sample(n=N_NUTS, seed=SEED + 100).sort("source_id")
sub_model_data_np = prepare_model_data(sub_sample, grid, r_min=10.0, r_max=20_000.0)
sub_model_data = {
    key: jnp.asarray(value) if isinstance(value, np.ndarray) else value
    for key, value in sub_model_data_np.items()
}

# %% colab={"base_uri": "https://localhost:8080/"} id="NxRc5Ri4ZZCt" outputId="f8c16018-96d8-4688-fc71-72c3b1a472c2"
mcmc_0, idata_nuts0 = run_nuts(model_0_parallax_only, sub_model_data, seed=SEED + 50)

# %% colab={"base_uri": "https://localhost:8080/"} id="fmNM-eiaZpxy" outputId="889ad325-aa5d-4183-af2a-c633295c837e"
mcmc_1, idata_nuts1 = run_nuts(model_1_cmd_marginalized, sub_model_data, seed=SEED + 60)

# %% colab={"base_uri": "https://localhost:8080/"} id="brpAX2agZrWn" outputId="fc7e2859-f079-48f3-f72d-42a404692fd7"
mcmc_2, idata_nuts2 = run_nuts(model_2_cmd_lambda, sub_model_data, seed=SEED + 70)

# %% colab={"base_uri": "https://localhost:8080/"} id="77822eac" outputId="fa3714e8-24b9-4fe6-8a1e-11987351bdbc"
idata_nuts0.to_netcdf(MODELS_DIR / "modelo_0_nuts.nc")
idata_nuts1.to_netcdf(MODELS_DIR / "modelo_1_nuts.nc")
idata_nuts2.to_netcdf(MODELS_DIR / "modelo_2_nuts.nc")

# %% colab={"base_uri": "https://localhost:8080/"} id="969c2270" outputId="29a6e343-d39e-49ad-a055-c059d67e2f95"
true_dist_nuts = sub_sample["barycentric_distance"].to_numpy()
metrics_nuts0 = evaluate_distance_posterior(idata_nuts0, true_dist_nuts, variable="distance_pc")
metrics_nuts1 = evaluate_distance_posterior(idata_nuts1, true_dist_nuts, variable="distance_pc")
metrics_nuts2 = evaluate_distance_posterior(idata_nuts2, true_dist_nuts, variable="distance_pc")

metrics_nuts0, metrics_nuts1, metrics_nuts2

# %% [markdown] id="d0a5b4bb"
# ### Visualización de Diagnósticos y Resultados
# Generamos gráficos para evaluar la convergencia (diagnósticos) y la calidad de la predicción de distancias.

# %% id="6f4d11e4"
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Comparación Visual: Distancia Real vs Estimada (Modelo 2 NUTS)
plt.figure(figsize=(8, 6))
median_dist = idata_nuts2.posterior["distance_pc"].median(dim=["chain", "draw"])

plt.scatter(true_dist_nuts, median_dist, alpha=0.3, s=10, label="Estrellas (NUTS)")
plt.plot([0, 20000], [0, 20000], color="red", linestyle="--", label="Identidad (Ideal)")
plt.xlabel("Distancia Real (pc)")
plt.ylabel("Distancia Estimada (Mediana Posterior, pc)")
plt.title("Modelo 2: Real vs Estimado")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# %% colab={"base_uri": "https://localhost:8080/", "height": 558} id="0ab4943e" outputId="330c9bfe-aa95-4da8-9d22-ab6eee800d6d"
# 2. Diagnóstico de Convergencia para lambda_cmd (Modelo 2)
az.plot_trace(idata_nuts2, var_names=["lambda_cmd"])
plt.tight_layout()
plt.show()

# 3. Resumen de diagnósticos (R-hat y ESS)
summary = az.summary(idata_nuts2, var_names=["lambda_cmd", "weights"], hdi_prob=0.95)
print("Diagnósticos para parámetros globales del Modelo 2:")
display(summary.head(10))


# %% [markdown] id="f72d5161"
# ### Comparación Final de Residuos
# Graficamos el error relativo $(d_{est} - d_{true}) / d_{true}$ para ver el comportamiento del sesgo en función de la distancia real.

# %% id="ca4700aa"
def get_median_dist(idata):
    return idata.posterior["distance_pc"].median(dim=["chain", "draw"]).values

dists = {
    "Modelo 0 (Solo Par)": get_median_dist(idata_nuts0),
    "Modelo 1 (CMD)": get_median_dist(idata_nuts1),
    "Modelo 2 (CMD+Lambda)": get_median_dist(idata_nuts2)
}

plt.figure(figsize=(10, 6))
for name, d_median in dists.items():
    residual_rel = (d_median - true_dist_nuts) / true_dist_nuts
    sns.scatterplot(x=true_dist_nuts, y=residual_rel, alpha=0.2, s=15, label=name)

plt.axhline(0, color="black", linestyle="--")
plt.ylim(-1, 2)
plt.xlabel("Distancia Real (pc)")
plt.ylabel("Error Relativo (Est - Real) / Real")
plt.title("Comparación de Sesgos: NUTS")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


# %% [markdown] id="c4a91689"
# ### Comparación Final de Residuos
# Graficamos el error relativo $(d_{est} - d_{true}) / d_{true}$ para ver el comportamiento del sesgo en función de la distancia real.

# %% id="e1450da9"
def get_median_dist(idata):
    return idata.posterior["distance_pc"].median(dim=["chain", "draw"]).values

dists = {
    "Modelo 0 (Solo Par)": get_median_dist(idata_nuts0),
    "Modelo 1 (CMD)": get_median_dist(idata_nuts1),
    "Modelo 2 (CMD+Lambda)": get_median_dist(idata_nuts2)
}

plt.figure(figsize=(10, 6))
for name, d_median in dists.items():
    residual_rel = (d_median - true_dist_nuts) / true_dist_nuts
    sns.scatterplot(x=true_dist_nuts, y=residual_rel, alpha=0.2, s=15, label=name)

plt.axhline(0, color="black", linestyle="--")
plt.ylim(-1, 2)
plt.xlabel("Distancia Real (pc)")
plt.ylabel("Error Relativo (Est - Real) / Real")
plt.title("Comparación de Sesgos: NUTS")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# %% id="dUulRz1CgXBQ"
