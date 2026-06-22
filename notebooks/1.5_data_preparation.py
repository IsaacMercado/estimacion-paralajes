# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: default
#     language: python
#     name: python3
# ---

# %%
import hashlib
from pathlib import Path

import arviz as az
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import pymc_extras as pmx
import pytensor.tensor as pt
import scipy as sp
from astropy.table import Table
from astropy.visualization import quantity_support
from astroquery.gaia import Gaia

Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"
quantity_support()

# %%
ROOT = Path(".").resolve().parent
DATA_DIR = ROOT / "data"
output_dir = DATA_DIR / 'raw'
fields = [
    'source_id',
    'ra',
    'ra_error',
    'dec',
    'dec_error',
    'parallax',
    'parallax_error',
    'phot_g_n_obs',
    'phot_g_mean_flux',
    'phot_g_mean_flux_error',
    'phot_g_mean_mag',
    'phot_bp_n_obs',
    'phot_bp_mean_flux',
    'phot_bp_mean_flux_error',
    'phot_bp_mean_mag',
    'phot_rp_n_obs',
    'phot_rp_mean_flux',
    'phot_rp_mean_flux_error',
    'phot_rp_mean_mag',
]

# %%
job = Gaia.launch_job("""
SELECT vmag, b_v
FROM public.hipparcos
WHERE vmag IS NOT NULL AND b_v IS NOT NULL
""")
r = job.get_results()

color = np.array(r["b_v"])
mag = np.array(r["vmag"])
mask = np.isfinite(color) & np.isfinite(mag)

plt.figure(figsize=(7, 6))
plt.scatter(
    color[mask],
    mag[mask],
    s=6,
    c=color[mask],
    cmap="plasma",
    alpha=0.7,
    edgecolors="none",
)
plt.gca().invert_yaxis()
plt.xlabel("B - V")
plt.ylabel("V (mag)")
plt.title("Diagrama color-magnitud (Hipparcos)")
plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()


# %%
def get_filename(query: str) -> str:
    _hash = hashlib.md5(query.encode()).hexdigest()
    return f"GaiaQuery_{_hash}.ecsv"

def query_data(query: str):
    filename = get_filename(query)

    if (output_dir / filename).exists():
        return Table.read(output_dir / filename)

    job = Gaia.launch_job_async(query)
    job.get_results().write(output_dir / filename)
    return job.get_results()


# %%
query = """
    SELECT
        source_id,
        phot_g_mean_mag + 5.0 * log10(parallax) - 10.0 as g_mag_abs,
        bp_rp
    FROM gaiadr3.gaia_source_lite
    WHERE
        parallax_over_error >= 5
        AND phot_bp_mean_flux_over_error > 0
        AND phot_rp_mean_flux_over_error > 0
        AND SQRT(POWER(2.5/log(10) / phot_bp_mean_flux_over_error, 2) + POWER(2.5/log(10) / phot_rp_mean_flux_over_error, 2)) <= 0.05
        AND random_index BETWEEN 0 AND 10000000
    """


# %%
import polars as pl

df = pl.read_csv(
    output_dir / get_filename(query),
    separator=" ",
    comment_prefix="#",
)
df

# %%
query = """
SELECT
    gum.source_id,
    gum.ra,
    gum.dec,
    gum.barycentric_distance,
    gum.mag_g,
    gum.mag_bp,
    gum.mag_rp,
    gss.ra AS ra_sim,
    gss.ra_error,
    gss.dec AS dec_sim,
    gss.dec_error,
    gss.parallax,
    gss.parallax_error,
    gss.phot_g_mean_flux,
    gss.phot_g_mean_flux_error,
    gss.phot_g_mean_mag,
    gss.phot_bp_mean_flux,
    gss.phot_bp_mean_flux_error,
    gss.phot_bp_mean_mag,
    gss.phot_rp_mean_flux,
    gss.phot_rp_mean_flux_error,
    gss.phot_rp_mean_mag
FROM gaiadr3.gaia_universe_model gum 
    INNER JOIN gaiadr3.gaia_source_simulation gss ON (gum.source_id = gss.source_id)
"""

df = pl.read_csv(
    output_dir / get_filename(query),
    separator=" ",
    comment_prefix="#",
)
df

# %%
df.columns

# %%
from astropy.io import ascii

polars_dtypes = {
    np.dtype('float32'): pl.Float32,
    np.dtype('float64'): pl.Float64,
    np.dtype('int64'): pl.Int64,
}

def query_data_with_header(query: str) -> pl.DataFrame:

    with open(output_dir / get_filename(query), 'r') as file:
        header_lines = []

        while True:
            tell = file.tell()
            line = file.readline()
            header_lines.append(line)
            if not line.startswith("#"):
                file.seek(tell)
                break

        header = ascii.read(header_lines, format="ecsv")

        df = pl.read_csv(
            file,
            separator=" ",
            comment_prefix="#",
            schema={
                column.name: polars_dtypes[column.dtype]
                for column in header.columns.values()
            }
        )

    return df, header

df, header = query_data_with_header(query)
df.drop_nans().sample(100)


# %%
sample = df.drop_nans().sample(100000)
color = sample["mag_bp"] - sample["mag_rp"]
mag = sample["mag_g"] - 5.0 * np.log10(sample["barycentric_distance"] * 1e3) + 5.0


# %%
def plot_cmd(
    color,
    magnitude,
    *,
    ax=None,
    figsize=(7, 6),
    s=6,
    cmap="plasma",
    alpha=0.7,
    color_label="Color",
    mag_label="Magnitud",
    title="Diagrama color-magnitud (Hertzsprung-Russell)",
    grid_alpha=0.25,
):
    color = np.asarray(color)
    magnitude = np.asarray(magnitude)
    mask = np.isfinite(color) & np.isfinite(magnitude)

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    scatter = ax.scatter(
        color[mask],
        magnitude[mask],
        s=s,
        c=color[mask],
        cmap=cmap,
        alpha=alpha,
        edgecolors="none",
    )
    ax.invert_yaxis()
    ax.set_xlabel(color_label)
    ax.set_ylabel(mag_label)
    ax.set_title(title)
    ax.grid(alpha=grid_alpha)
    plt.tight_layout()

    return ax, scatter


plot_cmd(
    color,
    mag,
    color_label="B - V",
    mag_label="V (mag)",
    title="Diagrama color-magnitud (Hipparcos)",
)
plt.show()


# %%
def plot_cmd_hist(
    color,
    magnitude,
    *,
    ax=None,
    figsize=(7, 6),
    bins=300,
    cmap="plasma",
    log_scale=True,
    color_label="Color",
    mag_label="Magnitud",
    title="Diagrama color-magnitud (Hertzsprung-Russell)",
    grid_alpha=0.15,
    add_colorbar=True,
):
    color = np.asarray(color)
    magnitude = np.asarray(magnitude)
    mask = np.isfinite(color) & np.isfinite(magnitude)

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    hist_kwargs = {
        "bins": bins,
        "cmap": cmap,
    }

    if log_scale:
        hist_kwargs["norm"] = mcolors.LogNorm()
        hist_kwargs["cmin"] = 1

    h = ax.hist2d(color[mask], magnitude[mask], **hist_kwargs)

    ax.invert_yaxis()
    ax.set_xlabel(color_label)
    ax.set_ylabel(mag_label)
    ax.set_title(title)
    ax.grid(alpha=grid_alpha)

    cbar = None
    if add_colorbar:
        cbar = plt.colorbar(h[3], ax=ax)
        cbar.set_label("Número de estrellas")

    plt.tight_layout()
    return ax, h, cbar


plot_cmd_hist(
    color,
    mag,
    color_label="BP - RP",
    mag_label="M_G",
    title="Diagrama color-magnitud (Hertzsprung-Russell) con histograma",
)
plt.show()

# %%
