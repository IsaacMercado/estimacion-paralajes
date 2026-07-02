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

# %%
# !pip install git+https://github.com/IsaacMercado/estimacion-paralajes.git#egg=project[colab]

# %%
import matplotlib.pyplot as plt
import numpy as np

from project.utils.data import query_data
from project.utils.plots import plot_cmd, plot_cmd_hist

# %%
query_data(
    "SELECT vmag, b_v from public.hipparcos",
    "hipparcos_cm.ecsv",
)

# %%
query_data(
    """
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
    """,
    "gaiadr3_lite_cm_sample.ecsv",
)

# %%
fields = [
    "source_id",
    "ra",
    "ra_error",
    "dec",
    "dec_error",
    "parallax",
    "parallax_error",
    "phot_g_n_obs",
    "phot_g_mean_flux",
    "phot_g_mean_flux_error",
    "phot_g_mean_mag",
    "phot_bp_n_obs",
    "phot_bp_mean_flux",
    "phot_bp_mean_flux_error",
    "phot_bp_mean_mag",
    "phot_rp_n_obs",
    "phot_rp_mean_flux",
    "phot_rp_mean_flux_error",
    "phot_rp_mean_mag",
]

query_data(
    f"""
    SELECT {", ".join(fields)}
    FROM gaiadr3.gaia_source
    WHERE random_index BETWEEN 0 AND 10000000
    """,
    "gaiadr3_sample.ecsv",
)

# %%
table = query_data(
    """
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
    """,
    "simulation_data.ecsv",
    fast=True,
)

# %%
sample = table.to_pandas().dropna().sample(100_000)
color = sample["mag_bp"] - sample["mag_rp"]
mag = sample["mag_g"] - 5.0 * np.log10(sample["barycentric_distance"] * 1e3) + 5.0

# %%
plot_cmd(
    color,
    mag,
    color_label="B - V",
    mag_label="V (mag)",
    title="Diagrama color-magnitud (Hipparcos)",
)
plt.show()

# %%
plot_cmd_hist(
    color,
    mag,
    color_label="BP - RP",
    mag_label="M_G",
    title="Diagrama color-magnitud (Hertzsprung-Russell) con histograma",
)
plt.show()
