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
# <div>
# <img src="https://www.cosmos.esa.int/documents/3414525/0/Logo_Gaia_may_23_2022.png/cf6be61e-609e-27dc-3ad6-03ac5209bdc4?t=1653299950248" width="300" align="right"/>
# </div>
#
#
# # Cone search > Bulk download
#
# <br />
# <br />
# <br />
# <br />
# <br />
#
#
# **Release number:**
# v1.1 (2022-08-06)
#
#
# **Applicable Gaia Data Releases:**
# Gaia EDR3, Gaia DR3
#
# **Author:**
# Héctor Cánovas Cabrera; hector.canovas@esa.int
#
# **Summary:**
#
# This code computes the list of Gaia (E)DR3 files associated to a circular region in the sky defined by the user. The granularity of this region is set by the [HEALPix](https://healpix.sourceforge.io) level selected.
#
# Input parameters:
# * target catalogue (e.g., gaia_source, auxiliary/agn_cross_id, or auxiliary/frame_rotator_source),
# * the cone-search parameters (centre and radius), and
# * the desired healpix level.
#
# Once the variables above are set the notebook creates a reference file that contains the min/max [HEALPix](https://healpix.sourceforge.io) index (levels: 6,7,8, and 9) encompassed by each gaia_source file available in the (E)DR3 [bulk download directory.](http://cdn.gea.esac.esa.int/Gaia/). The convertion between the different [HEALPix](https://healpix.sourceforge.io) levels is done by means of bit-shifting operations.
#
#
# **Useful URLs:**
#
# * [Questions or suggestions](https://www.cosmos.esa.int/web/gaia/questions)
# * [Tutorials, documentation, and more](https://www.cosmos.esa.int/web/gaia-users/archive)
# * [Known issues in the Gaia data](https://www.cosmos.esa.int/web/gaia-users/known-issues)
# * [Gaia data credits and acknowledgements](https://www.cosmos.esa.int/web/gaia-users/credits)

# %%
import numpy as np
import pandas as pd
import polars as pl
import polars.selectors as cs
from astropy import units as u
from astropy_healpix import HEALPix

from project.utils import DATA_RAW_DIR
from project.utils.download import download_url
from project.utils.load import get_gaia_md5sum, load_gz_with_polars

# %% [markdown]
# ## Set input variables
#
# Default input paramers:
# * DR3 = True ; Default Value. Set it to False to retrieve EDR3 files
# * target_table = 'gaia_source' ; Alternative values: 'Astrophysical_parameters/astrophysical_parameters', 'Variability/vari_cepheid', etc - see all the content in: http://cdn.gea.esac.esa.int/Gaia/gdr3/ & http://cdn.gea.esac.esa.int/Gaia/gedr3/
# * Cone-search parameters: radius = 0.5 degrees, centred in the Large Magallanic Cloud (in ICRS coordinates).
# * Healpix-level = 6 (choose a larger one to increase granularity, and viceversa).

# %%
# Set input parameters below ===========
DR3 = True  # Set it to False to select EDR3
target_table = "gaia_source"  # Alternative values: 'Astrophysical_parameters/astrophysical_parameters/', etc
hpx_level = 6
lon = 80.894 * u.deg  # Right Ascencion (ICRS)
lat = -69.756 * u.deg  # Declination (ICRS)
radius = 0.5 * u.deg
output_dir = DATA_RAW_DIR

print("Input Variables: ")
print(f"* HEALPix level                      = {hpx_level} ")
print(f"* ICRS longitude (~ Right Ascension) = {lon} ")
print(f"* ICRS latitude  (~ Declination)     = {lat} ")
print(f"* Radius                             = {radius} ")
print()

# %% [markdown]
# ## Create reference file

# %%
reference_file = get_gaia_md5sum(f"gdr3/{target_table}")

# %% [markdown]
# ## Compute Healpix indexes associated to the selected  circular region

# %%
print("=" * 120)
print(
    f"Computing HEALPix Level {hpx_level} encompasing a Cone Search (Radius, longitude, latitude): "
    f"{radius.value} {radius.unit},  {lon.value} {lon.unit}, {lat.value} {lat.unit}"
)
print("=" * 120)

hp = HEALPix(nside=2**hpx_level, order="nested")
hp_cone_search = hp.cone_search_lonlat(lon, lat, radius=radius)

# %% [markdown]
# ## Download files
#
# A .txt file with the list of files to be downloaded will be firts created. This file will be read and a secuencial download of all the files listed will start. A progress message will be in the terminal from where this Notebook was launched.
#

# %%
label_min = f"healpix{hpx_level}_min"
label_max = f"healpix{hpx_level}_max"


def _filter_row(row):
    hp_min, hp_max = row[label_min], row[label_max]
    return np.any(np.logical_and(hp_min <= hp_cone_search, hp_cone_search <= hp_max))


subset = reference_file["url"][reference_file.apply(_filter_row, axis=1)].tolist()
print("=" * 120)
print(f"A total of {len(subset)} files for download")
print("=" * 120)

# %%
print("=" * 120)
print(f"Bulk download files are stored in directory: {output_dir}")
print("=" * 120)

for url in subset:
    download_url(url, output_dir)

# %%

df, columns = load_gz_with_polars(
    "GaiaSource_*.csv.gz",
    columns=[
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
    ],
)

df = df.with_columns(
    (pl.col("phot_bp_mean_mag") - pl.col("phot_rp_mean_mag")).alias("bp_rp"),
    (pl.col("phot_g_mean_mag") + 5.0 * pl.col("parallax").log10() - 10.0).alias(
        "g_mag_abs"
    ),
).collect()

# %%
df.filter(pl.col("parallax").is_not_nan() | pl.col("parallax").is_not_null()).select(
    pl.col("parallax").count().alias("count"),
    pl.col("parallax").filter(pl.col("parallax") > 0).count().alias("count_positive"),
)

# %%
1613359 / 3285543

# %%
df.select(cs.float().fill_null(float("nan"))).head()

# %%
df.select(
    cs.float().fill_null(float("nan")),
).select(
    pl.all().is_null().sum().name.suffix("_is_null"),
).head()

# %%
pd.concat(
    [
        df["parallax"].isna().value_counts(),
        df["g_mag_abs"].isna().value_counts(),
    ],
    axis=1,
    keys=["parallax", "g_mag_abs"],
)

# %%
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt


def diagram_cm(color, mag):
    gammas = [0.8, 0.5, 0.3]

    fig, axs = plt.subplots(nrows=2, ncols=2)

    axs[0, 0].set_title("Linear normalization")
    axs[0, 0].hist2d(color, mag, bins=100)

    for ax, gamma in zip(axs.flat[1:], gammas):
        ax.set_title(r"Power law $(\gamma=%1.1f)$" % gamma)
        ax.hist2d(color, mag, bins=100, norm=mcolors.PowerNorm(gamma))

    fig.tight_layout()
    plt.show()


odf = df.filter(
    pl.col("bp_rp").is_not_nan()
    & pl.col("bp_rp").is_not_null()
    & pl.col("g_mag_abs").is_not_nan()
    & pl.col("g_mag_abs").is_not_null()
).select("bp_rp", "g_mag_abs")

diagram_cm(
    odf["bp_rp"],
    odf["g_mag_abs"],
)

# %%
from project.utils.plots import plot_cmd_hist

plot_cmd_hist(
    odf["bp_rp"],
    odf["g_mag_abs"],
)

# %%
plt.figure(figsize=(10, 10))
plt.scatter(
    df["phot_bp_mean_mag"] - df["phot_rp_mean_mag"],
    df["phot_g_mean_mag"],
    s=1,
    c="black",
)
plt.gca().invert_yaxis()
plt.xlabel("BP - RP")
plt.ylabel("G")
plt.title("Color-Magnitude Diagram")
plt.show()

# %%
