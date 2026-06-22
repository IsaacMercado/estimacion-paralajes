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
output_dir = DATA_DIR / "raw"
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

# %%
job = Gaia.launch_job("SELECT vmag, b_v from public.hipparcos")
r = job.get_results()
# r.write('values.ecsv', overwrite=True)

# %%
print(r)


# %%
def query_data(query: str):
    _hash = hashlib.md5(query.encode()).hexdigest()
    filename = f"GaiaQuery_{_hash}.ecsv"

    if (output_dir / filename).exists():
        return Table.read(output_dir / filename)

    job = Gaia.launch_job_async(query)
    job.get_results().write(output_dir / filename)
    return job.get_results()


# %%
results = query_data(
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
    """
)

# %%
results = query_data(
    f"""
    SELECT {", ".join(fields)}
    FROM gaiadr3.gaia_source
    WHERE random_index BETWEEN 0 AND 10000000
    """
)

# %%
results["bp_rp"] = results["phot_bp_mean_mag"] - results["phot_rp_mean_mag"]
results["g_mag_abs"] = (
    results["phot_g_mean_mag"] + 5.0 * np.log10(results["parallax"]) - 10.0
)

# %%
plt.figure(figsize=(10, 10))
plt.scatter(results["bp_rp"], results["g_mag_abs"], s=1, c="black")
plt.gca().invert_yaxis()
plt.xlabel("BP - RP")
plt.ylabel("G")
plt.title("Color-Magnitude Diagram")
plt.show()


# %%
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


# %%
diagram_cm(results["bp_rp"], results["g_mag_abs"])

# %%
mu = 30
sig = 1

# %%
with pm.Model() as model:
    # Define the normal distribution
    normal_dist = pm.Normal("normal_dist", mu=mu, sigma=sig)

    # Apply the logarithmic transformation
    log_transform = pm.Deterministic("log_transform", pt.log(normal_dist))

    # Sample from the model
    trace = pm.sample(1000, return_inferencedata=False)

# Plot the transformed distribution
plt.hist(trace["log_transform"], bins=30, histtype="stepfilled", alpha=0.5)
plt.xlabel("Log-transformed values")
plt.ylabel("Frequency")
plt.title("Histogram of Log-transformed Normal Distribution")
plt.show()

# %%
from pathlib import Path

import pytensor

file = Path(pytensor.__file__).resolve().parent / "misc/check_blas.py"
# !python "$file"

# %%
with model:
    idata = pmx.fit(method="pathfinder", num_draws=1000, inference_backend="blackjax")

# %%
az.plot_trace(idata)
plt.tight_layout()
# %%
with model:
    idata = pmx.fit(method="pathfinder", num_draws=1000)

# %%
az.plot_trace(idata)
plt.tight_layout()
# %%
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
from scipy.stats import norm
from xarray_einstats.stats import XrContinuousRV

# %config InlineBackend.figure_format = 'retina'
RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)
az.style.use("arviz-darkgrid")

k = 3
ndata = 500
centers = np.array([-5, 0, 5])
sds = np.array([0.5, 2.0, 0.75])
idx = rng.integers(0, k, ndata)
x = rng.normal(loc=centers[idx], scale=sds[idx], size=ndata)
plt.hist(x, 40)

# %%
with pm.Model(coords={"cluster": range(k)}) as model:
    μ = pm.Normal(
        "μ",
        mu=0,
        sigma=5,
        transform=pm.distributions.transforms.ordered,
        initval=[-4, 0, 4],
        dims="cluster",
    )
    σ = pm.HalfNormal("σ", sigma=1, dims="cluster")
    weights = pm.Dirichlet("w", np.ones(k), dims="cluster")
    pm.NormalMixture("x", w=weights, mu=μ, sigma=σ, observed=x)

# %%
pm.model_to_graphviz(model)

# %%
with model:
    idata = pmx.fit(method="pathfinder", num_draws=1000, inference_backend="blackjax")

# %%
xi = np.linspace(-7, 7, 500)
post = idata.posterior
pdf_components = XrContinuousRV(norm, post["μ"], post["σ"]).pdf(xi) * post["w"]
pdf = pdf_components.sum("cluster")

fig, ax = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
# empirical histogram
ax[0].hist(x, 50)
ax[0].set(title="Data", xlabel="x", ylabel="Frequency")
# pdf
pdf_components.mean(dim=["chain", "draw"]).sum("cluster").plot.line(ax=ax[1])
ax[1].set(title="PDF", xlabel="x", ylabel="Probability\ndensity")
# plot group membership probabilities
(pdf_components / pdf).mean(dim=["chain", "draw"]).plot.line(hue="cluster", ax=ax[2])
ax[2].set(title="Group membership", xlabel="x", ylabel="Probability")
# %%
x = pm.draw(pm.Normal.dist(mu=mu, sigma=sig), 10000)
plt.hist(x, bins=30, histtype="stepfilled", alpha=0.5)
# %%
y = np.log(x)
plt.hist(y, bins=30, histtype="stepfilled", alpha=0.5)
# %%
x = pm.draw(pm.Normal.dist(mu=5, sigma=1), 10000)
plt.hist(x, bins=30, histtype="stepfilled", alpha=0.5)
# %%
y = 22.5 - 2.5 * np.log10(x)
plt.hist(y, bins=30, histtype="stepfilled", alpha=0.5)
# %%
z = pm.draw(
    pm.TruncatedNormal.dist(mu=mu, sigma=sig, lower=0, upper=float("inf")), 10000
)
plt.hist(z, bins=30, histtype="stepfilled", alpha=0.5)
# %%
t = pm.draw(pm.Gumbel.dist(mu=mu, beta=sig), 10000)
plt.hist(t, bins=30, histtype="stepfilled", alpha=0.5)
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
import contextlib
import gzip
import io
from urllib.request import urlopen, urlretrieve

import numpy as np
import pandas as pd
from astropy import units as u
from astropy_healpix import HEALPix
from tqdm import tqdm

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

print("Input Variables: ")
print(f"* HEALPix level                      = {hpx_level} ")
print(f"* ICRS longitude (~ Right Ascension) = {lon} ")
print(f"* ICRS latitude  (~ Declination)     = {lat} ")
print(f"* Radius                             = {radius} ")
print()

# %% [markdown]
# ## Create reference file

# %%
gaia_dr_flag = "DR3" if DR3 else "EDR3"

print("=" * 120)
print(f'Preparing selection of Gaia {gaia_dr_flag}: ""{target_table}" files')
print("=" * 120)

url_prefix = f"http://cdn.gea.esac.esa.int/Gaia/g{gaia_dr_flag.lower()}/{target_table}/"
md5sum_file_url = url_prefix + "_MD5SUM.txt"
md5sum_file = pd.read_csv(
    md5sum_file_url,
    header=None,
    sep=r"\s+",
    names=["md5Sum", "file"],
)

if DR3:
    # The last row in the "_MD5SUM.txt" file in the DR3 directories includes the md5Sum value of the _MD5SUM.txt file
    md5sum_file.drop(md5sum_file.tail(1).index, inplace=True)

md5sum_file

# %%
# Extract HEALPix level-8 from file name ======================================
healpix_8_min = [
    int(file[file.find("_") + 1 : file.rfind("-")]) for file in md5sum_file["file"]
]
healpix_8_max = [
    int(file[file.rfind("-") + 1 : file.rfind(".csv")]) for file in md5sum_file["file"]
]
reference_file = pd.DataFrame(
    {
        "file": md5sum_file["file"],
        "healpix8_min": healpix_8_min,
        "healpix8_max": healpix_8_max,
    }
).reset_index(drop=True)

# Compute HEALPix levels 6,7, and 9 ===========================================
reference_file["healpix7_min"] = [inp >> 2 for inp in reference_file["healpix8_min"]]
reference_file["healpix7_max"] = [inp >> 2 for inp in reference_file["healpix8_max"]]

reference_file["healpix6_min"] = [inp >> 2 for inp in reference_file["healpix7_min"]]
reference_file["healpix6_max"] = [inp >> 2 for inp in reference_file["healpix7_max"]]

reference_file["healpix9_min"] = [inp << 2 for inp in reference_file["healpix8_min"]]
reference_file["healpix9_max"] = [
    (inp << 2) + 3 for inp in reference_file["healpix8_max"]
]

# Generate reference file =====================================================
ncols = [
    "file",
    "healpix6_min",
    "healpix6_max",
    "healpix7_min",
    "healpix7_max",
    "healpix8_min",
    "healpix8_max",
    "healpix9_min",
    "healpix9_max",
]
reference_file = reference_file[ncols]
reference_file

# %% [markdown]
# ## Compute Healpix indexes associated to the selected  circular region

# %%
print("=" * 120)
print(
    f"Computing HEALPix Level {hpx_level} encompasing a Cone Search (Radius, longitude, latitude): {radius.value} {radius.unit},  {lon.value} {lon.unit}, {lat.value} {lat.unit}"
)
print("=" * 120)

hp = HEALPix(nside=2**hpx_level, order="nested")
hp_cone_search = hp.cone_search_lonlat(lon, lat, radius=radius)

# %% [markdown]
# ## Download files
#
# A .txt file with the list of files to be downloaded will be firts created. This file will be read and a secuencial download of all the files listed will start. A progress message will be in the terminal from where this Notebook was launched.

# %%
subset = []
for index in reference_file.index:
    row = reference_file.iloc[index]
    hp_min, hp_max = row[f"healpix{hpx_level}_min"], row[f"healpix{hpx_level}_max"]
    if np.any(np.logical_and(hp_min <= hp_cone_search, hp_cone_search <= hp_max)):
        subset.append(url_prefix + row["file"])

print("=" * 120)
print(f"A total of {len(subset)} files for download")
print("=" * 120)

# %%
print("=" * 120)
print(f"Bulk download files are stored in directory: {output_dir}")
print("=" * 120)


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url: str):
    name = url.strip().split("/")[-1]
    filename = output_dir / name

    if filename.exists():
        statinfo = filename.stat()

        with contextlib.closing(urlopen(url)) as fp:
            headers = fp.info()
            length = int(headers["content-length"])

        if statinfo.st_size == length:
            print(f"Skipping {name}, file already downloaded")
            return
        else:
            print(f"Redownloading {name}")

    with DownloadProgressBar(
        unit="B",
        unit_scale=True,
        miniters=1,
        desc=name,
    ) as bar:
        urlretrieve(url, filename=filename, reporthook=bar.update_to)


for url in subset:
    download_url(url)


# %%
def extract_header_table(filename, **kwargs):
    with gzip.open(filename, mode="rt", newline="") as file, io.BytesIO() as virtual:
        for line in file:
            virtual.write(line.encode())
            if not line.startswith("#"):
                break
        virtual.seek(0)
        return Table.read(
            virtual,
            format="ascii.ecsv",
            **kwargs,
        ).columns


columns = extract_header_table(
    next(output_dir.glob("*.csv.gz")),
    include_names=fields,
    fill_values=("null", "0"),
)

# %%
df = pd.concat(
    pd.read_csv(
        path,
        comment="#",
        index_col="source_id",
        usecols=fields,
        dtype={col: columns[col].dtype for col in columns},
    )
    for path in output_dir.glob("*.csv.gz")
)
df.info()

# %%
result = Table.from_pandas(
    df,
    units={col: columns[col].unit for col in columns if columns[col].unit},
)

# %%
df["g_mag_abs"] = df["phot_g_mean_mag"] + 5.0 * np.log10(df["parallax"]) - 10.0
df["bp_rp"] = df["phot_bp_mean_mag"] - df["phot_rp_mean_mag"]

# %%
parallax = df["parallax"][~df["parallax"].isna()]
parallax.count(), parallax[parallax > 0].count()

# %%
np.log10(parallax[parallax > 0])

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
odf = df.dropna(subset=["bp_rp", "g_mag_abs"])
diagram_cm(
    odf["phot_bp_mean_mag"] - odf["phot_rp_mean_mag"],
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
