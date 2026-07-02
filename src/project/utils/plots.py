import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


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
