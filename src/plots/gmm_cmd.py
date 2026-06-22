import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import Ellipse
import matplotlib

ROOT_DIR = Path(__file__).parent.parent.parent
FIGURES_DIR = ROOT_DIR / "reports" / "figures"

def plot_gmm_cmd():
    matplotlib.rcParams.update({
        "font.family": "serif",
        "text.usetex": True,
    })

    fig, ax = plt.subplots(figsize=(6, 8))

    # Fake Main Sequence Background
    np.random.seed(42)
    color_ms = np.random.uniform(0.0, 3.0, 5000)
    mag_ms = 4.0 * color_ms + 1.0 + np.random.normal(0, 0.4, 5000)
    
    # Red Giants
    color_rg = np.random.uniform(1.5, 3.5, 1000)
    mag_rg = -1.0 * (color_rg - 2.5)**2 + 1.0 + np.random.normal(0, 0.5, 1000)

    color = np.concatenate([color_ms, color_rg])
    mag = np.concatenate([mag_ms, mag_rg])
    
    # Filter bounds
    mask = (mag > -2) & (mag < 15) & (color > -0.5) & (color < 4.0)
    color = color[mask]
    mag = mag[mask]

    # Plot empirical density
    ax.hexbin(color, mag, gridsize=50, cmap='Greys', alpha=0.5, mincnt=1)

    # Superimpose Gaussian Mixture Components
    centers_color = np.linspace(0.2, 2.8, 6)
    centers_mag = 4.0 * centers_color + 1.0
    
    # Add some RG centers
    rg_centers_c = np.array([2.0, 2.5, 3.0])
    rg_centers_m = np.array([0.5, 0.0, -0.5])
    
    all_centers_c = np.concatenate([centers_color, rg_centers_c])
    all_centers_m = np.concatenate([centers_mag, rg_centers_m])
    
    for i, (c, m) in enumerate(zip(all_centers_c, all_centers_m)):
        # Plot ellipsis
        ellipse = Ellipse((c, m), width=0.4, height=1.5, angle=-70 if m > 0.5 else 0, 
                          edgecolor='red', facecolor='none', lw=1.5, zorder=3)
        ax.add_patch(ellipse)
        ax.plot(c, m, 'r+', markersize=8, zorder=4)

    ax.invert_yaxis()
    ax.set_xlabel(r'$G_{\mathrm{BP}} - G_{\mathrm{RP}}$')
    ax.set_ylabel(r'$M_G$')
    ax.set_title("Aproximación del Prior del CMD mediante Mezcla Gaussiana")
    
    ax.set_xlim(-0.5, 4.0)
    ax.set_ylim(15, -2)
    ax.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "gmm_cmd.pdf", bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    plot_gmm_cmd()