from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = Path(__file__).parent.parent.parent
FIGURES_DIR = ROOT_DIR / "reports" / "figures"


def plot_extinction_effect():
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "text.usetex": True,
        }
    )

    fig, ax = plt.subplots(figsize=(6, 6))

    # Fake Main Sequence background
    np.random.seed(42)
    color_ms = np.random.uniform(-0.5, 3.0, 2000)
    mag_ms = 3.5 * color_ms + 1.0 + np.random.normal(0, 0.5, 2000)
    
    # Filter to keep to a nice diagonal
    mask = (mag_ms > -2) & (mag_ms < 12)
    ax.scatter(color_ms[mask], mag_ms[mask], color='lightgray', s=5, zorder=1, alpha=0.5, label="Distribución intrínseca (Secuencia Principal)")

    # Star without extinction (intrinsic)
    true_color = 0.5
    true_mag = 3.5
    ax.plot(true_color, true_mag, marker='*', markersize=18, color='blue', markeredgecolor='black', zorder=3, label="Estrella intrínseca")

    # Star with extinction (observed)
    # Extinction makes it fainter (higher mag) and redder (higher color)
    a_g = 2.5      # Extinction in G band (magnitudes)
    e_bp_rp = 1.2  # Reddening (color index)
    
    obs_color = true_color + e_bp_rp
    obs_mag = true_mag + a_g
    ax.plot(obs_color, obs_mag, marker='*', markersize=18, color='red', markeredgecolor='black', zorder=3, label="Estrella observada")

    # Reddening vector arrow
    ax.annotate("", xy=(obs_color, obs_mag), xytext=(true_color, true_mag),
                arrowprops=dict(arrowstyle="->", color="black", lw=2.5), zorder=4)
    
    ax.text((true_color + obs_color)/2 - 0.1, (true_mag + obs_mag)/2 - 0.6, 
            r"Vector de enrojecimiento", fontsize=12, rotation=np.degrees(np.arctan2(-a_g, e_bp_rp)),
            ha='center', va='center', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))
            
    ax.text(true_color - 0.2, true_mag - 0.4, r"$( \mathrm{color}_0, M_G )$", fontsize=11, ha="center")
    ax.text(obs_color + 0.3, obs_mag + 0.4, r"$( \mathrm{color}_0 + E, M_G + A_G )$", fontsize=11, ha="center")

    ax.invert_yaxis()
    ax.set_xlim(-1, 4)
    ax.set_ylim(14, -2)
    
    ax.set_xlabel(r'Color $G_{\mathrm{BP}} - G_{\mathrm{RP}}$')
    ax.set_ylabel(r'Magnitud Absoluta $M_G$')
    
    ax.grid(True, linestyle='--', alpha=0.5, zorder=0)
    ax.legend(loc="lower left", framealpha=0.9)
    plt.title("Efecto del polvo interestelar en el CMD", pad=15)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "efecto_extincion.pdf", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    plot_extinction_effect()