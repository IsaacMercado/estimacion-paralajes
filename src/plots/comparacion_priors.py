import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import norm

ROOT_DIR = Path(__file__).parent.parent.parent
FIGURES_DIR = ROOT_DIR / "reports" / "figures"

def edsd(r, L):
    """Prior Exponentially Decreasing Space Density (EDSD) de Bailer-Jones 2015"""
    prob = (1.0 / (2 * L**3)) * (r**2) * np.exp(-r / L)
    return prob

def likelihood(r, w, sig_w):
    """Likelihood de la paralaje dada la distancia"""
    # Se añade un pequeño epsilon a r para evitar división por cero
    return norm.pdf(w, loc=1.0/(r + 1e-10), scale=sig_w)

def plot_priors_and_posterior():
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "text.usetex": True,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
        }
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.9), constrained_layout=True)

    r = np.linspace(0.01, 5.0, 500)  # Distancias en kpc

    # 1. PRIORS
    # Uniforme impropio (constante)
    prior_uniform = np.ones_like(r) * 0.2  # Escala arbitraria para visualización
    
    # Uniforme truncado
    r_lim = 2.0
    prior_trunc = np.where(r <= r_lim, 0.5, 0.0)
    
    # EDSD
    L = 1.35 # kpc
    prior_edsd = edsd(r, L)

    ax1.plot(r, prior_uniform, '--', color='gray', label='Uniforme impropio')
    ax1.plot(r, prior_trunc, '-.', color='orange', label=f'Uniforme truncado ($r_{{lim}}={r_lim}$ kpc)')
    ax1.plot(r, prior_edsd, '-', color='blue', label=f'EDSD ($L={L}$ kpc)')

    ax1.set_xlabel('Distancia $r$ (kpc)')
    ax1.set_ylabel('$P(r)$')
    ax1.set_title('Priors de Distancia')
    ax1.set_xlim(0, 5)
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.5)

    # 2. POSTERIORS (Prior x Likelihood)
    # Observación para una estrella con mala señal-ruido
    w_hat = 0.5    # mas
    sig_w_hat = 0.3 # mas

    # Calculamos likelihood en la grilla r (w en mas -> 1/r con r en kpc)
    like = likelihood(r, w_hat, sig_w_hat)

    # Posteriors no normalizados (o normalizados numéricamente)
    post_uniform = prior_uniform * like
    post_trunc = prior_trunc * like
    post_edsd = prior_edsd * like

    # Normalización numérica simple
    post_uniform /= np.trapezoid(post_uniform, r)
    post_trunc /= np.trapezoid(post_trunc, r)
    post_edsd /= np.trapezoid(post_edsd, r)

    ax2.plot(r, post_uniform, '--', color='gray', label='Post. (Uniforme)')
    ax2.plot(r, post_trunc, '-.', color='orange', label='Post. (Truncado)')
    ax2.plot(r, post_edsd, '-', color='blue', label='Post. (EDSD)')
    
    # Mostrar la inversión simple r = 1/w
    r_naive = 1.0 / w_hat
    ax2.axvline(r_naive, color='red', linestyle=':', label=rf'$r = 1/\varpi$ ({r_naive:.1f} kpc)')

    ax2.set_xlabel('Distancia $r$ (kpc)')
    ax2.set_ylabel(r'$P(r \mid \varpi)$')
    ax2.set_title(r'Posterior para $\hat{\varpi}=0.5$ mas' + '\n' + r'$\sigma_{\varpi}=0.3$ mas')
    ax2.legend()
    ax2.set_xlim(0, 5)
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.savefig(FIGURES_DIR / "comparacion_priors.pdf", bbox_inches="tight", dpi=300)
    plt.close()

if __name__ == "__main__":
    plot_priors_and_posterior()