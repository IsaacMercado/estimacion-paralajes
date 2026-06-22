import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib

ROOT_DIR = Path(__file__).parent.parent.parent
FIGURES_DIR = ROOT_DIR / "reports" / "figures"

def plot_shrinkage():
    matplotlib.rcParams.update({
        "font.family": "serif",
        "text.usetex": True,
    })

    # Datos simulados conceptuales
    np.random.seed(11)
    N = 25
    true_distance = 1.0 # kpc
    # Inversión directa (alta varianza, sesgada hacia mayores distancias)
    var_direct = np.random.uniform(0.1, 0.8, N) 
    direct_est = true_distance + np.random.normal(0, var_direct) + (var_direct**2)*0.5
    
    # Inferencia Jerárquica Bayesiana (contracción hacia la media, menor varianza)
    # Entre más grande era la varianza original, más se contrae hacia el prior (1.0)
    hierarchical_est = true_distance + (direct_est - true_distance) / (1 + var_direct*5)
    var_hierarchical = var_direct * 0.4

    # Ordenar por el error de observación original para observar el efecto de "embudo" (funnel)
    sort_idx = np.argsort(var_direct)
    direct_est = direct_est[sort_idx]
    var_direct = var_direct[sort_idx]
    hierarchical_est = hierarchical_est[sort_idx]
    var_hierarchical = var_hierarchical[sort_idx]
    
    y_pos = np.arange(N)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 6), sharey=True)

    # Panel 1: Inversión directa
    ax1.errorbar(direct_est, y_pos, xerr=var_direct, fmt='o', color='red', ecolor='salmon', capsize=0, markersize=4)
    ax1.axvline(true_distance, color='black', linestyle='--', alpha=0.5, label='Distancia real')
    ax1.set_title("Inversión de Paralaje ($1/\\varpi$)")
    ax1.set_xlabel("Distancia Estimada (kpc)")
    ax1.set_xlim(0, 3)
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    # Mostrar conexiones entre estimaciones
    for i in range(len(y_pos)):
        if i % 3 == 0: # conector solo para algunas para no rellenar
            # dibujar línea entre los paneles (conceptualmente)
            pass

    # Panel 2: Estimación Jerárquica Bayesiana
    ax2.errorbar(hierarchical_est, y_pos, xerr=var_hierarchical, fmt='o', color='blue', ecolor='lightblue', capsize=0, markersize=4)
    ax2.axvline(true_distance, color='black', linestyle='--', alpha=0.5, label='Prior poblacional')
    ax2.set_title("Modelo Jerárquico Bayesiano")
    ax2.set_xlabel("Distancia Estimada (kpc)")
    ax2.set_xlim(0, 3)
    ax2.grid(True, linestyle='--', alpha=0.3)

    plt.suptitle("Ilustración del Fenómeno de Contracción Bayesiana (Shrinkage)", y=1.02, fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "contraccion_bayesiana.pdf", bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    plot_shrinkage()