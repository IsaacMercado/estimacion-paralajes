import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

ROOT_DIR = Path(__file__).parent.parent.parent
FIGURES_DIR = ROOT_DIR / "reports" / "figures"


def plot_cmd(data_file_path: Path):
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "text.usetex": True,
        }
    )

    df = pl.read_csv(
        str(data_file_path),
        separator=" ",
        comment_prefix="#",
    )

    # Filtrar datos con paralaje válido y alta relación señal-ruido (> 10)
    # y buena precisión radiométrica para poder ver claramente la forma del CMD
    df = df.filter(
        (pl.col("parallax") > 0) &
        ((pl.col("parallax") / pl.col("parallax_error")) > 10) &
        (pl.col("phot_g_mean_mag") < 17.0)
    ).drop_nulls(subset=["phot_g_mean_mag", "phot_bp_mean_mag", "phot_rp_mean_mag", "parallax"])

    bp = df["phot_bp_mean_mag"].to_numpy()
    rp = df["phot_rp_mean_mag"].to_numpy()
    g = df["phot_g_mean_mag"].to_numpy()
    parallax = df["parallax"].to_numpy() # asumiendo milisegundos de arco (mas)

    color = bp - rp
    mg = g + 5 * np.log10(parallax) - 10

    # Eliminar valores atípicos espurios o ruidosos que expanden la gráfica
    mask = (color > -0.5) & (color < 4.0) & (mg > -3) & (mg < 16)
    color = color[mask]
    mg = mg[mask]

    plt.figure(figsize=(8, 6))
    
    # Graficar con hexbin más denso para asemejarse al CMD teórico
    hb = plt.hexbin(color, mg, gridsize=200, extent=[-0.5, 4.0, -3, 16], cmap='viridis', mincnt=1, bins='log')
    cb = plt.colorbar(hb, label=r'$\log_{10}(N)$')
    
    plt.gca().invert_yaxis()
    
    plt.xlabel(r'$G_{\mathrm{BP}} - G_{\mathrm{RP}}$')
    plt.ylabel(r'$M_G$')
    
    # Textos de referencia acomodados a la nueva escala
    plt.text(1.2, 4, 'Secuencia\nPrincipal', fontsize=10, #rotation=-55,
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    plt.text(2.6, 0.5, 'Gigantes\nRojas', fontsize=10,
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    plt.text(0.1, 11, 'Enanas\nBlancas', fontsize=10,
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    plt.grid(True, linestyle='--', alpha=0.5)

    plt.savefig(FIGURES_DIR / "cmd.pdf", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        data_file = Path(sys.argv[1])
    else:
        data_file = (
            ROOT_DIR
            / "data"
            / "raw"
            / "GaiaQuery_49a3213639c8b391d52b3de55381e2bd.ecsv"
        )

    plot_cmd(Path(data_file).resolve())
