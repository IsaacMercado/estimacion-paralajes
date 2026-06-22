import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import matplotlib

ROOT_DIR = Path(__file__).parent.parent.parent
FIGURES_DIR = ROOT_DIR / "reports" / "figures"

def draw_box(ax, x, y, width, height, text, bgcolor='white'):
    box = patches.FancyBboxPatch((x, y), width, height,
                                 boxstyle="round,pad=0.1",
                                 edgecolor="black",
                                 facecolor=bgcolor,
                                 zorder=2)
    ax.add_patch(box)
    ax.text(x + width/2, y + height/2, text, ha='center', va='center', 
            fontsize=11, zorder=3, wrap=True)
    return (x + width/2, y + height/2), box

def draw_flow_arrow(ax, start, end, label=""):
    ax.annotate("",
                xy=end, xycoords='data',
                xytext=start, textcoords='data',
                arrowprops=dict(arrowstyle="->", color="black", lw=1.5,
                                connectionstyle="arc3,rad=0.0"), zorder=1)
    if label:
        mp = ((start[0]+end[0])/2, (start[1]+end[1])/2)
        ax.text(mp[0], mp[1]+0.1, label, ha='center', va='bottom', fontsize=9, zorder=3, 
                bbox=dict(facecolor='white', edgecolor='none', pad=0.5, alpha=0.8))

def plot_gaia_simulations_flow():
    matplotlib.rcParams.update({
        "font.family": "serif",
        "text.usetex": True,
    })

    fig, ax = plt.subplots(figsize=(8, 6))

    # Definición de bloques y coordenadas (x, y)
    w, h = 2.5, 0.8
    
    # Capa simulada arriba
    c_gums, b_gums = draw_box(ax, 1.0, 4.5, w, h, "Universo Simulado\n(Truthing/GUMS)\n$r, M_G, C$ reales", bgcolor="#e6f2ff")
    c_gog, b_gog = draw_box(ax, 1.0, 2.5, w, h, "Observaciones Simuladas\n(GOG)\n$\\hat{\\varpi}, \\hat{m}, \\hat{C}$ simulados", bgcolor="#ffe6e6")
    
    c_dev, b_dev = draw_box(ax, 4.5, 3.5, w, h, "Desarrollo y\nDepuración de Modelos\n(Comparación contra Verdad)", bgcolor="#f2e6ff")
    
    # Capa real abajo
    c_dr3, b_dr3 = draw_box(ax, 1.0, 0.5, w, h, "Gaia DR3 Real\n$\\hat{\\varpi}, \\hat{m}, \\hat{C}$ observados", bgcolor="#e6ffe6")
    c_final, b_final = draw_box(ax, 4.5, 0.5, w, h, "Inferencia Final\nEstimaciones Oficiales", bgcolor="#f9f9f9")

    # Flechas
    # GUMS -> GOG
    draw_flow_arrow(ax, (c_gums[0], c_gums[1]-h/2-0.1), (c_gog[0], c_gog[1]+h/2+0.1), "Añadir ruido\n(Pipeline de Gaia)")
    
    # GOG -> Dev
    draw_flow_arrow(ax, (c_gog[0]+w/2+0.1, c_gog[1]), (c_dev[0]-w/2-0.1, c_dev[1]-0.2))
    
    # GUMS -> Dev (comparación)
    draw_flow_arrow(ax, (c_gums[0]+w/2+0.1, c_gums[1]), (c_dev[0]-w/2-0.1, c_dev[1]+0.2), "Verdad\nground-truth")
    
    # Dev -> DR3 Final Model deploy
    draw_flow_arrow(ax, (c_dev[0], c_dev[1]-h/2-0.1), (c_final[0], c_final[1]+h/2+0.1), "Modelo Validado")
    
    # DR3 -> Final
    draw_flow_arrow(ax, (c_dr3[0]+w/2+0.1, c_dr3[1]), (c_final[0]-w/2-0.1, c_final[1]), "Aplicación\na datos reales")

    ax.set_xlim(0, 8)
    ax.set_ylim(-0.5, 6)
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "flujo_simulaciones.pdf", bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    plot_gaia_simulations_flow()