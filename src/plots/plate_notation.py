import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import matplotlib

ROOT_DIR = Path(__file__).parent.parent.parent
FIGURES_DIR = ROOT_DIR / "reports" / "figures"


def draw_node(ax, x, y, label, is_observed=False, radius=0.4):
    """Dibuja un nodo circular (variable aleatoria)"""
    facecolor = 'lightgray' if is_observed else 'white'
    circle = mpatches.Circle((x, y), radius, facecolor=facecolor, edgecolor='black', linewidth=1.5, zorder=3)
    ax.add_patch(circle)
    ax.text(x, y, label, ha='center', va='center', fontsize=14, zorder=4)

def draw_arrow(ax, start, end, radius=0.4):
    """Dibuja una flecha entre dos nodos"""
    import math
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    angle = math.atan2(dy, dx)
    
    # Acortar la flecha para que no pise los círculos
    start_x = start[0] + radius * math.cos(angle)
    start_y = start[1] + radius * math.sin(angle)
    end_x = end[0] - radius * math.cos(angle) - 0.05 * math.cos(angle)  # margen extra
    end_y = end[1] - radius * math.sin(angle) - 0.05 * math.sin(angle)
    
    ax.annotate("",
                xy=(end_x, end_y), xycoords='data',
                xytext=(start_x, start_y), textcoords='data',
                arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
                zorder=2)

def plot_plate_notation():
    matplotlib.rcParams.update({
        "font.family": "serif",
        "text.usetex": True,
    })

    fig, ax = plt.subplots(figsize=(6, 6))

    # Coordenadas de los nodos
    # Nivel poblacional (hiperparámetros)
    x_alpha, y_alpha = 4, 5
    x_phi, y_phi = 4, 3.5      # f_b (pesos de las componentes)
    
    # Nivel individual (latentes)
    x_r, y_r = 2, 2
    x_M, y_M = 4, 2
    x_C, y_C = 6, 2
    
    # Nivel individual (observables)
    x_w, y_w = 2, 0.5
    x_m, y_m = 4, 0.5
    x_cobs, y_cobs = 6, 0.5

    # Dibujar la placa (Plate) para N estrellas
    plate = mpatches.Rectangle((0.8, -0.2), 6.4, 3.2, fill=False, edgecolor='black', linewidth=1.2, zorder=1)
    ax.add_patch(plate)
    ax.text(6.8, -0.0, r"$k \in \{1, \dots, N\}$", ha='right', va='bottom', fontsize=12)

    # Nodos hiperparámetros
    draw_node(ax, x_alpha, y_alpha, r"$\alpha$")               # Hiperprior hiperparámetro de Dirichlet/pesos
    draw_node(ax, x_phi, y_phi, r"$\mathbf{f}$")               # Pesos CMD f_b

    # Nodos latentes
    draw_node(ax, x_r, y_r, r"$r_k$")
    draw_node(ax, x_M, y_M, r"$M_k$")
    draw_node(ax, x_C, y_C, r"$C_k$")

    # Nodos observados
    draw_node(ax, x_w, y_w, r"$\hat{\varpi}_k$", is_observed=True)
    draw_node(ax, x_m, y_m, r"$\hat{m}_k$", is_observed=True)
    draw_node(ax, x_cobs, y_cobs, r"$\hat{C}_k$", is_observed=True)

    # Conexiones: Hiperparámetros -> Latentes
    draw_arrow(ax, (x_alpha, y_alpha), (x_phi, y_phi))
    draw_arrow(ax, (x_phi, y_phi), (x_M, y_M))  # f afecta el prior de M y C
    draw_arrow(ax, (x_phi, y_phi), (x_C, y_C))
    
    # M y C están correlacionados en el prior conjunto basado en el CMD
    # O simplificado: la latente general "tipo estelar" genera M y C.
    
    # Conexiones: Latentes -> Observables
    draw_arrow(ax, (x_r, y_r), (x_w, y_w))
    
    draw_arrow(ax, (x_r, y_r), (x_m, y_m))    # m_k depende de r_k
    draw_arrow(ax, (x_M, y_M), (x_m, y_m))    # m_k depende de M_k
    
    draw_arrow(ax, (x_C, y_C), (x_cobs, y_cobs)) # C_obs depend de C_true

    ax.set_xlim(0, 8)
    ax.set_ylim(-1, 6)
    ax.axis("off")
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "plate_notation.pdf", bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    plot_plate_notation()