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
            fontsize=12, zorder=3, wrap=True)
    return (x + width/2, y + height/2)

def draw_arrow(ax, start, end, label=""):
    ax.annotate("",
                xy=end, xycoords='data',
                xytext=start, textcoords='data',
                arrowprops=dict(arrowstyle="->", color="black", lw=2,
                                connectionstyle="arc3"), zorder=1)
    if label:
        mp = ((start[0]+end[0])/2, (start[1]+end[1])/2)
        ax.text(mp[0], mp[1]+0.15, label, ha='center', va='bottom', fontsize=10, zorder=3)

def plot_iterative_flow():
    matplotlib.rcParams.update({
        "font.family": "serif",
        "text.usetex": True,
    })

    fig, ax = plt.subplots(figsize=(10, 3))
    
    w, h = 2.5, 1.2
    
    c1 = draw_box(ax, 0.5, 1, w, h, "Modelo 0\n(Línea Base)\nEstimación individual\npor estrella", bgcolor="#e6ffe6")
    c2 = draw_box(ax, 4.0, 1, w, h, "Modelo 1\n(Jerárquico)\nContracción hacia\nla población", bgcolor="#e6f2ff")
    c3 = draw_box(ax, 7.5, 1, w, h, "Modelo 2\n(GMM CMD)\nInformación\nfotométrica", bgcolor="#ffe6e6")
    
    draw_arrow(ax, (c1[0] + w/2 + 0.1, c1[1]), (c2[0] - w/2 - 0.1, c2[1]), "Validación cruzada")
    draw_arrow(ax, (c2[0] + w/2 + 0.1, c2[1]), (c3[0] - w/2 - 0.1, c3[1]), "Refinamiento")
    
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 3.5)
    ax.axis("off")
    plt.title("Flujo de Desarrollo Iterativo de Modelos", fontsize=14, pad=10)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "flujo_iterativo.pdf", bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    plot_iterative_flow()