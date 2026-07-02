from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = Path(__file__).parent.parent.parent
FIGURES_DIR = ROOT_DIR / "reports" / "figures"


def plot_trigonometric_parallax():
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "text.usetex": True,
        }
    )

    fig, ax = plt.subplots(figsize=(6, 7))

    # Órbita (eclíptica)
    orbit = matplotlib.patches.Ellipse(
        (0, 0),
        width=6,
        height=1.5,
        fill=False,
        edgecolor="black",
        linewidth=1,
        zorder=1,
    )
    ax.add_patch(orbit)

    # Líneas principales
    p_y = 5  # Altura de P más cerca
    ax.plot([0, 0], [0, p_y], color="gray", linewidth=1, zorder=1)  # Sol a P
    ax.plot([0, 3], [0, 0], color="gray", linewidth=1, zorder=1)  # Sol a Tierra
    ax.plot([3, 0], [0, p_y], color="gray", linewidth=1, zorder=1)  # Tierra a P

    # Sol
    ax.plot(
        0,
        0,
        marker="o",
        markersize=18,
        markerfacecolor="yellow",
        markeredgecolor="red",
        zorder=3,
    )
    ax.text(-0.35, 0, "Sol", fontsize=14, ha="right", va="center")

    # Tierra
    ax.plot(
        3,
        0,
        marker="o",
        markersize=14,
        markerfacecolor="cyan",
        markeredgecolor="black",
        zorder=3,
    )
    ax.text(3, -0.4, "Tierra", fontsize=12, ha="center", va="top")
    ax.text(3.1, 0.3, "1 de Enero", fontsize=11, ha="left", va="bottom")

    # 1 de Julio
    ax.text(-3, 0.2, "1 de Julio", fontsize=11, ha="center", va="bottom")

    # Eclíptica
    ax.text(-1.5, -1.0, "eclíptica", fontsize=11, ha="center", va="top")

    # P
    ax.plot(0, p_y, marker="*", markersize=16, markerfacecolor="white", markeredgecolor="black", zorder=3)
    ax.text(0.3, p_y + 0.1, "P", fontsize=14, ha="left", va="bottom")

    # Estrellas de fondo
    np.random.seed(42)
    bg_x = np.random.uniform(-3, 3, 15)
    bg_y = np.random.uniform(p_y + 0.6, p_y + 2.0, 15)
    ax.scatter(bg_x, bg_y, marker="*", s=80, color="darkgray", zorder=2)
    ax.text(0, p_y + 2.2, "Fondo", fontsize=12, ha="center", va="bottom")

    # Distancias
    ax.text(-0.1, p_y / 2, "1 pc", fontsize=12, ha="right", va="center")
    ax.text(1.5, 0.1, "1 U.A.", fontsize=12, ha="center", va="bottom")

    # Ángulo 1"
    angle = np.degrees(np.arctan2(-p_y, 3))
    arc = matplotlib.patches.Arc(
        (0, p_y),
        width=3,
        height=3,
        angle=0,
        theta1=270,
        theta2=angle + 360,
        color="black",
        linewidth=1,
    )
    ax.add_patch(arc)
    ax.text(0.1, p_y - 1.8, r"$\varpi = 1^{\prime\prime}$", fontsize=12, ha="left", va="center")

    ax.set_xlim(-4, 5)
    ax.set_ylim(-2, p_y + 3.0)
    ax.axis("off")
    ax.set_aspect("equal")
    fig.tight_layout()

    plt.savefig(FIGURES_DIR / "trigonometric_parallax.pdf", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    plot_trigonometric_parallax()
