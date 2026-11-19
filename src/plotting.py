"""Common figure style."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .data import FIGURES  # noqa: E402

BLUE, ORANGE, GREEN, RED, GREY, PURPLE = "#1f4e79", "#d9822b", "#2e7d32", "#b23a48", "#7f7f7f", "#6a4c93"

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 160, "font.size": 10, "axes.titlesize": 11,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.alpha": 0.3, "legend.frameon": False,
})


def save(fig, name):
    FIGURES.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES / name, bbox_inches="tight")
    plt.close(fig)
    print(f"saved figures/{name}")


def millions(ax, axis="y"):
    fmt = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1e6:g}")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)
