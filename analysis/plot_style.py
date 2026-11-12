"""Shared figure style: one fixed color per method across every figure (color follows the entity)."""
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"

# Categorical slots in fixed order (validated palette): 1 blue, 2 orange, 3 aqua, 4 yellow, 5 magenta, 6 green
ENTITY_COLORS = {
    "F2-fourier": "#2a78d6",
    "H": "#eb6834",
    "A": "#1baf7a",
    "F2-scatter": "#eda100",
    "C": "#e87ba4",
    "B": "#008300",
}
LABELS = {
    "A": "A identity", "B": "B random linear", "C": "C random quadratic", "H": "H random Fourier features",
    "F2-scatter": "F2-scatter (optical, v2)", "F2-fourier": "F2-fourier (optical, v2)",
}


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, length=0)
