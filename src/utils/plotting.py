"""Shared plotting helpers for consistent style across all scripts."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
DPI = 150


def new_fig(figsize=(8, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def save_fig(fig, path):
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
