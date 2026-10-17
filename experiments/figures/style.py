"""Shared model labels, typography, colors and figure export settings."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib.colors import to_rgb

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "experiments/outputs/figures"
WIDTH = 3.03
MODELS = {
    "qwen3-8b": "Qwen3-8B",
    "qwen3-14b": "Qwen3-14B",
    "qwen3-32b-awq": "Qwen3-32B-AWQ",
    "llama31-8b-instruct": "Llama-3.1-8B",
    "gemma3-12b-it": "Gemma-3-12B",
    "olmo2-13b-instruct": "OLMo-2-13B",
}
INK = "#0b0b0b"
GRID = "#e4e3df"
COLORS = {
    "lure": "#D55E00",
    "gold": "#0072B2",
    "abstain": "#F2B701",
    "other": "#009E73",
}
STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 7.5,
    "axes.labelsize": 7.3,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#8a8984",
    "axes.linewidth": 0.6,
    "xtick.color": "#52514e",
    "ytick.color": INK,
    "legend.fontsize": 6.8,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def label_color(color):
    """Choose a contrasting label color from the fill's luminance."""
    linear = [v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4
              for v in to_rgb(color)]
    luminance = sum(w * v for w, v in zip((0.2126, 0.7152, 0.0722), linear))
    return "white" if luminance < 0.18 else INK


def save_figure(fig, stem, output_dir):
    """Export the publication figure as a vector PDF."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{stem}.pdf"
    fig.savefig(path)
    print(path.resolve())
