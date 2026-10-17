"""Fig. 3(b): Original and context-faithful outcome distributions on fixed cohorts."""

import argparse
import json
from math import isclose, isfinite
from pathlib import Path

from style import COLORS, GRID, INK, OUTPUT, ROOT, STYLE, WIDTH, label_color, save_figure

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

RESULTS = ROOT / "experiments/outputs/analysis/context-faithful-r3/context_summary.json"
CONDITIONS = {
    "S-only | original B": "S-only\n" + r"($\mathcal{B}_m$)",
    "G-only | all items": "G-only\n(all)",
    "G+S mean | original A": "G+S\n" + r"($\mathcal{A}_m$)",
}
PROMPTS = {"reference_rate": "Original", "treatment_rate": "CF"}


def read_outcomes(path):
    """Saved values are percentages and already use equal model weights."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if len(data["by_model"]) != 6:
        raise ValueError("Context results must contain the six-model panel.")
    rows = []
    for condition in CONDITIONS:
        saved = data["panel_outcomes_pct"][condition]
        for prompt in PROMPTS:
            values = [saved[outcome][prompt] for outcome in COLORS]
            invalid = saved["invalid"][prompt]
            if (any(value is None or not isfinite(value) or not 0 <= value <= 100
                    for value in values)
                    or invalid != 0 or not isclose(sum(values), 100, abs_tol=1e-8)):
                raise ValueError(f"{condition}/{prompt}: incomplete or invalid outcome distribution.")
            rows.append(values)
    return rows


def plot_outcomes(rows):
    fig, ax = plt.subplots(figsize=(WIDTH, 1.70))
    fig.subplots_adjust(left=0.36, right=0.96, top=0.95, bottom=0.315)
    positions = [0, 0.72, 1.92, 2.64, 3.84, 4.56]
    for y, values in zip(positions, rows):
        left = 0
        for color, value in zip(COLORS.values(), values):
            ax.barh(y, value, left=left, height=0.55, color=color,
                    edgecolor="white", linewidth=0.4)
            if value >= 6:
                ax.text(left + value / 2, y, f"{value:.1f}", ha="center", va="center",
                        fontsize=6.3, color=label_color(color))
            left += value
    for y in (1.32, 3.24):
        ax.axhline(y, color=GRID, linewidth=0.6, clip_on=False)
    for i, label in enumerate(CONDITIONS.values()):
        ax.text(-0.23, positions[2 * i] + 0.36, label, transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=6.6, color=INK)
    ax.set_yticks(positions, labels=list(PROMPTS.values()) * 3, fontsize=6.5)
    ax.tick_params(axis="y", length=0, pad=3)
    ax.set_ylim(4.98, -0.42)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.tick_params(axis="x", length=2, width=0.5, pad=2, labelsize=6.5)
    ax.set_xlabel("Responses (%)", fontsize=6.8, labelpad=2)
    ax.spines["left"].set_visible(False)
    fig.legend(handles=[Patch(color=color, label=name.title()) for name, color in COLORS.items()],
               loc="lower center", bbox_to_anchor=(0.5, 0), ncol=4,
               handlelength=0.9, columnspacing=0.8, fontsize=6.5)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULTS, help="Local context_summary.json")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    rows = read_outcomes(args.results)
    with plt.rc_context(STYLE):
        fig = plot_outcomes(rows)
        save_figure(fig, "fig3b_original_vs_context_faithful_outcomes", args.output_dir)
        plt.close(fig)
    for i, condition in enumerate(CONDITIONS):
        for j, prompt in enumerate(PROMPTS.values()):
            values = rows[2 * i + j]
            print(f"{condition} / {prompt}: "
                  + ", ".join(f"{name} {value:.2f}%" for name, value in zip(COLORS, values)))


if __name__ == "__main__":
    main()
