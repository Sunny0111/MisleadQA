"""Fig. 3(a): S correctly classified as inapplicable, by model and group."""

import argparse
import json
from math import isclose
from pathlib import Path

from style import COLORS, MODELS, OUTPUT, ROOT, STYLE, WIDTH, label_color, save_figure

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

RESULTS = ROOT / "experiments/outputs/analysis/scope_diagnostic/main-scope-full-v2/groups.json"
GROUPS = {
    "AmbigDocs/entity": "Entity",
    "FRED/time": "Time",
    "FRED/geography": "Location",
    "SituatedQA/time": "Time",
    "SituatedQA/geography": "Location",
}


def read_rates(path):
    """Keep Cannot determine in each 40-item denominator; do not filter by B_m."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if set(data) != set(MODELS):
        raise ValueError("Scope results must contain the six-model panel.")
    rows = []
    for model in MODELS:
        if set(data[model]) != set(GROUPS):
            raise ValueError(f"{model}: expected the five source-scope groups.")
        row = []
        for group in GROUPS:
            entry = data[model][group]
            metric = entry["metrics"]["S.correct"]
            rate = metric["rate"]
            if (entry["items"] != 40 or metric["denominator"] != 40
                    or rate is None or not 0 <= rate <= 1
                    or not isclose(rate, metric["numerator"] / 40, abs_tol=1e-12)):
                raise ValueError(f"{model}/{group}: invalid 40-item S classification rate.")
            row.append(100 * rate)
        rows.append(row)
    return rows


def plot_rates(rates):
    fig, ax = plt.subplots(figsize=(WIDTH, 1.45))
    fig.subplots_adjust(left=0.33, right=0.87, bottom=0.09, top=0.795)
    cmap = LinearSegmentedColormap.from_list("gold_blue", ["#f5f8fa", COLORS["gold"]])
    heatmap = ax.imshow(rates, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    ax.set_yticks(range(len(MODELS)), labels=MODELS.values(), fontsize=6.6)
    ax.set_xticks(range(len(GROUPS)), labels=GROUPS.values(), fontsize=6.3)
    ax.xaxis.tick_top()
    ax.tick_params(which="both", length=0, pad=2)
    ax.set_xticks([i - 0.5 for i in range(6)], minor=True)
    ax.set_yticks([i - 0.5 for i in range(7)], minor=True)
    ax.grid(which="minor", color="white", linewidth=0.7)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for x, name in ((0.1, "AmbigDocs"), (0.4, "FRED"), (0.8, "SituatedQA")):
        ax.text(x, 1.16, name, transform=ax.transAxes, ha="center", va="bottom", fontsize=6.4)
    for y, row in enumerate(rates):
        for x, value in enumerate(row):
            ax.text(x, y, f"{value:.1f}", ha="center", va="center",
                    fontsize=6.3, color=label_color(cmap(value / 100)))
    cax = fig.add_axes([0.90, 0.09, 0.025, 0.705])
    colorbar = fig.colorbar(heatmap, cax=cax, ticks=[0, 25, 50, 75, 100])
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(length=2, width=0.5, pad=1, labelsize=6.3)
    colorbar.ax.set_title("%", fontsize=6.3, pad=4)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULTS, help="Local scope groups.json")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    rates = read_rates(args.results)
    with plt.rc_context(STYLE):
        fig = plot_rates(rates)
        save_figure(fig, "fig3a_s_scope_recognition_by_model_group", args.output_dir)
        plt.close(fig)
    print("Columns: " + ", ".join(GROUPS))
    for model, row in zip(MODELS, rates):
        print(f"{model}: " + ", ".join(f"{value:.1f}%" for value in row))


if __name__ == "__main__":
    main()
