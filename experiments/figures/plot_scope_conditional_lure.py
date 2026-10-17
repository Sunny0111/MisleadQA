"""Plot S-only lure rates conditional on correct, separately elicited scope judgments."""

import argparse
import json
from math import isclose
from pathlib import Path
from statistics import mean

from style import MODELS, OUTPUT, ROOT, STYLE, WIDTH, save_figure

import matplotlib.pyplot as plt

RESULTS = ROOT / "experiments/outputs/analysis/scope_diagnostic/main-scope-full-v2"
METRIC = "lure_given_inapplicable"
STEM = "fig3_scope_conditional_lure"
ORANGE, INK, SECONDARY, GRID = "#EB6834", "#1F1F1F", "#555555", "#C8C8C8"


def read_rates(directory):
    """Validate saved conditional counts, the model mean, and source-cluster intervals."""
    stats = json.loads((directory / "statistics.json").read_text(encoding="utf-8"))
    bootstrap = json.loads((directory / "bootstrap.json").read_text(encoding="utf-8"))
    if set(stats["models"]) != set(MODELS) or set(bootstrap["panel_models"]) != set(MODELS):
        raise ValueError("Expected the same six models in statistics and bootstrap results.")
    if "source, cluster_id" not in bootstrap["method"]:
        raise ValueError("Expected source-cluster bootstrap intervals.")

    rows, baseline_rates = [], []
    for model, label in [*MODELS.items(), ("equal_model_mean", "Mean")]:
        if model == "equal_model_mean":
            metric = stats[model][METRIC]
            expected = mean(row[1] for row in rows) / 100
            counts = "-"
        else:
            entry = stats["models"][model]
            metric = entry["metrics"][METRIC]
            n, d = metric["numerator"], metric["denominator"]
            outcomes = entry["cross_table_on_B_m"]["Not applicable"]
            if (not 0 <= n <= d <= entry["B_m_size"] <= entry["items"] or d == 0
                    or n != outcomes["lure"] or d != sum(outcomes.values())):
                raise ValueError(f"{model}: invalid conditional counts.")
            expected, counts = n / d, f"{n}/{d}"
            table = entry["cross_table_on_B_m"]
            if sum(sum(outcomes.values()) for outcomes in table.values()) != entry["B_m_size"]:
                raise ValueError(f"{model}: cross-table counts do not exhaust B_m.")
            baseline_rates.append(sum(outcomes["lure"] for outcomes in table.values())
                                  / entry["B_m_size"])

        rate = metric["rate"]
        ci = bootstrap["intervals"][model][METRIC]
        lower, upper = ci["lower"], ci["upper"]
        if (not isclose(rate, expected, abs_tol=1e-12)
                or not isclose(rate, ci["estimate"], abs_tol=1e-12)
                or ci["confidence"] != 0.95 or not 0 <= lower <= rate <= upper <= 1
                or ci["valid_replicates"] != bootstrap["replicates"]):
            raise ValueError(f"{model}: inconsistent rate or incomplete 95% interval.")
        rows.append((label, 100 * rate, 100 * lower, 100 * upper, counts))
    return rows, 100 * mean(baseline_rates)


def plot_rates(rows, baseline):
    """Draw full-precision intervals; round only annotation text."""
    fig, ax = plt.subplots(figsize=(WIDTH, 1.95))
    fig.subplots_adjust(left=0.35, right=0.97, bottom=0.20, top=0.97)
    positions = [*range(len(rows) - 1), len(rows) - 0.6]
    ax.set(xlim=(0, 90), ylim=(positions[-1] + 0.5, -0.8),
           xlabel="Lure rate with S alone (%)")
    ax.set_xticks(range(0, 81, 20))
    ax.set_xticks(range(10, 90, 20), minor=True)
    ax.set_yticks(positions, labels=[row[0] for row in rows])
    ax.tick_params(axis="both", which="both", length=0, pad=4)
    ax.set_axisbelow(True)
    ax.grid(axis="both", color=GRID, alpha=0.55, linewidth=0.4)
    ax.grid(axis="x", which="minor", color=GRID, alpha=0.30, linewidth=0.3)
    ax.axhline(len(rows) - 1.3, color=GRID, linewidth=0.4)
    for spine in ax.spines.values():
        spine.set(visible=True, color=INK, linewidth=0.5)
    for label in ax.get_yticklabels():
        label.set_x(-0.52)
        label.set_horizontalalignment("left")
        label.set_color(INK)
    ax.get_yticklabels()[-1].set_fontweight("bold")

    ax.axvline(baseline, color=SECONDARY, alpha=0.7,
               linewidth=0.5, linestyle=(0, (2, 1.5)))
    ax.annotate(f"all items: {baseline:.1f}", (baseline, -0.45),
                xytext=(2, 0), textcoords="offset points",
                va="center", color=SECONDARY, fontsize=6)
    for y, (label, rate, lower, upper, counts) in zip(positions, rows):
        summary = label == "Mean"
        color = INK if summary else ORANGE
        ax.hlines(y, lower, upper, color=color, alpha=0.40 if summary else 0.45,
                  linewidth=1.4, zorder=2)
        ax.plot(rate, y, "D" if summary else "o", color=color,
                markersize=4, markeredgecolor="white", markeredgewidth=0.6, zorder=3)
        ax.annotate(f"{rate:.1f}" if summary else counts,
                    (lower if summary else upper, y),
                    xytext=(-3 if summary else 4, 0), textcoords="offset points",
                    ha="right" if summary else "left", va="center", fontsize=6,
                    color=INK if summary else SECONDARY,
                    bbox=dict(facecolor="white", edgecolor="none", pad=0.3))
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=RESULTS,
                        help="Directory containing statistics.json and bootstrap.json")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    rows, baseline = read_rates(args.results_dir)
    with plt.rc_context(STYLE):
        fig = plot_rates(rows, baseline)
        save_figure(fig, STEM, args.output_dir)
        plt.close(fig)
    for label, rate, lower, upper, counts in rows:
        print(f"{label}: {rate:.2f}% [{lower:.2f}, {upper:.2f}]; lure/n={counts}")
    print(f"All diagnostic B_m items: {baseline:.2f}% (equal model weights).")


if __name__ == "__main__":
    main()
