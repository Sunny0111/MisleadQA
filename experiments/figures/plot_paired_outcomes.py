"""Plot F-mixed outcomes conditional on S-mixed correctness, by passage order.

Read saved paired statistics; no model requests or bootstrap recomputation.
The two orders can have different conditioning cohorts within each model.
"""

import argparse
import json
from math import isclose, isfinite
from pathlib import Path
from statistics import mean

from style import COLORS, OUTPUT, ROOT, STYLE, WIDTH, label_color, save_figure

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


RESULTS = ROOT / "experiments/outputs/analysis/paired-configurations-main-v1/statistics.json"
STEM = "fig2_paired_conditional_outcomes"
ORDERS = {
    "G_first": "G first",
    "misleading_first": "Misleading\nfirst",
}
PAIRS = {
    "G_first": ["g_then_s", "g_then_f"],
    "misleading_first": ["s_then_g", "f_then_g"],
}
OUTCOME_COLORS = {**COLORS, "invalid": "#999999"}


def read_outcomes(path):
    """Verify saved percentage estimates against per-model conditional counts."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if data["order_pairs"] != PAIRS:
        raise ValueError("Expected within-order S/G to F/G pairing.")
    source = data["source_cluster"]
    models = source["panel_models"]
    if len(models) != 6 or len(set(models)) != 6:
        raise ValueError("Expected the fixed panel of six distinct models.")
    rows = {}
    for order in ORDERS:
        values, denominators = {}, None
        totals = dict.fromkeys(models, 0)
        for outcome in OUTCOME_COLORS:
            key = f"{order}.F_given_S_gold.{outcome}"
            entry = source["panel"][key]
            nums, dens = entry["model_numerators"], entry["model_denominators"]
            if set(nums) != set(models) or set(dens) != set(models):
                raise ValueError(f"{key}: counts do not match the fixed model panel.")
            if denominators is None:
                denominators = dens
            elif dens != denominators:
                raise ValueError(f"{key}: outcome categories use different denominators.")
            for model in models:
                n, d = nums[model], dens[model]
                if not isinstance(n, int) or not isinstance(d, int) or not 0 <= n <= d or d == 0:
                    raise ValueError(f"{key}/{model}: invalid conditional counts.")
                totals[model] += n
            value = entry["estimate"]
            expected = mean(100 * nums[model] / dens[model] for model in models)
            if (entry["unit"] != "percent" or value is None or not isfinite(value)
                    or not 0 <= value <= 100 or not isclose(value, expected, abs_tol=1e-8)):
                raise ValueError(f"{key}: estimate is not the unweighted mean over models.")
            values[outcome] = value
        if totals != denominators or not isclose(sum(values.values()), 100, abs_tol=1e-8):
            raise ValueError(f"{order}: incomplete outcome distribution; refusing to renormalize.")
        rows[order] = values
    return rows


def plot_outcomes(rows, width=WIDTH, height=1.85):
    # Omit a zero-width invalid category from the legend only when both rates are zero.
    shown = [name for name in OUTCOME_COLORS
             if name != "invalid" or any(row[name] > 0 for row in rows.values())]
    fig, ax = plt.subplots(figsize=(width, height))
    fig.subplots_adjust(left=0.29, right=0.96, top=0.95, bottom=0.32)

    for y, order in enumerate(ORDERS):
        left = 0.0
        for outcome in shown:
            value, color = rows[order][outcome], OUTCOME_COLORS[outcome]
            # No white outlines: they would obscure the very narrow Other segments.
            ax.barh(y, value, left=left, height=0.48, color=color,
                    edgecolor="none", linewidth=0)
            center = left + value / 2
            if value >= 6:
                ax.text(center, y, f"{value:.1f}", ha="center", va="center",
                        fontsize=6.8, color=label_color(color))
            elif outcome in ("gold", "lure") and value > 0:
                ax.annotate(f"{value:.1f}", xy=(center, y - 0.24),
                            xytext=(min(center, 92), y - 0.48),
                            ha="center", va="bottom", fontsize=6.8, color=color,
                            arrowprops={"arrowstyle": "-", "color": color,
                                        "linewidth": 0.6, "shrinkA": 1, "shrinkB": 1})
            left += value

    ax.set_yticks(range(len(ORDERS)), labels=list(ORDERS.values()), fontsize=6.9)
    ax.tick_params(axis="y", length=0, pad=5)
    ax.set_ylim(1.55, -0.70)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.tick_params(axis="x", length=2, width=0.5, pad=2, labelsize=6.5)
    ax.set_xlabel("Conditional responses (%)", fontsize=6.8, labelpad=2)
    ax.spines["left"].set_visible(False)
    fig.legend(handles=[Patch(facecolor=OUTCOME_COLORS[name], label=name.title())
                        for name in shown], loc="lower center", bbox_to_anchor=(0.5, 0.01),
               ncol=len(shown), handlelength=0.9, columnspacing=0.8, fontsize=6.5)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULTS,
                        help="Paired-configurations statistics.json")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--width", type=float, default=WIDTH, help="Figure width in inches")
    parser.add_argument("--height", type=float, default=1.85, help="Figure height in inches")
    args = parser.parse_args()
    if not all(isfinite(v) and v > 0 for v in (args.width, args.height)):
        parser.error("Figure width and height must be positive finite values.")
    if not args.results.is_file():
        parser.error(f"Saved paired statistics not found: {args.results}")
    rows = read_outcomes(args.results)
    with plt.rc_context(STYLE):
        fig = plot_outcomes(rows, width=args.width, height=args.height)
        save_figure(fig, STEM, args.output_dir)
        plt.close(fig)
    print("Percentages are unweighted means over the six models; counts are not pooled.")
    for order, values in rows.items():
        print(f"{order}: " + ", ".join(f"{name} {value:.2f}%" for name, value in values.items()))
    print("Conditioning cohorts can differ across orders; this is not a fixed-cohort order contrast.")


if __name__ == "__main__":
    main()
