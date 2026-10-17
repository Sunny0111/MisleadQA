"""Plot Fig. 2: S-only outcomes on B_m and S/F lure rates on A_m.

Group outcomes average the six model-specific rates with equal weights.
The saved A_m rates are already equal-model means; only convert to percent.
"""

import argparse
import json
from math import isclose
from pathlib import Path
from statistics import mean

from style import COLORS, GRID, OUTPUT, ROOT, STYLE, WIDTH, save_figure

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

RESULTS = ROOT / "experiments/outputs/01a1040f-0ef4-71f2-9563-55b4e111a9a1/all_results.json"
GROUPS = {
    "FRED:time": "FRED time",
    "FRED:geography": "FRED location",
    "SituatedQA:time": "SituatedQA time",
    "SituatedQA:geography": "SituatedQA location",
    "AmbigDocs:entity": "AmbigDocs entity",
}
OUTCOMES = {
    "Lure": ("LR_S", COLORS["lure"]),
    "Gold": ("LR_S.gold", COLORS["gold"]),
    "Abstain": ("LR_S.abstain", COLORS["abstain"]),
    "Other": ("LR_S.other", COLORS["other"]),
}
LURE_KEYS = {
    "S": ("Single_S.on_A.lure", "DR_S.g_then_s", "DR_S.s_then_g"),
    "F": ("Single_F.on_A.lure", "DR_F.g_then_f", "DR_F.f_then_g"),
}
def group_outcomes(results, models):
    """Use saved rates on each model's B_m, without pooling counts."""
    strata = results["experiment_evidence"]["strata"]
    if len(models) != 6 or len(set(models)) != 6 or set(strata) != set(GROUPS):
        raise ValueError("Fig. 2(a) requires six distinct models and the five source-scope groups.")
    rows = []
    for group, label in GROUPS.items():
        entries = strata[group]["models"]
        if set(entries) != set(models):
            raise ValueError(f"{group}: model panel differs from the saved six-model panel.")
        values = []
        for model in models:
            entry = entries[model]
            rates = {}
            for outcome, (key, _) in OUTCOMES.items():
                metric = entry["metrics"][key]
                denominator, rate = metric["denominator"], metric["rate"]
                if (denominator != entry["B_m_size"] or denominator <= 0
                        or rate is None or not 0 <= rate <= 1
                        or not isclose(rate, metric["numerator"] / denominator, abs_tol=1e-12)):
                    raise ValueError(f"{group}/{model}/{key}: invalid B_m rate or denominator.")
                rates[outcome] = 100 * rate
            if not isclose(sum(rates.values()), 100, abs_tol=1e-8):
                raise ValueError(f"{group}/{model}: four outcomes do not sum to 100%.")
            values.append(rates)
        rows.append((label, {outcome: mean(r[outcome] for r in values) for outcome in OUTCOMES}))
    return sorted(rows, key=lambda row: row[1]["Lure"], reverse=True)


def lure_rates(results):
    """Read the existing panel means on A_m; do not average them again."""
    saved = results["review"]["cohort_rates"]["all_selected"]["rates"]
    rates = {}
    for evidence, keys in LURE_KEYS.items():
        if any(saved[key] is None or not 0 <= saved[key] <= 1 for key in keys):
            raise ValueError(f"{evidence}: missing or invalid A_m panel rate.")
        rates[evidence] = [100 * saved[key] for key in keys]
    return rates


def plot_group_outcomes(rows):
    fig, ax = plt.subplots(figsize=(WIDTH, 1.9))
    fig.subplots_adjust(left=0.37, right=0.96, top=0.98, bottom=0.30)
    for y, (_, values) in enumerate(rows):
        left = 0
        for outcome, (_, color) in OUTCOMES.items():
            value = values[outcome]
            ax.barh(y, value, left=left, height=0.64, color=color,
                    edgecolor="white", linewidth=0.7)
            if value >= 6:
                ax.text(left + value / 2, y, f"{value:.0f}", ha="center", va="center",
                        fontsize=6.3, color="#0b0b0b" if outcome == "Abstain" else "white")
            left += value
    ax.set_yticks(range(len(rows)), labels=[label for label, _ in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel(r"S-only responses on $\mathcal{B}_m$ (%)", labelpad=2)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    fig.legend(handles=[Patch(color=color, label=outcome) for outcome, (_, color) in OUTCOMES.items()],
               loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=4,
               handlelength=1, columnspacing=0.9)
    return fig


def plot_lure_rates(rates):
    fig, ax = plt.subplots(figsize=(WIDTH, 1.85))
    fig.subplots_adjust(left=0.17, right=0.98, top=0.97, bottom=0.20)
    x = range(3)
    for evidence, marker, label in (("S", "o", "S"), ("F", "s", "F (reference)")):
        color = OUTCOMES["Lure" if evidence == "S" else "Gold"][1]
        ax.plot(x, rates[evidence], color=color, linewidth=1.5, marker=marker,
                markersize=4.3, markeredgecolor=color,
                label=label, zorder=3, clip_on=False)
        for xi, value in zip(x, rates[evidence]):
            # Move the two near-zero S labels above and to the right.
            if evidence == "S" and xi > 0:
                ax.text(xi + 0.1, value + 1, f"{value:.1f}", color=color,
                        fontsize=6.3, ha="left", va="bottom")
            else:
                ax.text(xi, value + (5 if evidence == "F" else -5), f"{value:.1f}",
                        color=color, fontsize=6.3, ha="center",
                        va="bottom" if evidence == "F" else "top")
    ax.set_xticks(x, labels=["Alone", "G first", "S/F first"])
    ax.set_xlim(-0.3, 2.3)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel(r"Lure rate on $\mathcal{A}_m$ (%)", labelpad=2)
    ax.yaxis.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", handlelength=1.6)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULTS, help="Consolidated all_results.json")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT, help="Directory for vector PDF figures")
    args = parser.parse_args()
    package = json.loads(args.results.read_text(encoding="utf-8"))
    rows = group_outcomes(package["results"], package["models"])
    rates = lure_rates(package["results"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(STYLE):
        for stem, fig in (
            ("fig2a_s_only_outcomes_by_group", plot_group_outcomes(rows)),
            ("fig2b_lure_rates_with_gold", plot_lure_rates(rates)),
        ):
            save_figure(fig, stem, args.output_dir)
            plt.close(fig)
    for label, values in rows:
        print(f"{label}: " + ", ".join(f"{key} {value:.2f}%" for key, value in values.items()))
    for evidence, values in rates.items():
        print(f"{evidence} (Alone, G first, S/F first): " + ", ".join(f"{value:.2f}%" for value in values))


if __name__ == "__main__":
    main()
