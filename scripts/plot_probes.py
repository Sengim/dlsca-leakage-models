"""Figures for probe_sweep.py results: what each leakage model's networks encode about the shares.

1. <out>_gains: heatmap of trained-minus-init probe scores per leakage model (rows) and share x target (columns),
   at hidden layers 1 and 2. Accuracy targets (bit 0, bit 1, high/low HW) in percentage points; HW regression in
   R^2 x 100 in its own panel.
2. <out>_accuracy: absolute probe accuracy per leakage model for a few key probes, at hidden layer 1 and at the
   last hidden layer, with the untrained initialisation (hidden 1) as reference.

Example:
    uv run scripts/plot_probes.py results/probes_sweep_ascadr_27_09_2026_19_17_52/probes.csv --set paired \
        --out docs/figures/probes_ascadr_mlp_paired
"""

import argparse
import csv
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from plot_sweep import COLORS, INK, INK_MUTED, style

LEAKAGE_MODELS = ["out:0,1", "out:0,1+in:0,1", "in:0,1", "HW3", "HW", "out:2,3", "ID", "out:2,3,4,5,6,7"]
VARIABLES = ["r[2]", "r_in", "r_out", "Sbox(p^k) ^ r[2]", "Sbox(p^k) ^ r_out", "p^k ^ r_in", "Sbox(p^k)", "p^k"]
SHORT = {"r[2]": "r[2]", "r_in": "r_in", "r_out": "r_out", "Sbox(p^k) ^ r[2]": "S⊕r[2]", "Sbox(p^k) ^ r_out": "S⊕r_out",
         "p^k ^ r_in": "x⊕r_in", "Sbox(p^k)": "S (unmasked)", "p^k": "x (unmasked)"}
ACC_TARGETS = ["bit 0", "bit 1", "high/low"]
# diverging blue (gain) <-> red (loss) with a neutral grey midpoint
CMAP = LinearSegmentedColormap.from_list("gain", ["#e34948", "#f0efec", "#2a78d6"])
KEY_PROBES = [("Sbox(p^k) ^ r_out", "high/low"), ("r_out", "high/low"), ("Sbox(p^k) ^ r[2]", "bit 0"),
              ("r[2]", "bit 0"), ("r_in", "bit 0"), ("Sbox(p^k) ^ r[2]", "bit 1"),
              ("Sbox(p^k)", "high/low"), ("Sbox(p^k)", "bit 0")]


def load(path, which):
    """values[(lm, run, network, layer, variable, target)] -> score, and the last hidden layer of each model."""
    values, last = {}, {}
    with open(path) as f:
        for r in csv.DictReader(f):
            if r[f"in_{which}"] != "True" or r["layer"] == "logits":
                continue
            key = (r["leakage_model"], int(r["run"]))
            values[key + (r["network"], r["layer"], r["variable"], r["target"])] = float(r["value"])
            last[key] = f"hidden {int(r['n_layers']) - 1}"
    return values, last


def per_run(values, lm, network, layer, var, target, runs):
    return np.array([values[(lm, i, network, layer, var, target)] for i in runs])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("probes_csv")
    parser.add_argument("--set", default="paired", choices=["paired", "best"])
    parser.add_argument("--out", required=True, help="output prefix (without _gains / _accuracy and extension)")
    args = parser.parse_args()

    values, last = load(args.probes_csv, args.set)
    runs = defaultdict(list)
    for lm, i in last:
        runs[lm].append(i)
    lms = [lm for lm in LEAKAGE_MODELS if lm in runs]
    set_label = {"paired": "paired set (the same 10 runs for every leakage model)",
                 "best": "best-10 set (each leakage model's 10 lowest-GE runs)"}[args.set]

    # 1. gain heatmaps
    fig, axes = plt.subplots(2, 2, figsize=(17, 8.2), gridspec_kw={"width_ratios": [3, 1]})
    for row, layer in enumerate(("hidden 1", "hidden 2")):
        for col, (targets, scale, unit, lim) in enumerate(((ACC_TARGETS, 100, "pp", 6), (["HW"], 100, "R² × 100", 15))):
            ax = axes[row, col]
            cols = [(v, t) for v in VARIABLES for t in targets]
            grid = np.array([[scale * np.mean(per_run(values, lm, "trained", layer, v, t, runs[lm])
                                              - per_run(values, lm, "init", layer, v, t, runs[lm])) for v, t in cols]
                             for lm in lms])
            ax.imshow(grid, cmap=CMAP, vmin=-lim, vmax=lim, aspect="auto")
            for (y, x), g in np.ndenumerate(grid):
                ax.text(x, y, f"{g:+.1f}", ha="center", va="center", fontsize=7.5,
                        color="white" if abs(g) > 0.6 * lim else INK)
            ax.set_yticks(range(len(lms)), lms if col == 0 else [""] * len(lms))
            if len(targets) > 1:
                ax.set_xticks(range(len(cols)), [t.replace("high/low", "hi/lo") for _, t in cols], fontsize=8)
                for k in range(1, len(VARIABLES)):
                    ax.axvline(k * len(targets) - 0.5, color="white", lw=3)
                for k, v in enumerate(VARIABLES):
                    ax.text(k * len(targets) + (len(targets) - 1) / 2, -0.8, SHORT[v], ha="center", va="bottom",
                            fontsize=9, color=INK)
            else:
                ax.set_xticks(range(len(cols)), [SHORT[v] for v, _ in cols], rotation=45, ha="right", fontsize=8)
            ax.tick_params(length=0, colors=INK_MUTED, labelcolor=INK)
            for s in ax.spines.values():
                s.set_visible(False)
            ax.set_title(f"{layer}: gain over init ({unit})", loc="left", fontsize=10,
                         color=INK, pad=22 if len(targets) > 1 else 6)
    fig.suptitle(f"ASCADr MLPs, {set_label}: probe score of the trained network minus its initialisation "
                 f"(mean over runs; S = Sbox(p⊕k), x = p⊕k)", color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}_gains.{ext}", dpi=200, bbox_inches="tight", facecolor="#fcfcfb")

    # 2. absolute accuracies for key probes
    fig, axes = plt.subplots(2, 4, figsize=(17, 7.5), sharex=True)
    x = np.arange(len(lms))
    for ax, (var, target) in zip(axes.flat, KEY_PROBES):
        init = [100 * per_run(values, lm, "init", "hidden 1", var, target, runs[lm]).mean() for lm in lms]
        ax.scatter(x, init, marker="_", s=260, lw=2, color=INK_MUTED, zorder=2, label="init (hidden 1)")
        for dx, (layer_name, color) in zip((-0.14, 0.14), (("hidden 1", COLORS["best"]), ("last hidden", COLORS["final"]))):
            vals = [100 * np.array([values[(lm, i, "trained", layer_name if layer_name != "last hidden" else last[(lm, i)],
                                            var, target)] for i in runs[lm]]) for lm in lms]
            ax.errorbar(x + dx, [v.mean() for v in vals], yerr=[v.std(ddof=1) / np.sqrt(len(v)) for v in vals],
                        fmt="o", ms=6, color=color, mec="white", mew=0.8, elinewidth=1.2, capsize=0, zorder=3,
                        label=f"trained, {layer_name}")
        ax.set_title(f"{SHORT[var]}: {target}", loc="left", fontsize=10, color=INK)
        ax.set_xticks(x, lms, rotation=45, ha="right", fontsize=8.5)
        style(ax)
        ax.xaxis.grid(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("probe accuracy (%)", color=INK)
    fig.suptitle(f"ASCADr MLPs, {set_label}: probe accuracy per leakage model (mean ± s.e.; chance = 50%)",
                 color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper right", ncol=3, frameon=False, fontsize=9,
               labelcolor=INK, bbox_to_anchor=(0.995, 0.985))
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}_accuracy.{ext}", dpi=200, bbox_inches="tight", facecolor="#fcfcfb")
    print(f"Saved {args.out}_gains.png and {args.out}_accuracy.png (+ pdf)")


if __name__ == "__main__":
    main()
