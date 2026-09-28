"""Violin plot of a sweep on one combined attack-performance scale per leakage model.

Runs that end with GE < 1 are placed by their number of attack traces to GE < 1 (bottom half,
n_attack at the midline -> 0 at the bottom); runs that don't are placed by their final GE (top half,
0 at the midline -> 256 at the top). The scale is monotonic: lower is always a better attack.

Each violin is split: left half = best-val-loss checkpoint, right half = final epoch. Individual runs
are overlaid as dots. Leakage models are ordered by decreasing median score of the best checkpoint.
With --final-only, the best-val-loss checkpoint is dropped and each violin is a full final-epoch violin
(models ordered by the final-epoch median instead); the GE < 32 layer then gets its own color.

If the sweep tracked GE < 32 (ntge32_* columns), selected leakage models get a second layer at the
same x: the runs that reach GE < 32, placed by their traces to GE < 32 (hatched outline, hollow dots,
dashed medians). By default these are the models with an S-box input term (whose GE can't reach 0,
e.g. in:0,1 bottoms out at 31.5) plus ID as a reference; override with --ge32.

Example:
    uv run scripts/plot_sweep.py results/sweep_ascadr_27_09_2026_18_33_28
    uv run scripts/plot_sweep.py results/sweep_ascadr_27_09_2026_18_33_28 --final-only
"""

import argparse
import csv
import json
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from sca_tg.leakage import parse_leakage_model

COLORS = {"best": "#2a78d6", "final": "#eb6834", "ge32": "#7b52c4"}
LABELS = {"best": "best val-loss checkpoint", "final": "final epoch"}
INK, INK_MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"


def score(ge, ntge, n_attack, threshold=1):
    """Map a run to the combined scale: [0, 1] = NTGE / n_attack if final GE < threshold, else 1 + GE / 256."""
    if not np.isfinite(ge):
        return np.nan  # diverged run
    return ntge / n_attack if ge < threshold else 1 + ge / 256


def load(sweep_dir):
    """data[leakage_model][(threshold, checkpoint)] -> list of scores."""
    with open(f"{sweep_dir}/args.json") as f:
        n_attack = json.load(f)["n_attack"]
    data = defaultdict(lambda: defaultdict(list))
    with open(f"{sweep_dir}/summary.csv") as f:
        for row in csv.DictReader(f):
            for which in ("best", "final"):
                ge = float(row[f"ge_{which}"])
                for threshold, col in ((1, f"ntge_{which}"), (32, f"ntge32_{which}")):
                    if col not in row:
                        continue  # older sweeps have no GE<32 column
                    ntge = float(row[col]) if row[col] else np.nan
                    if threshold != 1 and not ge < threshold:
                        ntge, ge = np.nan, np.nan  # only runs that reach the threshold get a second point
                    data[row["leakage_model"]][(threshold, which)].append(score(ge, ntge, n_attack, threshold))
    return data, n_attack


def half_violin(ax, values, x, side, color, hollow=False, width=0.8):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    sign = {"left": -1, "right": 1, "both": 1}[side]
    lo, hi = (-width / 2, width / 2) if side == "both" else sorted((0, sign * width / 2))
    if len(values) >= 2 and np.ptp(values) > 0:
        body = ax.violinplot(values, positions=[x], widths=width, showextrema=False)["bodies"][0]
        verts = body.get_paths()[0].vertices
        if side != "both":
            verts[:, 0] = np.clip(verts[:, 0], -np.inf, x) if side == "left" else np.clip(verts[:, 0], x, np.inf)
        body.set_edgecolor(color)
        body.set_linewidth(1)
        if hollow:
            body.set_facecolor("none")
            body.set_hatch("////")
            body.set_linestyle("--")
            body.set_alpha(0.8)
        else:
            body.set_facecolor(color)
            body.set_alpha(0.35)
    elif len(values):  # no spread to estimate a density from: draw a flat tick instead
        ax.plot([x + lo, x + hi], [values[0]] * 2, color=color, lw=2, solid_capstyle="round",
                ls="--" if hollow else "-")
    if len(values):
        rng = np.random.default_rng(1 if hollow else 0)
        if side == "both":
            jitter = rng.uniform(-0.2, 0.2, len(values))
        else:
            jitter = rng.uniform(0.05, 0.25, len(values)) * sign
        ax.scatter(x + jitter, values, s=18, zorder=3, linewidth=1 if hollow else 0.8,
                   color="white" if hollow else color, edgecolor=color if hollow else "white")
        ax.plot([x + lo * 0.875, x + hi * 0.875], [np.median(values)] * 2, color=INK, lw=1.5, zorder=4, ls="--" if hollow else "-")


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK_MUTED)
    ax.tick_params(colors=INK_MUTED, labelcolor=INK)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sweep_dir")
    parser.add_argument("--ge32", nargs="*", default=None,
                        help="leakage models that get the GE<32 layer (default: models with an in: term, plus ID)")
    parser.add_argument("--final-only", action="store_true",
                        help="drop the best-val-loss checkpoint and plot full final-epoch violins")
    parser.add_argument("--out", default=None,
                        help="output path without extension (default: <sweep_dir>/violins[_final])")
    args = parser.parse_args()

    data, n_attack = load(args.sweep_dir)
    checkpoints = [("final", "both")] if args.final_only else [("best", "left"), ("final", "right")]
    order_by = checkpoints[0][0]
    models = sorted(data, key=lambda m: -np.nanmedian(data[m][(1, order_by)]))
    has_ge32 = bool(data[models[0]][(32, "best")])

    if args.ge32 is None:
        ge32_models = {m for m in models if m == "ID" or any(src == "in" for src, _ in parse_leakage_model(m).terms)}
    else:
        ge32_models = set(args.ge32)

    def thresholds_for(m):
        return [1, 32] if has_ge32 and m in ge32_models else [1]

    xs = np.arange(len(models))

    fig, ax = plt.subplots(figsize=(1.9 * len(models) + 2, 6.5))
    for x, m in zip(xs, models):
        counts = []
        for t in thresholds_for(m):
            for which, side in checkpoints:
                color = COLORS["ge32"] if args.final_only and t != 1 else COLORS[which]
                half_violin(ax, data[m][(t, which)], x, side, color, hollow=t != 1)
            n = len(data[m][(t, "final")])
            broke = [int((np.asarray(data[m][(t, w)]) <= 1).sum()) for w, _ in checkpoints]
            counts.append(f"<{t}: " + " | ".join(f"{b}/{n}" for b in broke))
        ax.annotate("\n".join(counts), (x, 0), xytext=(0, -4), textcoords="offset points",
                    ha="center", va="top", fontsize=7.5, color=INK_MUTED, linespacing=1.3)

    ax.axhline(1, color=INK_MUTED, lw=1, ls="--")
    ge_ticks = [0, 64, 128, 192, 256]
    nt_ticks = np.linspace(0, n_attack, 5)
    ax.set_yticks([t / n_attack for t in nt_ticks] + [1 + t / 256 for t in ge_ticks[1:]],
                  [f"{int(t)}" for t in nt_ticks[:-1]] + [f"{int(n_attack)} / 0"] + [str(t) for t in ge_ticks[1:]])
    ax.set_ylim(-0.08, 2.05)
    for y, label in ((1.5, "final GE (didn't reach threshold)"), (0.5, "attack traces to GE < threshold")):
        ax.annotate(label, (0, y), xycoords=ax.get_yaxis_transform(), xytext=(-62, 0), textcoords="offset points",
                    rotation=90, ha="right", va="center", color=INK)
    ax.set_xticks(xs, models)
    ax.tick_params(axis="x", length=0, pad=24 if any(len(thresholds_for(m)) > 1 for m in models) else 14)
    ax.set_xlim(-0.6, len(models) - 0.4)
    style(ax)
    ax.text(1.0, -0.14, "runs reaching threshold: " + " | ".join(w for w, _ in checkpoints), transform=ax.transAxes, ha="right",
            fontsize=8, color=INK_MUTED)

    handles = [plt.Line2D([], [], marker="o", ls="", color=COLORS[w], label=LABELS[w]) for w, _ in checkpoints]
    handles.append(plt.Line2D([], [], color=INK, lw=1.5, label="GE < 1 (filled)"))
    if any(32 in thresholds_for(m) for m in models):
        ge32_color = COLORS["ge32"] if args.final_only else INK
        handles.append(plt.Line2D([], [], color=ge32_color, lw=1.5, ls="--", marker="o", markerfacecolor="white",
                                  label="GE < 32 (hatched, hollow)"))
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=9, labelcolor=INK)
    first = next(iter(data.values()))
    ax.set_title(f"ASCADr sweep: {len(first[(1, 'final')])} runs per leakage model (random subsets + hyperparameters)",
                    color=INK, fontsize=11, loc="left")

    out = args.out or f"{args.sweep_dir}/violins{'_final' if args.final_only else ''}"
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=200, bbox_inches="tight", facecolor="#fcfcfb")
    print(f"Saved {out}.png and {out}.pdf")


if __name__ == "__main__":
    main()
