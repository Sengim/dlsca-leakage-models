"""Guessing entropy vs. number of attack traces for every run of a sweep, one panel per leakage model.

Thin lines are individual runs, the band is the interquartile range across runs and the bold line the
median. Panels are ordered like plot_sweep.py (best median first). Leakage models whose GE cannot reach 0
(S-box input terms: tied key guesses) get their floor drawn as a dashed line.

--log-x puts the attack traces on a log axis; --log-y puts GE on a symmetric-log axis (linear 0..1, log above),
so GE = 0 stays visible while the 1..32 range around the GE < 1 / GE < 32 thresholds is spread out.

Example:
    uv run scripts/plot_ge_curves.py results/sweep_ascadr_27_09_2026_19_17_52 --checkpoint final
    uv run scripts/plot_ge_curves.py results/sweep_ascadr_27_09_2026_19_17_52 --log-x --log-y
"""

import argparse
import csv
import json

import matplotlib.pyplot as plt
import numpy as np

from plot_sweep import COLORS, GRID, INK, INK_MUTED, style, title_prefix
from sca_tg.leakage import parse_leakage_model


def ge_floor(lm):
    """GE floor from key guesses tied with the correct one: (n_tied - 1) / 2, n_tied = 2^(8 - fixed key bits)."""
    in_bits = set()
    out_term = False
    for src, transform in lm.terms:
        if src == "out":
            out_term = True  # S-box output labels separate all key guesses in practice
        else:
            in_bits |= set(range(8)) if isinstance(transform, str) else set(transform)  # ID / HW / HW3: all bits
    return 0.0 if out_term else (2 ** (8 - len(in_bits)) - 1) / 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sweep_dir")
    parser.add_argument("--checkpoint", default="final", choices=["final", "best"])
    parser.add_argument("--log-x", action="store_true", help="log scale for the number of attack traces")
    parser.add_argument("--log-y", action="store_true", help="symmetric-log scale for GE (linear below 1)")
    parser.add_argument("--out", default=None,
                        help="output path without extension (default: <sweep_dir>/ge_curves_<checkpoint>[_logx][_logy])")
    args = parser.parse_args()

    with open(f"{args.sweep_dir}/args.json") as f:
        sweep_args = json.load(f)
    with open(f"{args.sweep_dir}/summary.csv") as f:
        rows = list(csv.DictReader(f))
    runs = sorted({int(r["run"]) for r in rows})
    models = list(dict.fromkeys(r["leakage_model"] for r in rows))

    curves = {}
    for m in models:
        name = m.replace(":", "").replace(",", "").replace("+", "_")
        c = [np.load(f"{args.sweep_dir}/{name}_run{i}.npz")[f"ge_{args.checkpoint}"] for i in runs
             if any(r["leakage_model"] == m and int(r["run"]) == i for r in rows)]
        curves[m] = np.array([x for x in c if np.isfinite(x).all()])  # drop diverged runs
    models.sort(key=lambda m: np.median(curves[m][:, -1]))

    ncols = min(4, len(models))
    nrows = -(-len(models) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.3 * ncols, 2.7 * nrows), sharex=True, sharey=True,
                             squeeze=False)
    color = COLORS[args.checkpoint]
    n = curves[models[0]].shape[1]
    x = np.arange(1, n + 1)
    for ax, m in zip(axes.flat, models):
        c = curves[m]
        for run in c:
            ax.plot(x, run, color=color, lw=0.4, alpha=0.12)
        q1, med, q3 = np.percentile(c, [25, 50, 75], axis=0)
        ax.fill_between(x, q1, q3, color=color, alpha=0.25, lw=0)
        ax.plot(x, med, color=color, lw=2)
        floor = ge_floor(parse_leakage_model(m))
        if floor:
            ax.axhline(floor, color=INK_MUTED, lw=1, ls="--")
            ax.annotate(f"floor {floor:g}", (n, floor), xytext=(0, 3), textcoords="offset points",
                        ha="right", va="bottom", fontsize=7.5, color=INK_MUTED)
        ax.set_title(f"{m}  (median at end: {med[-1]:.0f})", fontsize=9.5, color=INK, loc="left")
        if args.log_y:
            ax.set_yscale("symlog", linthresh=1, linscale=0.5)
            ax.set_ylim(0, 256)
            ax.set_yticks([0, 1, 4, 16, 64, 256], ["0", "1", "4", "16", "64", "256"])
            ax.yaxis.set_minor_locator(plt.NullLocator())
        else:
            ax.set_ylim(-5, 256)
            ax.set_yticks([0, 64, 128, 192, 256])
        if args.log_x:
            ax.set_xscale("log")
            ax.set_xlim(1, n)
        style(ax)
        ax.xaxis.grid(True, color=GRID, lw=0.8)
    for ax in axes.flat[len(models):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("guessing entropy", color=INK)
    for ax in axes[-1]:
        ax.set_xlabel("attack traces", color=INK)

    label = {"final": "final epoch", "best": "best val-loss checkpoint"}[args.checkpoint]
    fig.suptitle(f"{title_prefix(sweep_args)} sweep: GE of {label}, {len(runs)} runs per leakage model "
                 f"(line = median, band = IQR)", color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    suffix = ("_logx" if args.log_x else "") + ("_logy" if args.log_y else "")
    out = args.out or f"{args.sweep_dir}/ge_curves_{args.checkpoint}{suffix}"
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=200, bbox_inches="tight", facecolor="#fcfcfb")
    print(f"Saved {out}.png and {out}.pdf")


if __name__ == "__main__":
    main()
