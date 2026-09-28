"""Print markdown tables summarising a sweep (for docs/): per-leakage-model results and paired comparisons.

Per leakage model: runs that end with GE < 1 / GE < 32 (final epoch and best-val-loss checkpoint), median
attack traces to reach them among those runs, median final GE and median best epoch.
Paired: for each run, which of two leakage models attacks better on the combined scale of plot_sweep.py
(traces to GE < 1 if reached, else final GE), with ties when neither reaches GE < 1 and both final GEs match.

Example:
    uv run scripts/summarize_sweep.py results/sweep_ascadr_27_09_2026_19_17_52 --reference "out:0,1"
"""

import argparse
import csv
import json
from collections import defaultdict

import numpy as np

from plot_sweep import score


def num(v):
    return float(v) if v not in ("", None) else np.nan


def med(values, fmt="{:.0f}"):
    values = [v for v in values if np.isfinite(v)]
    return fmt.format(np.median(values)) if values else "–"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sweep_dir")
    parser.add_argument("--reference", default="out:0,1", help="leakage model the paired table compares against")
    parser.add_argument("--checkpoint", default="final", choices=["final", "best"], help="checkpoint for the paired table")
    args = parser.parse_args()

    with open(f"{args.sweep_dir}/args.json") as f:
        n_attack = json.load(f)["n_attack"]
    with open(f"{args.sweep_dir}/summary.csv") as f:
        rows = list(csv.DictReader(f))
    by_model = defaultdict(list)
    for r in rows:
        by_model[r["leakage_model"]].append(r)

    def run_score(r, which):
        return score(num(r[f"ge_{which}"]), num(r[f"ntge_{which}"]), n_attack)

    models = sorted(by_model, key=lambda m: np.nanmedian([run_score(r, "final") for r in by_model[m]]))

    print("| leakage model | classes | runs | GE<1 final | traces to GE<1 (median) | GE<32 final | traces to GE<32 (median) "
          "| median final GE | GE<1 best ckpt | median best epoch |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for m in models:
        rs = by_model[m]
        ge = np.array([num(r["ge_final"]) for r in rs])
        hit1, hit32 = ge < 1, ge < 32
        nt1 = [num(r["ntge_final"]) for r, h in zip(rs, hit1) if h]
        nt32 = [num(r["ntge32_final"]) for r, h in zip(rs, hit32) if h]
        best1 = sum(num(r["ge_best"]) < 1 for r in rs)
        print(f"| `{m}` | {rs[0]['num_classes']} | {len(rs)} | {hit1.sum()} | {med(nt1)} | {hit32.sum()} | {med(nt32)} "
              f"| {med(ge, '{:.1f}')} | {best1} | {med([num(r['best_epoch']) for r in rs])} |")

    ref = {int(r["run"]): run_score(r, args.checkpoint) for r in by_model.get(args.reference, [])}
    if not ref:
        return
    ckpt = {"final": "final epoch", "best": "best val-loss checkpoint"}[args.checkpoint]
    print(f"\nPaired against `{args.reference}` (same subset and hyperparameters per run, {ckpt}):\n")
    print(f"| leakage model | runs | `{args.reference}` better | other better | tie |")
    print("|---|---|---|---|---|")
    for m in models:
        if m == args.reference:
            continue
        pairs = [(ref[int(r["run"])], run_score(r, args.checkpoint)) for r in by_model[m] if int(r["run"]) in ref]
        pairs = [(a, b) for a, b in pairs if np.isfinite(a) and np.isfinite(b)]
        wins = sum(a < b for a, b in pairs)
        losses = sum(a > b for a, b in pairs)
        print(f"| `{m}` | {len(pairs)} | {wins} | {losses} | {len(pairs) - wins - losses} |")


if __name__ == "__main__":
    main()
