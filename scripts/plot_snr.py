"""SNR of every share of the target byte on ASCADr, to check the mask indices and where each share leaks.

One panel per variable, all on the same log y scale with the noise floor dashed: the masks and masked values
the implementation processes should show clear peaks; the unmasked values and mismatched mask pairs are
controls and should stay at the noise floor (a swapped r_in / r_out index would move peaks from the shares to
the "wrong mask" row).

Example:
    uv run scripts/plot_snr.py --dataset_path ascad-variable.h5 --n_traces 100000
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from plot_sweep import GRID, INK, INK_MUTED, style
from sca_tg.ascadr import load_ascadr
from sca_tg.shares import ascadr_shares, snr

GROUP_COLORS = {"masks": "#2a78d6", "masked values": "#2a78d6", "unmasked": INK_MUTED, "wrong mask": INK_MUTED}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default=os.environ.get("ASCADR_PATH", "ascad-variable.h5"))
    parser.add_argument("--n_traces", type=int, default=100000, help="profiling traces used for the SNR")
    parser.add_argument("--target_byte", type=int, default=2)
    parser.add_argument("--out", default="results/snr_ascadr", help="output path without extension")
    args = parser.parse_args()

    ds = load_ascadr(args.dataset_path, n_profiling=args.n_traces, n_attack=1, target_byte=args.target_byte)
    groups = ascadr_shares(ds.profiling_plaintexts, ds.profiling_keys, ds.profiling_masks, args.target_byte)
    curves = {(g, name): snr(ds.x_profiling, v) for g, vs in groups.items() for name, v in vs.items()}

    print(f"{'variable':22s} {'group':11s} {'max SNR':>8s} {'at sample':>9s}")
    for (g, name), c in curves.items():
        print(f"{name:22s} {g:11s} {c.max():8.4f} {int(c.argmax()):9d}")
    # Which raw mask byte leaks where: a direct check of the index layout.
    print("\nmax SNR of each raw mask byte:", " ".join(f"{j}:{snr(ds.x_profiling, ds.profiling_masks[:, j]).max():.3f}"
                                                    for j in range(ds.profiling_masks.shape[1])))

    ncols = 3
    rows = [list(vs) for vs in groups.values()]
    fig, axes = plt.subplots(len(rows), ncols, figsize=(4.2 * ncols, 2.3 * len(rows)), sharex=True, sharey=True)
    ymax = max(c.max() for c in curves.values()) * 2
    # SNR of a non-leaking variable is ~ 1 / (traces per class); draw it as the reference level
    floor = 256 / len(ds.x_profiling)
    for r, (g, names) in enumerate(zip(groups, rows)):
        for c_idx in range(ncols):
            ax = axes[r, c_idx]
            if c_idx >= len(names):
                ax.set_visible(False)
                continue
            curve = curves[(g, names[c_idx])]
            ax.plot(curve, color=GROUP_COLORS[g], lw=1)
            ax.axhline(floor, color=INK_MUTED, lw=1, ls="--")
            ax.set_title(f"{names[c_idx]}  (max {curve.max():.3f})", fontsize=9.5, color=INK, loc="left")
            ax.set_yscale("log")
            ax.set_ylim(floor / 20, ymax)
            style(ax)
            ax.xaxis.grid(True, color=GRID, lw=0.8)
        axes[r, 0].set_ylabel(f"{g}\nSNR", color=INK)
    for ax in axes[-1]:
        ax.set_xlabel("sample", color=INK)
    fig.suptitle(f"ASCADr ({os.path.basename(args.dataset_path)}), target byte {args.target_byte}: SNR on "
                 f"{len(ds.x_profiling)} profiling traces (dashed: noise floor 256 / n)", color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}", dpi=200, bbox_inches="tight", facecolor="#fcfcfb")
    print(f"Saved {args.out}.png and {args.out}.pdf")


if __name__ == "__main__":
    main()
