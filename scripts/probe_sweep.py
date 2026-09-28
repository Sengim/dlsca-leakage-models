"""Linear probes for the shares of the target byte in networks from a sweep (ASCADr for now).

Models: a paired set (the first --n_models runs in which --reference breaks the key, every leakage model trained
on each) and a best set (the --n_models lowest-GE runs per leakage model). The sweeps don't store weights, so each
model is retrained from its saved subset and config (exact for MLPs; the retrained GE is checked against the
sweep's) and cached under <out>/models. Every layer of the trained network and of the same network at its
initialisation (the baseline) is probed on --n_probe profiling traces the model never trained on (a fixed random
draw per run from outside its training subset); the per-trace signal of the unmasked values is small, so a large
probe set matters (10k test traces at the default 40k: ~0.5 points standard error on an accuracy).

Results go to <out>/probes.csv, one row per (leakage model, run, network, layer, variable, target); rows already
there are skipped, so a stopped run resumes.

Example:
    uv run scripts/probe_sweep.py results/sweep_ascadr_27_09_2026_19_17_52
"""

import argparse
import csv
import json
import os

import numpy as np
import torch

from plot_sweep import score
from sca_tg.datasets import load_dataset
from sca_tg.leakage import parse_leakage_model
from sca_tg.probing import layer_outputs, probe_layer, probe_targets, split_indices
from sca_tg.shares import ascadr_shares
from sca_tg.train import TrainConfig, build_model, train_and_evaluate

FIELDS = ["leakage_model", "run", "in_paired", "in_best", "model_ge_final", "network", "layer", "layer_index",
          "n_layers", "group", "variable", "target", "metric", "value"]


def num(v):
    return float(v) if v not in ("", None) else np.nan


def select_models(rows, n_attack, reference, n_models, labels):
    """{(leakage model, run): (in_paired, in_best)}."""
    by = {}
    for r in rows:
        by.setdefault(r["leakage_model"], {})[int(r["run"])] = score(num(r["ge_final"]), num(r["ntge_final"]), n_attack)
    paired_runs = sorted(i for i, s in by[reference].items() if s <= 1)[:n_models]
    sel = {}
    for lm in labels:
        best = sorted(by[lm], key=lambda i: by[lm][i])[:n_models]
        for i in set(paired_runs) | set(best):
            sel[(lm, i)] = (i in paired_runs, i in best)
    return sel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sweep_dir")
    parser.add_argument("--labels", nargs="*", default=None,
                        help="leakage models to probe (default: every one with results for all runs)")
    parser.add_argument("--reference", default="out:0,1", help="leakage model whose successful runs form the paired set")
    parser.add_argument("--n_models", type=int, default=10)
    parser.add_argument("--n_probe", type=int, default=40000, help="probe traces per model (fit / val / test 60/15/25)")
    parser.add_argument("--out", default=None, help="default: results/probes_<sweep dir name>")
    args = parser.parse_args()

    with open(f"{args.sweep_dir}/args.json") as f:
        sweep = json.load(f)
    if sweep.get("dataset", "ascadr") != "ascadr":
        raise SystemExit("share definitions are only implemented for ASCADr so far")
    with open(f"{args.sweep_dir}/summary.csv") as f:
        rows = list(csv.DictReader(f))
    n_runs = len(sweep["runs"])
    counts = {lm: sum(r["leakage_model"] == lm for r in rows) for lm in sweep["leakage_models"]}
    labels = args.labels or [lm for lm, c in counts.items() if c == n_runs]
    print("leakage models:", labels, "| skipped (incomplete):", [lm for lm in counts if lm not in labels])
    sel = select_models(rows, sweep["n_attack"], args.reference, args.n_models, labels)
    ge_sweep = {(r["leakage_model"], int(r["run"])): num(r["ge_final"]) for r in rows}

    out = args.out or f"results/probes_{os.path.basename(os.path.normpath(args.sweep_dir))}"
    os.makedirs(f"{out}/models", exist_ok=True)
    csv_path = f"{out}/probes.csv"
    done = set()
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            done = {(r["leakage_model"], int(r["run"]), r["network"]) for r in csv.DictReader(f)}
    else:
        with open(csv_path, "w", newline="") as f:
            csv.DictWriter(f, FIELDS).writeheader()

    ds = load_dataset("ascadr", sweep["dataset_path"], n_profiling=sweep["n_profiling"], n_attack=sweep["n_attack"],
                      target_byte=sweep["target_byte"])
    val_idx = np.arange(len(ds.x_profiling) - sweep["n_validation"], len(ds.x_profiling))
    splits = split_indices(args.n_probe)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{len(sel)} models, probing on {args.n_probe} unseen traces each -> {csv_path}")

    for k, ((lm_spec, i), (in_paired, in_best)) in enumerate(sorted(sel.items())):
        if {(lm_spec, i, "trained"), (lm_spec, i, "init")} <= done:
            continue
        lm = parse_leakage_model(lm_spec)
        cfg = TrainConfig(**sweep["runs"][i])
        name = lm_spec.replace(":", "").replace(",", "").replace("+", "_")
        path = f"{out}/models/{name}_run{i}.pt"
        train_idx = np.load(f"{args.sweep_dir}/train_idx_run{i}.npy")
        if os.path.exists(path):
            saved = torch.load(path)
        else:
            r = train_and_evaluate(ds, lm, train_idx, val_idx, cfg, log=lambda *_: None)
            saved = {"state_dict": r["state_dict"], "input_mean": r["input_mean"], "input_std": r["input_std"],
                     "ge_final": float(r["ge_final"][-1])}
            torch.save(saved, path)
            if abs(saved["ge_final"] - ge_sweep[(lm_spec, i)]) > 0.5:
                print(f"  WARNING {lm_spec} run {i}: retrained GE {saved['ge_final']:.1f}, sweep {ge_sweep[(lm_spec, i)]}")

        unseen = np.setdiff1d(np.arange(len(ds.x_profiling)), train_idx)
        probe_idx = np.sort(np.random.default_rng(i).choice(unseen, args.n_probe, replace=False))
        groups = ascadr_shares(ds.profiling_plaintexts[probe_idx], ds.profiling_keys[probe_idx],
                               ds.profiling_masks[probe_idx], sweep["target_byte"])
        groups.pop("wrong mask")
        group_of = {v: g for g, vs in groups.items() for v in vs}
        targets = probe_targets({v: x for vs in groups.values() for v, x in vs.items()})
        x = (torch.from_numpy(ds.x_profiling[probe_idx]) - saved["input_mean"]) / saved["input_std"]
        for network in ("trained", "init"):
            if (lm_spec, i, network) in done:
                continue
            torch.manual_seed(cfg.seed)  # the exact initialisation the trained model started from
            model = build_model(cfg, x.shape[1], lm.num_classes)
            if network == "trained":
                model.load_state_dict(saved["state_dict"])
            model.to(device)
            layers = layer_outputs(model, x.to(device))
            out_rows = []
            for j, (layer, acts) in enumerate(layers):
                for var, target, metric, value in probe_layer(acts, targets, splits, device):
                    out_rows.append({"leakage_model": lm_spec, "run": i, "in_paired": in_paired, "in_best": in_best,
                                     "model_ge_final": saved["ge_final"], "network": network, "layer": layer,
                                     "layer_index": j, "n_layers": len(layers), "group": group_of[var],
                                     "variable": var, "target": target, "metric": metric, "value": round(value, 4)})
            with open(csv_path, "a", newline="") as f:
                csv.DictWriter(f, FIELDS).writerows(out_rows)
            del model, layers
            torch.cuda.empty_cache()
        print(f"[{k + 1}/{len(sel)}] {lm_spec} run {i}: GE {saved['ge_final']:.1f}")
    print(f"Done: {csv_path}")


if __name__ == "__main__":
    main()
