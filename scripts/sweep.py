"""Random-search sweep over leakage models on ASCAD-variable with a fixed number of training traces.

Each run i draws a random subset of --n_train profiling traces and a random (small, fast) set of
hyperparameters. Run i uses the same subset and hyperparameters for every leakage model, so the
leakage models are compared on identical training data and configurations.

--model cnn samples small CNNs instead (CNN_SEARCH_SPACE), e.g. for ascad-variable-desync50.h5.

Example:
    uv run scripts/sweep.py --dataset_path /path/to/ascad-variable.h5 \
        --leakage_models ID HW "out:0,1" "out:2,3,4,5,6,7" "in:0,1+out:0,1" --n_runs 8 --quiet
    uv run scripts/sweep.py --dataset_path /path/to/ascad-variable-desync50.h5 --model cnn --n_runs 8 --quiet
    uv run scripts/sweep.py --dataset eshard --dataset_path /path/to/eshard.h5 --n_runs 8 --quiet

--extend <sweep_dir> trains extra leakage models on an existing sweep's saved subsets and per-run configs
(everything else is read from its args.json) and appends to its summary.csv. (leakage model, run) pairs
already in summary.csv are skipped, so it also resumes an interrupted sweep. Don't extend a running sweep.
    uv run scripts/sweep.py --extend results/sweep_ascadr_27_09_2026_19_17_52 --leakage_models "out:4,5" --quiet
"""

import argparse
import csv
import json
import os
from dataclasses import asdict
from datetime import datetime

import numpy as np

from sca_tg.datasets import DATASETS, default_path, load_dataset
from sca_tg.leakage import parse_leakage_model
from sca_tg.selection import parse_fix, restrict_to_fixed
from sca_tg.train import TrainConfig, train_and_evaluate

DEFAULT_LEAKAGE_MODELS = ["ID", "HW", "out:0,1", "in:0,1", "out:2,3,4,5,6,7", "in:0,1+out:0,1"]

# Search space: small MLPs so a run takes seconds to a few minutes.
SEARCH_SPACE = {
    "layers": [2, 3, 4, 5, 6],
    "hidden": [32, 64, 100, 128, 200],
    "batch_size": [128, 256, 400, 512],
    "activation": ["elu", "relu", "selu"],
    "lr": (1e-4, 3e-3),  # log-uniform
}

# CNN: conv blocks (conv -> act -> BN -> avg pool, filters doubling per block) + a small MLP head.
CNN_SEARCH_SPACE = {
    "conv_layers": [1, 2, 3, 4],
    "filters": [4, 8, 16, 32],
    "kernel_size": [5, 11, 21, 31, 41, 51],
    "pool_size": [2, 4, 6, 8, 10],
    "layers": [1, 2, 3],
    "hidden": [32, 64, 100, 128, 200],
    "batch_size": [128, 256, 400, 512],
    "activation": ["elu", "relu", "selu"],
    "lr": (1e-4, 3e-3),  # log-uniform
}


def sample_config(rng, epochs, seed, model="mlp"):
    if model == "cnn":
        return sample_cnn_config(rng, epochs, seed)
    lo, hi = SEARCH_SPACE["lr"]
    return TrainConfig(
        epochs=epochs,
        layers=int(rng.choice(SEARCH_SPACE["layers"])),
        hidden=int(rng.choice(SEARCH_SPACE["hidden"])),
        batch_size=int(rng.choice(SEARCH_SPACE["batch_size"])),
        activation=str(rng.choice(SEARCH_SPACE["activation"])),
        lr=float(np.exp(rng.uniform(np.log(lo), np.log(hi)))),
        seed=seed,
    )


def sample_cnn_config(rng, epochs, seed, input_dim=1400):
    sp = CNN_SEARCH_SPACE
    conv_layers = int(rng.choice(sp["conv_layers"]))
    # pooling must leave at least one sample after the last block
    pool_sizes = [p for p in sp["pool_size"] if input_dim // p ** conv_layers >= 1]
    lo, hi = sp["lr"]
    return TrainConfig(
        model="cnn",
        epochs=epochs,
        conv_layers=conv_layers,
        filters=int(rng.choice(sp["filters"])),
        kernel_size=int(rng.choice(sp["kernel_size"])),
        pool_size=int(rng.choice(pool_sizes)),
        layers=int(rng.choice(sp["layers"])),
        hidden=int(rng.choice(sp["hidden"])),
        batch_size=int(rng.choice(sp["batch_size"])),
        activation=str(rng.choice(sp["activation"])),
        lr=float(np.exp(rng.uniform(np.log(lo), np.log(hi)))),
        seed=seed,
    )


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extend", default=None, metavar="SWEEP_DIR",
                        help="add --leakage_models to an existing sweep (other options are read from its args.json)")
    parser.add_argument("--dataset", default="ascadr", choices=list(DATASETS))
    parser.add_argument("--dataset_path", default=None,
                        help="default: see sca_tg.datasets ($ASCADR_PATH or ascad-variable.h5 for ascadr); "
                             "with --extend, the sweep's own dataset_path")
    parser.add_argument("--results_root_path", default="results")
    parser.add_argument("--n_profiling", type=int, default=190000)
    parser.add_argument("--n_attack", type=int, default=10000)
    parser.add_argument("--n_validation", type=int, default=10000)
    parser.add_argument("--target_byte", type=int, default=2)
    parser.add_argument("--leakage_models", nargs="+", default=None,
                        help=f"default: {' '.join(DEFAULT_LEAKAGE_MODELS)}; required with --extend")
    parser.add_argument("--n_train", type=int, default=20000)
    parser.add_argument("--model", default="mlp", choices=["mlp", "cnn"])
    parser.add_argument("--n_runs", type=int, default=8, help="random (subset, hyperparameter) draws per leakage model")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0, help="seed for drawing subsets and hyperparameters")
    parser.add_argument("--fix", default=None,
                        help='only train/validate on traces with these label values, e.g. "in:0,1+out:0,1=0"; '
                             "n_train is capped at the number of matching traces")
    parser.add_argument("--quiet", action="store_true", help="don't print per-epoch lines")
    args = parser.parse_args()
    if args.extend and not args.leakage_models:
        parser.error("--extend needs --leakage_models")
    return args


def summary_fields(model):
    cnn_fields = ["conv_layers", "filters", "kernel_size", "pool_size"] if model == "cnn" else []
    return ["leakage_model", "num_classes", "fix", "n_train", "run", "model", *cnn_fields,
            "layers", "hidden", "activation", "lr", "batch_size",
            "ge_final", "ntge_final", "ntge32_final", "ge_best", "ntge_best", "ntge32_best", "best_epoch", "final_val_acc"]


def load_validation(args):
    ds = load_dataset(args.dataset, args.dataset_path, n_profiling=args.n_profiling, n_attack=args.n_attack,
                      target_byte=args.target_byte)
    n_candidates = len(ds.x_profiling) - args.n_validation
    val_idx = restrict_to_fixed(ds, np.arange(n_candidates, len(ds.x_profiling)), args.fix)
    return ds, val_idx, n_candidates


def new_sweep(args):
    """Create a sweep dir and draw its runs. Returns (sweep_dir, ds, val_idx, runs, fields, done)."""
    args.dataset_path = args.dataset_path or default_path(args.dataset)
    args.leakage_models = args.leakage_models or DEFAULT_LEAKAGE_MODELS
    ds, val_idx, n_candidates = load_validation(args)  # before creating the sweep dir, so a bad path leaves nothing behind
    # record what was actually loaded (smaller datasets, e.g. CHES CTF, have fewer traces than the defaults)
    for k, x in (("n_profiling", ds.x_profiling), ("n_attack", ds.x_attack)):
        if len(x) < getattr(args, k):
            print(f"{args.dataset} has only {len(x)} traces for {k}={getattr(args, k)}; using {len(x)}")
            setattr(args, k, len(x))
    candidates = restrict_to_fixed(ds, np.arange(n_candidates), args.fix)
    n_train = min(args.n_train, len(candidates))
    if args.fix:
        print(f"--fix {args.fix}: {len(candidates)} training candidates, {len(val_idx)} validation traces")
    if n_train < args.n_train:
        print(f"WARNING: only {len(candidates)} training candidates, using n_train={n_train} instead of {args.n_train}")

    now = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    sweep_dir = f"{args.results_root_path}/sweep_{args.dataset}_{now}"
    os.makedirs(sweep_dir, exist_ok=True)

    # Draw all runs up front so they are identical across leakage models.
    rng = np.random.default_rng(args.seed)
    runs = []
    for i in range(args.n_runs):
        train_idx = np.sort(rng.choice(candidates, size=n_train, replace=False))
        runs.append((i, train_idx, sample_config(rng, args.epochs, seed=args.seed * 1000 + i, model=args.model)))
        np.save(f"{sweep_dir}/train_idx_run{i}.npy", train_idx)
    with open(f"{sweep_dir}/args.json", "w") as f:
        search_space = CNN_SEARCH_SPACE if args.model == "cnn" else SEARCH_SPACE
        json.dump({**vars(args), "search_space": search_space, "runs": [asdict(c) for _, _, c in runs]}, f, indent=2)

    fields = summary_fields(args.model)
    with open(f"{sweep_dir}/summary.csv", "w", newline="") as f:
        csv.DictWriter(f, fields).writeheader()
    return sweep_dir, ds, val_idx, runs, fields, set()


def extend_sweep(args):
    """Reload an existing sweep's settings and runs. Returns (sweep_dir, ds, val_idx, runs, fields, done)."""
    sweep_dir = args.extend
    with open(f"{sweep_dir}/args.json") as f:
        saved = json.load(f)
    new_lms = args.leakage_models
    dataset_path = args.dataset_path
    for k in ("dataset_path", "n_profiling", "n_attack", "n_validation", "target_byte", "fix", "seed"):
        setattr(args, k, saved[k])
    args.dataset_path = dataset_path or args.dataset_path
    args.model = saved.get("model", "mlp")  # sweeps from before --model are MLP sweeps
    args.dataset = saved.get("dataset", "ascadr")  # ... and ASCADr sweeps

    runs = [(i, np.load(f"{sweep_dir}/train_idx_run{i}.npy"), TrainConfig(**c)) for i, c in enumerate(saved["runs"])]
    with open(f"{sweep_dir}/summary.csv", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames  # older sweeps have fewer columns; keep the file's own header
        done = {(r["leakage_model"], int(r["run"])) for r in reader}

    saved["leakage_models"] += [lm for lm in new_lms if lm not in saved["leakage_models"]]
    with open(f"{sweep_dir}/args.json", "w") as f:
        json.dump(saved, f, indent=2)

    ds, val_idx, _ = load_validation(args)
    todo = sum((lm, i) not in done for lm in new_lms for i, _, _ in runs)
    print(f"Extending {sweep_dir} ({args.model}, {args.dataset} at {args.dataset_path}): {todo} runs to train, "
          f"{len(new_lms) * len(runs) - todo} already in summary.csv")
    return sweep_dir, ds, val_idx, runs, fields, done


def main():
    args = get_arguments()
    lms = [parse_leakage_model(s) for s in args.leakage_models or DEFAULT_LEAKAGE_MODELS]  # fail before loading
    if args.fix:
        parse_fix(args.fix)

    sweep_dir, ds, val_idx, runs, fields, done = (extend_sweep if args.extend else new_sweep)(args)
    summary_path = f"{sweep_dir}/summary.csv"

    log = (lambda *_: None) if args.quiet else print
    for i, train_idx, cfg in runs:
        for lm in lms:
            if (lm.spec, i) in done:
                continue
            r = train_and_evaluate(ds, lm, train_idx, val_idx, cfg, log=log)

            name = f"{lm.spec.replace(':', '').replace(',', '').replace('+', '_')}_run{i}"
            np.savez(f"{sweep_dir}/{name}.npz", ge_final=r["ge_final"], ge_best=r["ge_best"], **r["history"])
            row = {"leakage_model": lm.spec, "num_classes": lm.num_classes, "fix": args.fix or "", "n_train": len(train_idx),
                   "run": i, "model": cfg.model,
                   **{k: getattr(cfg, k) for k in ("conv_layers", "filters", "kernel_size", "pool_size")},
                   "layers": cfg.layers, "hidden": cfg.hidden, "activation": cfg.activation,
                   "lr": f"{cfg.lr:.2e}", "batch_size": cfg.batch_size,
                   "ge_final": round(float(r["ge_final"][-1]), 2), "ntge_final": r["ntge_final"],
                   "ntge32_final": r["ntge32_final"],
                   "ge_best": round(float(r["ge_best"][-1]), 2), "ntge_best": r["ntge_best"],
                   "ntge32_best": r["ntge32_best"],
                   "best_epoch": r["best_epoch"], "final_val_acc": round(float(r["history"]["val_acc"][-1]), 4)}
            with open(summary_path, "a", newline="") as f:
                csv.DictWriter(f, fields, extrasaction="ignore").writerow(row)
            print(", ".join(f"{k}={row[k]}" for k in fields))

    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
