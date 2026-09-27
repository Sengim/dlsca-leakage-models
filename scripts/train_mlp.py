"""Train a single MLP on ASCAD-variable and report guessing entropy.

Example:
    uv run scripts/train_mlp.py --dataset_path /path/to/ascad-variable.h5 --leakage_model "out:0,1" --n_train 50000
"""

import argparse
import json
import os
from dataclasses import asdict
from datetime import datetime

import numpy as np
import torch

from sca_tg.ascadr import load_ascadr
from sca_tg.leakage import parse_leakage_model
from sca_tg.selection import restrict_to_fixed, select_training_traces
from sca_tg.train import TrainConfig, train_and_evaluate


def get_arguments():
    d = TrainConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default=os.environ.get("ASCADR_PATH", "ascad-variable.h5"))
    parser.add_argument("--results_root_path", default="results")
    parser.add_argument("--n_profiling", type=int, default=190000, help="profiling traces loaded (train candidates + validation)")
    parser.add_argument("--n_attack", type=int, default=10000)
    parser.add_argument("--n_validation", type=int, default=10000, help="taken from the end of the loaded profiling set")
    parser.add_argument("--target_byte", type=int, default=2)
    parser.add_argument("--leakage_model", default="ID", help='see sca_tg/leakage.py, e.g. "ID", "HW", "out:0,1", "in:0,1+out:0,1"')
    parser.add_argument("--n_train", type=int, default=None, help="number of training traces (default: all candidates)")
    parser.add_argument("--selection", default="random", choices=["random", "first"])
    parser.add_argument("--indices_file", default=None, help=".npy with explicit training indices")
    parser.add_argument("--fix", default=None,
                        help='only train/validate on traces with these label values, e.g. "in:0,1+out:0,1=0" or "out:0,1=0,3"')
    parser.add_argument("--epochs", type=int, default=d.epochs)
    parser.add_argument("--batch_size", type=int, default=d.batch_size)
    parser.add_argument("--lr", type=float, default=d.lr)
    parser.add_argument("--hidden", type=int, default=d.hidden)
    parser.add_argument("--layers", type=int, default=d.layers)
    parser.add_argument("--seed", type=int, default=d.seed)
    return parser.parse_args()


def main():
    args = get_arguments()
    lm = parse_leakage_model(args.leakage_model)
    cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, hidden=args.hidden,
                      layers=args.layers, seed=args.seed)

    ds = load_ascadr(args.dataset_path, n_profiling=args.n_profiling, n_attack=args.n_attack, target_byte=args.target_byte)
    n_candidates = len(ds.x_profiling) - args.n_validation
    val_idx = restrict_to_fixed(ds, np.arange(n_candidates, len(ds.x_profiling)), args.fix)
    candidates = restrict_to_fixed(ds, np.arange(n_candidates), args.fix)
    if args.fix:
        print(f"--fix {args.fix}: {len(candidates)} training candidates, {len(val_idx)} validation traces")
    train_idx = select_training_traces(candidates, args.n_train, args.selection,
                                       np.random.default_rng(args.seed), args.indices_file)

    result = train_and_evaluate(ds, lm, train_idx, val_idx, cfg)
    print(f"Final model: GE={result['ge_final'][-1]:.2f}, traces to GE<1: {result['ntge_final']}, to GE<32: {result['ntge32_final']}")
    print(f"Best val-loss model (epoch {result['best_epoch']}): GE={result['ge_best'][-1]:.2f}, "
          f"traces to GE<1: {result['ntge_best']}, to GE<32: {result['ntge32_best']}")

    now = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    safe_lm = args.leakage_model.replace(":", "").replace(",", "").replace("+", "_")
    result_dir = f"{args.results_root_path}/mlp_ascadr_{safe_lm}_{now}"
    os.makedirs(result_dir, exist_ok=True)
    with open(f"{result_dir}/args.json", "w") as f:
        json.dump({**vars(args), "train_config": asdict(cfg)}, f, indent=2)
    torch.save(result["state_dict"], f"{result_dir}/model.pt")
    np.savez(f"{result_dir}/metrics.npz", train_idx=train_idx, ge_final=result["ge_final"], ge_best=result["ge_best"],
             **result["history"])
    print(f"Saved results to {result_dir}")


if __name__ == "__main__":
    main()
