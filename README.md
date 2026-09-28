# sca_training_groups

Profiling side-channel analysis experiments: how the leakage model used for the training labels, and the choice of training traces, affect attack performance. PyTorch + uv.

Motivation: Karayalcin et al., *Interpreting Emergent Features in Deep Learning-based Side-channel Analysis* (NeurIPS 2025) show that MLPs trained on ASCADr (ID labels) mostly exploit the two LSBs of the S-box input `p^k` and of the S-box output `Sbox[p^k]`. Questions here:
- What happens if we train only on those bits, or only on *other* bits?
- Do bit-restricted labels need fewer profiling traces?

## Setup
```
uv sync
```

## Data
ASCAD-variable (`ascad-variable.h5`, 1400 samples/trace, 200k profiling / 100k attack traces):
https://static.data.gouv.fr/resources/ascad-atmega-8515-variable-key/20190903-083349/ascad-variable.h5

Note: the paper uses a 20k-sample window of the raw traces resampled to 2000 points, not this 1400-sample extracted window.

## Leakage models
Specified as strings (see `src/sca_tg/leakage.py`): terms `<in|out>:<ID|HW|bit list>` joined by `+`.

| spec | labels | classes |
|---|---|---|
| `ID` / `HW` | S-box output byte / its Hamming weight | 256 / 9 |
| `out:0,1` | two LSBs of the S-box output | 4 |
| `out:2,3,4,5,6,7` | all but the two LSBs | 64 |
| `in:0,1+out:0,1` | the paper's 16 classes Y_i | 16 |

Labels that only use S-box *input* bits depend on only those key bits, so many key guesses tie. GE counts ties as half a rank, so it bottoms out above 0 (e.g. 31.5 for `in:0,1`).

## Single run
```
uv run scripts/train_mlp.py --dataset_path /path/to/ascad-variable.h5 --leakage_model "out:0,1" --n_train 20000
```
Training traces: `--n_train` with `--selection random|first`, or `--indices_file idx.npy`. The last `--n_validation` loaded profiling traces are held out for validation.

Excluding variation: `--fix "SPEC=V1,V2"` (both scripts) keeps only training/validation traces whose label under SPEC is one of the values, e.g. `--fix "in:0,1+out:0,1=0"` trains only on traces with `p^k & 3 == 0` and `Sbox[p^k] & 3 == 0`. Joint values are mixed radix (`4*in01 + out01`). The attack set is not filtered. One Y_i value leaves ~1/16 of the traces (~11k of 180k), and the sweep caps `n_train` at that with a warning.

## Sweep (random search at fixed n_train)
```
uv run scripts/sweep.py --dataset_path /path/to/ascad-variable.h5 \
    --leakage_models ID HW "out:0,1" "out:2,3,4,5,6,7" "in:0,1+out:0,1" \
    --n_train 20000 --n_runs 8 --epochs 30 --quiet
```
Each run draws a random `--n_train` subset of the profiling traces and random small-MLP hyperparameters (layers 2-6, width 32-200, ELU/ReLU/SELU, lr 1e-4..3e-3 log-uniform, batch 128-512; see `SEARCH_SPACE` in the script). Run i uses the same subset and hyperparameters for every leakage model, so comparisons are paired. Outputs go to `results/sweep_ascadr_<time>/`: `summary.csv` (hyperparameters, final GE and traces-to-GE<1 for the final and best-validation-loss model), per-run GE/loss curves, and the training indices of each run.

### CNNs (desynchronised traces)
Both scripts take `--model cnn`, e.g. for `ascad-variable-desync50.h5` (same layout, traces shifted by up to 50 samples):
```
uv run scripts/sweep.py --dataset_path /path/to/ascad-variable-desync50.h5 --model cnn --n_runs 8 --epochs 50 --quiet
```
The CNN is 1-4 blocks of Conv1d -> activation -> BatchNorm -> AvgPool (filters doubling per block), then an MLP head. The sweep samples conv blocks, first-block filters 4-32, kernel 5-51, pool 2-10 and a 1-3 layer head (see `CNN_SEARCH_SPACE`); with `train_mlp.py` set them via `--conv_layers --filters --kernel_size --pool_size`.

`train_mlp.py` defaults follow the paper's ASCADr model (6x100, ELU, Adam lr 5e-4, batch 400, 100 epochs).

## Tests
```
uv run --with pytest pytest tests
```
