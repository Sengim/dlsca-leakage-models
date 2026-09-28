"""MLP / CNN training loop shared by the single-run and sweep scripts."""

import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from sca_tg.ascadr import guessing_entropy, traces_to_ge


@dataclass
class TrainConfig:
    # Defaults follow the ASCADr MLP in Karayalcin et al. (NeurIPS 2025), App. C (from Perin et al. 2022).
    epochs: int = 100
    batch_size: int = 400
    lr: float = 5e-4
    hidden: int = 100
    layers: int = 6
    activation: str = "elu"
    seed: int = 0
    ge_experiments: int = 100
    # "mlp", or "cnn" = conv blocks followed by an MLP head (hidden x layers), for desynchronised traces.
    model: str = "mlp"
    conv_layers: int = 2
    filters: int = 8  # doubled after every conv block
    kernel_size: int = 11
    pool_size: int = 2


ACTIVATIONS = {"elu": nn.ELU, "relu": nn.ReLU, "selu": nn.SELU}


def mlp(input_dim, num_classes, hidden, layers, activation="elu"):
    blocks, d = [], input_dim
    for _ in range(layers):
        blocks += [nn.Linear(d, hidden), ACTIVATIONS[activation]()]
        d = hidden
    return nn.Sequential(*blocks, nn.Linear(d, num_classes))


def cnn(input_dim, num_classes, conv_layers, filters, kernel_size, pool_size, hidden, layers, activation="elu"):
    """Conv1d -> activation -> BatchNorm -> AvgPool blocks (filters doubling per block), then an MLP head."""
    blocks, c, n = [nn.Unflatten(1, (1, input_dim))], 1, input_dim
    for i in range(conv_layers):
        out_c = filters * 2 ** i
        blocks += [nn.Conv1d(c, out_c, kernel_size, padding="same"), ACTIVATIONS[activation](),
                   nn.BatchNorm1d(out_c), nn.AvgPool1d(pool_size)]
        c, n = out_c, n // pool_size
    if n < 1:
        raise ValueError(f"{conv_layers} pooling layers of size {pool_size} leave no samples of {input_dim}")
    return nn.Sequential(*blocks, nn.Flatten(), mlp(c * n, num_classes, hidden, layers, activation))


def build_model(cfg, input_dim, num_classes):
    if cfg.model == "mlp":
        return mlp(input_dim, num_classes, cfg.hidden, cfg.layers, cfg.activation)
    if cfg.model == "cnn":
        return cnn(input_dim, num_classes, cfg.conv_layers, cfg.filters, cfg.kernel_size, cfg.pool_size,
                   cfg.hidden, cfg.layers, cfg.activation)
    raise ValueError(f"unknown model {cfg.model!r}")


def _batched(model, x, batch_size=2048):
    return torch.cat([model(x[i:i + batch_size]) for i in range(0, len(x), batch_size)])


def fit_batch_size(model, x, y, loss_fn, batch_size, budget_gib, min_batch_size=16):
    """Halve batch_size until one training step (forward + backward) peaks under budget_gib of GPU memory.

    Some CNN shapes make cuDNN pick a convolution algorithm with a huge workspace (e.g. Conv1d 16->32, kernel 51
    on 140 samples: ~24 MB per trace, 9.5 GiB at batch 400); on Windows/WSL that spills into shared system memory
    and slows training ~10x. The probe runs on a copy of the model, so the real model (BatchNorm statistics,
    gradients) and the RNG are untouched, and configs that fit train exactly as without the probe.
    """
    if x.device.type != "cuda":
        return batch_size
    probe = copy.deepcopy(model).train()
    while batch_size > min_batch_size:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(x.device)
        base = torch.cuda.memory_allocated(x.device)
        try:
            loss_fn(probe(x[:batch_size]), y[:batch_size]).backward()
            fits = torch.cuda.max_memory_allocated(x.device) - base <= budget_gib * 2 ** 30
        except torch.OutOfMemoryError:
            fits = False
        probe.zero_grad(set_to_none=True)
        if fits:
            break
        batch_size //= 2
    del probe
    torch.cuda.empty_cache()  # release the probe's workspace instead of keeping it cached
    return batch_size


def train_and_evaluate(ds, leakage_model, train_idx, val_idx, cfg, device=None, log=print, gpu_mem_budget_gib=4.0):
    """Train an MLP or CNN (cfg.model) on ds.x_profiling[train_idx] labelled with leakage_model, then attack ds.x_attack.

    Returns a dict with the training history, GE curves and traces-to-GE<1 for both the final model
    and the checkpoint with the lowest validation loss, and the state dict of the final model with its
    input standardisation (input_mean, input_std).
    Both GE curves use the same attack-trace permutations. If training diverged (NaN outputs, or no
    finite validation loss for the best checkpoint), the affected GE curve is all NaN.
    On GPU, cfg.batch_size is halved until a training step fits in gpu_mem_budget_gib (see fit_batch_size);
    the batch size actually used is returned as "batch_size".
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)

    y_prof = torch.from_numpy(ds.profiling_labels(leakage_model))
    x_train = torch.from_numpy(ds.x_profiling[train_idx]).to(device)
    mean, std = x_train.mean(0), x_train.std(0).clamp_min(1e-8)

    def prep(x):
        return (torch.as_tensor(x, device=device) - mean) / std

    x_train = (x_train - mean) / std
    y_train = y_prof[train_idx].to(device)
    x_val, y_val = prep(ds.x_profiling[val_idx]), y_prof[val_idx].to(device)
    x_attack = prep(ds.x_attack)

    model = build_model(cfg, x_train.shape[1], leakage_model.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.CrossEntropyLoss()
    batch_size = fit_batch_size(model, x_train, y_train, loss_fn, cfg.batch_size, gpu_mem_budget_gib)
    if batch_size != cfg.batch_size:
        log(f"[{leakage_model}] batch size {cfg.batch_size} needs > {gpu_mem_budget_gib} GiB of GPU memory, "
            f"using {batch_size}")

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val, best_state, best_epoch = float("inf"), None, -1
    g = torch.Generator(device=device).manual_seed(cfg.seed)
    for epoch in range(cfg.epochs):
        model.train()
        perm = torch.randperm(len(x_train), device=device, generator=g)
        total, n_batches = 0.0, 0
        for i in range(0, len(perm), batch_size):
            b = perm[i:i + batch_size]
            optimizer.zero_grad()
            loss = loss_fn(model(x_train[b]), y_train[b])
            loss.backward()
            optimizer.step()
            total += loss.item()
            n_batches += 1

        model.eval()
        with torch.no_grad():
            out = _batched(model, x_val)
            val_loss = loss_fn(out, y_val).item()
            val_acc = (out.argmax(1) == y_val).float().mean().item()
        history["train_loss"].append(total / n_batches)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        if np.isfinite(val_loss) and val_loss < best_val:
            best_val, best_state, best_epoch = val_loss, copy.deepcopy(model.state_dict()), epoch
        log(f"[{leakage_model} n={len(train_idx)}] epoch {epoch}: train_loss={total / n_batches:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

    labels_kh = ds.key_hypothesis_labels(leakage_model)

    def attack():
        model.eval()
        with torch.no_grad():
            log_probs = torch.log_softmax(_batched(model, x_attack), dim=1).cpu().numpy()
        # same seed for every call, so final and best GE are averaged over the same permutations
        return guessing_entropy(log_probs, labels_kh, ds.correct_key, cfg.ge_experiments,
                                np.random.default_rng(cfg.seed))

    ge_final = attack()
    final_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    input_mean, input_std = mean.cpu(), std.cpu()  # needed to feed new traces to the saved model
    if best_state is None:
        ge_best = np.full(len(x_attack), np.nan)
    else:
        model.load_state_dict(best_state)
        ge_best = attack()

    # Free everything this run put on the GPU. Otherwise small tensors of the next run land inside cached
    # blocks (e.g. a ~10 GiB cuDNN workspace) and pin them, and the next large workspace needs a second one.
    del model, optimizer, best_state, x_train, y_train, x_val, y_val, x_attack, mean, std
    if device == "cuda" or str(device).startswith("cuda"):
        torch.cuda.empty_cache()

    return {
        "history": {k: np.array(v) for k, v in history.items()},
        "ge_final": ge_final, "ntge_final": traces_to_ge(ge_final), "ntge32_final": traces_to_ge(ge_final, 32),
        "ge_best": ge_best, "ntge_best": traces_to_ge(ge_best), "ntge32_best": traces_to_ge(ge_best, 32),
        "best_epoch": best_epoch,
        "batch_size": batch_size,
        "state_dict": final_state,
        "input_mean": input_mean, "input_std": input_std,
    }
