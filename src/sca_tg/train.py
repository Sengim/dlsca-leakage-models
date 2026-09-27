"""MLP training loop shared by the single-run and sweep scripts."""

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


ACTIVATIONS = {"elu": nn.ELU, "relu": nn.ReLU, "selu": nn.SELU}


def mlp(input_dim, num_classes, hidden, layers, activation="elu"):
    blocks, d = [], input_dim
    for _ in range(layers):
        blocks += [nn.Linear(d, hidden), ACTIVATIONS[activation]()]
        d = hidden
    return nn.Sequential(*blocks, nn.Linear(d, num_classes))


def _batched(model, x, batch_size=8192):
    return torch.cat([model(x[i:i + batch_size]) for i in range(0, len(x), batch_size)])


def train_and_evaluate(ds, leakage_model, train_idx, val_idx, cfg, device=None, log=print):
    """Train an MLP on ds.x_profiling[train_idx] labelled with leakage_model, then attack ds.x_attack.

    Returns a dict with the training history, GE curves and traces-to-GE<1 for both the final model
    and the checkpoint with the lowest validation loss, and the state dict of the final model.
    Both GE curves use the same attack-trace permutations. If training diverged (NaN outputs, or no
    finite validation loss for the best checkpoint), the affected GE curve is all NaN.
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

    model = mlp(x_train.shape[1], leakage_model.num_classes, cfg.hidden, cfg.layers, cfg.activation).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val, best_state, best_epoch = float("inf"), None, -1
    g = torch.Generator(device=device).manual_seed(cfg.seed)
    for epoch in range(cfg.epochs):
        model.train()
        perm = torch.randperm(len(x_train), device=device, generator=g)
        total, n_batches = 0.0, 0
        for i in range(0, len(perm), cfg.batch_size):
            b = perm[i:i + cfg.batch_size]
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
    final_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        ge_best = np.full(len(x_attack), np.nan)
    else:
        model.load_state_dict(best_state)
        ge_best = attack()

    return {
        "history": {k: np.array(v) for k, v in history.items()},
        "ge_final": ge_final, "ntge_final": traces_to_ge(ge_final), "ntge32_final": traces_to_ge(ge_final, 32),
        "ge_best": ge_best, "ntge_best": traces_to_ge(ge_best), "ntge32_best": traces_to_ge(ge_best, 32),
        "best_epoch": best_epoch,
        "state_dict": final_state,
    }
