"""Linear probes on a network's layers: which shares (and unmasked values) it encodes, layer by layer.

For each variable (a byte per trace, e.g. a mask or masked S-box output) the targets are its 8 bits and whether
its Hamming weight is high or low (logistic probes, accuracy; HW 4 excluded from high/low) and its Hamming
weight (ridge regression, R^2). All targets of a layer are fitted at once on the GPU; the L2 strength is picked
per target on a validation split and the reported score is on a separate test split.
"""

import numpy as np
import torch
from torch import nn

from sca_tg.aes import HW

ACTIVATIONS = (nn.ELU, nn.ReLU, nn.SELU)
LAMBDAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)


@torch.no_grad()
def layer_outputs(model, x, batch_size=2048):
    """[(name, (n, d) float32 CPU tensor)]: every hidden layer and the logits.

    MLP hidden layers are taken after their activation; CNN conv blocks after their AvgPool (flattened). The input
    is left out: within a run it is the same for every leakage model, and the untrained network is the baseline.
    """
    model.eval()
    modules = [m for m in model.modules() if not isinstance(m, nn.Sequential)]
    record = []
    for i, m in enumerate(modules):
        nxt = modules[i + 1] if i + 1 < len(modules) else None
        record.append(isinstance(m, nn.AvgPool1d) or (isinstance(m, ACTIVATIONS) and not isinstance(nxt, nn.BatchNorm1d)))
    record[-1] = True  # logits
    outs = [[] for _ in range(sum(record))]
    for s in range(0, len(x), batch_size):
        h = x[s:s + batch_size]
        k = 0
        for m, rec in zip(modules, record):
            h = m(h)
            if rec:
                outs[k].append(h.flatten(1).cpu())
                k += 1
    names = [f"hidden {j}" for j in range(1, len(outs))] + ["logits"]
    return [(n, torch.cat(o)) for n, o in zip(names, outs)]


def probe_targets(values):
    """values: name -> (n,) uint8. Returns (binary targets (n, T), their masks (n, T), [(variable, target)]) and
    (HW targets (n, V), [(variable, "HW")])."""
    ys, ms, names, hws, hw_names = [], [], [], [], []
    for var, v in values.items():
        v = v.astype(np.int64)
        h = HW[v]
        for b in range(8):
            ys.append((v >> b) & 1)
            ms.append(np.ones(len(v), bool))
            names.append((var, f"bit {b}"))
        ys.append((h > 4).astype(np.int64))
        ms.append(h != 4)
        names.append((var, "high/low"))
        hws.append(h)
        hw_names.append((var, "HW"))
    return np.stack(ys, 1), np.stack(ms, 1), names, np.stack(hws, 1), hw_names


def split_indices(n, seed=0, fractions=(0.6, 0.15)):
    """Fixed random fit / validation / test split of n probe traces."""
    perm = np.random.default_rng(seed).permutation(n)
    a, b = int(fractions[0] * n), int((fractions[0] + fractions[1]) * n)
    return perm[:a], perm[a:b], perm[b:]


def _standardise(x, fit):
    mean, std = x[fit].mean(0), x[fit].std(0)
    keep = std > 1e-6  # dead units (e.g. ReLU) carry nothing
    return (x[:, keep] - mean[keep]) / std[keep]


def _logistic(x, y, m, splits, device, iters=100):
    fit, val, test = splits
    d = x.shape[1]
    xf, yf, mf = x[fit], y[fit], m[fit]
    best_val = torch.full((y.shape[1],), -1.0, device=device)
    best_test = torch.zeros(y.shape[1], device=device)
    for lam in LAMBDAS:
        w = torch.zeros(d, y.shape[1], device=device, requires_grad=True)
        b = torch.zeros(y.shape[1], device=device, requires_grad=True)
        opt = torch.optim.LBFGS([w, b], max_iter=iters, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            bce = nn.functional.binary_cross_entropy_with_logits(xf @ w + b, yf, reduction="none")
            # heads are independent, so one joint objective = the sum of the per-target ones
            loss = (bce * mf).sum(0).div(mf.sum(0)).sum() + lam * w.pow(2).sum()
            loss.backward()
            return loss

        opt.step(closure)
        with torch.no_grad():
            acc = [(((x[s] @ w + b) > 0).float() == y[s]).float().mul(m[s]).sum(0) / m[s].sum(0) for s in (val, test)]
        better = acc[0] > best_val
        best_val = torch.where(better, acc[0], best_val)
        best_test = torch.where(better, acc[1], best_test)
    return best_test.cpu().numpy()


def _ridge(x, y, splits, device):
    fit, val, test = splits
    xf, yf = x[fit], y[fit]
    y_mean = yf.mean(0)
    gram, xty = xf.T @ xf, xf.T @ (yf - y_mean)
    best_val = torch.full((y.shape[1],), -np.inf, device=device)
    best_test = torch.zeros(y.shape[1], device=device)

    def r2(s, w):
        resid = y[s] - (x[s] @ w + y_mean)
        return 1 - resid.pow(2).sum(0) / (y[s] - y[s].mean(0)).pow(2).sum(0)

    for lam in LAMBDAS:
        w = torch.linalg.solve(gram + lam * len(fit) * torch.eye(len(gram), device=device), xty)
        v, t = r2(val, w), r2(test, w)
        better = v > best_val
        best_val = torch.where(better, v, best_val)
        best_test = torch.where(better, t, best_test)
    return best_test.cpu().numpy()


def probe_layer(acts, targets, splits, device="cuda"):
    """Fit all probes on one layer's activations. Returns [(variable, target, metric, test score)]."""
    y, m, names, hw, hw_names = targets
    x = _standardise(acts.to(device, torch.float32), torch.as_tensor(splits[0], device=device))
    splits = [torch.as_tensor(s, device=device) for s in splits]
    acc = _logistic(x, torch.as_tensor(y, dtype=torch.float32, device=device),
                    torch.as_tensor(m, dtype=torch.float32, device=device), splits, device)
    r2 = _ridge(x, torch.as_tensor(hw, dtype=torch.float32, device=device), splits, device)
    return ([(v, t, "accuracy", float(a)) for (v, t), a in zip(names, acc)]
            + [(v, t, "R2", float(r)) for (v, t), r in zip(hw_names, r2)])
