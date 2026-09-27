"""Choosing which profiling traces are used for training."""

import numpy as np

from sca_tg.leakage import parse_leakage_model


def select_training_traces(candidates, n_train=None, strategy="random", rng=None, indices_file=None):
    """Return indices (into the profiling set) of the traces to train on.

    candidates:   indices eligible for training (i.e. not the validation split)
    n_train:      how many to use (None = all candidates)
    strategy:     "first"  -> the first n_train candidates
                  "random" -> a uniform random subset of size n_train
    indices_file: .npy file with explicit indices; overrides strategy, must be a subset of candidates
    """
    candidates = np.asarray(candidates)
    if indices_file is not None:
        idx = np.load(indices_file)
        if not np.isin(idx, candidates).all():
            raise ValueError(f"{indices_file} contains indices outside the training candidates")
        return idx if n_train is None else idx[:n_train]

    n_train = len(candidates) if n_train is None else n_train
    if n_train > len(candidates):
        raise ValueError(f"n_train={n_train} but only {len(candidates)} training candidates")
    if strategy == "first":
        return candidates[:n_train]
    if strategy == "random":
        rng = np.random.default_rng() if rng is None else rng
        return np.sort(rng.choice(candidates, size=n_train, replace=False))
    raise ValueError(f"unknown selection strategy '{strategy}'")


def parse_fix(fix):
    """Parse "SPEC=V1,V2,..." into (LeakageModel, [values]), e.g. "in:0,1+out:0,1=0"."""
    spec, sep, values = fix.rpartition("=")
    if not sep or not spec:
        raise ValueError(f"--fix must look like SPEC=V1,V2 (got '{fix}')")
    lm = parse_leakage_model(spec)
    vals = [int(v) for v in values.split(",")]
    if any(v < 0 or v >= lm.num_classes for v in vals):
        raise ValueError(f"values in '{fix}' must be in [0, {lm.num_classes})")
    return lm, vals


def restrict_to_fixed(ds, indices, fix):
    """Keep only the profiling indices whose label under the --fix spec is one of the given values.

    This removes the variation in those bits from the training data. For joint specs the value is the
    mixed-radix label, e.g. for "in:0,1+out:0,1" value = 4 * (p^k & 3) + (Sbox[p^k] & 3).
    """
    if fix is None:
        return indices
    lm, vals = parse_fix(fix)
    return indices[np.isin(ds.profiling_labels(lm)[indices], vals)]
