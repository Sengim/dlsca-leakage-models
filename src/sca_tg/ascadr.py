"""Loader for ASCAD-variable (ascad-variable.h5), ported from diff_release without the TF dependency.

Traces and metadata are loaded once; labels are computed on demand for any leakage model
(see sca_tg.leakage), so a sweep over leakage models does not reload the file.
"""

from dataclasses import dataclass

import h5py
import numpy as np


@dataclass
class ASCADr:
    x_profiling: np.ndarray  # (n_profiling, n_samples) float32
    x_attack: np.ndarray     # (n_attack, n_samples) float32
    profiling_plaintexts: np.ndarray
    profiling_keys: np.ndarray
    profiling_masks: np.ndarray
    attack_plaintexts: np.ndarray
    attack_keys: np.ndarray
    attack_masks: np.ndarray
    target_byte: int

    @property
    def correct_key(self):
        # ASCADr uses a fixed key in the attack set.
        return int(self.attack_keys[0, self.target_byte])

    def profiling_labels(self, leakage_model):
        b = self.target_byte
        return leakage_model(self.profiling_plaintexts[:, b], self.profiling_keys[:, b])

    def attack_labels(self, leakage_model):
        b = self.target_byte
        return leakage_model(self.attack_plaintexts[:, b], self.attack_keys[:, b])

    def key_hypothesis_labels(self, leakage_model):
        """(256, n_attack): labels of the attack traces under every key guess."""
        guesses = np.arange(256, dtype=np.uint8)[:, None]
        return leakage_model(self.attack_plaintexts[None, :, self.target_byte], guesses)


def load_ascadr(file_path, n_profiling=200000, n_attack=10000, target_byte=2, first_sample=0, number_of_samples=1400):
    window = slice(first_sample, first_sample + number_of_samples)
    with h5py.File(file_path, "r") as f:
        x_prof = f["Profiling_traces/traces"][:n_profiling, window].astype(np.float32)
        x_att = f["Attack_traces/traces"][:n_attack, window].astype(np.float32)
        meta_prof = f["Profiling_traces/metadata"][:n_profiling]
        meta_att = f["Attack_traces/metadata"][:n_attack]

    return ASCADr(
        x_profiling=x_prof, x_attack=x_att,
        profiling_plaintexts=meta_prof["plaintext"], profiling_keys=meta_prof["key"], profiling_masks=meta_prof["masks"],
        attack_plaintexts=meta_att["plaintext"], attack_keys=meta_att["key"], attack_masks=meta_att["masks"],
        target_byte=target_byte,
    )


def guessing_entropy(log_probs, labels_key_hypothesis, correct_key, n_experiments=100, rng=None):
    """Average rank (0 = best) of the correct key vs. number of attack traces.

    log_probs: (n_attack, num_classes) model log-probabilities on attack traces.
    Returns an array of length n_attack; entry i is the GE after i+1 traces.

    Key guesses that give the correct key's labels on every trace (e.g. with bit leakage models on
    the S-box input, which only depend on some key bits) score identically; ties count as half a
    rank each, so GE then bottoms out at (n_tied - 1) / 2 instead of reporting a false 0.
    """
    rng = np.random.default_rng() if rng is None else rng
    n_attack = log_probs.shape[0]
    if np.isnan(log_probs).any():
        # diverged model: NaN scores would compare as neither better nor tied and fake a rank of -0.5
        return np.full(n_attack, np.nan)
    # (256, n_attack): log-likelihood of each trace under each key guess
    scores = log_probs[np.arange(n_attack)[None, :], labels_key_hypothesis]
    ranks = np.zeros(n_attack)
    for _ in range(n_experiments):
        cum = np.cumsum(scores[:, rng.permutation(n_attack)], axis=1)
        correct = cum[correct_key]
        ranks += (cum > correct).sum(axis=0) + 0.5 * ((cum == correct).sum(axis=0) - 1)
    return ranks / n_experiments


def traces_to_ge(ge, threshold=1.0):
    """Number of attack traces until GE drops below threshold (None if it never does)."""
    below = ge < threshold
    return int(np.argmax(below)) + 1 if below.any() else None
