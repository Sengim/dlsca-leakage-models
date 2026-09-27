import numpy as np
import pytest

from sca_tg.aes import AES_SBOX
from sca_tg.ascadr import ASCADr, guessing_entropy
from sca_tg.leakage import parse_leakage_model
from sca_tg.train import TrainConfig, train_and_evaluate


def _leaky_ds(n_prof=3000, n_att=300, seed=0):
    rng = np.random.default_rng(seed)

    def make(n, key):
        pt = rng.integers(0, 256, (n, 16)).astype(np.uint8)
        k = np.full((n, 16), key, np.uint8) if key is not None else rng.integers(0, 256, (n, 16)).astype(np.uint8)
        z = AES_SBOX[pt[:, 2] ^ k[:, 2]]
        x = rng.normal(0, 1, (n, 20)).astype(np.float32)
        x[:, 3] += (z & 1) * 2
        x[:, 7] += ((z >> 1) & 1) * 2
        return x, pt, k, np.zeros((n, 18), np.uint8)

    xp, pp, kp, mp = make(n_prof, None)
    xa, pa, ka, ma = make(n_att, 0x42)
    return ASCADr(xp, xa, pp, kp, mp, pa, ka, ma, target_byte=2)


CFG = dict(batch_size=128, lr=3e-3, hidden=32, layers=2, ge_experiments=10, seed=1)
LOG = lambda *_: None  # noqa: E731


def test_ge_best_is_the_best_val_loss_checkpoint():
    ds, lm = _leaky_ds(), parse_leakage_model("out:0,1")
    train_idx, val_idx = np.arange(2500), np.arange(2500, 3000)
    r = train_and_evaluate(ds, lm, train_idx, val_idx, TrainConfig(epochs=8, **CFG), device="cpu", log=LOG)
    assert r["best_epoch"] == int(np.argmin(r["history"]["val_loss"]))

    # Retraining with the same seed for best_epoch + 1 epochs ends exactly at the best checkpoint.
    r2 = train_and_evaluate(ds, lm, train_idx, val_idx, TrainConfig(epochs=r["best_epoch"] + 1, **CFG),
                            device="cpu", log=LOG)
    np.testing.assert_allclose(r2["history"]["val_loss"][-1], r["history"]["val_loss"][r["best_epoch"]], rtol=1e-5)
    np.testing.assert_allclose(r2["ge_final"], r["ge_best"])
    assert r["ge_best"][-1] < 1  # leaky toy data should be broken


def test_diverged_run_does_not_crash():
    ds = _leaky_ds()
    ds.x_profiling[5, 0] = np.nan  # poisons the standardisation -> NaN everywhere
    r = train_and_evaluate(ds, parse_leakage_model("ID"), np.arange(2500), np.arange(2500, 3000),
                           TrainConfig(epochs=2, **CFG), device="cpu", log=LOG)
    assert np.isnan(r["ge_final"]).all() and np.isnan(r["ge_best"]).all()
    assert r["ntge_final"] is None and r["ntge_best"] is None and r["best_epoch"] == -1


def test_ge_matches_naive_reference():
    rng = np.random.default_rng(0)
    lm = parse_leakage_model("HW")
    pt = rng.integers(0, 256, 50).astype(np.uint8)
    kh = lm(pt[None, :], np.arange(256, dtype=np.uint8)[:, None])
    log_probs = np.log(rng.dirichlet(np.ones(9), 50))
    ge = guessing_entropy(log_probs, kh, 7, n_experiments=3, rng=np.random.default_rng(5))

    perm_rng, ref = np.random.default_rng(5), np.zeros(50)
    for _ in range(3):
        perm = perm_rng.permutation(50)
        for n in range(1, 51):
            s = [sum(log_probs[t, kh[g, t]] for t in perm[:n]) for g in range(256)]
            ref[n - 1] += sum(v > s[7] for v in s) + 0.5 * (sum(v == s[7] for v in s) - 1)
    np.testing.assert_allclose(ge, ref / 3)


def test_nan_scores_give_nan_ge():
    kh = np.zeros((256, 4), dtype=np.int64)
    assert np.isnan(guessing_entropy(np.full((4, 9), np.nan), kh, 0, n_experiments=2)).all()
