import numpy as np
import pytest

from sca_tg.aes import AES_SBOX, HW
from sca_tg.ascadr import guessing_entropy
from sca_tg.leakage import parse_leakage_model

P = np.arange(256, dtype=np.uint8)
K = np.uint8(0x5A)


def test_id_and_hw_match_classic_labels():
    assert (parse_leakage_model("ID")(P, K) == AES_SBOX[P ^ K]).all()
    assert (parse_leakage_model("HW")(P, K) == HW[AES_SBOX[P ^ K]]).all()
    assert parse_leakage_model("ID").num_classes == 256
    assert parse_leakage_model("HW").num_classes == 9


def test_hw3_buckets_low_med_high():
    lm = parse_leakage_model("HW3")
    hw = HW[AES_SBOX[P ^ K]]
    assert (lm(P, K) == np.where(hw < 4, 0, np.where(hw == 4, 1, 2))).all()
    assert lm.num_classes == 3 and np.bincount(lm(P, K)).tolist() == [93, 70, 93]


def test_bits():
    lm = parse_leakage_model("out:0,1")
    assert lm.num_classes == 4
    assert (lm(P, K) == AES_SBOX[P ^ K] % 4).all()
    hi = parse_leakage_model("out:2,3,4,5,6,7")
    assert hi.num_classes == 64
    assert (hi(P, K) == AES_SBOX[P ^ K] >> 2).all()


def test_joint_in_out_bits_is_paper_Yi():
    # i = concat(bits 1,0 of p^k, bits 1,0 of Sbox[p^k])
    lm = parse_leakage_model("in:0,1+out:0,1")
    assert lm.num_classes == 16
    assert (lm(P, K) == ((P ^ K) % 4) * 4 + AES_SBOX[P ^ K] % 4).all()


def test_key_hypothesis_broadcast():
    lm = parse_leakage_model("out:0,1")
    kh = lm(P[None, :], np.arange(256, dtype=np.uint8)[:, None])
    assert kh.shape == (256, 256)
    assert (kh[K] == lm(P, K)).all()


@pytest.mark.parametrize("bad", ["foo:ID", "out:8", "out:0,0", "out:"])
def test_invalid_specs(bad):
    with pytest.raises(ValueError):
        parse_leakage_model(bad)


def test_ge_ties_do_not_report_false_success():
    # Labels on 2 LSBs of the S-box *input* only depend on 2 key bits -> 64 guesses are indistinguishable.
    lm = parse_leakage_model("in:0,1")
    rng = np.random.default_rng(0)
    pt = rng.integers(0, 256, 500).astype(np.uint8)
    kh = lm(pt[None, :], np.arange(256, dtype=np.uint8)[:, None])
    perfect = np.full((500, 4), -20.0)
    perfect[np.arange(500), kh[K]] = 0.0
    ge = guessing_entropy(perfect, kh, int(K), n_experiments=5, rng=rng)
    assert ge[-1] == pytest.approx(31.5)
