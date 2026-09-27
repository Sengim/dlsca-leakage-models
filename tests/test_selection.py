import numpy as np
import pytest

from sca_tg.aes import AES_SBOX
from sca_tg.ascadr import ASCADr
from sca_tg.selection import parse_fix, restrict_to_fixed


def _fake_ds(n=4096, seed=0):
    rng = np.random.default_rng(seed)
    pt = rng.integers(0, 256, (n, 16)).astype(np.uint8)
    k = rng.integers(0, 256, (n, 16)).astype(np.uint8)
    z = np.zeros((n, 18), np.uint8)
    return ASCADr(np.zeros((n, 1), np.float32), np.zeros((1, 1), np.float32), pt, k, z, pt[:1], k[:1], z[:1], target_byte=2)


def test_fix_removes_variation_in_bits():
    ds = _fake_ds()
    idx = restrict_to_fixed(ds, np.arange(4096), "in:0,1+out:0,1=6")  # in01 = 1, out01 = 2
    x = ds.profiling_plaintexts[idx, 2] ^ ds.profiling_keys[idx, 2]
    assert len(idx) > 0
    assert (x & 3 == 1).all() and (AES_SBOX[x] & 3 == 2).all()
    assert 4096 / 16 * 0.7 < len(idx) < 4096 / 16 * 1.3


def test_fix_multiple_values_and_none():
    ds = _fake_ds()
    idx = restrict_to_fixed(ds, np.arange(4096), "out:0,1=0,3")
    assert set(np.unique(AES_SBOX[ds.profiling_plaintexts[idx, 2] ^ ds.profiling_keys[idx, 2]] & 3)) == {0, 3}
    assert len(restrict_to_fixed(ds, np.arange(10), None)) == 10


@pytest.mark.parametrize("bad", ["out:0,1", "=1", "out:0,1=4"])
def test_invalid_fix(bad):
    with pytest.raises(ValueError):
        parse_fix(bad)
