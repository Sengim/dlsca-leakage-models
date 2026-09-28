import h5py
import numpy as np
import pytest

from sca_tg.ches_ctf import CHES_CTF_KEY, load_ches_ctf
from sca_tg.datasets import load_dataset
from sca_tg.eshard import ESHARD_KEY, load_eshard


def _write_ascad_format(path, attack_key_hex, n_samples, masks=True, n_prof=50, n_att=20, seed=0):
    rng = np.random.default_rng(seed)
    fields = [("plaintext", np.uint8, (16,)), ("key", np.uint8, (16,))] + ([("masks", np.uint8, (16,))] if masks else [])
    with h5py.File(path, "w") as f:
        for group, n, key in (("Profiling_traces", n_prof, None), ("Attack_traces", n_att, attack_key_hex)):
            meta = np.zeros(n, dtype=fields)
            meta["plaintext"] = rng.integers(0, 256, (n, 16))
            meta["key"] = rng.integers(0, 256, (n, 16)) if key is None else np.frombuffer(bytes.fromhex(key), np.uint8)
            if masks:
                meta["masks"] = rng.integers(0, 256, (n, 16))
            f[f"{group}/traces"] = rng.integers(-128, 128, (n, n_samples)).astype(np.int8)
            f[f"{group}/metadata"] = meta
    return path


def test_eshard_loads_with_masks(tmp_path):
    ds = load_eshard(_write_ascad_format(tmp_path / "eshard.h5", ESHARD_KEY, 1400))
    assert ds.x_profiling.shape == (50, 1400) and ds.x_attack.shape == (20, 1400)
    assert ds.x_profiling.dtype == np.float32 and ds.attack_masks.shape == (20, 16)
    assert ds.correct_key == 0xDA  # byte 2 of 077BDA0F...


def test_ches_ctf_loads_without_masks(tmp_path):
    path = _write_ascad_format(tmp_path / "ches_ctf.h5", CHES_CTF_KEY, 2200, masks=False)
    ds = load_dataset("ches_ctf", path, n_profiling=30, n_attack=None, target_byte=0)
    assert ds.x_profiling.shape == (30, 2200) and ds.x_attack.shape == (20, 2200)
    assert ds.profiling_masks is None and ds.attack_masks is None
    assert ds.correct_key == 0x17


def test_sample_window(tmp_path):
    ds = load_ches_ctf(_write_ascad_format(tmp_path / "c.h5", CHES_CTF_KEY, 2200, masks=False),
                       first_sample=100, number_of_samples=500)
    assert ds.x_attack.shape == (20, 500)


def test_wrong_file_for_dataset_raises(tmp_path):
    with pytest.raises(ValueError, match="expected"):
        load_eshard(_write_ascad_format(tmp_path / "c.h5", CHES_CTF_KEY, 1400, masks=False))


def test_non_fixed_attack_key_raises(tmp_path):
    path = _write_ascad_format(tmp_path / "e.h5", ESHARD_KEY, 1400)
    with h5py.File(path, "r+") as f:
        meta = f["Attack_traces/metadata"][:]
        meta["key"][1, 2] ^= 1
        f["Attack_traces/metadata"][...] = meta
    with pytest.raises(ValueError, match="not fixed"):
        load_eshard(path)
