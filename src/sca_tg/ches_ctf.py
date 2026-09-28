"""Loader for CHES CTF 2018 (extracted ches_ctf.h5), ported from the energy repo's ReadCHESCTF.

Same ASCAD layout as ASCADr but without masks in the metadata (attack_masks / profiling_masks are None):
2200 samples per trace, labels Sbox[p ^ k], a fixed key for the attack set. Returns an ASCADr container.
"""

from sca_tg.ascadr import read_ascad_format

CHES_CTF_KEY = "175cf2997a8583413c77dfac7e6c59d8"


def load_ches_ctf(file_path, n_profiling=None, n_attack=None, target_byte=2, first_sample=0, number_of_samples=2200):
    return read_ascad_format(file_path, n_profiling, n_attack, target_byte, first_sample, number_of_samples,
                             attack_key=CHES_CTF_KEY)
