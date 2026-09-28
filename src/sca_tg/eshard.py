"""Loader for eShard (extracted eshard.h5 used in Karayalcin et al., NeurIPS 2025), ported from its ReadEshard.

Masked AES software implementation, same ASCAD layout as ASCADr: 1400 samples per trace, labels
Sbox[p ^ k], a fixed key for the attack set, and masks in the metadata. Returns an ASCADr container.
"""

from sca_tg.ascadr import read_ascad_format

ESHARD_KEY = "077BDA0FAB1E5501EAD0150AB1E020FE"


def load_eshard(file_path, n_profiling=None, n_attack=None, target_byte=2, first_sample=0, number_of_samples=1400):
    return read_ascad_format(file_path, n_profiling, n_attack, target_byte, first_sample, number_of_samples,
                             attack_key=ESHARD_KEY)
