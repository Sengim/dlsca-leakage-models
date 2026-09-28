"""Masking shares of the target byte and SNR, for checking leakage in the traces (and later for probing).

ASCADr metadata masks are 18 bytes per trace: r[0..15] (Boolean masks of the state bytes), r_in = masks[16]
(S-box input mask) and r_out = masks[17] (S-box output mask), following ANSSI's ASCAD v1 implementation and
the share definitions in Karayalcin et al. (NeurIPS 2025).
"""

import numpy as np

from sca_tg.aes import AES_SBOX

R_IN, R_OUT = 16, 17


def ascadr_shares(plaintexts, keys, masks, target_byte):
    """name -> (n,) uint8 values of every share of the target byte, plus unmasked values and mismatched pairs.

    "masks" and "masked values" are the shares the masked AES actually processes. "unmasked" and "wrong mask"
    should show no first-order leakage; they are controls for the mask indices.
    """
    b = target_byte
    x = plaintexts[:, b] ^ keys[:, b]
    z = AES_SBOX[x]
    r, r_in, r_out = masks[:, b], masks[:, R_IN], masks[:, R_OUT]
    return {
        "masks": {f"r[{b}]": r, "r_in": r_in, "r_out": r_out},
        "masked values": {
            f"Sbox(p^k) ^ r[{b}]": z ^ r,
            "Sbox(p^k) ^ r_out": z ^ r_out,
            "p^k ^ r_in": x ^ r_in,
        },
        "unmasked": {"Sbox(p^k)": z, "p^k": x},
        "wrong mask": {"Sbox(p^k) ^ r_in": z ^ r_in, "p^k ^ r_out": x ^ r_out},
    }


def snr(traces, labels):
    """Per-sample SNR Var_c(E[X | c]) / E_c(Var[X | c]) over the classes present in labels."""
    order = np.argsort(labels, kind="stable")
    sorted_labels = labels[order]
    _, starts, counts = np.unique(sorted_labels, return_index=True, return_counts=True)
    x = traces[order]
    sums = np.add.reduceat(x, starts, axis=0, dtype=np.float64)
    sq = np.add.reduceat(np.square(x, dtype=np.float64), starts, axis=0)
    means = sums / counts[:, None]
    variances = sq / counts[:, None] - means ** 2
    return means.var(axis=0) / variances.mean(axis=0)
