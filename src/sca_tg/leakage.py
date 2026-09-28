"""Leakage models as label functions of (plaintext byte, key byte).

A leakage model spec is one or more terms joined by "+". Each term is "<source>:<transform>":
  source:    "in"  -> p ^ k            (S-box input)
             "out" -> Sbox[p ^ k]      (S-box output, default if the source is omitted)
  transform: "ID"  -> the full byte (256 classes)
             "HW"  -> Hamming weight (9 classes)
             "HW3" -> Hamming weight bucketed into low (0-3) / med (4) / high (5-8) (3 classes)
             "0,1" -> the listed bits, bit 0 = LSB (2^len classes)
Multiple terms are combined into one joint label (mixed radix), e.g.

  "ID"                 standard identity model on the S-box output
  "HW"                 Hamming weight of the S-box output
  "HW3"                low / med / high Hamming weight of the S-box output
  "out:0,1"            two LSBs of the S-box output (4 classes)
  "out:2,3,4,5,6,7"    everything except the two LSBs (64 classes)
  "in:0,1+out:0,1"     the 16 classes Y_i from the ASCADr analysis in Karayalcin et al. (NeurIPS 2025)
"""

from dataclasses import dataclass

import numpy as np

from sca_tg.aes import AES_SBOX, HW


@dataclass(frozen=True)
class LeakageModel:
    spec: str
    terms: tuple  # ((source, transform), ...) with transform "ID", "HW" or a tuple of bit indices

    @property
    def num_classes(self):
        n = 1
        for _, transform in self.terms:
            n *= _term_classes(transform)
        return n

    def __call__(self, plaintext, key):
        """Labels for plaintext/key bytes; arrays broadcast (e.g. key guesses (256, 1) vs plaintexts (1, n))."""
        p = np.asarray(plaintext, dtype=np.uint8)
        k = np.asarray(key, dtype=np.uint8)
        labels = 0
        for source, transform in self.terms:
            v = p ^ k if source == "in" else AES_SBOX[p ^ k]
            labels = labels * _term_classes(transform) + _apply(transform, v)
        return np.asarray(labels, dtype=np.int64)

    def __str__(self):
        return self.spec


def _term_classes(transform):
    return {"ID": 256, "HW": 9, "HW3": 3}.get(transform) if isinstance(transform, str) else 2 ** len(transform)


def _apply(transform, v):
    if transform == "ID":
        return v.astype(np.int64)
    if transform == "HW":
        return HW[v]
    if transform == "HW3":
        return np.sign(HW[v].astype(np.int64) - 4) + 1  # 0: HW < 4, 1: HW = 4, 2: HW > 4
    out = np.zeros(v.shape, dtype=np.int64)
    for i, b in enumerate(transform):
        out |= ((v >> b) & 1).astype(np.int64) << i
    return out


def parse_leakage_model(spec):
    terms = []
    for term in spec.split("+"):
        source, _, transform = term.strip().rpartition(":")
        source = source or "out"
        if source not in ("in", "out"):
            raise ValueError(f"unknown source '{source}' in leakage model '{spec}'")
        if transform not in ("ID", "HW", "HW3"):
            bits = tuple(int(b) for b in transform.split(","))
            if not bits or any(b < 0 or b > 7 for b in bits) or len(set(bits)) != len(bits):
                raise ValueError(f"invalid bit list '{transform}' in leakage model '{spec}'")
            transform = bits
        terms.append((source, transform))
    return LeakageModel(spec, tuple(terms))
