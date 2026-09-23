"""Versioned, stateless Philox4x32-10 streams, indexed by global trial ID.

The same integer algorithm is implemented in kernels.cu. Random decisions do
not depend on batch size, rank, candidate enumeration, or checkpoint boundaries.
Counter words: cell/index, step low, step high, (rule << 8) | domain.
"""
from __future__ import annotations

import hashlib
import struct

import numpy as np

RNG_VERSION = "philox4x32-10-v1"
GLOBAL, RULE, ORDER, EVENT = 1, 2, 3, 4
MASK = np.uint64(0xFFFFFFFF)


def trial_keys(seed: int, trial_ids) -> np.ndarray:
    if not 0 <= seed < 2**64:
        raise ValueError("Independent RNG seed must be an unsigned 64-bit integer")
    keys = []
    for trial in trial_ids:
        if not 0 <= int(trial) < 2**64:
            raise ValueError("Trial IDs must be unsigned 64-bit integers")
        digest = hashlib.blake2b(struct.pack("<QQ", seed, int(trial)),
                                 digest_size=8, person=b"PyBCA-RNG-v1").digest()
        keys.append(struct.unpack("<II", digest))
    return np.asarray(keys, dtype=np.uint32)


def philox_u32(keys, index, step: int, rule=0, domain=GLOBAL):
    """Return the first word of Philox; integer operations match CUDA exactly."""
    keys = np.asarray(keys, dtype=np.uint64)
    index, rule, k0, k1 = np.broadcast_arrays(
        np.asarray(index, dtype=np.uint64), np.asarray(rule, dtype=np.uint64),
        keys[..., 0], keys[..., 1])
    c0 = index & MASK
    c1 = np.full_like(c0, step & 0xFFFFFFFF)
    c2 = np.full_like(c0, step >> 32)
    c3 = (rule << np.uint64(8)) | np.uint64(domain)
    for _ in range(10):
        p0 = c0 * np.uint64(0xD2511F53)
        p1 = c2 * np.uint64(0xCD9E8D57)
        c0, c1, c2, c3 = ((p1 >> np.uint64(32)) ^ c1 ^ k0,
                          p1 & MASK,
                          (p0 >> np.uint64(32)) ^ c3 ^ k1,
                          p0 & MASK)
        k0 = (k0 + np.uint64(0x9E3779B9)) & MASK
        k1 = (k1 + np.uint64(0xBB67AE85)) & MASK
    return c0.astype(np.uint32)


def uniform(keys, index, step: int, rule=0, domain=GLOBAL):
    # Exactly representable float32 in [0, 1), including on V100.
    return (philox_u32(keys, index, step, rule, domain) >> np.uint32(8)).astype(np.float32) * np.float32(2**-24)


def permutations(keys, n: int, step: int) -> np.ndarray:
    """Unbiased Fisher-Yates; rejection avoids modulo bias."""
    t = len(keys)
    result = np.tile(np.arange(n, dtype=np.int32), (t, 1))
    draws = philox_u32(keys[:, None, :], np.arange(n)[None], step, domain=ORDER)
    for trial in range(t):
        for i in range(n - 1, 0, -1):
            bound = i + 1
            threshold = 2**32 % bound
            value = int(draws[trial, i])
            attempt = 0
            while value < threshold:
                attempt += 1
                value = int(philox_u32(keys[trial], i, step, attempt, ORDER))
            j = value % bound
            result[trial, i], result[trial, j] = result[trial, j], result[trial, i]
    return result
