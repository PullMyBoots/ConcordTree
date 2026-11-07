"""Reversible transition matrices for a fixed supplied model.

Diagonalize only the symmetric 4x4 generator on the CPU. Evaluate the O(NK)
exponentials on the GPU in FP64, using I + V expm1(t*r*lambda) V^-1 to retain
short-edge off-diagonal precision. This is the same matrix exponential as
the pruning objective; no model fit, rate approximation or message cache.
"""
import numpy as np
import torch
import triton
from spectral_cuda import _forward


@torch.no_grad()
def fixed_transitions(q, pi, lengths, rates, device):
    q = np.asarray(q, dtype=np.float64)
    pi = np.asarray(pi, dtype=np.float64)
    lengths = np.asarray(lengths, dtype=np.float64).reshape(-1)
    rates = np.asarray(rates, dtype=np.float64).reshape(-1)
    root = np.sqrt(pi)
    sym = root[:, None] * q / root[None, :]
    lam, u = np.linalg.eigh((sym + sym.T) * .5)
    v = u / root[:, None]
    inv = u.T * root[None, :]
    packed = torch.tensor(np.r_[lam, v.reshape(-1), inv.reshape(-1),
                                rates, lengths], device=device, dtype=torch.float64)
    k = len(rates)
    out = torch.empty((len(lengths), k, 4, 4), device=device, dtype=torch.float64)
    _forward[(triton.cdiv(len(lengths) * k, 32),)](
        packed[36+k:], packed[36:36+k], packed[:4],
        packed[4:20].view(4,4), packed[20:36].view(4,4),
        out, len(lengths), k, 32, enable_fp_fusion=False)
    return out
