"""Experimental bounded-site parameter estimation, NOT full-site ML.

Uniform sampling without replacement estimates the fixed-topology likelihood
objective using at most ``fit_sites`` columns. Constants, ambiguous columns and
missingness are not filtered. The caller must still score topology on the full
alignment; this estimator does not itself infer topology. No neural/reference
access. Fixed sample size does not imply full-data consistency as L -> infinity.
"""
from __future__ import annotations

import hashlib
import numbers
import time
import numpy as np


def select_parameter_sites(sites, fit_sites=16384, seed=20261022):
    """Sorted distinct positions; O(min(L, cap)) storage, never an L-sized mask."""
    for key, value in (("sites", sites), ("fit_sites", fit_sites), ("seed", seed)):
        if isinstance(value, bool) or not isinstance(value, numbers.Integral):
            raise ValueError(f"{key} must be an integer")
    if sites < 1 or fit_sites < 1 or seed < 0:
        raise ValueError("positive lengths/cap and nonnegative seed required")
    take = min(int(sites), int(fit_sites))
    if take == sites:
        return np.arange(sites, dtype=np.int64)
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    # choice with shuffle=False uses bounded-size selection for this small cap.
    return np.sort(rng.choice(int(sites), size=take, replace=False, shuffle=False))


def parameter_sample(states, fit_sites=16384, seed=20261022):
    x = np.asarray(states)
    if x.ndim != 2 or min(x.shape) < 1 or not np.issubdtype(x.dtype, np.integer):
        raise ValueError("nonempty integer taxa x sites matrix required")
    indices = select_parameter_sites(x.shape[1], fit_sites, seed)
    sampled = np.ascontiguousarray(x[:, indices])
    if sampled.min() < 0 or sampled.max() > 4:
        raise ValueError("sampled DNA states must be 0..4")
    if not np.all(np.any(sampled < 4, axis=1)):
        raise ValueError("parameter sample leaves a taxon entirely missing; no silent fallback")
    return sampled, indices


def fit_bounded_parameters(tree, names, states, *, fit_sites=16384, seed=20261022,
                           device="cuda:0", max_iter=256):
    """Use the retained GPU optimizer on a fresh sample, returning its audit.

    Physical convergence refers ONLY to the sampled objective. All setup,
    sampling and optimization are included in ``bounded_estimation_seconds``.
    No cached fitted lengths/model are accepted or silently reused.
    """
    from fitter import fit_reversible_reference_optimizer
    started = time.perf_counter()
    sampled, indices = parameter_sample(states, fit_sites, seed)
    base = tree.copy()
    for ed in base.lengths:
        base.lengths[ed] = None
    # Same fused path/settings as the retained all-site real10 anchor; only its
    # estimation input changes. Slab cap bounds scratch, not total site coverage.
    cap = max(1, min(262144, (4 << 30) // ((len(names) - 2) * 5 * 56)))
    block = 1 << (int(cap).bit_length() - 1)
    fitted, metrics = fit_reversible_reference_optimizer(
        base, names, sampled, model="GTR+I+G4", device=device,
        max_iter=max_iter, precision="mixed", length_coordinate="sqrt",
        seed_method="parsimony", block_sites=block, plan_backend="fused",
        transition_backend="spectral-cuda", coordinate_metric="incumbent",
        stopping_policy="physical", model_backend="device",
        score_backend="analytic-cuda", optimizer_backend="lbfgsb", graph=True)
    metrics["parameter_estimation"] = {
        "method": "uniform-without-replacement fixed-site approximate ML",
        "original_sites": int(states.shape[1]), "fitted_sites": len(indices),
        "site_cap": int(fit_sites), "seed": int(seed), "rng": "numpy.PCG64",
        "sample_indices_sha256": hashlib.sha256(indices.astype("<i8").tobytes()).hexdigest(),
        "all_original_sites_used_for_fit": len(indices) == states.shape[1],
        "constant_or_missing_column_filter": False,
        "full_data_ml_convergence_claim": False,
        "bounded_estimation_seconds": time.perf_counter() - started,
    }
    return fitted, metrics, indices
