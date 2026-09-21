"""Matrix-free aggregate-profile neighbor joining.

For complete alignments, expected mismatch between two mergeable nucleotide
profiles is bilinear.  Consequently every NJ row sum can be computed exactly
from one aggregate profile, without materialising the quadratic distance
matrix.  Sparse projection candidates and reciprocal-best contractions then
give a parallel scaffold whose intended work is O(n s log n) for fixed
projection width and geometrically shrinking rounds.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable

import numpy as np

from concordtree._core.scaleqf import TreeNode, tree_to_graph, validate_topology
from concordtree._core.sctb_reachability import (
    ProjectionCandidateConfig,
    _pair_profile_distance,
    projection_order_candidates,
)


@dataclass
class AggregateNJStats:
    rounds: int = 0
    merges: int = 0
    candidate_pairs: int = 0
    maximum_candidate_degree: int = 0
    maximum_active: int = 0
    aggregate_coordinates: int = 0
    candidate_seconds: float = 0.0
    candidate_profile_seconds: float = 0.0
    candidate_projection_seconds: float = 0.0
    candidate_pool_seconds: float = 0.0
    candidate_distance_seconds: float = 0.0
    candidate_ranking_seconds: float = 0.0
    candidate_fold_seconds: float = 0.0
    candidate_graph_seconds: float = 0.0
    row_sum_seconds: float = 0.0
    row_sum_stack_cast_seconds: float = 0.0
    row_sum_validation_seconds: float = 0.0
    row_sum_aggregate_seconds: float = 0.0
    row_sum_matvec_seconds: float = 0.0
    row_sum_self_dot_seconds: float = 0.0
    row_sum_finalize_seconds: float = 0.0
    q_score_seconds: float = 0.0
    selection_seconds: float = 0.0
    merge_seconds: float = 0.0
    sparse_distance_seconds: float = 0.0
    gpu_candidate_rounds: int = 0
    streamed_candidate_rounds: int = 0
    candidate_pool_pairs: int = 0
    reused_candidate_distances: int = 0
    candidate_distance_round_seconds: list[float] | None = None
    candidate_pool_round_pairs: list[int] | None = None


def imputed_site_frequencies(states: np.ndarray) -> np.ndarray:
    """Return the historical float32 missing-cell distribution per site."""

    states = np.asarray(states, dtype=np.uint8)
    if states.ndim != 2:
        raise ValueError("states must have shape [taxa, sites]")
    n_sites = states.shape[1]
    site_frequency = np.zeros((n_sites, 4), dtype=np.float32)
    for base in range(4):
        site_frequency[:, base] = np.count_nonzero(states == base, axis=0)
    totals = site_frequency.sum(axis=1, keepdims=True)
    return np.divide(
        site_frequency,
        totals,
        out=np.full_like(site_frequency, 0.25),
        where=totals > 0,
    )


def initial_imputed_profiles(
    states: np.ndarray,
    *,
    immutable_complete_slab: np.ndarray | None = None,
) -> dict[int, np.ndarray]:
    """Encode bases as distributions, imputing missing cells from their site.

    The empirical site distribution is a neutral soft representation that
    preserves every column.  An all-missing sampled site falls back to the
    uniform distribution.  Every taxon/site vector therefore has unit mass,
    which is the only requirement of the aggregate row-sum identity.
    """

    states = np.asarray(states, dtype=np.uint8)
    if states.ndim != 2:
        raise ValueError("states must have shape [taxa, sites]")
    n_taxa, n_sites = states.shape
    # A pre-fork caller may already have constructed this exact leaf-profile
    # map once.  The slab is deliberately immutable: leaf profiles are only
    # popped from the dictionary, while every merged profile is newly
    # allocated.  Forked Views can therefore share the physical leaf pages
    # without coupling their independently evolving NJ states.
    if immutable_complete_slab is not None:
        slab = np.asarray(immutable_complete_slab)
        expected_shape = (n_taxa, n_sites, 4)
        if slab.shape != expected_shape or slab.dtype != np.float32:
            raise ValueError(
                "immutable_complete_slab must have shape [taxa, sites, 4] "
                "and dtype float32"
            )
        if slab.flags.writeable:
            raise ValueError("immutable_complete_slab must be read-only")
        return {taxon: slab[taxon] for taxon in range(n_taxa)}

    # When every sampled cell is observed, imputation is algebraically dead.
    # One indexed identity matrix produces exactly the same float32 one-hot
    # leaves as the general loop, while the dictionary keeps zero-copy views
    # of one contiguous slab.  Missing inputs retain the historical path.
    if bool(np.all(states < 4)):
        slab = np.eye(4, dtype=np.float32)[states]
        return {taxon: slab[taxon] for taxon in range(n_taxa)}
    site_frequency = imputed_site_frequencies(states)
    result: dict[int, np.ndarray] = {}
    for taxon in range(n_taxa):
        profile = np.empty((n_sites, 4), dtype=np.float32)
        profile[:] = site_frequency
        valid = states[taxon] < 4
        profile[valid] = 0.0
        profile[np.nonzero(valid)[0], states[taxon, valid]] = 1.0
        result[taxon] = profile
    return result


def imputed_profile_slab(states: np.ndarray) -> np.ndarray:
    """Return the historical leaf-profile map as one contiguous float32 slab.

    This is the same deterministic encoding used by
    :func:`initial_imputed_profiles`. Keeping the construction separate lets
    the pre-CUDA parent compute it once and let forked Views inherit its
    read-only pages, including for gap-heavy inputs.
    """

    states = np.asarray(states, dtype=np.uint8)
    if states.ndim != 2:
        raise ValueError("states must have shape [taxa, sites]")
    if bool(np.all(states < 4)):
        slab = np.eye(4, dtype=np.float32)[states]
    else:
        n_taxa, n_sites = states.shape
        site_frequency = imputed_site_frequencies(states)
        slab = np.empty((n_taxa, n_sites, 4), dtype=np.float32)
        slab[:] = site_frequency
        for taxon in range(n_taxa):
            valid = states[taxon] < 4
            slab[taxon, valid] = 0.0
            slab[taxon, np.nonzero(valid)[0], states[taxon, valid]] = 1.0
    slab.flags.writeable = False
    return slab


def observed_profile_slab(states: np.ndarray) -> np.ndarray:
    """Encode only observed A/C/G/T states; missing cells have zero mass.

    Unlike empirical imputation, this representation keeps the observation
    mask explicit in the profile mass.  It is the sufficient statistic used
    by the marginalized-missing distance below.
    """

    states = np.asarray(states, dtype=np.uint8)
    if states.ndim != 2:
        raise ValueError("states must have shape [taxa, sites]")
    n_taxa, n_sites = states.shape
    slab = np.zeros((n_taxa, n_sites, 4), dtype=np.float32)
    valid = states < 4
    rows, sites = np.nonzero(valid)
    slab[rows, sites, states[rows, sites]] = 1.0
    slab.flags.writeable = False
    return slab


def site_mismatch_prior(states: np.ndarray) -> np.ndarray:
    """Estimate each site's marginal mismatch probability from observations."""

    frequencies = imputed_site_frequencies(states).astype(np.float64, copy=False)
    return np.asarray(1.0 - np.einsum("ij,ij->i", frequencies, frequencies), dtype=np.float64)


def site_observation_fraction(states: np.ndarray) -> np.ndarray:
    """Return the empirical probability that a taxon observes each site."""

    states = np.asarray(states, dtype=np.uint8)
    if states.ndim != 2 or states.shape[0] == 0:
        raise ValueError("states must have shape [taxa, sites]")
    return np.count_nonzero(states < 4, axis=0).astype(np.float64) / states.shape[0]


def site_pair_observation_probability(states: np.ndarray) -> np.ndarray:
    """Return the plug-in probability that both members observe each site."""

    fraction = site_observation_fraction(states)
    return fraction * fraction


def marginalized_profile_distance(
    left: np.ndarray,
    right: np.ndarray,
    mismatch_prior: np.ndarray,
) -> float:
    """Expected mismatch after marginalizing unobserved state pairs.

    If both profiles carry observed mass, their actual expected mismatch
    replaces the site prior.  Otherwise that site's population mismatch prior
    remains.  The expression is affine-bilinear, so profile averaging and the
    aggregate NJ row-sum identity remain valid.  With complete observations it
    reduces exactly to ordinary profile mismatch.
    """

    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    prior = np.asarray(mismatch_prior, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 2 or left.shape[1] != 4:
        raise ValueError("profiles must have matching shape [sites, 4]")
    if prior.shape != (left.shape[0],):
        raise ValueError("mismatch_prior must have one value per site")
    left_mass = left.sum(axis=1)
    right_mass = right.sum(axis=1)
    correction = np.sum(
        (1.0 - prior) * left_mass * right_mass
        - np.einsum("ij,ij->i", left, right)
    )
    return float(prior.mean() + correction / len(prior))


def coverage_calibrated_profile_distance(
    left: np.ndarray,
    right: np.ndarray,
    mismatch_prior: np.ndarray,
    site_frequency: np.ndarray,
    imputation_weight: np.ndarray,
) -> float:
    """Expected mismatch with coverage-calibrated one-sided imputation.

    ``imputation_weight`` is the empirical probability that both members of a
    random taxon pair observe the site. At well-covered sites the expression
    approaches empirical state imputation; at sparsely observed sites one-sided missing
    comparisons are marginalized to the site's mismatch prior.  No dataset
    threshold or fitted scalar is used.
    """

    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    prior = np.asarray(mismatch_prior, dtype=np.float64)
    frequency = np.asarray(site_frequency, dtype=np.float64)
    weight = np.asarray(imputation_weight, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 2 or left.shape[1] != 4:
        raise ValueError("profiles must have matching shape [sites, 4]")
    sites = left.shape[0]
    if prior.shape != (sites,) or weight.shape != (sites,) or frequency.shape != (sites, 4):
        raise ValueError("coverage-calibrated site statistics have invalid shape")
    left_mass = left.sum(axis=1)
    right_mass = right.sum(axis=1)
    residual = (1.0 - frequency) - prior[:, None]
    left_residual = np.einsum("ij,ij->i", left, residual)
    right_residual = np.einsum("ij,ij->i", right, residual)
    value = prior + (1.0 - prior) * left_mass * right_mass
    value -= np.einsum("ij,ij->i", left, right)
    value += weight * (
        left_residual * (1.0 - right_mass)
        + right_residual * (1.0 - left_mass)
    )
    return float(value.mean())


def aggregate_coverage_calibrated_row_sums(
    nodes: list[int],
    profiles: dict[int, np.ndarray],
    offsets: dict[int, float],
    mismatch_prior: np.ndarray,
    site_frequency: np.ndarray,
    imputation_weight: np.ndarray,
    timing: dict[str, float] | None = None,
) -> dict[int, float]:
    """Exact aggregate row sums for coverage-calibrated profile distance."""

    if not nodes:
        raise ValueError("cannot aggregate an empty active set")
    phase = perf_counter()
    raw = np.stack([profiles[node] for node in nodes]).astype(np.float64, copy=False)
    if timing is not None:
        timing["stack_cast_seconds"] = perf_counter() - phase
        timing["validation_seconds"] = 0.0
    prior = np.asarray(mismatch_prior, dtype=np.float64)
    frequency = np.asarray(site_frequency, dtype=np.float64)
    weight = np.asarray(imputation_weight, dtype=np.float64)
    sites = raw.shape[1]
    if prior.shape != (sites,) or weight.shape != (sites,) or frequency.shape != (sites, 4):
        raise ValueError("coverage-calibrated site statistics have invalid shape")
    mass = raw.sum(axis=2)
    flat = raw.reshape(len(nodes), -1)
    residual = (1.0 - frequency) - prior[:, None]
    h = np.einsum("nsb,sb->ns", raw, residual)

    phase = perf_counter()
    aggregate_profile = flat.sum(axis=0)
    aggregate_mass = mass.sum(axis=0)
    aggregate_h = h.sum(axis=0)
    if timing is not None:
        timing["aggregate_seconds"] = perf_counter() - phase

    phase = perf_counter()
    base_all = len(nodes) * prior[None, :]
    base_all = base_all + (1.0 - prior)[None, :] * mass * aggregate_mass
    base_all -= (raw * aggregate_profile.reshape(sites, 4)[None, :, :]).sum(axis=2)
    base_all += weight[None, :] * (
        h * (len(nodes) - aggregate_mass)[None, :]
        + (1.0 - mass) * aggregate_h[None, :]
    )
    base_all = base_all.sum(axis=1) / sites
    if timing is not None:
        timing["matvec_seconds"] = perf_counter() - phase

    phase = perf_counter()
    base_self = prior[None, :] + (1.0 - prior)[None, :] * mass * mass
    base_self -= (raw * raw).sum(axis=2)
    base_self += 2.0 * weight[None, :] * h * (1.0 - mass)
    base_self = base_self.sum(axis=1) / sites
    if timing is not None:
        timing["self_dot_seconds"] = perf_counter() - phase

    phase = perf_counter()
    offset_total = sum(float(offsets[node]) for node in nodes)
    result = {
        node: float(base_all[row] - base_self[row])
        + (len(nodes) - 2) * float(offsets[node])
        + offset_total
        for row, node in enumerate(nodes)
    }
    if timing is not None:
        timing["finalize_seconds"] = perf_counter() - phase
    return result


def aggregate_marginalized_row_sums(
    nodes: list[int],
    profiles: dict[int, np.ndarray],
    offsets: dict[int, float],
    mismatch_prior: np.ndarray,
    timing: dict[str, float] | None = None,
) -> dict[int, float]:
    """Return exact NJ row sums for marginalized-missing profile distance."""

    if not nodes:
        raise ValueError("cannot aggregate an empty active set")
    phase = perf_counter()
    raw = np.stack([profiles[node] for node in nodes]).astype(np.float64, copy=False)
    if timing is not None:
        timing["stack_cast_seconds"] = perf_counter() - phase
        timing["validation_seconds"] = 0.0
    prior = np.asarray(mismatch_prior, dtype=np.float64)
    if prior.shape != (raw.shape[1],):
        raise ValueError("mismatch_prior must have one value per site")
    sites = raw.shape[1]
    if sites == 0:
        raise ValueError("at least one retained site is required")
    mass = raw.sum(axis=2)
    flat = raw.reshape(len(nodes), -1)

    phase = perf_counter()
    aggregate_profile = flat.sum(axis=0)
    aggregate_mass = mass.sum(axis=0)
    if timing is not None:
        timing["aggregate_seconds"] = perf_counter() - phase

    phase = perf_counter()
    base_all = len(nodes) * float(prior.mean()) + (
        mass @ ((1.0 - prior) * aggregate_mass)
        - flat @ aggregate_profile
    ) / sites
    if timing is not None:
        timing["matvec_seconds"] = perf_counter() - phase

    phase = perf_counter()
    base_self = float(prior.mean()) + (
        np.sum((1.0 - prior) * mass * mass, axis=1)
        - np.einsum("ij,ij->i", flat, flat)
    ) / sites
    if timing is not None:
        timing["self_dot_seconds"] = perf_counter() - phase

    phase = perf_counter()
    offset_total = sum(float(offsets[node]) for node in nodes)
    result = {
        node: float(base_all[row] - base_self[row])
        + (len(nodes) - 2) * float(offsets[node])
        + offset_total
        for row, node in enumerate(nodes)
    }
    if timing is not None:
        timing["finalize_seconds"] = perf_counter() - phase
    return result


def aggregate_mismatch_row_sums(
    nodes: list[int],
    profiles: dict[int, np.ndarray],
    offsets: dict[int, float],
    timing: dict[str, float] | None = None,
    validate_profiles: bool = True,
) -> dict[int, float]:
    """Return exact row sums for the profile-mismatch NJ representation.

    Profiles must have unit mass at every retained site.  This is preserved by
    equal-weight NJ profile merges when the input sketch contains no missing
    states.  The diagonal is explicitly removed because a mixed profile's
    self-mismatch is not zero even though a distance-matrix diagonal is.
    """

    if not nodes:
        raise ValueError("cannot aggregate an empty active set")
    phase = perf_counter()
    raw = np.stack([profiles[node] for node in nodes]).astype(np.float64, copy=False)
    if timing is not None:
        timing["stack_cast_seconds"] = perf_counter() - phase

    phase = perf_counter()
    if validate_profiles and not np.allclose(raw.sum(axis=2), 1.0, atol=1e-6):
        raise ValueError(
            "aggregate mismatch row sums currently require complete sampled sites"
        )
    if timing is not None:
        timing["validation_seconds"] = perf_counter() - phase
    sites = raw.shape[1]
    if sites == 0:
        raise ValueError("at least one retained site is required")
    flat = raw.reshape(len(nodes), -1)

    phase = perf_counter()
    aggregate = flat.sum(axis=0)
    if timing is not None:
        timing["aggregate_seconds"] = perf_counter() - phase

    phase = perf_counter()
    base_all = len(nodes) - (flat @ aggregate) / sites
    if timing is not None:
        timing["matvec_seconds"] = perf_counter() - phase

    phase = perf_counter()
    base_self = 1.0 - np.einsum("ij,ij->i", flat, flat) / sites
    if timing is not None:
        timing["self_dot_seconds"] = perf_counter() - phase

    phase = perf_counter()
    offset_total = sum(float(offsets[node]) for node in nodes)
    result = {
        node: float(base_all[row] - base_self[row])
        + (len(nodes) - 2) * float(offsets[node])
        + offset_total
        for row, node in enumerate(nodes)
    }
    if timing is not None:
        timing["finalize_seconds"] = perf_counter() - phase
    return result


def finalize_mismatch_row_sums(
    nodes: list[int],
    base_row_sums: np.ndarray,
    offsets: dict[int, float],
) -> dict[int, float]:
    """Add the exact NJ offset terms to precomputed bilinear base row sums."""

    base = np.asarray(base_row_sums, dtype=np.float64)
    if base.shape != (len(nodes),):
        raise ValueError("base_row_sums must have one value per active node")
    offset_total = sum(float(offsets[node]) for node in nodes)
    return {
        node: float(base[row])
        + (len(nodes) - 2) * float(offsets[node])
        + offset_total
        for row, node in enumerate(nodes)
    }


def build_aggregate_parallel_nj(
    states: np.ndarray,
    projections: int = 16,
    window: int = 8,
    candidate_cap: int = 32,
    seed: int = 20260903,
    candidate_device: str | None = None,
    pair_batch_size: int = 256,
    candidate_distance_backend: str | None = None,
    row_sum_backend: str = "cpu",
    immutable_complete_slab: np.ndarray | None = None,
    compact_imputed_first_round: bool = False,
    profile_stream_sites: int | None = None,
    missing_distance_model: str = "imputed",
    progress_callback: Callable[[dict[str, int]], None] | None = None,
) -> tuple[dict[int, set[int]], AggregateNJStats]:
    """Build a complete binary tree using sparse exact-Q reciprocal joins."""

    states = np.asarray(states, dtype=np.uint8)
    if states.ndim != 2 or len(states) < 4:
        raise ValueError("states must contain at least four taxa")
    n_taxa = len(states)
    if missing_distance_model not in {"imputed", "marginalized", "coverage"}:
        raise ValueError(
            "missing_distance_model must be imputed, marginalized, or coverage"
        )
    # The two formulas are algebraically identical on a complete matrix.  Use
    # the established representation there to preserve its exact reduction
    # order as well as its topology.
    marginalized_prior: np.ndarray | None = None
    coverage_frequency: np.ndarray | None = None
    coverage_weight: np.ndarray | None = None
    proposal_profiles: dict[int, np.ndarray] | None = None
    if missing_distance_model in {"marginalized", "coverage"} and bool(np.any(states >= 4)):
        proposal_profiles = initial_imputed_profiles(
            states,
            immutable_complete_slab=immutable_complete_slab,
        )
        slab = observed_profile_slab(states)
        profiles = {taxon: slab[taxon] for taxon in range(n_taxa)}
        marginalized_prior = site_mismatch_prior(states)
        if missing_distance_model == "coverage":
            coverage_frequency = imputed_site_frequencies(states).astype(
                np.float64, copy=False
            )
            coverage_weight = site_pair_observation_probability(states)
        immutable_complete_slab = slab
    else:
        profiles = initial_imputed_profiles(
            states,
            immutable_complete_slab=immutable_complete_slab,
        )
    offsets = {node: 0.0 for node in range(n_taxa)}
    members = {node: (node,) for node in range(n_taxa)}
    roots = {node: TreeNode(leaf=node) for node in range(n_taxa)}
    active: set[int] = set(range(n_taxa))
    next_node = n_taxa
    stats = AggregateNJStats(maximum_active=n_taxa)
    stats.candidate_distance_round_seconds = []
    stats.candidate_pool_round_pairs = []

    while len(active) > 3:
        nodes = sorted(active, key=lambda node: members[node])
        tie_keys = {node: members[node] for node in nodes}
        config = ProjectionCandidateConfig(
            projections=projections,
            window=window,
            candidate_cap=candidate_cap,
            seed=seed + stats.rounds * 1_000_003,
        )
        native_pool = None
        if candidate_device is None:
            candidates = projection_order_candidates(
                nodes,
                proposal_profiles if proposal_profiles is not None else profiles,
                config,
                tie_keys=tie_keys,
            )
            if coverage_weight is not None:
                evidence_candidates = projection_order_candidates(
                    nodes, profiles, config, tie_keys=tie_keys
                )
                for node in nodes:
                    candidates[node].update(evidence_candidates[node])
            exact_pool_distances = None
        else:
            from concordtree._core.sctb_gpu_candidate import (
                projection_pool_batch_gpu,
            )

            native_pool = projection_pool_batch_gpu(
                nodes,
                profiles,
                config,
                device=candidate_device,
                pair_batch_size=pair_batch_size,
                validate_profiles=False,
                distance_backend=candidate_distance_backend,
                row_sum_backend=(
                    "native" if marginalized_prior is not None else row_sum_backend
                ),
                prestacked_profiles=(
                    immutable_complete_slab if stats.rounds == 0 else None
                ),
                complete_leaf_states=(
                    states
                    if stats.rounds == 0
                    and marginalized_prior is None
                    and immutable_complete_slab is not None
                    and bool(np.all(states < 4))
                    and candidate_distance_backend == "native"
                    else None
                ),
                imputed_leaf_states=(
                    states
                    if compact_imputed_first_round
                    and stats.rounds == 0
                    and marginalized_prior is None
                    and immutable_complete_slab is not None
                    and bool(np.any(states >= 4))
                    and candidate_distance_backend == "native"
                    else None
                ),
                imputed_site_frequency=(
                    imputed_site_frequencies(states)
                    if compact_imputed_first_round
                    and stats.rounds == 0
                    and marginalized_prior is None
                    and immutable_complete_slab is not None
                    and bool(np.any(states >= 4))
                    and candidate_distance_backend == "native"
                    else None
                ),
                profile_stream_sites=profile_stream_sites,
                site_mismatch_prior=marginalized_prior,
                site_frequency=coverage_frequency,
                site_imputation_weight=coverage_weight,
                projection_profiles=(
                    (proposal_profiles, profiles)
                    if coverage_weight is not None
                    and proposal_profiles is not None
                    else proposal_profiles
                ),
            )
            candidate_timing = native_pool.timing
            stats.candidate_seconds += (
                candidate_timing.projection_seconds
                + candidate_timing.pool_seconds
                + candidate_timing.distance_seconds
            )
            stats.candidate_profile_seconds += candidate_timing.profile_seconds
            stats.candidate_projection_seconds += candidate_timing.projection_seconds
            stats.candidate_pool_seconds += candidate_timing.pool_seconds
            stats.candidate_distance_seconds += candidate_timing.distance_seconds
            stats.candidate_distance_round_seconds.append(
                candidate_timing.distance_seconds
            )
            stats.candidate_ranking_seconds += candidate_timing.ranking_seconds
            stats.gpu_candidate_rounds += 1
            if candidate_timing.distance_backend == "native-streamed64":
                stats.streamed_candidate_rounds += 1
            stats.candidate_pool_pairs += candidate_timing.unique_pool_pairs
            stats.candidate_pool_round_pairs.append(
                candidate_timing.unique_pool_pairs
            )
        if native_pool is None:
            phase_started = perf_counter()
            stats.maximum_candidate_degree = max(
                stats.maximum_candidate_degree,
                max((len(value) for value in candidates.values()), default=0),
            )
            pairs = sorted(
                {
                    tuple(sorted((left, right), key=lambda node: members[node]))
                    for left in nodes
                    for right in candidates[left]
                    if left != right
                },
                key=lambda pair: (members[pair[0]], members[pair[1]]),
            )
            if not pairs:
                raise RuntimeError("sparse candidate graph has no edges")
            stats.candidate_graph_seconds += perf_counter() - phase_started

        phase_started = perf_counter()
        row_sum_timing: dict[str, float] = {}
        if native_pool is not None and native_pool.base_row_sums is not None:
            row_sum_timing.update(native_pool.row_sum_timing or {})
            finalize_started = perf_counter()
            # The GPU pool uses numeric node order, whereas Aggregate-NJ's
            # outer loop uses the lineage-member tie order.  Bind each
            # sufficient statistic to the order in which it was computed.
            row_nodes = list(native_pool.nodes)
            row_sums = finalize_mismatch_row_sums(
                row_nodes, native_pool.base_row_sums, offsets
            )
            row_sum_timing["finalize_seconds"] = perf_counter() - finalize_started
        elif coverage_weight is not None:
            assert marginalized_prior is not None
            assert coverage_frequency is not None
            row_sums = aggregate_coverage_calibrated_row_sums(
                nodes,
                profiles,
                offsets,
                marginalized_prior,
                coverage_frequency,
                coverage_weight,
                timing=row_sum_timing,
            )
        elif marginalized_prior is not None:
            row_sums = aggregate_marginalized_row_sums(
                nodes,
                profiles,
                offsets,
                marginalized_prior,
                timing=row_sum_timing,
            )
        else:
            row_sums = aggregate_mismatch_row_sums(
                nodes,
                profiles,
                offsets,
                timing=row_sum_timing,
                validate_profiles=False,
            )
        stats.aggregate_coordinates += len(nodes) * states.shape[1] * 4
        stats.row_sum_seconds += sum(row_sum_timing.values())
        stats.row_sum_stack_cast_seconds += row_sum_timing["stack_cast_seconds"]
        stats.row_sum_validation_seconds += row_sum_timing["validation_seconds"]
        stats.row_sum_aggregate_seconds += row_sum_timing["aggregate_seconds"]
        stats.row_sum_matvec_seconds += row_sum_timing["matvec_seconds"]
        stats.row_sum_self_dot_seconds += row_sum_timing["self_dot_seconds"]
        stats.row_sum_finalize_seconds += row_sum_timing["finalize_seconds"]

        if native_pool is not None:
            from concordtree.assets import load_candidate_graph_backend
            from concordtree._core.sctb_gpu_candidate import _candidate_cpu_workers

            pool_nodes = list(native_pool.nodes)
            tie_order = sorted(
                range(len(pool_nodes)), key=lambda row: members[pool_nodes[row]]
            )
            tie_rank = np.empty(len(pool_nodes), dtype=np.int64)
            tie_rank[np.asarray(tie_order, dtype=np.int64)] = np.arange(
                len(pool_nodes), dtype=np.int64
            )
            row_sum_array = np.asarray(
                [row_sums[node] for node in pool_nodes], dtype=np.float64
            )
            offset_array = np.asarray(
                [offsets[node] for node in pool_nodes], dtype=np.float64
            )
            phase_started = perf_counter()
            native_selected, candidate_count, maximum_degree = (
                load_candidate_graph_backend().select_nj_merges(
                    native_pool.pairs,
                    native_pool.distances,
                    tie_rank,
                    row_sum_array,
                    offset_array,
                    candidate_cap,
                    len(nodes) - 3,
                    _candidate_cpu_workers(),
                )
            )
            native_selected = np.asarray(native_selected, dtype=np.int64)
            fold_seconds = perf_counter() - phase_started
            stats.candidate_seconds += fold_seconds
            stats.candidate_ranking_seconds += fold_seconds
            stats.candidate_fold_seconds += fold_seconds
            stats.candidate_pairs += int(candidate_count)
            stats.reused_candidate_distances += int(candidate_count)
            stats.maximum_candidate_degree = max(
                stats.maximum_candidate_degree, int(maximum_degree)
            )
            selected_with_base = [
                (
                    pool_nodes[int(left_row)],
                    pool_nodes[int(right_row)],
                    float(native_pool.distances[int(pool_row)]),
                )
                for left_row, right_row, pool_row in native_selected
            ]
        else:
            def base_distance(left: int, right: int) -> float:
                if marginalized_prior is not None:
                    if coverage_weight is not None:
                        assert coverage_frequency is not None
                        return coverage_calibrated_profile_distance(
                            profiles[left],
                            profiles[right],
                            marginalized_prior,
                            coverage_frequency,
                            coverage_weight,
                        )
                    return marginalized_profile_distance(
                        profiles[left], profiles[right], marginalized_prior
                    )
                return _pair_profile_distance(profiles[left], profiles[right])

            def distance(left: int, right: int) -> float:
                return base_distance(left, right) + offsets[left] + offsets[right]

            phase_started = perf_counter()
            q_scores = {
                pair: (len(nodes) - 2) * distance(*pair)
                - row_sums[pair[0]]
                - row_sums[pair[1]]
                for pair in pairs
            }
            stats.candidate_pairs += len(pairs)
            stats.q_score_seconds += perf_counter() - phase_started

            phase_started = perf_counter()
            best: dict[int, tuple[float, tuple[int, ...], int]] = {}
            for left, right in pairs:
                score = float(q_scores[(left, right)])
                for node, other in ((left, right), (right, left)):
                    rank = (score, members[other], other)
                    if node not in best or rank < best[node]:
                        best[node] = rank

            selected: list[tuple[int, int]] = []
            consumed: set[int] = set()
            maximum_merges = len(nodes) - 3
            for pair in sorted(
                pairs,
                key=lambda value: (
                    q_scores[value], members[value[0]], members[value[1]]
                ),
            ):
                left, right = pair
                if len(selected) >= maximum_merges:
                    break
                if left in consumed or right in consumed:
                    continue
                if best[left][2] == right and best[right][2] == left:
                    selected.append(pair)
                    consumed.update(pair)
            if not selected:
                selected = [
                    min(
                        pairs,
                        key=lambda pair: (
                            q_scores[pair], members[pair[0]], members[pair[1]]
                        ),
                    )
                ]
            stats.selection_seconds += perf_counter() - phase_started
            selected_with_base = [
                (left, right, base_distance(left, right)) for left, right in selected
            ]

        phase_started = perf_counter()
        for left, right, base in selected_with_base:
            profiles[next_node] = 0.5 * (profiles.pop(left) + profiles.pop(right))
            if proposal_profiles is not None:
                proposal_profiles[next_node] = 0.5 * (
                    proposal_profiles.pop(left) + proposal_profiles.pop(right)
                )
            # This offset makes on-demand profile distance obey the conventional
            # NJ reduction d(ij,k)=(d(i,k)+d(j,k)-d(i,j))/2 exactly.
            offsets[next_node] = -0.5 * base
            offsets.pop(left)
            offsets.pop(right)
            members[next_node] = tuple(
                sorted((*members.pop(left), *members.pop(right)))
            )
            roots[next_node] = TreeNode(children=[roots.pop(left), roots.pop(right)])
            active.remove(left)
            active.remove(right)
            active.add(next_node)
            next_node += 1
        stats.merge_seconds += perf_counter() - phase_started
        stats.merges += len(selected_with_base)
        stats.rounds += 1
        if progress_callback is not None:
            progress_callback(
                {
                    "round": stats.rounds,
                    "active": len(active),
                    "merges_this_round": len(selected_with_base),
                    "merges_total": stats.merges,
                    "candidate_pairs_total": stats.candidate_pairs,
                }
            )

    final = [roots[node] for node in sorted(active, key=lambda node: members[node])]
    graph = tree_to_graph(TreeNode(children=final), n_taxa)
    validate_topology(graph, n_taxa)
    return graph, stats
