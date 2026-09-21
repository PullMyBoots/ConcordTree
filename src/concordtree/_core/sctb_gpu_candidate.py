"""Equivalence-preserving GPU batching for exact sparse profile distances."""

from __future__ import annotations

from dataclasses import dataclass
import os
from time import perf_counter
from typing import Iterable

import numpy as np
import torch

from concordtree.assets import load_backends, load_candidate_graph_backend
from concordtree._core.sctb_reachability import ProjectionCandidateConfig


@dataclass(frozen=True)
class CandidateTiming:
    profile_seconds: float
    projection_seconds: float
    pool_seconds: float
    distance_seconds: float
    ranking_seconds: float
    unique_pool_pairs: int
    distance_backend: str


@dataclass(frozen=True)
class ProjectionCandidateBatch:
    """A candidate graph together with its already evaluated exact edges.

    Projection pooling is only a sparse edge-discovery device: every edge in
    the returned graph has already received the same exact profile mismatch
    used by the downstream NJ objective.  Keeping those values makes that
    mathematical relationship explicit and avoids evaluating the selected
    subset a second time.
    """

    candidates: dict[int, set[int]]
    exact_distances: dict[tuple[int, int], float]
    timing: CandidateTiming


@dataclass(frozen=True)
class ProjectionPoolBatch:
    """Projection-window pool with exact distances in stable pool-row order."""

    nodes: tuple[int, ...]
    pairs: np.ndarray
    distances: np.ndarray
    timing: CandidateTiming
    base_row_sums: np.ndarray | None = None
    row_sum_timing: dict[str, float] | None = None


def canonical_node_pair(left: int, right: int) -> tuple[int, int]:
    """Return one stable dictionary key for an undirected node pair."""

    return (left, right) if left < right else (right, left)


def _candidate_cpu_workers() -> int:
    value = int(os.environ.get("CONCORDTREE_CANDIDATE_CPU_WORKERS", "8"))
    if value < 1 or value > 64:
        raise ValueError("CONCORDTREE_CANDIDATE_CPU_WORKERS must be between 1 and 64")
    return value


def _candidate_graph_backend_name() -> str:
    value = os.environ.get("CONCORDTREE_CANDIDATE_GRAPH_BACKEND", "native")
    if value not in {"native", "numpy"}:
        raise ValueError("CONCORDTREE_CANDIDATE_GRAPH_BACKEND must be 'native' or 'numpy'")
    return value


def _active_profile_slab(
    nodes: list[int],
    profiles: dict[int, np.ndarray],
    prestacked_profiles: np.ndarray | None = None,
) -> np.ndarray:
    """Bind an already ordered immutable slab, or build the historical stack.

    The optional path is deliberately narrow: it is valid only for the first
    Aggregate-NJ round, where active node ids are the leaf ids in numeric order
    and every dictionary value is a view into the supplied slab.  Later rounds
    contain newly allocated merged profiles and retain the historical stack.
    """

    if prestacked_profiles is None:
        rows = [profiles[node] for node in nodes]
        if _candidate_graph_backend_name() == "native":
            native_stack = getattr(
                load_candidate_graph_backend(), "stack_profile_rows", None
            )
            if native_stack is not None:
                return np.asarray(
                    native_stack(rows, _candidate_cpu_workers()),
                    dtype=np.float32,
                )
        return np.stack(rows).astype(np.float32, copy=False)
    slab = np.asarray(prestacked_profiles)
    if slab.dtype != np.float32 or slab.ndim != 3 or slab.shape[2] != 4:
        raise ValueError(
            "prestacked_profiles must have shape (nodes, sites, 4) and dtype float32"
        )
    if slab.shape[0] != len(nodes) or not slab.flags.c_contiguous:
        raise ValueError("prestacked_profiles must be C-contiguous and match active nodes")
    if nodes != list(range(len(nodes))):
        raise ValueError("prestacked_profiles requires first-round numeric leaf order")
    if slab.flags.writeable:
        raise ValueError("prestacked_profiles must be immutable")
    # ``initial_imputed_profiles`` constructs every leaf as a direct row view
    # of this exact slab. Checking that O(1) base identity for each row avoids
    # NumPy's general overlap solver, whose repeated exact interval analysis is
    # measurable for ten thousand leaves and a gigabyte-scale missing-data
    # slab.
    if any(profiles[node].base is not slab for node in nodes):
        raise ValueError("prestacked_profiles rows must back the active leaf profiles")
    return slab


def _fused_sparse_profile_distance(
    raw: torch.Tensor, left: torch.Tensor, right: torch.Tensor
) -> torch.Tensor:
    """Evaluate only requested profile pairs without materializing an n-by-n matrix."""

    return 1.0 - (raw[left] * raw[right]).sum(dim=(1, 2)) / raw.shape[1]


# The optional compiled backend is useful for frontier experiments.  Release
# inference defaults to the eager backend below because that preserves rc11's
# floating reduction order exactly.  Dynamic shapes let one compiled reduction
# serve every NJ round; the default compiler mode avoids graph-per-shape CUDA
# capture.  Compilation remains lazy and CUDA-only at the call site.
_compiled_sparse_profile_distance = torch.compile(
    _fused_sparse_profile_distance,
    dynamic=True,
    fullgraph=True,
)


def exact_profile_pair_distances_gpu(
    pairs: list[tuple[int, int]],
    profiles: dict[int, np.ndarray],
    *,
    device: str | torch.device,
    pair_batch_size: int = 256,
) -> dict[tuple[int, int], float]:
    """Evaluate unit-mass profile mismatch for an explicit sparse pair list."""

    if not pairs:
        return {}
    nodes = sorted({node for pair in pairs for node in pair})
    position = {node: row for row, node in enumerate(nodes)}
    raw = np.stack([profiles[node] for node in nodes]).astype(np.float32, copy=False)
    if not np.allclose(raw.sum(axis=2), 1.0, atol=1e-5):
        raise ValueError("GPU exact-distance shortcut requires unit-mass profiles")
    target = torch.device(device)
    raw_tensor = torch.from_numpy(raw).to(target)
    output: dict[tuple[int, int], float] = {}
    with torch.no_grad():
        for start in range(0, len(pairs), pair_batch_size):
            chunk = pairs[start : start + pair_batch_size]
            left = torch.as_tensor(
                [position[pair[0]] for pair in chunk], dtype=torch.long, device=target
            )
            right = torch.as_tensor(
                [position[pair[1]] for pair in chunk], dtype=torch.long, device=target
            )
            match = (
                raw_tensor.index_select(0, left)
                .mul(raw_tensor.index_select(0, right))
                .sum(dim=(1, 2))
                / raw.shape[1]
            )
            for pair, value in zip(chunk, (1.0 - match).cpu().numpy()):
                output[pair] = float(value)
    return output


def projection_pool_batch_gpu(
    active: Iterable[int],
    profiles: dict[int, np.ndarray],
    config: ProjectionCandidateConfig,
    *,
    device: str | torch.device,
    pair_batch_size: int = 256,
    validate_profiles: bool = True,
    distance_backend: str | None = None,
    row_sum_backend: str = "cpu",
    prestacked_profiles: np.ndarray | None = None,
    complete_leaf_states: np.ndarray | None = None,
    imputed_leaf_states: np.ndarray | None = None,
    imputed_site_frequency: np.ndarray | None = None,
    profile_stream_sites: int | None = None,
    site_mismatch_prior: np.ndarray | None = None,
    site_frequency: np.ndarray | None = None,
    site_imputation_weight: np.ndarray | None = None,
    projection_profiles: (
        dict[int, np.ndarray] | tuple[dict[int, np.ndarray], ...] | None
    ) = None,
) -> ProjectionPoolBatch:
    """Build the stable projection pool and evaluate every pool edge exactly.

    The caller must supply unit-mass site profiles, as produced by
    ``initial_imputed_profiles`` and preserved by equal-child averaging.
    """

    nodes = sorted(active)
    if config.projections <= 0 or config.window <= 0 or config.candidate_cap <= 0:
        raise ValueError("projection configuration must be positive")
    if pair_batch_size <= 0:
        raise ValueError("pair_batch_size must be positive")
    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    leaf_states: np.ndarray | None = None
    state_tensor: torch.Tensor | None = None
    imputed_state_tensor: torch.Tensor | None = None
    site_frequency_tensor: torch.Tensor | None = None
    if site_mismatch_prior is not None and (
        imputed_leaf_states is not None or complete_leaf_states is not None
    ):
        raise ValueError("marginalized distance requires explicit observed profiles")
    if (site_frequency is None) != (site_imputation_weight is None):
        raise ValueError(
            "site_frequency and site_imputation_weight must be supplied together"
        )
    if site_frequency is not None and site_mismatch_prior is None:
        raise ValueError("coverage calibration requires site_mismatch_prior")
    if imputed_leaf_states is not None:
        if complete_leaf_states is not None:
            raise ValueError("complete and imputed compact states are mutually exclusive")
        candidate_states = np.asarray(imputed_leaf_states)
        candidate_frequency = np.asarray(imputed_site_frequency)
        if (
            candidate_states.dtype != np.uint8
            or candidate_states.ndim != 2
            or candidate_states.shape[0] != len(nodes)
            or not candidate_states.flags.c_contiguous
            or candidate_frequency.dtype != np.float32
            or candidate_frequency.shape != (candidate_states.shape[1], 4)
            or not candidate_frequency.flags.c_contiguous
            or nodes != list(range(len(nodes)))
            or prestacked_profiles is None
        ):
            raise ValueError(
                "imputed compact states require contiguous first-round uint8 states, "
                "a matching float32 site-frequency table, and the historical profile slab"
            )
        if target.type != "cuda":
            raise ValueError("imputed compact states currently require CUDA")
        leaf_states = candidate_states
        imputed_state_tensor = torch.tensor(
            leaf_states, dtype=torch.uint8, device=target
        )
        site_frequency_tensor = torch.tensor(
            candidate_frequency, dtype=torch.float32, device=target
        )
    if complete_leaf_states is not None:
        candidate_states = np.asarray(complete_leaf_states)
        if (
            candidate_states.dtype != np.uint8
            or candidate_states.ndim != 2
            or candidate_states.shape[0] != len(nodes)
            or not candidate_states.flags.c_contiguous
            or bool(np.any(candidate_states >= 4))
            or nodes != list(range(len(nodes)))
        ):
            raise ValueError(
                "complete_leaf_states must be contiguous first-round uint8 "
                "states matching the profile slab"
            )
        if target.type != "cuda":
            raise ValueError("complete_leaf_states quotient currently requires CUDA")
        leaf_states = candidate_states
        state_tensor = torch.tensor(leaf_states, dtype=torch.uint8, device=target)

    phase = perf_counter()
    raw: np.ndarray | None = None
    if state_tensor is None:
        raw = _active_profile_slab(nodes, profiles, prestacked_profiles)
        if validate_profiles and not np.allclose(raw.sum(axis=2), 1.0, atol=1e-5):
            raise ValueError("GPU exact-distance shortcut requires unit-mass profiles")
    profile_seconds = perf_counter() - phase

    phase = perf_counter()
    n_sites = leaf_states.shape[1] if leaf_states is not None else int(raw.shape[1])
    rng = np.random.default_rng(config.seed)
    projection_count = config.projections
    directions = rng.choice(
        np.asarray([-1.0, 1.0], dtype=np.float32),
        size=(n_sites * 4, projection_count),
    )
    if state_tensor is None:
        assert raw is not None
        if projection_profiles is None:
            assert raw is not None
            projected = raw.reshape(len(nodes), -1) @ directions
        elif isinstance(projection_profiles, tuple):
            projection_banks = projection_profiles
        else:
            projection_banks = (projection_profiles,)
        if projection_profiles is not None:
            if not projection_banks:
                raise ValueError("at least one projection profile bank is required")
            projected = np.concatenate(
                [
                    _active_profile_slab(nodes, bank).reshape(len(nodes), -1)
                    @ directions
                    for bank in projection_banks
                ],
                axis=1,
            )
    else:
        _sequence_backend, pattern_backend = load_backends()
        signed_directions = np.ascontiguousarray(
            directions.reshape(n_sites, 4, config.projections),
            dtype=np.int8,
        )
        direction_tensor = torch.tensor(
            signed_directions, dtype=torch.int8, device=target
        )
        projected = (
            pattern_backend.complete_state_projections_cuda(
                state_tensor, direction_tensor
            )
            .cpu()
            .numpy()
        )
    projection_seconds = perf_counter() - phase

    phase = perf_counter()
    graph_backend_name = _candidate_graph_backend_name()
    candidate_workers = _candidate_cpu_workers()
    if graph_backend_name == "native":
        graph_backend = load_candidate_graph_backend()
        pair_array = np.asarray(
            graph_backend.projection_pairs(
                np.ascontiguousarray(projected), config.window, candidate_workers
            ),
            dtype=np.int64,
        )
    else:
        pair_blocks: list[np.ndarray] = []
        for column in range(config.projections):
            order = np.argsort(projected[:, column], kind="stable")
            for offset in range(1, min(config.window, len(nodes) - 1) + 1):
                left = order[:-offset].astype(np.int64, copy=False)
                right = order[offset:].astype(np.int64, copy=False)
                pair_blocks.append(
                    np.column_stack(
                        (np.minimum(left, right), np.maximum(left, right))
                    )
                )
        pair_array = np.unique(np.concatenate(pair_blocks, axis=0), axis=0)
    pool_seconds = perf_counter() - phase

    phase = perf_counter()
    raw_tensor = None
    streamed_coverage_base_row_sums: np.ndarray | None = None
    streamed_coverage_row_seconds = 0.0
    stream_setting = os.environ.get("CONCORDTREE_STREAM_PROFILE_SITES")
    if profile_stream_sites is not None:
        stream_profile_sites = int(profile_stream_sites)
    elif stream_setting is None:
        # Keep resident execution while one active profile slab is modest, and
        # switch to the same coordinate-wise sufficient statistics before a
        # single slab can consume a material fraction of a 24--48 GiB device.
        # This is a storage-capacity boundary, not a data- or score-dependent
        # estimator route.
        stream_profile_sites = (
            2048
            if raw is not None and raw.nbytes >= (8 << 30)
            else 0
        )
    else:
        stream_profile_sites = int(stream_setting)
    if stream_profile_sites < 0:
        raise ValueError("CONCORDTREE_STREAM_PROFILE_SITES must be nonnegative")
    stream_profiles = (
        stream_profile_sites > 0
        and state_tensor is None
        and imputed_state_tensor is None
        and target.type == "cuda"
    )
    if state_tensor is None and imputed_state_tensor is None and not stream_profiles:
        assert raw is not None
        raw_tensor = (
            torch.tensor(raw, dtype=torch.float32, device=target)
            if prestacked_profiles is not None
            else torch.from_numpy(raw).to(target)
        )
    distances = np.empty(len(pair_array), dtype=np.float64)
    requested_backend = distance_backend or os.environ.get(
        "CONCORDTREE_CANDIDATE_DISTANCE_BACKEND", "eager"
    )
    if requested_backend not in {"compiled", "eager", "native"}:
        raise ValueError(
            "CONCORDTREE_CANDIDATE_DISTANCE_BACKEND must be 'compiled', 'eager', or 'native'"
        )
    distance_backend = (
        requested_backend
        if target.type == "cuda" and requested_backend in {"compiled", "native"}
        else "eager"
    )
    if stream_profiles and distance_backend != "native":
        raise ValueError("streamed profiles require the native distance backend")
    with torch.no_grad():
        if site_mismatch_prior is not None:
            prior = np.asarray(site_mismatch_prior, dtype=np.float32)
            if prior.shape != (n_sites,):
                raise ValueError("site_mismatch_prior must have one value per site")
            frequency = None
            weight = None
            if site_frequency is not None:
                frequency = np.asarray(site_frequency, dtype=np.float32)
                weight = np.asarray(site_imputation_weight, dtype=np.float32)
                if frequency.shape != (n_sites, 4) or weight.shape != (n_sites,):
                    raise ValueError("coverage-calibrated site statistics have invalid shape")
            if stream_profiles:
                if requested_backend != "native" or frequency is None or weight is None:
                    raise ValueError(
                        "streamed latent profiles require native coverage calibration"
                    )
                assert raw is not None
                _sequence_backend, pattern_backend = load_backends()
                pair_tensor = torch.from_numpy(pair_array).to(target)
                distance_numerator = torch.zeros(
                    len(pair_array), dtype=torch.float64, device=target
                )
                row_numerator = torch.zeros(
                    len(nodes), dtype=torch.float64, device=target
                )
                row_started = perf_counter()
                for site_start in range(0, n_sites, stream_profile_sites):
                    site_stop = min(n_sites, site_start + stream_profile_sites)
                    site_count = site_stop - site_start
                    profile_chunk = np.ascontiguousarray(
                        raw[:, site_start:site_stop, :], dtype=np.float32
                    )
                    chunk_tensor = torch.from_numpy(profile_chunk).to(target)
                    prior32 = torch.from_numpy(
                        np.ascontiguousarray(prior[site_start:site_stop])
                    ).to(target)
                    frequency32 = torch.from_numpy(
                        np.ascontiguousarray(frequency[site_start:site_stop])
                    ).to(target)
                    weight32 = torch.from_numpy(
                        np.ascontiguousarray(weight[site_start:site_stop])
                    ).to(target)
                    values = pattern_backend.sparse_coverage_profile_distances_cuda(
                        chunk_tensor,
                        prior32,
                        frequency32,
                        weight32,
                        pair_tensor,
                    )
                    distance_numerator.add_(
                        values.to(dtype=torch.float64), alpha=float(site_count)
                    )

                    # The coverage row statistic is affine-bilinear.  Its only
                    # cross-node sufficient statistic is the per-site profile
                    # sum, so the exact same law can be reduced independently
                    # over bounded site tiles and accumulated in float64.
                    aggregate_profile = torch.from_numpy(
                        np.sum(profile_chunk, axis=0, dtype=np.float64)
                    ).to(target)
                    prior64 = torch.as_tensor(
                        np.asarray(
                            site_mismatch_prior[site_start:site_stop],
                            dtype=np.float64,
                        ),
                        dtype=torch.float64,
                        device=target,
                    )
                    frequency64 = torch.as_tensor(
                        np.asarray(
                            site_frequency[site_start:site_stop], dtype=np.float64
                        ),
                        dtype=torch.float64,
                        device=target,
                    )
                    weight64 = torch.as_tensor(
                        np.asarray(
                            site_imputation_weight[site_start:site_stop],
                            dtype=np.float64,
                        ),
                        dtype=torch.float64,
                        device=target,
                    )
                    rows = pattern_backend.coverage_profile_row_sums_cuda(
                        chunk_tensor,
                        prior64,
                        frequency64,
                        weight64,
                        aggregate_profile,
                    )
                    row_numerator.add_(rows, alpha=float(site_count))
                    del (
                        rows,
                        aggregate_profile,
                        weight64,
                        frequency64,
                        prior64,
                        values,
                        weight32,
                        frequency32,
                        prior32,
                        chunk_tensor,
                        profile_chunk,
                    )
                distances[:] = (distance_numerator / n_sites).cpu().numpy()
                streamed_coverage_base_row_sums = (
                    row_numerator / n_sites
                ).cpu().numpy()
                streamed_coverage_row_seconds = perf_counter() - row_started
                distance_backend = "coverage-native-streamed64"
            else:
                assert raw_tensor is not None
                prior_tensor = torch.from_numpy(prior).to(target)
                frequency_tensor = None
                weight_tensor = None
                if frequency is not None:
                    assert weight is not None
                    frequency_tensor = torch.from_numpy(frequency).to(target)
                    weight_tensor = torch.from_numpy(weight).to(target)
                if requested_backend == "native":
                    _sequence_backend, pattern_backend = load_backends()
                    pair_tensor = torch.from_numpy(pair_array).to(target)
                    if frequency_tensor is None:
                        values = pattern_backend.sparse_marginalized_profile_distances_cuda(
                            raw_tensor, prior_tensor, pair_tensor
                        )
                        distance_backend = "marginalized-native"
                    else:
                        assert weight_tensor is not None
                        values = pattern_backend.sparse_coverage_profile_distances_cuda(
                            raw_tensor,
                            prior_tensor,
                            frequency_tensor,
                            weight_tensor,
                            pair_tensor,
                        )
                        distance_backend = "coverage-native"
                    distances[:] = values.cpu().numpy()
                else:
                    prior_mean = prior_tensor.mean()
                    for start in range(0, len(pair_array), pair_batch_size):
                        chunk = pair_array[start : start + pair_batch_size]
                        left = torch.from_numpy(chunk[:, 0]).to(target)
                        right = torch.from_numpy(chunk[:, 1]).to(target)
                        left_profile = raw_tensor.index_select(0, left)
                        right_profile = raw_tensor.index_select(0, right)
                        left_mass = left_profile.sum(dim=2)
                        right_mass = right_profile.sum(dim=2)
                        correction = (
                            (1.0 - prior_tensor) * left_mass * right_mass
                            - (left_profile * right_profile).sum(dim=2)
                        )
                        if frequency_tensor is not None:
                            assert weight_tensor is not None
                            residual = (1.0 - frequency_tensor) - prior_tensor[:, None]
                            left_h = (left_profile * residual).sum(dim=2)
                            right_h = (right_profile * residual).sum(dim=2)
                            correction = correction + weight_tensor * (
                                left_h * (1.0 - right_mass)
                                + right_h * (1.0 - left_mass)
                            )
                        correction = correction.sum(dim=1) / n_sites
                        distances[start : start + len(chunk)] = (
                            prior_mean + correction
                        ).cpu().numpy()
                    distance_backend = "marginalized-eager"
        elif distance_backend in {"compiled", "native"}:
            pair_tensor = torch.from_numpy(pair_array).to(target)
            if distance_backend == "compiled":
                assert raw_tensor is not None
                values = _compiled_sparse_profile_distance(
                    raw_tensor, pair_tensor[:, 0], pair_tensor[:, 1]
                )
            else:
                _sequence_backend, pattern_backend = load_backends()
                if imputed_state_tensor is not None:
                    assert site_frequency_tensor is not None
                    values = pattern_backend.sparse_imputed_state_distances_cuda(
                        imputed_state_tensor, site_frequency_tensor, pair_tensor
                    )
                elif state_tensor is None:
                    if stream_profiles:
                        assert raw is not None
                        matches = torch.zeros(
                            len(pair_array), dtype=torch.float64, device=target
                        )
                        for site_start in range(0, n_sites, stream_profile_sites):
                            site_stop = min(n_sites, site_start + stream_profile_sites)
                            profile_chunk = np.ascontiguousarray(
                                raw[:, site_start:site_stop, :], dtype=np.float32
                            )
                            chunk_tensor = torch.from_numpy(profile_chunk).to(target)
                            pattern_backend.accumulate_sparse_profile_matches_cuda(
                                chunk_tensor, pair_tensor, matches
                            )
                            del chunk_tensor, profile_chunk
                        values = 1.0 - matches / n_sites
                        distance_backend = "native-streamed64"
                    else:
                        assert raw_tensor is not None
                        values = pattern_backend.sparse_profile_distances_cuda(
                            raw_tensor, pair_tensor
                        )
                else:
                    values = pattern_backend.sparse_state_distances_cuda(
                        state_tensor, pair_tensor
                    )
            distances[:] = values.cpu().numpy()
        else:
            for start in range(0, len(pair_array), pair_batch_size):
                chunk = pair_array[start : start + pair_batch_size]
                left = torch.from_numpy(chunk[:, 0]).to(target)
                right = torch.from_numpy(chunk[:, 1]).to(target)
                if raw_tensor is None:
                    raise AssertionError(
                        "complete-state first round requires the native distance backend"
                    )
                match = (
                    raw_tensor.index_select(0, left)
                    .mul(raw_tensor.index_select(0, right))
                    .sum(dim=(1, 2))
                    / raw.shape[1]
                )
                distances[start : start + len(chunk)] = (1.0 - match).cpu().numpy()
    distance_seconds = perf_counter() - phase

    base_row_sums = None
    row_sum_timing = None
    if row_sum_backend != "cpu":
        if row_sum_backend not in {"cpu-reuse", "native", "gpu32", "gpu64"}:
            raise ValueError(
                "row_sum_backend must be 'cpu', 'cpu-reuse', 'native', 'gpu32', or 'gpu64'"
            )
        row_sum_timing = {
            "stack_cast_seconds": 0.0,
            "validation_seconds": 0.0,
            "aggregate_seconds": 0.0,
            "matvec_seconds": 0.0,
            "self_dot_seconds": 0.0,
            "finalize_seconds": 0.0,
        }
        sites = n_sites
        phase = perf_counter()
        if site_mismatch_prior is not None:
            if streamed_coverage_base_row_sums is not None:
                base_row_sums = streamed_coverage_base_row_sums
                row_sum_timing["matvec_seconds"] = streamed_coverage_row_seconds
                return ProjectionPoolBatch(
                    nodes=tuple(nodes),
                    pairs=pair_array,
                    distances=distances,
                    timing=CandidateTiming(
                        profile_seconds=profile_seconds,
                        projection_seconds=projection_seconds,
                        pool_seconds=pool_seconds,
                        distance_seconds=distance_seconds,
                        ranking_seconds=0.0,
                        unique_pool_pairs=len(pair_array),
                        distance_backend=distance_backend,
                    ),
                    base_row_sums=base_row_sums,
                    row_sum_timing=row_sum_timing,
                )
            if raw_tensor is None:
                raise ValueError(
                    "marginalized native row sums require resident profiles"
                )
            prior64 = torch.as_tensor(
                np.asarray(site_mismatch_prior, dtype=np.float64),
                dtype=torch.float64,
                device=target,
            )
            # ``torch.sum(dtype=float64)`` may materialize a full double copy
            # of the N x L x 4 slab before reducing it.  At 100K taxa that
            # transient is larger than the resident float slab itself.  NumPy
            # promotes only the reduction accumulator, yielding the same small
            # L x 4 sufficient statistic without an O(NL) double buffer.
            assert raw is not None
            aggregate_profile = torch.from_numpy(
                np.sum(raw, axis=0, dtype=np.float64)
            ).to(target)
            row_sum_timing["aggregate_seconds"] = perf_counter() - phase
            phase = perf_counter()
            _sequence_backend, pattern_backend = load_backends()
            if site_frequency is None:
                values = pattern_backend.marginalized_profile_row_sums_cuda(
                    raw_tensor, prior64, aggregate_profile
                )
            else:
                frequency64 = torch.as_tensor(
                    np.asarray(site_frequency, dtype=np.float64),
                    dtype=torch.float64,
                    device=target,
                )
                weight64 = torch.as_tensor(
                    np.asarray(site_imputation_weight, dtype=np.float64),
                    dtype=torch.float64,
                    device=target,
                )
                values = pattern_backend.coverage_profile_row_sums_cuda(
                    raw_tensor,
                    prior64,
                    frequency64,
                    weight64,
                    aggregate_profile,
                )
            base_row_sums = values.cpu().numpy()
            row_sum_timing["matvec_seconds"] = perf_counter() - phase
            return ProjectionPoolBatch(
                nodes=tuple(nodes),
                pairs=pair_array,
                distances=distances,
                timing=CandidateTiming(
                    profile_seconds=profile_seconds,
                    projection_seconds=projection_seconds,
                    pool_seconds=pool_seconds,
                    distance_seconds=distance_seconds,
                    ranking_seconds=0.0,
                    unique_pool_pairs=len(pair_array),
                    distance_backend=distance_backend,
                ),
                base_row_sums=base_row_sums,
                row_sum_timing=row_sum_timing,
            )
        if stream_profiles:
            assert raw is not None
            base_row_sums = np.asarray(
                load_candidate_graph_backend().aggregate_mismatch_base_rows(
                    raw, _candidate_cpu_workers()
                ),
                dtype=np.float64,
            )
            row_sum_timing["stack_cast_seconds"] = perf_counter() - phase
            return ProjectionPoolBatch(
                nodes=tuple(nodes), pairs=pair_array, distances=distances,
                timing=CandidateTiming(
                    profile_seconds=profile_seconds,
                    projection_seconds=projection_seconds,
                    pool_seconds=pool_seconds,
                    distance_seconds=distance_seconds,
                    ranking_seconds=0.0,
                    unique_pool_pairs=len(pair_array),
                    distance_backend=distance_backend,
                ),
                base_row_sums=base_row_sums,
                row_sum_timing=row_sum_timing,
            )
        if imputed_state_tensor is not None and row_sum_backend in {"gpu32", "gpu64"}:
            assert raw is not None
            assert site_frequency_tensor is not None
            # This sufficient statistic is identical to summing the historical
            # expanded slab, but occupies only one [sites, 4] array.
            # Missing-state sufficient statistics are accumulated in float64.
            # Later rounds retain the caller's historical backend; only this
            # compact first-round quotient needs the conservative accumulator.
            aggregate_dtype = np.float64
            aggregate_profile = np.ascontiguousarray(
                raw.sum(axis=0, dtype=aggregate_dtype), dtype=aggregate_dtype
            )
            aggregate_tensor = torch.tensor(
                aggregate_profile,
                dtype=torch.float64,
                device=target,
            )
            row_sum_timing["stack_cast_seconds"] = perf_counter() - phase
            phase = perf_counter()
            _sequence_backend, pattern_backend = load_backends()
            base_row_sums = (
                pattern_backend.imputed_state_row_sums_cuda(
                    imputed_state_tensor,
                    site_frequency_tensor,
                    aggregate_tensor,
                    True,
                )
                .to(dtype=torch.float64)
                .cpu()
                .numpy()
            )
            row_sum_timing["matvec_seconds"] = perf_counter() - phase
            return ProjectionPoolBatch(
                nodes=tuple(nodes), pairs=pair_array, distances=distances,
                timing=CandidateTiming(
                    profile_seconds=profile_seconds,
                    projection_seconds=projection_seconds,
                    pool_seconds=pool_seconds,
                    distance_seconds=distance_seconds,
                    ranking_seconds=0.0,
                    unique_pool_pairs=len(pair_array),
                    distance_backend="native-compact-imputed",
                ),
                base_row_sums=base_row_sums,
                row_sum_timing=row_sum_timing,
            )
        if state_tensor is not None and row_sum_backend in {"gpu32", "gpu64"}:
            assert leaf_states is not None
            site_counts = np.ascontiguousarray(
                np.column_stack(
                    [
                        np.count_nonzero(leaf_states == state, axis=0)
                        for state in range(4)
                    ]
                ),
                dtype=np.int64,
            )
            counts_tensor = torch.tensor(
                site_counts, dtype=torch.int64, device=target
            )
            row_sum_timing["stack_cast_seconds"] = perf_counter() - phase
            phase = perf_counter()
            _sequence_backend, pattern_backend = load_backends()
            base_row_sums = (
                pattern_backend.complete_state_row_sums_cuda(
                    state_tensor,
                    counts_tensor,
                    row_sum_backend == "gpu64",
                )
                .to(dtype=torch.float64)
                .cpu()
                .numpy()
            )
            row_sum_timing["matvec_seconds"] = perf_counter() - phase
            return ProjectionPoolBatch(
                nodes=tuple(nodes),
                pairs=pair_array,
                distances=distances,
                timing=CandidateTiming(
                    profile_seconds=profile_seconds,
                    projection_seconds=projection_seconds,
                    pool_seconds=pool_seconds,
                    distance_seconds=distance_seconds,
                    ranking_seconds=0.0,
                    unique_pool_pairs=len(pair_array),
                    distance_backend=distance_backend,
                ),
                base_row_sums=base_row_sums,
                row_sum_timing=row_sum_timing,
            )
        if row_sum_backend == "native":
            assert raw is not None
            base_row_sums = np.asarray(
                load_candidate_graph_backend().aggregate_mismatch_base_rows(
                    raw, _candidate_cpu_workers()
                ),
                dtype=np.float64,
            )
            row_sum_timing["stack_cast_seconds"] = perf_counter() - phase
            return ProjectionPoolBatch(
                nodes=tuple(nodes),
                pairs=pair_array,
                distances=distances,
                timing=CandidateTiming(
                    profile_seconds=profile_seconds,
                    projection_seconds=projection_seconds,
                    pool_seconds=pool_seconds,
                    distance_seconds=distance_seconds,
                    ranking_seconds=0.0,
                    unique_pool_pairs=len(pair_array),
                    distance_backend=distance_backend,
                ),
                base_row_sums=base_row_sums,
                row_sum_timing=row_sum_timing,
            )
        if row_sum_backend == "cpu-reuse":
            assert raw is not None
            row_raw = raw.astype(np.float64, copy=False)
            flat_rows = row_raw.reshape(len(nodes), -1)
        else:
            row_dtype = torch.float32 if row_sum_backend == "gpu32" else torch.float64
            assert raw_tensor is not None
            row_raw = raw_tensor.to(dtype=row_dtype)
            flat_rows = row_raw.reshape(len(nodes), -1)
        row_sum_timing["stack_cast_seconds"] = perf_counter() - phase

        phase = perf_counter()
        aggregate = flat_rows.sum(axis=0) if row_sum_backend == "cpu-reuse" else flat_rows.sum(dim=0)
        row_sum_timing["aggregate_seconds"] = perf_counter() - phase

        phase = perf_counter()
        base_all = len(nodes) - (flat_rows @ aggregate) / sites
        if row_sum_backend != "cpu-reuse":
            torch.cuda.synchronize(target) if target.type == "cuda" else None
        row_sum_timing["matvec_seconds"] = perf_counter() - phase

        phase = perf_counter()
        if row_sum_backend == "cpu-reuse":
            base_self = 1.0 - np.einsum("ij,ij->i", flat_rows, flat_rows) / sites
            base_row_sums = np.asarray(base_all - base_self, dtype=np.float64)
        else:
            # Express the self dot as a reduction, not a materialized
            # elementwise product.  The latter temporarily duplicates the
            # entire active profile slab and makes worker teardown dominate
            # the four-View critical path.
            base_self = 1.0 - torch.einsum("ij,ij->i", flat_rows, flat_rows) / sites
            base_row_sums = (base_all - base_self).to(dtype=torch.float64).cpu().numpy()
        row_sum_timing["self_dot_seconds"] = perf_counter() - phase

    return ProjectionPoolBatch(
        nodes=tuple(nodes),
        pairs=pair_array,
        distances=distances,
        timing=CandidateTiming(
            profile_seconds=profile_seconds,
            projection_seconds=projection_seconds,
            pool_seconds=pool_seconds,
            distance_seconds=distance_seconds,
            ranking_seconds=0.0,
            unique_pool_pairs=len(pair_array),
            distance_backend=distance_backend,
        ),
        base_row_sums=base_row_sums,
        row_sum_timing=row_sum_timing,
    )


def projection_candidate_batch_gpu(
    active: Iterable[int],
    profiles: dict[int, np.ndarray],
    config: ProjectionCandidateConfig,
    *,
    device: str | torch.device,
    tie_keys: dict[int, tuple[int, ...]] | None = None,
    pair_batch_size: int = 256,
    validate_profiles: bool = True,
) -> ProjectionCandidateBatch:
    """Return legacy candidates and exact distances selected from one pool."""

    pool = projection_pool_batch_gpu(
        active,
        profiles,
        config,
        device=device,
        pair_batch_size=pair_batch_size,
        validate_profiles=validate_profiles,
    )
    nodes = list(pool.nodes)
    pair_array = pool.pairs
    distances = pool.distances
    graph_backend_name = _candidate_graph_backend_name()
    candidate_workers = _candidate_cpu_workers()
    graph_backend = (
        load_candidate_graph_backend() if graph_backend_name == "native" else None
    )
    phase = perf_counter()
    if tie_keys is None:
        tie_keys = {node: (node,) for node in nodes}
    tie_order = sorted(range(len(nodes)), key=lambda row: tie_keys[nodes[row]])
    tie_rank = np.empty(len(nodes), dtype=np.int64)
    tie_rank[np.asarray(tie_order, dtype=np.int64)] = np.arange(
        len(nodes), dtype=np.int64
    )
    if graph_backend_name == "native":
        selected_directed = np.asarray(
            graph_backend.select_directed_pairs(
                pair_array,
                distances,
                tie_rank,
                config.candidate_cap,
                candidate_workers,
            ),
            dtype=np.int64,
        )
    else:
        owners = np.concatenate((pair_array[:, 0], pair_array[:, 1]))
        others = np.concatenate((pair_array[:, 1], pair_array[:, 0]))
        directed_distances = np.concatenate((distances, distances))
        ranking_order = np.lexsort((tie_rank[others], directed_distances, owners))
        ranked_owners = owners[ranking_order]
        _owner_rows, owner_starts, owner_counts = np.unique(
            ranked_owners, return_index=True, return_counts=True
        )
        within_owner = np.arange(len(ranking_order)) - np.repeat(
            owner_starts, owner_counts
        )
        selected_rows = ranking_order[within_owner < config.candidate_cap]
        selected_directed = np.column_stack(
            (owners[selected_rows], others[selected_rows])
        )
    result = {node: set() for node in nodes}
    for owner, other in selected_directed:
        result[nodes[int(owner)]].add(nodes[int(other)])

    selected_pairs = np.column_stack(
        (
            np.minimum(selected_directed[:, 0], selected_directed[:, 1]),
            np.maximum(selected_directed[:, 0], selected_directed[:, 1]),
        )
    )
    selected_pairs = np.unique(selected_pairs, axis=0)
    pool_keys = pair_array[:, 0] * len(nodes) + pair_array[:, 1]
    selected_keys = selected_pairs[:, 0] * len(nodes) + selected_pairs[:, 1]
    selected_pool_rows = np.searchsorted(pool_keys, selected_keys)
    if not np.array_equal(pair_array[selected_pool_rows], selected_pairs):
        raise AssertionError("selected candidate edge is absent from its projection pool")
    exact_distances = {
        canonical_node_pair(nodes[int(left)], nodes[int(right)]): float(distances[row])
        for row, (left, right) in zip(selected_pool_rows, selected_pairs)
    }
    ranking_seconds = perf_counter() - phase
    return ProjectionCandidateBatch(
        candidates=result,
        exact_distances=exact_distances,
        timing=CandidateTiming(
            profile_seconds=pool.timing.profile_seconds,
            projection_seconds=pool.timing.projection_seconds,
            pool_seconds=pool.timing.pool_seconds,
            distance_seconds=pool.timing.distance_seconds,
            ranking_seconds=ranking_seconds,
            unique_pool_pairs=pool.timing.unique_pool_pairs,
            distance_backend=pool.timing.distance_backend,
        ),
    )


def projection_order_candidates_gpu(
    active: Iterable[int],
    profiles: dict[int, np.ndarray],
    config: ProjectionCandidateConfig,
    *,
    device: str | torch.device,
    tie_keys: dict[int, tuple[int, ...]] | None = None,
    pair_batch_size: int = 256,
    validate_profiles: bool = True,
) -> tuple[dict[int, set[int]], CandidateTiming]:
    """Return the legacy API while sharing the fused batch implementation."""

    batch = projection_candidate_batch_gpu(
        active,
        profiles,
        config,
        device=device,
        tie_keys=tie_keys,
        pair_batch_size=pair_batch_size,
        validate_profiles=validate_profiles,
    )
    return batch.candidates, batch.timing
