"""Reference-free implementation of the ConcordTree inference pipeline."""

from __future__ import annotations

from collections import Counter, OrderedDict, defaultdict
import concurrent.futures as futures
from dataclasses import dataclass
from datetime import datetime, timezone
import itertools
import json
import math
import multiprocessing
import os
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import sys
import time
from typing import Any

import numpy as np
import torch

from concordtree import __version__
from concordtree.assets import (
    ASSET_SHA256,
    QUARTET_PREDICTORS,
    load_attention_pair_masks,
    load_backends,
    load_mlp,
    load_learned_nni_plan_backend,
    load_panel_score_backend,
    load_qf_bundle,
    load_split_compat_backend,
    sha256_file,
    verify_assets,
)
from concordtree._core.eapc_reachability import split_bitmasks
from concordtree._core.graphrank_laminar import (
    canonical_split,
    complete_compatible_splits,
)
from concordtree._core.learned_nni import (
    EdgeEvidence,
    apply_independent_nni,
    refine_learned_nni,
    tree_path_to_graph,
)
from concordtree._core.scaleqf import (
    _stratified_positions,
    graph_to_newick,
    load_phylip_sketch,
    tree_to_graph,
    validate_topology,
)
from concordtree._core.sctb_aggregate_nj import build_aggregate_parallel_nj
from concordtree._core.sctb_contextual_patch import (
    ContextualPanel,
    SparseContextualPanelPlan,
    branch_nodes_for_edge,
    build_contextual_panel_covers,
    build_contextual_tree_index,
    compile_sparse_contextual_panel_plan,
    consensus_evidence,
    internal_edges,
    score_contextual_panel_edges,
    score_sparse_contextual_panel_plan,
)
from concordtree._core.sctb_oracle_nni import normalized_split_distance
from concordtree._core.splitbank import (
    LaminarSplitSelector,
    split_set_to_tree_indexed,
)
from concordtree.sparse_attention import compress_block_layout, define_historical_quartet_tail
DEFAULT_VIEW_COUNT = 4
MAX_VIEW_COUNT = 8
MAX_CONCURRENT_VIEW_WORKERS = 4
# Internal compatibility alias: historical callers importing VIEW_COUNT mean
# the frozen default, not the maximum accepted evidence budget.
VIEW_COUNT = DEFAULT_VIEW_COUNT
MAX_SITES = 16_384
DEFAULT_VIEW_STOP_RATIO = 0.01
DEFAULT_VIEW_MAX_ROUNDS = 24
DEFAULT_COORDINATE_STOP_RATIO = 0.005
DEFAULT_COORDINATE_MAX_ROUNDS = 4
DEFAULT_SATURATION_STOP_RATIO = 0.005
DEFAULT_SATURATION_MAX_ROUNDS = 5
# A threshold-only run has no user-visible round budget.  This guard detects a
# pathological non-converging/cycling execution and raises instead of silently
# returning a budget-truncated tree.
REFINEMENT_EMERGENCY_MAX_ROUNDS = 10_000
# Compatibility aliases for internal callers from the pre-control release.
VIEW_MAX_PASSES = DEFAULT_VIEW_MAX_ROUNDS
COORDINATE_EXTRA_ROUNDS = DEFAULT_COORDINATE_MAX_ROUNDS - 1
SATURATING_EXTRA_ROUNDS = DEFAULT_SATURATION_MAX_ROUNDS - 1
PROJECTIONS = 16
WINDOW = 8
CANDIDATE_CAP = 32
NNI_MARGIN = 0.10
SCORER_BATCH_SIZE = 4
FAST_SCORER_BATCH_SIZE = 16
PANEL_CACHE_MAX_ENTRIES = 2048
MISSING_DATA_MODELS = {
    "standard": "imputed",
    "coverage-aware": "coverage",
}
QF_AUTOCAST_DTYPE: torch.dtype | None = torch.float16
POST_VIEW_SCHEDULE = "views-then-splitbank"
FORK_POST_VIEW_SCHEDULE = "views-then-splitbank"
SEEDS = tuple(
    (20260903 + 1009 * view, 20260934 + 1013 * view)
    for view in range(MAX_VIEW_COUNT)
)
LOCAL_TEMPLATE = np.asarray(
    list(itertools.combinations(range(24), 4)), dtype=np.int32
)
LOCAL_ROW_INDEX = np.full((24, 24, 24, 24), -1, dtype=np.int32)
for _row, _quartet in enumerate(LOCAL_TEMPLATE):
    for _permutation in itertools.permutations(int(value) for value in _quartet):
        LOCAL_ROW_INDEX[_permutation] = _row
if int(np.count_nonzero(LOCAL_ROW_INDEX >= 0)) != 24 * len(LOCAL_TEMPLATE):
    raise AssertionError("fixed quartet row index is incomplete")

ScorerBundle = tuple[
    Any,
    Any,
    torch.nn.Module,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]


@dataclass(frozen=True)
class ForkSharedViewInput:
    """Read-only alignment state inherited by pre-CUDA workers."""

    input_sha256: str
    msa: Path
    packed: np.ndarray
    raw_names: tuple[str, ...]
    variable_length: int
    scaffold_names: tuple[str, ...]
    scaffold_states: np.ndarray
    leaf_profile_slab: np.ndarray | None
    view_scaffold_states: tuple[np.ndarray, ...] = ()


def _fork_shared_input_nbytes(shared: ForkSharedViewInput) -> int:
    """Count physically distinct inherited arrays once."""

    arrays = [shared.packed, shared.scaffold_states, *shared.view_scaffold_states]
    if shared.leaf_profile_slab is not None:
        arrays.append(shared.leaf_profile_slab)
    unique = {id(array): array for array in arrays}
    return sum(int(array.nbytes) for array in unique.values())


# ``ProcessPoolExecutor(mp_context='fork')`` snapshots this object without
# pickle/copy traffic.  It is populated only before CUDA initialization and is
# never mutated by either parent or children.
_FORK_SHARED_VIEW_INPUT: ForkSharedViewInput | None = None

@dataclass(frozen=True)
class PostViewContext:
    """Immutable evidence and runtime state shared by all post-view operators."""

    inference_mode: str
    sequence_backend: Any
    pattern_backend: Any
    model: torch.nn.Module
    coeff: torch.Tensor | None
    active_block_indices: torch.Tensor | None
    active_block_counts: torch.Tensor | None
    attention_pair_masks: torch.Tensor | None
    quartet_matrix: torch.Tensor | None
    species: torch.Tensor | None
    sequences: torch.Tensor
    names: tuple[str, ...]
    effective_length: int
    view_tree_paths: tuple[Path, ...]
    view_graphs: tuple[dict[int, set[int]], ...]
    view_splits: tuple[frozenset[int], ...]
    view_counts: dict[int, int]
    view_count: int
    medoid_index: int
    device: torch.device
    local_template_device: torch.Tensor
    probability_cache: OrderedDict[tuple[int, ...], np.ndarray]
    panel_score_backend: Any | None = None
    split_compat_backend: Any | None = None


@dataclass
class QFExecutionTrace:
    """Opt-in CUDA/host timing for regular fixed-width scorer batches."""

    batches: int = 0
    requested_panels: int = 0
    panels: int = 0
    cache_hits: int = 0
    quartets: int = 0
    quartet_index_seconds: float = 0.0
    pattern_cuda_ms: float = 0.0
    model_cuda_ms: float = 0.0
    output_to_host_seconds: float = 0.0
    plan_compile_seconds: float = 0.0
    probability_seconds: float = 0.0
    score_reduce_seconds: float = 0.0

    def as_dict(self) -> dict[str, int | float]:
        return {
            "batches": self.batches,
            "requested_panels": self.requested_panels,
            "panels": self.panels,
            "cache_hits": self.cache_hits,
            "quartets": self.quartets,
            "quartet_index_seconds": self.quartet_index_seconds,
            "pattern_cuda_ms": self.pattern_cuda_ms,
            "model_cuda_ms": self.model_cuda_ms,
            "output_to_host_seconds": self.output_to_host_seconds,
            "plan_compile_seconds": self.plan_compile_seconds,
            "probability_seconds": self.probability_seconds,
            "score_reduce_seconds": self.score_reduce_seconds,
        }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def read_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        fields = handle.readline().split()
    if len(fields) != 2:
        raise ValueError("invalid sequential PHYLIP header; expected '<taxa> <sites>'")
    try:
        n_taxa, length = map(int, fields)
    except ValueError as error:
        raise ValueError("PHYLIP dimensions must be integers") from error
    if n_taxa < 24:
        raise ValueError("ConcordTree requires at least 24 taxa")
    if length < 1:
        raise ValueError("alignment length must be positive")
    return n_taxa, length


def stop_moves_for_ratio(n_taxa: int, ratio: float | None) -> int | None:
    """Convert a normalized accepted-move threshold to an exact move count."""

    if ratio is None:
        return None
    if not math.isfinite(ratio) or not 0.0 <= ratio <= 1.0:
        raise ValueError("stop ratio must lie in [0, 1] or be None")
    return math.floor(ratio * (n_taxa - 3))


def coordinate_stop_moves(n_taxa: int) -> int:
    """Return the historical default post-View continuation tolerance."""

    result = stop_moves_for_ratio(n_taxa, DEFAULT_COORDINATE_STOP_RATIO)
    assert result is not None
    return result


def validate_refinement_control(
    stage: str,
    stop_ratio: float | None,
    max_rounds: int | None,
) -> None:
    """Validate one independent convergence/budget control pair."""

    stop_moves_for_ratio(4, stop_ratio)
    if max_rounds is not None and max_rounds < 1:
        raise ValueError(f"{stage}_max_rounds must be positive or None")
    if stop_ratio is None and max_rounds is None:
        raise ValueError(
            f"{stage} refinement requires a stop ratio, a maximum round count, or both"
        )


def refinement_stop_reasons(
    moves: int,
    round_index: int,
    n_taxa: int,
    stop_ratio: float | None,
    max_rounds: int | None,
) -> tuple[str, ...]:
    """Return every stopping condition satisfied by a completed pass."""

    reasons: list[str] = []
    if moves == 0:
        reasons.append("no_moves")
    threshold = stop_moves_for_ratio(n_taxa, stop_ratio)
    if threshold is not None and moves <= threshold:
        reasons.append("threshold")
    if max_rounds is not None and round_index >= max_rounds:
        reasons.append("max_rounds")
    return tuple(reasons)


def resolve_candidate_distance_backend(n_taxa: int, inference_mode: str) -> str:
    """Choose the scale-qualified candidate-distance execution schedule."""

    requested = os.environ.get("CONCORDTREE_CANDIDATE_DISTANCE_BACKEND")
    if requested is not None:
        if requested not in {"compiled", "eager", "native"}:
            raise ValueError(
                "CONCORDTREE_CANDIDATE_DISTANCE_BACKEND must be compiled, eager, or native"
            )
        return requested
    return "native" if inference_mode == "fast" and n_taxa >= 4096 else "eager"


def resolve_scaffold_row_sum_backend(
    n_taxa: int,
    inference_mode: str,
) -> str:
    """Place the scaffold statistic without changing the topology estimator."""

    requested = os.environ.get("CONCORDTREE_SCAFFOLD_ROW_SUM_BACKEND")
    if requested is not None:
        requested = requested.strip().lower()
        if requested not in {"cpu", "cpu-reuse", "native", "gpu32", "gpu64"}:
            raise ValueError(
                "CONCORDTREE_SCAFFOLD_ROW_SUM_BACKEND must be cpu, cpu-reuse, native, gpu32, or gpu64"
            )
        return requested
    if n_taxa >= 32768:
        # Ultra-scale Views stream profile coordinates in bounded CUDA tiles.
        # Accumulate their row statistic in the native float64 reduction as
        # well, so the memory-saving path preserves the mathematical profile
        # sum instead of introducing a scale-amplified float32 rounding lane.
        return "native"
    if n_taxa >= 4096:
        # The native complete-state distance quotient does not materialize the
        # float profile slab required by the host-native row reducer.  The
        # established GPU row statistic consumes the same complete states;
        # Fast uses the lower-cost statistic; Transformer uses the numerically
        # conservative lane. Both compute the same aggregate identity.
        return "gpu32" if inference_mode == "fast" else "gpu64"
    return "native" if inference_mode == "fast" and n_taxa >= 4096 else "cpu"


def resolve_nni_reduction_backend(n_taxa: int, inference_mode: str) -> str:
    """Select fused exact NNI evidence reduction only for large-N Fast runs."""

    requested = os.environ.get("CONCORDTREE_NNI_REDUCTION_BACKEND")
    if requested is not None:
        requested = requested.strip().lower()
        if requested not in {"python", "native"}:
            raise ValueError("CONCORDTREE_NNI_REDUCTION_BACKEND must be python or native")
        return requested
    return "native" if inference_mode == "fast" and n_taxa >= 4096 else "python"


def resolve_missing_distance_model(missing_data_model: str) -> str:
    """Translate one public missing-data assumption to its scaffold law."""

    try:
        return MISSING_DATA_MODELS[missing_data_model]
    except KeyError as error:
        choices = ", ".join(MISSING_DATA_MODELS)
        raise ValueError(f"missing_data_model must be one of: {choices}") from error


def resolve_view_executor(n_taxa: int, inference_mode: str) -> str:
    """Choose one shared-context executor for the single-GPU release target."""

    requested = os.environ.get("CONCORDTREE_VIEW_EXECUTOR")
    if requested is not None:
        requested = requested.strip().lower()
        if requested not in {"fork", "subprocess", "thread"}:
            raise ValueError(
                "CONCORDTREE_VIEW_EXECUTOR must be 'fork', 'subprocess', or 'thread'"
            )
        if requested == "fork" and not sys.platform.startswith("linux"):
            raise ValueError("the fork View executor is supported only on Linux")
        if requested == "fork" and torch.cuda.is_initialized():
            raise RuntimeError(
                "fork View execution must begin before parent CUDA initialization"
            )
        return requested
    return "thread"


def resolve_view_workers(
    n_taxa: int,
    requested: int,
    view_count: int = DEFAULT_VIEW_COUNT,
) -> int:
    """Choose bounded single-GPU concurrency independently of View budget."""

    if requested < 0 or requested > MAX_CONCURRENT_VIEW_WORKERS:
        raise ValueError("view-workers must be between 0 (auto) and 4")
    if view_count < 2 or view_count > MAX_VIEW_COUNT:
        raise ValueError("view-count must be between 2 and 8")
    if requested > 0:
        return min(requested, view_count)
    # At 50K, two gap-heavy View workers can each retain about 21 GiB.  The
    # conservative large-N lane serializes them; users may explicitly raise
    # concurrency after checking their own device and alignment.
    return (
        1
        if n_taxa >= 32768
        else min(2, view_count)
    )


def _backend_label(inference_mode: str) -> str:
    if inference_mode == "transformer":
        return "qf"
    if inference_mode == "fast":
        return "mlp"
    raise ValueError(f"unknown inference mode: {inference_mode}")


def _load_post_view_bundle(
    inference_mode: str,
    device: torch.device,
    quartet_predictor: str = "heterogeneous",
) -> ScorerBundle:
    """Load exactly the scorer assets needed by one release mode."""

    if inference_mode == "transformer":
        return load_qf_bundle(device, quartet_predictor)
    if inference_mode == "fast":
        sequence_backend, pattern_backend = load_backends()
        return (
            sequence_backend,
            pattern_backend,
            load_mlp(device, quartet_predictor),
            None,
            None,
            None,
        )
    raise ValueError(f"unknown inference mode: {inference_mode}")


def _relative(path: Path, work_dir: Path) -> str:
    return str(path.resolve().relative_to(work_dir.resolve()))


def _prediction_medoid(split_sets: list[frozenset[int]]) -> int:
    ranked: list[tuple[float, int]] = []
    for index, current in enumerate(split_sets):
        total = sum(
            normalized_split_distance(current, other)
            for other_index, other in enumerate(split_sets)
            if other_index != index
        )
        ranked.append((total, index))
    return min(ranked)[1]


def _clone(graph: dict[int, set[int]]) -> dict[int, set[int]]:
    return {node: set(neighbors) for node, neighbors in graph.items()}


def _postorder_graph_handoff(
    graph: dict[int, set[int]], n_taxa: int
) -> dict[int, set[int]]:
    """Reproduce the internal ids assigned by the staged Newick roundtrip.

    ``graph_to_newick`` roots at the smallest internal id and emits children in
    sorted id order.  ETE then assigns internal ids in that emitted postorder.
    Keeping this finite relabel explicit preserves the historical tie-breaks
    without reparsing the Newick file that remains the durable stage artifact.
    """

    degree_two = sorted(
        node for node in graph if node >= n_taxa and len(graph[node]) == 2
    )
    internal = sorted(node for node in graph if node >= n_taxa)
    if not internal:
        raise ValueError("Topology has no internal node")
    root = degree_two[0] if degree_two else internal[0]
    postorder: list[int] = []
    visited: set[int] = set()
    stack: list[tuple[int, int | None, bool]] = [(root, None, False)]
    while stack:
        node, parent, closing = stack.pop()
        if closing:
            if node >= n_taxa:
                postorder.append(node)
            continue
        if node in visited:
            raise ValueError("adjacency is not a tree")
        visited.add(node)
        stack.append((node, parent, True))
        for child in sorted(
            (neighbor for neighbor in graph[node] if neighbor != parent),
            reverse=True,
        ):
            stack.append((child, node, False))
    if len(visited) != len(graph):
        raise ValueError("adjacency is disconnected")
    mapping = {
        node: n_taxa + position for position, node in enumerate(postorder)
    }
    relabeled = {
        mapping.get(node, node): {
            mapping.get(neighbor, neighbor) for neighbor in neighbors
        }
        for node, neighbors in graph.items()
    }
    # Match tree_path_to_graph's suppression of a degree-two parsed root.
    roots = sorted(
        node
        for node, neighbors in relabeled.items()
        if node >= n_taxa and len(neighbors) == 2
    )
    for node in roots:
        left, right = sorted(relabeled[node])
        relabeled[left].remove(node)
        relabeled[right].remove(node)
        relabeled[left].add(right)
        relabeled[right].add(left)
        del relabeled[node]
    return relabeled


def _uncertain_edges(
    graph: dict[int, set[int]],
    n_taxa: int,
    view_counts: dict[int, int],
    view_count: int,
) -> set[tuple[int, int]]:
    return {
        edge
        for edge, split in _internal_edge_splits(graph, n_taxa).items()
        if view_counts.get(split, 0) < view_count
    }


def _internal_edge_splits(
    graph: dict[int, set[int]], n_taxa: int
) -> dict[tuple[int, int], int]:
    """Return exact canonical splits without materializing both edge sides.

    The historical implementation built a directed message for both
    orientations of every edge.  Those two masks are complements, so on a
    large tree they contain exactly ``n_taxa`` bits per edge even though only
    one canonical split is consumed.  A single rooted postorder is sufficient:
    every undirected edge has one child-side subtree mask and its complement is
    obtained transiently from the tree-wide mask.  This is the same split
    algebra with no hash approximation or estimator change.
    """

    if not graph:
        return {}
    internals = [node for node in graph if node >= n_taxa]
    root = min(internals) if internals else min(graph)
    parent: dict[int, int | None] = {root: None}
    order = [root]
    for node in order:
        for neighbor in sorted(graph[node]):
            if neighbor == parent[node]:
                continue
            if neighbor in parent:
                raise ValueError("topology contains a cycle")
            parent[neighbor] = node
            order.append(neighbor)
    if len(order) != len(graph):
        raise ValueError("topology is disconnected")

    subtree: dict[int, int] = {}
    for node in reversed(order):
        mask = (1 << node) if node < n_taxa else 0
        for neighbor in graph[node]:
            if parent.get(neighbor) == node:
                mask |= subtree[neighbor]
        subtree[node] = mask
    total = subtree[root]
    leaf_count = total.bit_count()
    if leaf_count != n_taxa:
        raise ValueError(
            f"topology contains {leaf_count} leaves, expected {n_taxa}"
        )

    output: dict[tuple[int, int], int] = {}
    for node in order[1:]:
        ancestor = parent[node]
        assert ancestor is not None
        if node < n_taxa or ancestor < n_taxa:
            continue
        side = subtree[node]
        other = total ^ side
        side_count = side.bit_count()
        other_count = leaf_count - side_count
        if min(side_count, other_count) < 2:
            raise AssertionError("internal edge produced a trivial split")
        if side_count < other_count:
            canonical = side
        elif other_count < side_count:
            canonical = other
        else:
            canonical = min(side, other)
        edge = (node, ancestor) if node < ancestor else (ancestor, node)
        output[edge] = canonical
    return output


def _edge_splits(
    graph: dict[int, set[int]], n_taxa: int
) -> dict[tuple[int, int], int]:
    return _internal_edge_splits(graph, n_taxa)


def _decode_splitbank(
    counts: Counter[int],
    margins: dict[int, list[float]],
    anchor: frozenset[int],
    n_taxa: int,
    view_count: int = DEFAULT_VIEW_COUNT,
    selector_backend: Any | None = None,
) -> tuple[frozenset[int], dict[str, int]]:
    target = n_taxa - 3
    unanimous = sorted(
        split for split, count in counts.items() if count == view_count
    )
    selector = LaminarSplitSelector(n_taxa, target, backend=selector_backend)
    if len(selector.extend(unanimous)) != len(unanimous):
        raise AssertionError("unanimous split set is incompatible")
    minority: list[tuple[bool, int, float, float, int]] = []
    for split, count in counts.items():
        if count == view_count:
            continue
        values = margins.get(split)
        if values is None or len(values) != 2 * count:
            raise AssertionError(f"missing two-cover evidence for split count {count}")
        minority.append(
            (
                all(value >= 0.0 for value in values),
                count,
                float(np.median(values)),
                float(np.mean(values)),
                split,
            )
        )
    ranked_minority = [
        split
        for _stable, _count, _median, _mean, split in sorted(
        minority,
        key=lambda item: (-int(item[0]), -item[1], -item[2], -item[3], item[4]),
        )
    ]
    selector.extend(ranked_minority)
    bank_selected = len(selector.splits)
    selector.extend(sorted(anchor))
    selected = selector.splits
    selected_before_completion = len(selected)
    completed = complete_compatible_splits(
        selected,
        n_taxa,
        backend=selector_backend,
    )
    return completed, {
        "unanimous_locked": sum(
            count == view_count for count in counts.values()
        ),
        "model_bank_selected": bank_selected,
        "anchor_selected": selected_before_completion - bank_selected,
        "deterministic_completion": target - selected_before_completion,
    }


def build_view(
    msa: Path,
    target: Path,
    view: int,
    device_name: str,
    candidate_distance_backend: str = "eager",
    return_topology: bool = False,
    input_sha256: str | None = None,
    row_sum_backend: str | None = None,
    nni_reduction_backend: str | None = None,
    quartet_predictor: str = "heterogeneous",
    missing_distance_model: str = "imputed",
    view_stop_ratio: float | None = DEFAULT_VIEW_STOP_RATIO,
    view_max_rounds: int | None = DEFAULT_VIEW_MAX_ROUNDS,
) -> Any:
    """Build one frozen high-site r4 view in an isolated process."""

    entry_started = time.perf_counter()
    tree_path = target / "tree.nwk"
    metrics_path = target / "metrics.json"
    input_hash_started = time.perf_counter()
    if input_sha256 is None:
        input_sha256 = sha256_file(msa)
        input_hash_seconds = time.perf_counter() - input_hash_started
    else:
        input_hash_seconds = 0.0
    if candidate_distance_backend not in {"compiled", "eager", "native"}:
        raise ValueError(
            "candidate distance backend must be compiled, eager, or native"
        )
    dimensions_started = time.perf_counter()
    n_taxa, alignment_length = read_dimensions(msa)
    dimensions_seconds = time.perf_counter() - dimensions_started
    if row_sum_backend is None:
        row_sum_backend = resolve_scaffold_row_sum_backend(
            n_taxa,
            "fast" if candidate_distance_backend in {"compiled", "native"} else "transformer",
        )
    if row_sum_backend not in {"cpu", "cpu-reuse", "native", "gpu32", "gpu64"}:
        raise ValueError(
            "CONCORDTREE_SCAFFOLD_ROW_SUM_BACKEND must be cpu, cpu-reuse, native, gpu32, or gpu64"
        )
    if nni_reduction_backend is None:
        nni_reduction_backend = resolve_nni_reduction_backend(
            n_taxa,
            "fast" if candidate_distance_backend in {"compiled", "native"} else "transformer",
        )
    if nni_reduction_backend not in {"python", "native"}:
        raise ValueError("CONCORDTREE_NNI_REDUCTION_BACKEND must be python or native")
    if quartet_predictor not in QUARTET_PREDICTORS:
        raise ValueError(
            f"quartet_predictor must be one of {QUARTET_PREDICTORS}"
        )
    if missing_distance_model not in set(MISSING_DATA_MODELS.values()):
        raise ValueError("missing_distance_model must be imputed or coverage")
    validate_refinement_control("view", view_stop_ratio, view_max_rounds)
    if tree_path.is_file() and metrics_path.is_file():
        previous = json.loads(metrics_path.read_text())
        if previous.get("input_sha256") != input_sha256:
            raise ValueError(f"view work directory belongs to a different MSA: {target}")
        if previous.get("candidate_distance_backend", "eager") != candidate_distance_backend:
            raise ValueError(f"view work directory uses a different distance backend: {target}")
        if previous.get("scaffold_row_sum_backend", "cpu") != row_sum_backend:
            raise ValueError(f"view work directory uses a different row-sum backend: {target}")
        if previous.get("nni_reduction_backend", "python") != nni_reduction_backend:
            raise ValueError(f"view work directory uses a different NNI reduction backend: {target}")
        if previous.get("quartet_predictor", "heterogeneous") != quartet_predictor:
            raise ValueError(
                f"view work directory uses a different quartet predictor: {target}"
            )
        if previous.get("missing_distance_model", "imputed") != missing_distance_model:
            raise ValueError(
                f"view work directory uses a different missing-distance model: {target}"
            )
        if previous.get("stopping_semantics") != "per-round-or-v1":
            raise ValueError(
                f"view work directory uses legacy stopping semantics: {target}"
            )
        if previous.get("view_stop_ratio", DEFAULT_VIEW_STOP_RATIO) != view_stop_ratio:
            raise ValueError(
                f"view work directory uses a different View stop ratio: {target}"
            )
        if previous.get("view_max_rounds", DEFAULT_VIEW_MAX_ROUNDS) != view_max_rounds:
            raise ValueError(
                f"view work directory uses a different View round budget: {target}"
            )
        return {**previous, "resumed": True, "tree": str(tree_path.resolve())}

    target.mkdir(parents=True, exist_ok=True)
    sketch_seed, projection_seed = SEEDS[view]
    device = torch.device(device_name)
    if device.type != "cuda":
        raise RuntimeError("ConcordTree requires a CUDA device")
    # ``set_device`` below is the authoritative availability check.  Calling
    # ``torch.cuda.is_available`` in all four freshly forked workers first
    # serializes redundant driver probes on a single GPU.
    cuda_init_started = time.perf_counter()
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    cuda_init_seconds = time.perf_counter() - cuda_init_started
    started = time.perf_counter()

    scan_started = time.perf_counter()
    sequence_backend, pattern_backend = load_backends()
    packed: np.ndarray | None = None
    raw_names: list[str] | None = None
    variable_length: int | None = None
    immutable_leaf_profile_slab: np.ndarray | None = None
    fork_shared_input = False
    scaffold_input_backend = "python-stratified"
    shared_input = _FORK_SHARED_VIEW_INPUT
    if (
        shared_input is not None
        and shared_input.input_sha256 == input_sha256
        and shared_input.msa == msa.resolve()
    ):
        packed = shared_input.packed
        raw_names = list(shared_input.raw_names)
        variable_length = shared_input.variable_length
        scaffold_names = list(shared_input.scaffold_names)
        if shared_input.view_scaffold_states:
            if view >= len(shared_input.view_scaffold_states):
                raise ValueError("fork-shared input does not contain the requested View")
            scaffold_states = shared_input.view_scaffold_states[view]
            immutable_leaf_profile_slab = None
            scaffold_input_backend = "fork-shared-native-multiview-stratified"
        else:
            scaffold_states = shared_input.scaffold_states
            immutable_leaf_profile_slab = shared_input.leaf_profile_slab
            scaffold_input_backend = (
                "fork-shared-native-packed"
                if bool(np.all(scaffold_states < 4))
                else "fork-shared-native-packed-imputed"
            )
        fork_shared_input = True
    elif alignment_length <= MAX_SITES:
        packed_raw, loaded_names, loaded_length = (
            sequence_backend.load_phy_to_packed_tensor(str(msa), True)
        )
        complete_states = np.asarray(
            sequence_backend.unpack_packed_tensor(packed_raw, int(loaded_length)),
            dtype=np.uint8,
        )
        # When every alignment column was eligible for retention and the
        # effective matrix has no missing cells, both readers discard exactly
        # the same conserved sites.  Reuse this native parse for the scaffold.
        if bool(np.all(complete_states < 4)):
            packed = packed_raw
            raw_names = list(loaded_names)
            variable_length = int(loaded_length)
            order = sorted(range(n_taxa), key=lambda position: raw_names[position])
            scaffold_names = [raw_names[position] for position in order]
            scaffold_states = complete_states[order]
            scaffold_input_backend = "native-packed-reuse"
        else:
            sketch = load_phylip_sketch(
                msa, max_sites=MAX_SITES, seed=sketch_seed
            )
            order = sorted(
                range(sketch.n_taxa), key=lambda position: sketch.names[position]
            )
            scaffold_names = [sketch.names[position] for position in order]
            scaffold_states = sketch.states[order]
    else:
        sketch = load_phylip_sketch(msa, max_sites=MAX_SITES, seed=sketch_seed)
        order = sorted(
            range(sketch.n_taxa), key=lambda position: sketch.names[position]
        )
        scaffold_names = [sketch.names[position] for position in order]
        scaffold_states = sketch.states[order]
    scan_seconds = time.perf_counter() - scan_started

    stream_setting = os.environ.get("CONCORDTREE_STREAM_PROFILE_SITES")
    if stream_setting is None:
        expanded_profile_bytes = int(scaffold_states.size) * 4 * 4
        profile_stream_sites = 2048 if expanded_profile_bytes >= (8 << 30) else 0
    else:
        profile_stream_sites = int(stream_setting)
    if profile_stream_sites < 0:
        raise ValueError("CONCORDTREE_STREAM_PROFILE_SITES must be nonnegative")

    scaffold_started = time.perf_counter()
    scaffold, scaffold_stats = build_aggregate_parallel_nj(
        scaffold_states,
        projections=PROJECTIONS,
        window=WINDOW,
        candidate_cap=CANDIDATE_CAP,
        seed=projection_seed,
        candidate_device=device_name,
        candidate_distance_backend=candidate_distance_backend,
        row_sum_backend=row_sum_backend,
        immutable_complete_slab=immutable_leaf_profile_slab,
        compact_imputed_first_round=(
            os.environ.get("CONCORDTREE_COMPACT_IMPUTED_FIRST_ROUND", "1") == "1"
        ),
        profile_stream_sites=profile_stream_sites,
        missing_distance_model=missing_distance_model,
    )
    scaffold_seconds = time.perf_counter() - scaffold_started
    scaffold_path = target / "scaffold.nwk"
    scaffold_path.write_text(
        graph_to_newick(scaffold, scaffold_names, n_taxa) + "\n"
    )

    packed_started = time.perf_counter()
    if packed is None or raw_names is None or variable_length is None:
        packed_raw, loaded_names, loaded_length = (
            sequence_backend.load_phy_to_packed_tensor(str(msa), True)
        )
        packed = packed_raw
        raw_names = list(loaded_names)
        variable_length = int(loaded_length)
    names = list(raw_names)
    sequences = torch.from_numpy(packed).to(device)
    model = load_mlp(device, quartet_predictor)
    nni_plan_backend = load_learned_nni_plan_backend()
    packed_seconds = time.perf_counter() - packed_started
    adjacency = tree_path_to_graph(scaffold_path, names)

    def predict(quartets: np.ndarray) -> np.ndarray:
        indices = torch.as_tensor(quartets, dtype=torch.long, device=device)
        features = pattern_backend.compute_pattern_frequencies_cuda_packed(
            sequences, indices, int(variable_length)
        ).contiguous()
        with torch.no_grad():
            return torch.softmax(model(features), dim=-1).cpu().numpy()

    nni_started = time.perf_counter()
    stop_moves = stop_moves_for_ratio(n_taxa, view_stop_ratio)
    pass_limit = (
        view_max_rounds
        if view_max_rounds is not None
        else REFINEMENT_EMERGENCY_MAX_ROUNDS
    )
    repair = refine_learned_nni(
        adjacency,
        len(names),
        predict,
        passes=pass_limit,
        representatives=4,
        min_margin=NNI_MARGIN,
        stop_moves_at_most=stop_moves,
        plan_backend=nni_plan_backend,
        compact_reduction=nni_reduction_backend == "native",
    )
    executed_passes = len(repair["passes"])
    last_moves = (
        int(repair["passes"][-1]["moves"])
        if repair["passes"]
        else 0
    )
    stop_reasons = refinement_stop_reasons(
        last_moves,
        executed_passes,
        n_taxa,
        view_stop_ratio,
        view_max_rounds,
    )
    if not stop_reasons and executed_passes >= REFINEMENT_EMERGENCY_MAX_ROUNDS:
        raise RuntimeError(
            "View refinement failed to reach its threshold before the internal "
            "non-convergence guard"
        )
    nni_seconds = time.perf_counter() - nni_started
    validate_topology(adjacency, n_taxa)
    tree_path.write_text(graph_to_newick(adjacency, names, n_taxa) + "\n")
    metrics: dict[str, Any] = {
        "status": "complete",
        "view": view,
        "tree": str(tree_path.resolve()),
        "n_taxa": n_taxa,
        "alignment_length": alignment_length,
        "input_sha256": input_sha256,
        "preflight_seconds": started - entry_started,
        "input_hash_seconds": input_hash_seconds,
        "dimensions_seconds": dimensions_seconds,
        "cuda_init_seconds": cuda_init_seconds,
        "sketch_seed": sketch_seed,
        "projection_seed": projection_seed,
        "candidate_distance_backend": candidate_distance_backend,
        "scaffold_row_sum_backend": row_sum_backend,
        "missing_distance_model": missing_distance_model,
        "compact_imputed_first_round": (
            os.environ.get("CONCORDTREE_COMPACT_IMPUTED_FIRST_ROUND", "1") == "1"
        ),
        "profile_stream_policy": (
            f"fixed:{profile_stream_sites}"
            if stream_setting is not None
            else f"auto:{profile_stream_sites}-sites-at-8GiB"
        ),
        "stop_moves": stop_moves,
        "view_stop_ratio": view_stop_ratio,
        "view_max_rounds": view_max_rounds,
        "stopping_semantics": "per-round-or-v1",
        "retained_variable_sites": int(scaffold_states.shape[1]),
        "effective_variable_sites": int(variable_length),
        "scan_seconds": scan_seconds,
        "scaffold_input_backend": scaffold_input_backend,
        "fork_shared_input": fork_shared_input,
        "scaffold_seconds": scaffold_seconds,
        "packed_seconds": packed_seconds,
        "nni_seconds": nni_seconds,
        "nni_plan_backend": "native-exact",
        "nni_reduction_backend": nni_reduction_backend,
        "quartet_predictor": quartet_predictor,
        "total_seconds": time.perf_counter() - started,
        "nni_passes_executed": executed_passes,
        "nni_moves": int(repair["moves"]),
        "nni_last_moves": last_moves,
        "nni_last_move_ratio": last_moves / (n_taxa - 3),
        "nni_stop_reasons": list(stop_reasons),
        "nni_timing": repair.get("timing", {}),
        "scaffold_rounds": scaffold_stats.rounds,
        "scaffold_streamed_candidate_rounds": scaffold_stats.streamed_candidate_rounds,
        "scaffold_candidate_pairs": scaffold_stats.candidate_pairs,
        "scaffold_candidate_pool_pairs": scaffold_stats.candidate_pool_pairs,
        "scaffold_reused_candidate_distances": scaffold_stats.reused_candidate_distances,
        "scaffold_candidate_seconds": scaffold_stats.candidate_seconds,
        "scaffold_candidate_profile_seconds": scaffold_stats.candidate_profile_seconds,
        "scaffold_candidate_projection_seconds": scaffold_stats.candidate_projection_seconds,
        "scaffold_candidate_pool_seconds": scaffold_stats.candidate_pool_seconds,
        "scaffold_candidate_distance_seconds": scaffold_stats.candidate_distance_seconds,
        "scaffold_candidate_distance_round_seconds": scaffold_stats.candidate_distance_round_seconds,
        "scaffold_candidate_pool_round_pairs": scaffold_stats.candidate_pool_round_pairs,
        "scaffold_candidate_ranking_seconds": scaffold_stats.candidate_ranking_seconds,
        "scaffold_candidate_fold_seconds": scaffold_stats.candidate_fold_seconds,
        "scaffold_candidate_graph_seconds": scaffold_stats.candidate_graph_seconds,
        "scaffold_row_sum_seconds": scaffold_stats.row_sum_seconds,
        "scaffold_row_sum_stack_cast_seconds": scaffold_stats.row_sum_stack_cast_seconds,
        "scaffold_row_sum_validation_seconds": scaffold_stats.row_sum_validation_seconds,
        "scaffold_row_sum_aggregate_seconds": scaffold_stats.row_sum_aggregate_seconds,
        "scaffold_row_sum_matvec_seconds": scaffold_stats.row_sum_matvec_seconds,
        "scaffold_row_sum_self_dot_seconds": scaffold_stats.row_sum_self_dot_seconds,
        "scaffold_row_sum_finalize_seconds": scaffold_stats.row_sum_finalize_seconds,
        "scaffold_q_score_seconds": scaffold_stats.q_score_seconds,
        "scaffold_selection_seconds": scaffold_stats.selection_seconds,
        "scaffold_merge_seconds": scaffold_stats.merge_seconds,
        "scaffold_sparse_distance_seconds": scaffold_stats.sparse_distance_seconds,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "maximum_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "reference_access": "none",
    }
    atomic_json(metrics_path, metrics)
    if return_topology:
        canonical_graph = _postorder_graph_handoff(adjacency, n_taxa)
        return (
            metrics,
            canonical_graph,
            split_bitmasks(canonical_graph, n_taxa),
        )
    return metrics


def _run_view_process(
    msa: Path,
    work_dir: Path,
    view: int,
    device_name: str,
    candidate_distance_backend: str,
    input_sha256: str,
    row_sum_backend: str,
    nni_reduction_backend: str,
    quartet_predictor: str,
    missing_distance_model: str,
    view_stop_ratio: float | None,
    view_max_rounds: int | None,
) -> dict[str, Any]:
    target = work_dir / "views" / f"view{view}"
    metrics_path = target / "metrics.json"
    tree_path = target / "tree.nwk"
    if metrics_path.is_file() and tree_path.is_file():
        row = json.loads(metrics_path.read_text())
        if row.get("input_sha256") != input_sha256:
            raise ValueError(f"view work directory belongs to a different MSA: {target}")
        if row.get("candidate_distance_backend", "eager") != candidate_distance_backend:
            raise ValueError(f"view work directory uses a different distance backend: {target}")
        if row.get("scaffold_row_sum_backend", "cpu") != row_sum_backend:
            raise ValueError(f"view work directory uses a different row-sum backend: {target}")
        if row.get("nni_reduction_backend", "python") != nni_reduction_backend:
            raise ValueError(f"view work directory uses a different NNI reduction backend: {target}")
        if row.get("quartet_predictor", "heterogeneous") != quartet_predictor:
            raise ValueError(
                f"view work directory uses a different quartet predictor: {target}"
            )
        if row.get("missing_distance_model", "imputed") != missing_distance_model:
            raise ValueError(
                f"view work directory uses a different missing-distance model: {target}"
            )
        if row.get("stopping_semantics") != "per-round-or-v1":
            raise ValueError(f"view work directory uses legacy stopping semantics: {target}")
        if row.get("view_stop_ratio", DEFAULT_VIEW_STOP_RATIO) != view_stop_ratio:
            raise ValueError(f"view work directory uses a different stop ratio: {target}")
        if row.get("view_max_rounds", DEFAULT_VIEW_MAX_ROUNDS) != view_max_rounds:
            raise ValueError(f"view work directory uses a different round budget: {target}")
        row["tree"] = str(tree_path.resolve())
        row["resumed"] = True
        return row
    target.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "concordtree.cli",
        "_view",
        "--msa",
        str(msa),
        "--target",
        str(target),
        "--view",
        str(view),
        "--device",
        device_name,
        "--candidate-distance-backend",
        candidate_distance_backend,
        "--input-sha256",
        input_sha256,
        "--row-sum-backend",
        row_sum_backend,
        "--nni-reduction-backend",
        nni_reduction_backend,
        "--quartet-predictor",
        quartet_predictor,
        "--missing-distance-model",
        missing_distance_model,
        "--view-stop-ratio",
        "none" if view_stop_ratio is None else str(view_stop_ratio),
        "--view-max-rounds",
        "none" if view_max_rounds is None else str(view_max_rounds),
    ]
    environment = {
        **os.environ,
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    with (target / "driver.log").open("a", buffering=1) as log:
        log.write(f"[{utc_now()}] COMMAND {' '.join(command)}\n")
        done = subprocess.run(
            command,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        log.write(f"[{utc_now()}] EXIT {done.returncode}\n")
    if done.returncode:
        raise RuntimeError(
            f"view {view} failed with exit code {done.returncode}; see {target / 'driver.log'}"
        )
    row = json.loads(metrics_path.read_text())
    row["tree"] = str(tree_path.resolve())
    return row


def _run_view_forked(
    msa: Path,
    work_dir: Path,
    view: int,
    device_name: str,
    candidate_distance_backend: str,
    input_sha256: str,
    row_sum_backend: str,
    nni_reduction_backend: str,
    quartet_predictor: str,
    missing_distance_model: str,
    view_stop_ratio: float | None,
    view_max_rounds: int | None,
) -> dict[str, Any]:
    """Run one unchanged View in a pre-CUDA forked worker process."""

    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    target = work_dir / "views" / f"view{view}"
    target.mkdir(parents=True, exist_ok=True)
    log_path = target / "driver.log"
    with log_path.open("a", buffering=1) as log:
        log.write(f"[{utc_now()}] DIRECT_FORK view={view} device={device_name}\n")
        try:
            result = build_view(
                msa,
                target,
                view,
                device_name,
                candidate_distance_backend,
                return_topology=True,
                input_sha256=input_sha256,
                row_sum_backend=row_sum_backend,
                nni_reduction_backend=nni_reduction_backend,
                quartet_predictor=quartet_predictor,
                missing_distance_model=missing_distance_model,
                view_stop_ratio=view_stop_ratio,
                view_max_rounds=view_max_rounds,
            )
            if isinstance(result, tuple):
                row, graph, splits = result
                row["_canonical_graph"] = graph
                row["_canonical_splits"] = splits
            else:
                row = result
        except BaseException as error:
            log.write(f"[{utc_now()}] ERROR {type(error).__name__}: {error}\n")
            raise
        log.write(f"[{utc_now()}] EXIT 0\n")
    return row


def _run_view_threaded(
    msa: Path,
    work_dir: Path,
    view: int,
    device_name: str,
    candidate_distance_backend: str,
    input_sha256: str,
    row_sum_backend: str,
    nni_reduction_backend: str,
    quartet_predictor: str,
    missing_distance_model: str,
    view_stop_ratio: float | None,
    view_max_rounds: int | None,
) -> dict[str, Any]:
    """Run one unchanged View inside the parent's shared CUDA context."""

    target = work_dir / "views" / f"view{view}"
    target.mkdir(parents=True, exist_ok=True)
    log_path = target / "driver.log"
    with log_path.open("a", buffering=1) as log:
        log.write(f"[{utc_now()}] DIRECT_THREAD view={view} device={device_name}\n")
        try:
            result = build_view(
                msa,
                target,
                view,
                device_name,
                candidate_distance_backend,
                return_topology=True,
                input_sha256=input_sha256,
                row_sum_backend=row_sum_backend,
                nni_reduction_backend=nni_reduction_backend,
                quartet_predictor=quartet_predictor,
                missing_distance_model=missing_distance_model,
                view_stop_ratio=view_stop_ratio,
                view_max_rounds=view_max_rounds,
            )
            if isinstance(result, tuple):
                row, graph, splits = result
                row["_canonical_graph"] = graph
                row["_canonical_splits"] = splits
            else:
                row = result
        except BaseException as error:
            log.write(f"[{utc_now()}] ERROR {type(error).__name__}: {error}\n")
            raise
        log.write(f"[{utc_now()}] EXIT 0\n")
    return row


def _prepare_post_view_context(
    msa: Path,
    view_rows: list[dict[str, Any]],
    bundle: ScorerBundle,
    device: torch.device,
    n_taxa: int,
    inference_mode: str,
    predecoded_graphs: dict[int, dict[int, set[int]]] | None = None,
    predecoded_splits: dict[int, frozenset[int]] | None = None,
    preloaded_input: ForkSharedViewInput | None = None,
) -> tuple[PostViewContext, dict[str, Any]]:
    """Materialize invariant post-view state exactly once."""

    started = time.perf_counter()
    sequence_backend, pattern_backend, model, coeff, quartet_matrix, species = bundle
    layout_started = time.perf_counter()
    if inference_mode == "transformer":
        if coeff is None or quartet_matrix is None or species is None:
            raise AssertionError("transformer mode is missing QuartFormer assets")
        active_block_indices, active_block_counts = compress_block_layout(coeff)
        quartet_matrix = define_historical_quartet_tail(quartet_matrix)
        attention_pair_masks = load_attention_pair_masks(device)
    elif inference_mode == "fast":
        active_block_indices = None
        active_block_counts = None
        attention_pair_masks = None
    else:
        raise ValueError(f"unknown inference mode: {inference_mode}")
    layout_seconds = time.perf_counter() - layout_started
    msa_started = time.perf_counter()
    if (
        preloaded_input is not None
        and preloaded_input.msa == msa
        and preloaded_input.input_sha256
        == str(view_rows[0].get("input_sha256", ""))
    ):
        sequences = torch.from_numpy(preloaded_input.packed).to(device)
        names_list = list(preloaded_input.raw_names)
        effective_length = preloaded_input.variable_length
        msa_input_backend = "fork-shared-native-packed"
    else:
        sequences, names_list, effective_length = _load_msa_on_device(
            msa, sequence_backend, device
        )
        msa_input_backend = "native-packed-parse"
    msa_seconds = time.perf_counter() - msa_started
    view_started = time.perf_counter()
    view_tree_paths = tuple(Path(row["tree"]) for row in view_rows)
    view_indices = tuple(int(row["view"]) for row in view_rows)
    if (
        predecoded_graphs is not None
        and predecoded_splits is not None
        and all(index in predecoded_graphs for index in view_indices)
        and all(index in predecoded_splits for index in view_indices)
    ):
        view_graphs = tuple(predecoded_graphs[index] for index in view_indices)
        view_splits = tuple(predecoded_splits[index] for index in view_indices)
    else:
        view_graphs = tuple(
            tree_path_to_graph(path, names_list) for path in view_tree_paths
        )
        view_splits = tuple(split_bitmasks(graph, n_taxa) for graph in view_graphs)
    view_counts: Counter[int] = Counter(
        split for splits in view_splits for split in splits
    )
    medoid_index = _prediction_medoid(list(view_splits))
    view_seconds = time.perf_counter() - view_started
    context = PostViewContext(
        inference_mode=inference_mode,
        sequence_backend=sequence_backend,
        pattern_backend=pattern_backend,
        model=model,
        coeff=coeff,
        active_block_indices=active_block_indices,
        active_block_counts=active_block_counts,
        attention_pair_masks=attention_pair_masks,
        quartet_matrix=quartet_matrix,
        species=species,
        sequences=sequences,
        names=tuple(names_list),
        effective_length=effective_length,
        view_tree_paths=view_tree_paths,
        view_graphs=view_graphs,
        view_splits=view_splits,
        view_counts=dict(view_counts),
        view_count=len(view_rows),
        medoid_index=medoid_index,
        device=device,
        local_template_device=torch.from_numpy(LOCAL_TEMPLATE).to(device),
        probability_cache=OrderedDict(),
        panel_score_backend=load_panel_score_backend(),
        split_compat_backend=load_split_compat_backend(),
    )
    metrics: dict[str, Any] = {
        "seconds": time.perf_counter() - started,
        "scorer": _backend_label(inference_mode),
        "msa_load_to_device_seconds": msa_seconds,
        "msa_input_backend": msa_input_backend,
        "view_decode_and_split_seconds": view_seconds,
        "attention_layout_seconds": layout_seconds,
        "effective_length": effective_length,
        "packed_bytes": int(sequences.numel() * sequences.element_size()),
    }
    if inference_mode == "transformer":
        assert active_block_counts is not None
        assert attention_pair_masks is not None
        metrics.update(
            {
                "attention_active_blocks": int(active_block_counts.sum().item()),
                "attention_active_blocks_per_row_max": int(
                    active_block_counts.max().item()
                ),
                "attention_pair_mask_bytes": int(
                    attention_pair_masks.numel()
                    * attention_pair_masks.element_size()
                ),
            }
        )
    return context, metrics


def _panel_probabilities(
    panels: list[ContextualPanel],
    context: PostViewContext,
    trace: QFExecutionTrace | None = None,
    qf_autocast_dtype: torch.dtype | None = None,
) -> list[np.ndarray]:
    """Score panels once per ordered taxon set and reuse exact posteriors.

    The selected backend is a pure function of the immutable packed MSA and
    ``panel.taxa``.  Panel ids, target edges, covers and refinement rounds do
    not enter the model.  Coalescing equal ordered taxon sets is therefore
    common-subexpression elimination, not an approximation.
    """

    probabilities: list[np.ndarray | None] = [None] * len(panels)
    missing_panels: list[ContextualPanel] = []
    missing_positions: dict[tuple[int, ...], list[int]] = {}
    cache_hits = 0
    for position, panel in enumerate(panels):
        key = panel.taxa
        cached = context.probability_cache.get(key)
        if cached is not None:
            context.probability_cache.move_to_end(key)
            probabilities[position] = cached
            cache_hits += 1
            continue
        positions = missing_positions.get(key)
        if positions is None:
            missing_positions[key] = [position]
            missing_panels.append(panel)
        else:
            positions.append(position)
            cache_hits += 1
    if trace is not None:
        trace.requested_panels += len(panels)
        trace.cache_hits += cache_hits

    batch_size = (
        FAST_SCORER_BATCH_SIZE
        if context.inference_mode == "fast"
        else SCORER_BATCH_SIZE
    )
    for start in range(0, len(missing_panels), batch_size):
        batch = missing_panels[start : start + batch_size]
        index_started = time.perf_counter()
        # The local quartet template is invariant. Transfer only the 24 taxon
        # ids of each panel, then apply the exact integer gather on-device.
        # This preserves panel and quartet row order exactly.
        panel_taxa = torch.as_tensor(
            np.asarray([panel.taxa for panel in batch], dtype=np.int64),
            dtype=torch.long,
            device=context.device,
        )
        indices = panel_taxa[:, context.local_template_device].reshape(-1, 4)
        if trace is not None:
            trace.batches += 1
            trace.panels += len(batch)
            trace.quartets += len(batch) * len(LOCAL_TEMPLATE)
            trace.quartet_index_seconds += time.perf_counter() - index_started
            pattern_started = torch.cuda.Event(enable_timing=True)
            pattern_finished = torch.cuda.Event(enable_timing=True)
            model_finished = torch.cuda.Event(enable_timing=True)
            pattern_started.record()
        pattern = context.pattern_backend.compute_pattern_frequencies_cuda_packed(
            context.sequences, indices, int(context.effective_length)
        ).contiguous().view(len(batch), len(LOCAL_TEMPLATE), -1)
        if trace is not None:
            pattern_finished.record()
        with torch.no_grad(), torch.autocast(
            device_type=context.device.type,
            dtype=(
                torch.float16
                if qf_autocast_dtype is None
                else qf_autocast_dtype
            ),
            enabled=qf_autocast_dtype is not None,
        ):
            if context.inference_mode == "fast":
                logits = context.model(
                    pattern.reshape(-1, pattern.shape[-1])
                ).view(len(batch), len(LOCAL_TEMPLATE), 3)
            else:
                if any(
                    value is None
                    for value in (
                        context.coeff,
                        context.active_block_indices,
                        context.active_block_counts,
                        context.attention_pair_masks,
                        context.quartet_matrix,
                        context.species,
                    )
                ):
                    raise AssertionError("transformer mode is missing QuartFormer state")
                assert context.species is not None
                model_input = torch.cat(
                    [
                        context.species.unsqueeze(0).expand(len(batch), -1, -1),
                        pattern,
                    ],
                    dim=2,
                )
                padded = torch.cat(
                    [
                        model_input,
                        model_input.new_zeros(
                            (len(batch), 14, model_input.shape[2])
                        ),
                    ],
                    dim=1,
                )
                logits = context.model(
                    padded,
                    context.coeff,
                    context.quartet_matrix,
                    context.active_block_indices,
                    context.active_block_counts,
                    context.attention_pair_masks,
                )[:, : len(LOCAL_TEMPLATE), :]
            output_device = torch.softmax(logits, dim=-1)
        if trace is not None:
            model_finished.record()
        host_started = time.perf_counter()
        output = output_device.cpu().numpy()
        if trace is not None:
            torch.cuda.synchronize(context.device)
            trace.pattern_cuda_ms += pattern_started.elapsed_time(pattern_finished)
            trace.model_cuda_ms += pattern_finished.elapsed_time(model_finished)
            trace.output_to_host_seconds += time.perf_counter() - host_started
        for offset, panel in enumerate(batch):
            key = panel.taxa
            value = output[offset]
            context.probability_cache[key] = value
            context.probability_cache.move_to_end(key)
            if len(context.probability_cache) > PANEL_CACHE_MAX_ENTRIES:
                context.probability_cache.popitem(last=False)
            for position in missing_positions[key]:
                probabilities[position] = value
    if any(value is None for value in probabilities):
        raise AssertionError("panel probability cache left an unresolved request")
    return [value for value in probabilities if value is not None]



def _sparse_fast_panel_probabilities(
    plans: list[SparseContextualPanelPlan],
    context: PostViewContext,
    trace: QFExecutionTrace | None = None,
) -> list[np.ndarray]:
    """Evaluate only independent MLP rows used by contextual edge decisions."""

    if context.inference_mode != "fast":
        raise ValueError("sparse panel probabilities require Fast mode")
    if trace is not None:
        trace.requested_panels += len(plans)
    maximum_rows = FAST_SCORER_BATCH_SIZE * len(LOCAL_TEMPLATE)
    output: list[np.ndarray] = []
    start = 0
    while start < len(plans):
        end = start
        row_count = 0
        while end < len(plans):
            candidate_rows = len(plans[end].row_indices)
            if end > start and row_count + candidate_rows > maximum_rows:
                break
            row_count += candidate_rows
            end += 1
        batch = plans[start:end]
        index_started = time.perf_counter()
        panel_taxa = torch.as_tensor(
            np.asarray([plan.taxa for plan in batch], dtype=np.int64),
            dtype=torch.long,
            device=context.device,
        )
        local_rows = torch.as_tensor(
            np.concatenate([plan.row_indices for plan in batch]),
            dtype=torch.long,
            device=context.device,
        )
        plan_rows = torch.repeat_interleave(
            torch.arange(len(batch), dtype=torch.long, device=context.device),
            torch.as_tensor(
                [len(plan.row_indices) for plan in batch],
                dtype=torch.long,
                device=context.device,
            ),
        )
        local_quartets = context.local_template_device.index_select(
            0, local_rows
        ).long()
        indices = torch.gather(
            panel_taxa.index_select(0, plan_rows), 1, local_quartets
        )
        if trace is not None:
            trace.batches += 1
            trace.panels += len(batch)
            trace.quartets += row_count
            trace.quartet_index_seconds += time.perf_counter() - index_started
            pattern_started = torch.cuda.Event(enable_timing=True)
            pattern_finished = torch.cuda.Event(enable_timing=True)
            model_finished = torch.cuda.Event(enable_timing=True)
            pattern_started.record()
        pattern = context.pattern_backend.compute_pattern_frequencies_cuda_packed(
            context.sequences, indices, int(context.effective_length)
        ).contiguous()
        if trace is not None:
            pattern_finished.record()
        with torch.no_grad():
            probabilities_device = torch.softmax(context.model(pattern), dim=-1)
        if trace is not None:
            model_finished.record()
        host_started = time.perf_counter()
        probabilities = probabilities_device.cpu().numpy()
        if trace is not None:
            torch.cuda.synchronize(context.device)
            trace.pattern_cuda_ms += pattern_started.elapsed_time(pattern_finished)
            trace.model_cuda_ms += pattern_finished.elapsed_time(model_finished)
            trace.output_to_host_seconds += time.perf_counter() - host_started
        offset = 0
        for plan in batch:
            next_offset = offset + len(plan.row_indices)
            output.append(probabilities[offset:next_offset])
            offset = next_offset
        if offset != len(probabilities):
            raise AssertionError("sparse panel probability split is inconsistent")
        start = end
    return output


def _score_contextual_panels(
    context: PostViewContext,
    graph: dict[int, set[int]],
    panels: list[ContextualPanel],
    tree_index: Any,
    target_edges: set[tuple[int, int]],
    trace: QFExecutionTrace | None,
    *,
    compute_medians: bool = True,
) -> tuple[list[Any], int]:
    """Dispatch exact dense QF scoring or algebraically sparse Fast scoring."""

    if context.inference_mode != "fast":
        probabilities = _panel_probabilities(
            panels,
            context,
            trace,
            qf_autocast_dtype=QF_AUTOCAST_DTYPE,
        )
        scores: list[Any] = []
        for panel, probability in zip(panels, probabilities):
            scores.extend(
                score
                for score in score_contextual_panel_edges(
                    graph,
                    panel,
                    probability,
                    LOCAL_TEMPLATE,
                    tree_index=tree_index,
                    row_lookup=LOCAL_ROW_INDEX,
                    plan_backend=context.panel_score_backend,
                )
                if score.edge in target_edges
            )
        return scores, len(panels) * len(LOCAL_TEMPLATE)

    if context.panel_score_backend is None:
        raise RuntimeError("sparse Fast scoring requires the packaged panel backend")
    plans: list[SparseContextualPanelPlan] = []
    plan_started = time.perf_counter()
    for panel in panels:
        retained_edges = tuple(
            edge for edge in panel.target_edges if edge in target_edges
        )
        if not retained_edges:
            continue
        narrowed = ContextualPanel(
            cover=panel.cover,
            panel_id=panel.panel_id,
            taxa=panel.taxa,
            target_edges=retained_edges,
        )
        plans.append(
            compile_sparse_contextual_panel_plan(
                graph,
                narrowed,
                LOCAL_TEMPLATE,
                tree_index=tree_index,
                row_lookup=LOCAL_ROW_INDEX,
                plan_backend=context.panel_score_backend,
            )
        )
    if trace is not None:
        trace.plan_compile_seconds += time.perf_counter() - plan_started
    probability_started = time.perf_counter()
    probabilities = _sparse_fast_panel_probabilities(plans, context, trace)
    if trace is not None:
        trace.probability_seconds += time.perf_counter() - probability_started
    score_started = time.perf_counter()
    scores = []
    for plan, probability in zip(plans, probabilities):
        scores.extend(
            score_sparse_contextual_panel_plan(
                plan,
                probability,
                plan_backend=context.panel_score_backend,
                compute_medians=compute_medians,
            )
        )
    if trace is not None:
        trace.score_reduce_seconds += time.perf_counter() - score_started
    return scores, sum(len(plan.row_indices) for plan in plans)


def _load_msa_on_device(
    msa: Path, sequence_backend: Any, device: torch.device
) -> tuple[torch.Tensor, list[str], int]:
    packed, names_raw, effective_length = sequence_backend.load_phy_to_packed_tensor(
        str(msa), True
    )
    return torch.from_numpy(packed).to(device), list(names_raw), int(effective_length)


def _prepare_fork_shared_view_input(
    msa: Path,
    input_sha256: str,
    n_taxa: int,
    alignment_length: int,
    view_count: int = VIEW_COUNT,
) -> ForkSharedViewInput | None:
    """Factor exact packed and bounded-sketch input maps out of forked Views.

    Bounded alignments retain one common exact state matrix. Longer alignments
    keep the historical independent per-View stratified positions, but extract
    all of them from one native packed parse before workers fork. For missing
    inputs, the scaffold keeps the historical observed-variable filter and
    missing-state representation.
    """

    sequence_backend, _pattern_backend = load_backends()
    if alignment_length > MAX_SITES:
        raw_packed, names_raw, loaded_length = (
            sequence_backend.load_phy_to_packed_tensor(str(msa), False)
        )
        if int(loaded_length) != alignment_length or len(names_raw) != n_taxa:
            raise AssertionError("native raw packed input dimensions changed unexpectedly")
        position_rows = np.ascontiguousarray(
            np.stack(
                [
                    _stratified_positions(
                        alignment_length,
                        MAX_SITES,
                        SEEDS[view][0],
                    )
                    for view in range(view_count)
                ]
            ),
            dtype=np.int64,
        )
        # The sequence backend uses joined, one-shot C++ threads here. Unlike
        # an OpenMP pool, no worker runtime survives into the later Linux fork.
        workers = min(
            32,
            len(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else (os.cpu_count() or 1),
        )
        sampled = np.asarray(
            sequence_backend.extract_packed_positions(
                raw_packed,
                alignment_length,
                position_rows,
                workers,
            ),
            dtype=np.uint8,
        )
        packed, effective_length = sequence_backend.filter_conserved_packed(
            raw_packed,
            alignment_length,
            workers,
        )
        packed = np.asarray(packed)
        raw_names = tuple(str(name) for name in names_raw)
        order = sorted(range(n_taxa), key=lambda position: raw_names[position])
        view_states: list[np.ndarray] = []
        for view in range(view_count):
            states = sampled[view]
            observed_states = np.stack(
                [np.any(states == base, axis=0) for base in range(4)]
            )
            variable = np.count_nonzero(observed_states, axis=0) >= 2
            if bool(np.any(variable)):
                states = states[:, variable]
            states = np.ascontiguousarray(states[order])
            states.flags.writeable = False
            view_states.append(states)
        if any(states.shape[0] != n_taxa for states in view_states):
            raise AssertionError("native multiview sketch lost taxa")
        return ForkSharedViewInput(
            input_sha256=input_sha256,
            msa=msa,
            packed=packed,
            raw_names=raw_names,
            variable_length=int(effective_length),
            scaffold_names=tuple(raw_names[position] for position in order),
            scaffold_states=view_states[0],
            leaf_profile_slab=None,
            view_scaffold_states=tuple(view_states),
        )

    packed, names_raw, effective_length = sequence_backend.load_phy_to_packed_tensor(
        str(msa), True
    )
    native_states = np.asarray(
        sequence_backend.unpack_packed_tensor(packed, int(effective_length)),
        dtype=np.uint8,
    )
    if native_states.shape != (n_taxa, int(effective_length)):
        raise AssertionError("native packed input dimensions changed unexpectedly")
    raw_names = tuple(str(name) for name in names_raw)
    order = sorted(range(n_taxa), key=lambda position: raw_names[position])
    scaffold_states = native_states
    if not bool(np.all(native_states < 4)):
        # ``load_phylip_sketch`` retains a sampled column iff at least two
        # distinct observed A/C/G/T states occur. The native packed reader has
        # already removed only raw-constant columns, so this recovers the exact
        # scaffold subset without a second text scan.
        observed_states = np.stack(
            [np.any(native_states == base, axis=0) for base in range(4)]
        )
        variable = np.count_nonzero(observed_states, axis=0) >= 2
        # The historical reader deliberately keeps all sampled columns when
        # none is observed-variable. Native filtering cannot reconstruct that
        # degenerate case, so preserve the historical per-View path there.
        if not bool(np.any(variable)):
            return None
        scaffold_states = native_states[:, variable]
    scaffold_states = np.ascontiguousarray(scaffold_states[order])
    # Match the Python sketch's sentinel exactly. Downstream mathematics treats
    # every state >= 4 as missing, but 255 also makes the state matrix itself
    # byte-identical to the historical reader.
    if bool(np.any(scaffold_states >= 4)):
        scaffold_states[scaffold_states >= 4] = 255
    from concordtree._core.sctb_aggregate_nj import imputed_profile_slab

    leaf_profile_slab = imputed_profile_slab(scaffold_states)
    scaffold_states.flags.writeable = False
    return ForkSharedViewInput(
        input_sha256=input_sha256,
        msa=msa,
        packed=np.asarray(packed),
        raw_names=raw_names,
        variable_length=int(effective_length),
        scaffold_names=tuple(raw_names[position] for position in order),
        scaffold_states=scaffold_states,
        leaf_profile_slab=leaf_profile_slab,
    )


def _splitbank_model(
    context: PostViewContext,
    n_taxa: int,
    work_dir: Path,
    trace_performance: bool = False,
) -> tuple[dict[str, Any], dict[int, set[int]]]:
    operator_started = time.perf_counter()
    counts = Counter(context.view_counts)
    medoid = context.view_splits[context.medoid_index]
    margins: dict[int, list[float]] = defaultdict(list)
    panels_scored = 0
    decision_rows_scored = 0
    trace = QFExecutionTrace() if trace_performance else None
    phase_seconds: defaultdict[str, float] = defaultdict(float)
    started = time.perf_counter()
    for graph in context.view_graphs:
        phase_started = time.perf_counter()
        mapping = _edge_splits(graph, n_taxa)
        targets = {
            edge
            for edge, split in mapping.items()
            if counts[split] < context.view_count
        }
        phase_seconds["edge_mapping_seconds"] += time.perf_counter() - phase_started
        phase_started = time.perf_counter()
        tree_index = build_contextual_tree_index(
            graph,
            n_taxa,
            representative_limit=4,
            plan_backend=context.panel_score_backend,
        )
        phase_seconds["tree_index_seconds"] += time.perf_counter() - phase_started
        phase_started = time.perf_counter()
        panels = [
            panel
            for panel in build_contextual_panel_covers(
                graph,
                n_taxa,
                covers=(0, 1),
                panel_size=24,
                maximum_target_edges=12,
                required_taxa_cap=20,
                tree_index=tree_index,
                plan_backend=context.panel_score_backend,
            )
            if any(edge in targets for edge in panel.target_edges)
        ]
        phase_seconds["panel_build_seconds"] += time.perf_counter() - phase_started
        phase_started = time.perf_counter()
        scored, decision_rows = _score_contextual_panels(
            context,
            graph,
            panels,
            tree_index,
            targets,
            trace,
            compute_medians=False,
        )
        phase_seconds["panel_score_seconds"] += time.perf_counter() - phase_started
        phase_started = time.perf_counter()
        for score in scored:
            value = (
                (score.scores[0] - max(score.scores[1], score.scores[2]))
                * len(LOCAL_TEMPLATE)
                / score.discriminating_rows
            )
            margins[mapping[score.edge]].append(float(value))
        phase_seconds["margin_collect_seconds"] += time.perf_counter() - phase_started
        panels_scored += len(panels)
        decision_rows_scored += decision_rows
    phase_started = time.perf_counter()
    selected, stats = _decode_splitbank(
        counts,
        margins,
        medoid,
        n_taxa,
        view_count=context.view_count,
        selector_backend=context.split_compat_backend,
    )
    phase_seconds["decode_seconds"] += time.perf_counter() - phase_started
    phase_started = time.perf_counter()
    graph = tree_to_graph(
        split_set_to_tree_indexed(
            selected, n_taxa, backend=context.split_compat_backend
        ),
        n_taxa,
    )
    validate_topology(graph, n_taxa)
    path = work_dir / "splitbank" / f"{_backend_label(context.inference_mode)}.nwk"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(graph_to_newick(graph, list(context.names), n_taxa) + "\n")
    phase_seconds["materialize_seconds"] += time.perf_counter() - phase_started
    result = {
        "prediction": str(path.resolve()),
        "effective_length": context.effective_length,
        "union_candidates": len(counts),
        "minority_candidates": sum(
            count < context.view_count for count in counts.values()
        ),
        "panels": panels_scored,
        "decision_quartet_rows": decision_rows_scored,
        "dense_panel_quartet_rows": panels_scored * len(LOCAL_TEMPLATE),
        "seconds": time.perf_counter() - started,
        "total_seconds": time.perf_counter() - operator_started,
        "decode": stats,
    }
    if trace is not None:
        result["performance_trace"] = trace.as_dict()
        result["phase_seconds"] = dict(phase_seconds)
    return result, _postorder_graph_handoff(graph, n_taxa)


def _coordinate_pass(
    parent: dict[int, set[int]],
    context: PostViewContext,
    n_taxa: int,
    work_dir: Path,
    label: str,
    trace_performance: bool = False,
) -> tuple[dict[str, Any], dict[int, set[int]]]:
    operator_started = time.perf_counter()
    validate_topology(parent, n_taxa)
    target_edges = _uncertain_edges(
        parent, n_taxa, context.view_counts, context.view_count
    )
    tree_index = build_contextual_tree_index(
        parent,
        n_taxa,
        representative_limit=4,
        plan_backend=context.panel_score_backend,
    )
    panels = [
        panel
        for panel in build_contextual_panel_covers(
            parent,
            n_taxa,
            covers=(0, 1),
            panel_size=24,
            maximum_target_edges=12,
            required_taxa_cap=20,
            tree_index=tree_index,
            plan_backend=context.panel_score_backend,
        )
        if any(edge in target_edges for edge in panel.target_edges)
    ]
    started = time.perf_counter()
    trace = QFExecutionTrace() if trace_performance else None
    scores, decision_rows = _score_contextual_panels(
        context, parent, panels, tree_index, target_edges, trace
    )
    expected = len(target_edges) * 2
    if len(scores) != expected:
        raise AssertionError(
            f"uncertain edge cover mismatch: {len(scores)} != {expected}"
        )
    evidence = consensus_evidence(scores, covers=2, minimum_gain=0.0)
    graph = _clone(parent)
    selected = apply_independent_nni(graph, evidence, min_margin=0.0)
    validate_topology(graph, n_taxa)
    path = work_dir / label / f"{_backend_label(context.inference_mode)}.nwk"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(graph_to_newick(graph, list(context.names), n_taxa) + "\n")
    result = {
        "prediction": str(path.resolve()),
        "moves": len(selected),
        "panels": len(panels),
        "decision_quartet_rows": decision_rows,
        "dense_panel_quartet_rows": len(panels) * len(LOCAL_TEMPLATE),
        "seconds": time.perf_counter() - started,
        "total_seconds": time.perf_counter() - operator_started,
    }
    if trace is not None:
        result["performance_trace"] = trace.as_dict()
    return result, _postorder_graph_handoff(graph, n_taxa)



def _saturated_panel(
    graph: dict[int, set[int]],
    edge: tuple[int, int],
    tree_index: Any,
    panel_id: int,
    n_taxa: int,
) -> ContextualPanel:
    left, right = edge
    branches = branch_nodes_for_edge(graph, edge)
    owners = (left, left, right, right)
    groups = [
        list(tree_index.nearest[(branch, owner)])
        for branch, owner in zip(branches, owners)
    ]
    selected: list[int] = []
    rank = 0
    while len(selected) < 24:
        added = False
        for group in groups:
            if rank < len(group):
                selected.append(group[rank])
                added = True
                if len(selected) == 24:
                    break
        if not added:
            break
        rank += 1
    if len(selected) < 24:
        used = set(selected)
        selected.extend(taxon for taxon in range(n_taxa) if taxon not in used)
        selected = selected[:24]
    if len(selected) != 24 or len(set(selected)) != 24:
        raise AssertionError("cannot form a unique 24-taxon saturated panel")
    return ContextualPanel(
        cover=0,
        panel_id=panel_id,
        taxa=tuple(sorted(selected)),
        target_edges=(edge,),
    )


def _evidence_from_scores(scores: list[Any]) -> list[EdgeEvidence]:
    result: list[EdgeEvidence] = []
    for score in scores:
        if (
            score.best == 0
            or score.gain <= 0
            or score.alternative_median_gains[score.best - 1] <= 0
        ):
            continue
        result.append(
            EdgeEvidence(
                edge=score.edge,
                branch_nodes=score.branch_nodes,
                scores=score.scores,
                best=score.best,
                margin=score.gain,
                quartet_count=score.discriminating_rows,
            )
        )
    return result


def _saturating_pass(
    parent: dict[int, set[int]],
    context: PostViewContext,
    n_taxa: int,
    work_dir: Path,
    label: str,
    trace_performance: bool = False,
) -> tuple[dict[str, Any], dict[int, set[int]]]:
    """Apply one edge-local MLP continuation step."""

    operator_started = time.perf_counter()
    targets = _uncertain_edges(
        parent, n_taxa, context.view_counts, context.view_count
    )
    tree_index = build_contextual_tree_index(
        parent,
        n_taxa,
        representative_limit=24,
        plan_backend=context.panel_score_backend,
    )
    panels = [
        _saturated_panel(parent, edge, tree_index, panel_id, n_taxa)
        for panel_id, edge in enumerate(sorted(targets))
    ]
    started = time.perf_counter()
    trace = QFExecutionTrace() if trace_performance else None
    if context.inference_mode == "fast":
        # An NNI changes only quartets drawing one taxon from each of its four
        # incident branches. Every other panel row contributes the same term
        # to all three local topologies and cancels from both gain and median
        # comparisons. Reuse the exact sparse scorer already used by the
        # coordinate pass instead of evaluating 10,626 irrelevant MLP rows per
        # edge-local panel.
        scores, decision_rows = _score_contextual_panels(
            context,
            parent,
            panels,
            tree_index,
            targets,
            trace,
        )
    else:
        probabilities = _panel_probabilities(panels, context, trace)
        scores = []
        for panel, probability in zip(panels, probabilities):
            scores.extend(
                score_contextual_panel_edges(
                    parent,
                    panel,
                    probability,
                    LOCAL_TEMPLATE,
                    tree_index=tree_index,
                    row_lookup=LOCAL_ROW_INDEX,
                    plan_backend=context.panel_score_backend,
                )
            )
        decision_rows = len(panels) * len(LOCAL_TEMPLATE)
    evidence = _evidence_from_scores(scores)
    graph = _clone(parent)
    selected = apply_independent_nni(graph, evidence, min_margin=0.0)
    validate_topology(graph, n_taxa)
    path = work_dir / label / f"{_backend_label(context.inference_mode)}.nwk"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(graph_to_newick(graph, list(context.names), n_taxa) + "\n")
    result = {
        "prediction": str(path.resolve()),
        "target_edges": len(targets),
        "candidates": len(evidence),
        "moves": len(selected),
        "panels": len(panels),
        "decision_quartet_rows": decision_rows,
        "dense_panel_quartet_rows": len(panels) * len(LOCAL_TEMPLATE),
        "seconds": time.perf_counter() - started,
        "total_seconds": time.perf_counter() - operator_started,
    }
    if trace is not None:
        result["performance_trace"] = trace.as_dict()
    return result, _postorder_graph_handoff(graph, n_taxa)


def _run_refinement_stage(
    operator: Any,
    stage_label: str,
    parent: dict[int, set[int]],
    context: PostViewContext,
    n_taxa: int,
    work_dir: Path,
    stop_ratio: float | None,
    max_rounds: int | None,
    trace_performance: bool,
) -> tuple[list[dict[str, Any]], dict[int, set[int]], Path]:
    """Run one refinement stage under independent convergence/budget controls."""

    validate_refinement_control(stage_label, stop_ratio, max_rounds)
    pass_limit = (
        max_rounds if max_rounds is not None else REFINEMENT_EMERGENCY_MAX_ROUNDS
    )
    history: list[dict[str, Any]] = []
    current_graph = parent
    current_path: Path | None = None
    for round_index in range(1, pass_limit + 1):
        result, current_graph = operator(
            current_graph,
            context,
            n_taxa,
            work_dir,
            f"{stage_label}/round{round_index}",
            trace_performance,
        )
        moves = int(result["moves"])
        reasons = refinement_stop_reasons(
            moves,
            round_index,
            n_taxa,
            stop_ratio,
            max_rounds,
        )
        history.append(
            {
                "round": round_index,
                **result,
                "move_ratio": moves / (n_taxa - 3),
                "stop_reasons": list(reasons),
            }
        )
        current_path = Path(result["prediction"])
        if reasons:
            break
    if current_path is None:
        raise AssertionError(f"{stage_label} refinement executed no rounds")
    if not history[-1]["stop_reasons"]:
        raise RuntimeError(
            f"{stage_label} refinement failed to reach its threshold before the "
            "internal non-convergence guard"
        )
    return history, current_graph, current_path



def infer(
    msa: Path,
    output: Path,
    work_dir: Path,
    *,
    device_name: str = "cuda:0",
    view_workers: int = MAX_CONCURRENT_VIEW_WORKERS,
    view_count: int = DEFAULT_VIEW_COUNT,
    quartet_model: str = "mlp",
    trace_performance: bool = False,
    quartet_predictor: str = "heterogeneous",
    parallelism: int = 0,
    missing_data_model: str = "standard",
    view_stop_ratio: float | None = DEFAULT_VIEW_STOP_RATIO,
    view_max_rounds: int | None = DEFAULT_VIEW_MAX_ROUNDS,
    coordinate_stop_ratio: float | None = DEFAULT_COORDINATE_STOP_RATIO,
    coordinate_max_rounds: int | None = DEFAULT_COORDINATE_MAX_ROUNDS,
    saturation_stop_ratio: float | None = DEFAULT_SATURATION_STOP_RATIO,
    saturation_max_rounds: int | None = DEFAULT_SATURATION_MAX_ROUNDS,
) -> dict[str, Any]:
    """Run ConcordTree without opening a reference topology.

    View construction and MLP SplitBank initialization are common to both
    public backends.  ``quartet_model`` changes only the quartet potential used
    by the coordinate and saturating NNI refinement schedule.
    """

    msa = msa.resolve(strict=True)
    work_dir = work_dir.resolve()
    output = output.resolve()
    if work_dir == msa.parent:
        raise ValueError("work directory must not be the MSA input directory")
    if not output.is_relative_to(work_dir):
        raise ValueError("final output must be located inside --work-dir")
    if view_workers < 0 or view_workers > MAX_CONCURRENT_VIEW_WORKERS:
        raise ValueError("view-workers must be between 0 (auto) and 4")
    if view_count < 2 or view_count > MAX_VIEW_COUNT:
        raise ValueError("view-count must be between 2 and 8")
    if quartet_model not in {"mlp", "transformer"}:
        raise ValueError("quartet_model must be mlp or transformer")
    if quartet_predictor not in QUARTET_PREDICTORS:
        raise ValueError(
            f"quartet_predictor must be one of {QUARTET_PREDICTORS}"
        )
    validate_refinement_control("view", view_stop_ratio, view_max_rounds)
    validate_refinement_control(
        "coordinate", coordinate_stop_ratio, coordinate_max_rounds
    )
    validate_refinement_control(
        "saturation", saturation_stop_ratio, saturation_max_rounds
    )
    missing_distance_model = resolve_missing_distance_model(missing_data_model)
    n_taxa, alignment_length = read_dimensions(msa)
    refinement_controls = {
        "stopping_semantics": "per-round-or-v1",
        "view": {
            "stop_ratio": view_stop_ratio,
            "max_rounds": view_max_rounds,
            "stop_moves": stop_moves_for_ratio(n_taxa, view_stop_ratio),
        },
        "coordinate": {
            "stop_ratio": coordinate_stop_ratio,
            "max_rounds": coordinate_max_rounds,
            "stop_moves": stop_moves_for_ratio(n_taxa, coordinate_stop_ratio),
        },
        "saturation": {
            "stop_ratio": saturation_stop_ratio,
            "max_rounds": saturation_max_rounds,
            "stop_moves": stop_moves_for_ratio(n_taxa, saturation_stop_ratio),
        },
    }
    # Both public model choices share one MLP View and SplitBank estimator.
    # The model choice begins only at the common NNI refinement boundary.
    execution_mode = "fast"
    refinement_mode = "fast" if quartet_model == "mlp" else "transformer"
    scorer_label = "mlp" if quartet_model == "mlp" else "quartformer"
    requested_view_workers = view_workers
    view_workers = resolve_view_workers(
        n_taxa, requested_view_workers, view_count
    )
    candidate_distance_backend = resolve_candidate_distance_backend(
        n_taxa, execution_mode
    )
    row_sum_backend = resolve_scaffold_row_sum_backend(n_taxa, execution_mode)
    nni_reduction_backend = resolve_nni_reduction_backend(n_taxa, execution_mode)
    view_executor = resolve_view_executor(n_taxa, execution_mode)
    post_view_schedule = (
        FORK_POST_VIEW_SCHEDULE
        if view_executor == "fork"
        else POST_VIEW_SCHEDULE
    )
    input_sha256 = sha256_file(msa)
    device = torch.device(device_name)
    if device.type != "cuda":
        raise ValueError("ConcordTree supports CUDA devices only")
    work_dir.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)

    final_record = work_dir / "record.json"
    if final_record.is_file() and output.is_file():
        previous = json.loads(final_record.read_text())
        if (
            previous.get("input_sha256") != input_sha256
            or previous.get("algorithm_graph") != "concordtree-0.1.2"
            or previous.get("quartet_model", "mlp") != quartet_model
            or previous.get("quartet_predictor", "heterogeneous")
            != quartet_predictor
            or previous.get("candidate_distance_backend", "eager")
            != candidate_distance_backend
            or previous.get("missing_distance_model", "imputed")
            != missing_distance_model
            or previous.get("missing_data_model", "standard")
            != missing_data_model
            or previous.get("scaffold_row_sum_backend", "cpu") != row_sum_backend
            or previous.get("nni_reduction_backend", "python")
            != nni_reduction_backend
            or previous.get("view_executor", "subprocess") != view_executor
            or previous.get("post_view_schedule", "serial") != post_view_schedule
            or int(previous.get("view_count", DEFAULT_VIEW_COUNT)) != view_count
            or previous.get("refinement_controls", {
                "view": {
                    "stop_ratio": DEFAULT_VIEW_STOP_RATIO,
                    "max_rounds": DEFAULT_VIEW_MAX_ROUNDS,
                    "stop_moves": stop_moves_for_ratio(n_taxa, DEFAULT_VIEW_STOP_RATIO),
                },
                "coordinate": {
                    "stop_ratio": DEFAULT_COORDINATE_STOP_RATIO,
                    "max_rounds": DEFAULT_COORDINATE_MAX_ROUNDS,
                    "stop_moves": stop_moves_for_ratio(n_taxa, DEFAULT_COORDINATE_STOP_RATIO),
                },
                "saturation": {
                    "stop_ratio": DEFAULT_SATURATION_STOP_RATIO,
                    "max_rounds": DEFAULT_SATURATION_MAX_ROUNDS,
                    "stop_moves": stop_moves_for_ratio(n_taxa, DEFAULT_SATURATION_STOP_RATIO),
                },
            }) != refinement_controls
            or Path(str(previous.get("final_prediction", ""))).resolve() != output
        ):
            raise ValueError(
                "completed work directory belongs to a different input, quartet "
                "model/family, missing-data model, release graph, execution "
                "backend, or output"
            )
        previous["resumed_from_final"] = True
        return previous

    asset_hashes = verify_assets()
    manifest: dict[str, Any] = {
        "run_id": work_dir.name,
        "status": "running",
        "started_at": utc_now(),
        "version": __version__,
        "algorithm_graph": "concordtree-0.1.2",
        "algorithm": f"ConcordTree {quartet_model}",
        "pipeline": (
            f"{view_count} high-site r4 views -> "
            f"MLP SplitBank -> {scorer_label} coordinate continuation -> "
            f"{scorer_label} saturation"
        ),
        "input_msa": str(msa),
        "input_sha256": input_sha256,
        "n_taxa": n_taxa,
        "alignment_length": alignment_length,
        "output": str(output),
        "work_dir": str(work_dir),
        "reference_access": "none",
        "device": device_name,
        "view_workers": view_workers,
        "requested_view_workers": requested_view_workers,
        "requested_parallelism": parallelism,
        "view_count": view_count,
        "blas_threads_per_view": int(os.environ.get("OPENBLAS_NUM_THREADS", "1")),
        "candidate_cpu_workers_per_view": int(
            os.environ.get("CONCORDTREE_CANDIDATE_CPU_WORKERS", "8")
        ),
        "quartet_model": quartet_model,
        "execution_mode": execution_mode,
        "refinement_scorer": scorer_label,
        "quartet_predictor": quartet_predictor,
        "missing_data_model": missing_data_model,
        "candidate_distance_backend": candidate_distance_backend,
        "missing_distance_model": missing_distance_model,
        "scaffold_row_sum_backend": row_sum_backend,
        "nni_reduction_backend": nni_reduction_backend,
        "compact_imputed_first_round": (
            os.environ.get("CONCORDTREE_COMPACT_IMPUTED_FIRST_ROUND", "1") == "1"
        ),
        "profile_stream_policy": os.environ.get(
            "CONCORDTREE_STREAM_PROFILE_SITES", "auto:2048-sites-at-8GiB"
        ),
        "view_executor": view_executor,
        "post_view_schedule": post_view_schedule,
        "trace_performance": trace_performance,
        "refinement_controls": refinement_controls,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "assets": asset_hashes,
        "frozen_constants": {
            "view_count": view_count,
            "candidate_distance_backend": candidate_distance_backend,
            "missing_data_model": missing_data_model,
            "missing_distance_model": missing_distance_model,
            "scaffold_row_sum_backend": row_sum_backend,
            "nni_reduction_backend": nni_reduction_backend,
            "view_executor": view_executor,
            "post_view_schedule": post_view_schedule,
            "max_sites": MAX_SITES,
            "view_max_passes": view_max_rounds,
            "view_stop_ratio": view_stop_ratio,
            "view_stop_moves": refinement_controls["view"]["stop_moves"],
            "coordinate_max_rounds": coordinate_max_rounds,
            "coordinate_stop_ratio": coordinate_stop_ratio,
            "coordinate_stop_moves": refinement_controls["coordinate"]["stop_moves"],
            "initialization_scorer": "mlp",
            "refinement_scorer": scorer_label,
            "refinement_objective": "positive local quartet log-potential ascent",
            "saturation_max_rounds": saturation_max_rounds,
            "saturation_stop_ratio": saturation_stop_ratio,
            "saturation_stop_moves": refinement_controls["saturation"]["stop_moves"],
            "run_saturating_refinement": True,
            "scorer_batch_size": (
                FAST_SCORER_BATCH_SIZE
                if refinement_mode == "fast"
                else SCORER_BATCH_SIZE
            ),
            "quartet_index_expansion": "device-fixed-template",
            "panel_cache_max_entries": PANEL_CACHE_MAX_ENTRIES,
            "qf_attention_layout": (
                "compiled-static-pair-mask-bsr16"
                if quartet_model == "transformer"
                else "not-applicable"
            ),
            "qf_padding_attention": (
                "explicitly excluded"
                if quartet_model == "transformer"
                else "not-applicable"
            ),
        },
    }
    atomic_json(work_dir / "run_manifest.json", manifest)
    started = time.perf_counter()
    shared_input_started = time.perf_counter()
    global _FORK_SHARED_VIEW_INPUT
    shared_view_input = None
    if view_executor in {"fork", "thread"}:
        shared_view_input = _prepare_fork_shared_view_input(
            msa, input_sha256, n_taxa, alignment_length, view_count
        )
    _FORK_SHARED_VIEW_INPUT = shared_view_input
    shared_view_input_seconds = time.perf_counter() - shared_input_started
    view_phase_started = started
    view_pool_shutdown_seconds = 0.0
    jobs = [
        (
            msa,
            work_dir,
            view,
            device_name,
            candidate_distance_backend,
            input_sha256,
            row_sum_backend,
            nni_reduction_backend,
            quartet_predictor,
            missing_distance_model,
            view_stop_ratio,
            view_max_rounds,
        )
        for view in range(view_count)
    ]
    view_rows: list[dict[str, Any]] = []
    predecoded_graphs: dict[int, dict[int, set[int]]] = {}
    predecoded_splits: dict[int, frozenset[int]] = {}
    if view_executor == "fork":
        pool_type: Any = futures.ProcessPoolExecutor
        pool_options = {"mp_context": multiprocessing.get_context("fork")}
        view_runner = _run_view_forked
    elif view_executor == "thread":
        # Initialize the one shared context before concurrent workers enter
        # PyTorch.  No estimator work is performed here.
        torch.cuda.set_device(device)
        torch.empty(0, device=device)
        pool_type = futures.ThreadPoolExecutor
        pool_options = {}
        view_runner = _run_view_threaded
    else:
        pool_type = futures.ThreadPoolExecutor
        pool_options = {}
        view_runner = _run_view_process
    view_pool = pool_type(max_workers=view_workers, **pool_options)
    try:
        pending = {view_pool.submit(view_runner, *job): job[2] for job in jobs}
        for completed in futures.as_completed(pending):
            row = completed.result()
            graph = row.pop("_canonical_graph", None)
            splits = row.pop("_canonical_splits", None)
            if graph is not None and splits is not None:
                index = int(row["view"])
                predecoded_graphs[index] = graph
                predecoded_splits[index] = splits
            view_rows.append(row)
            view_rows.sort(key=lambda item: int(item["view"]))
            atomic_json(work_dir / "view_records.partial.json", view_rows)
    except BaseException:
        view_pool.shutdown(wait=True, cancel_futures=True)
        _FORK_SHARED_VIEW_INPUT = None
        raise
    if len(view_rows) != view_count:
        view_pool.shutdown(wait=True, cancel_futures=True)
        _FORK_SHARED_VIEW_INPUT = None
        raise AssertionError(f"not all {view_count} ConcordTree views completed")
    atomic_json(work_dir / "view_records.json", view_rows)
    views_complete_seconds = time.perf_counter() - view_phase_started
    _FORK_SHARED_VIEW_INPUT = None
    shutdown_started = time.perf_counter()
    view_pool.shutdown(wait=True, cancel_futures=True)
    view_pool_shutdown_seconds = time.perf_counter() - shutdown_started

    if not torch.cuda.is_available():
        raise RuntimeError("ConcordTree requires CUDA")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    initialization_bundle_started = time.perf_counter()
    initialization_bundle = _load_post_view_bundle(
        "fast", device, quartet_predictor
    )
    initialization_bundle_seconds = (
        time.perf_counter() - initialization_bundle_started
    )
    initialization_context, initialization_context_metrics = _prepare_post_view_context(
        msa,
        view_rows,
        initialization_bundle,
        device,
        n_taxa,
        "fast",
        predecoded_graphs or None,
        predecoded_splits or None,
        shared_view_input,
    )
    splitbank, current_graph = _splitbank_model(
        initialization_context,
        n_taxa,
        work_dir,
        trace_performance,
    )
    consensus = {
        "status": "not-materialized",
        "role": "diagnostic-only",
        "medoid_view": initialization_context.medoid_index,
        "reference_access": "none",
    }
    current_path = Path(splitbank["prediction"])
    if quartet_model == "mlp":
        refinement_context = initialization_context
        refinement_bundle_seconds = 0.0
        refinement_context_metrics = initialization_context_metrics
    else:
        refinement_bundle_started = time.perf_counter()
        refinement_bundle = _load_post_view_bundle(
            "transformer", device, quartet_predictor
        )
        refinement_bundle_seconds = time.perf_counter() - refinement_bundle_started
        refinement_context, refinement_context_metrics = _prepare_post_view_context(
            msa,
            view_rows,
            refinement_bundle,
            device,
            n_taxa,
            "transformer",
            predecoded_graphs or None,
            predecoded_splits or None,
            shared_view_input,
        )
    coordinate_history, current_graph, current_path = _run_refinement_stage(
        _coordinate_pass,
        "coordinate",
        current_graph,
        refinement_context,
        n_taxa,
        work_dir,
        coordinate_stop_ratio,
        coordinate_max_rounds,
        trace_performance,
    )
    saturating_history, current_graph, current_path = _run_refinement_stage(
        _saturating_pass,
        "saturating",
        current_graph,
        refinement_context,
        n_taxa,
        work_dir,
        saturation_stop_ratio,
        saturation_max_rounds,
        trace_performance,
    )

    output_temporary = output.with_suffix(output.suffix + ".tmp")
    shutil.copyfile(current_path, output_temporary)
    output_temporary.replace(output)
    output_sha256 = sha256_file(output)
    row: dict[str, Any] = {
        "status": "complete",
        "version": __version__,
        "algorithm_graph": "concordtree-0.1.2",
        "n_taxa": n_taxa,
        "alignment_length": alignment_length,
        "input_msa": str(msa),
        "input_sha256": manifest["input_sha256"],
        "final_prediction": str(output),
        "final_prediction_sha256": output_sha256,
        "reference_access": "none",
        "quartet_model": quartet_model,
        "execution_mode": execution_mode,
        "refinement_scorer": scorer_label,
        "quartet_predictor": quartet_predictor,
        "missing_data_model": missing_data_model,
        "candidate_distance_backend": candidate_distance_backend,
        "missing_distance_model": missing_distance_model,
        "scaffold_row_sum_backend": row_sum_backend,
        "nni_reduction_backend": nni_reduction_backend,
        "compact_imputed_first_round": (
            os.environ.get("CONCORDTREE_COMPACT_IMPUTED_FIRST_ROUND", "1") == "1"
        ),
        "profile_stream_policy": os.environ.get(
            "CONCORDTREE_STREAM_PROFILE_SITES", "auto:2048-sites-at-8GiB"
        ),
        "view_executor": view_executor,
        "post_view_schedule": post_view_schedule,
        "view_count": view_count,
        "view_workers": view_workers,
        "requested_view_workers": requested_view_workers,
        "requested_parallelism": parallelism,
        "blas_threads_per_view": int(os.environ.get("OPENBLAS_NUM_THREADS", "1")),
        "candidate_cpu_workers_per_view": int(
            os.environ.get("CONCORDTREE_CANDIDATE_CPU_WORKERS", "8")
        ),
        "views_complete_seconds": views_complete_seconds,
        "fork_shared_input_seconds": shared_view_input_seconds,
        "fork_shared_input_bytes": (
            _fork_shared_input_nbytes(shared_view_input)
            if shared_view_input is not None
            else 0
        ),
        "view_pool_shutdown_seconds": view_pool_shutdown_seconds,
        "later_stop_moves": (
            refinement_controls["coordinate"]["stop_moves"]
            if refinement_controls["coordinate"]["stop_moves"]
            == refinement_controls["saturation"]["stop_moves"]
            else None
        ),
        "refinement_controls": refinement_controls,
        "refinement_stops": {
            "views": [
                {
                    "view": int(item["view"]),
                    "executed_rounds": int(item["nni_passes_executed"]),
                    "last_moves": int(item["nni_last_moves"]),
                    "last_move_ratio": float(item["nni_last_move_ratio"]),
                    "reasons": list(item["nni_stop_reasons"]),
                }
                for item in view_rows
            ],
            "coordinate": {
                "executed_rounds": len(coordinate_history),
                "last_moves": int(coordinate_history[-1]["moves"]),
                "last_move_ratio": float(coordinate_history[-1]["move_ratio"]),
                "reasons": list(coordinate_history[-1]["stop_reasons"]),
            },
            "saturation": {
                "executed_rounds": len(saturating_history),
                "last_moves": int(saturating_history[-1]["moves"]),
                "last_move_ratio": float(saturating_history[-1]["move_ratio"]),
                "reasons": list(saturating_history[-1]["stop_reasons"]),
            },
        },
        "initialization_setup": {
            "bundle_load_seconds": initialization_bundle_seconds,
            **initialization_context_metrics,
        },
        "refinement_setup": {
            "bundle_load_seconds": refinement_bundle_seconds,
            **refinement_context_metrics,
        },
        "consensus": consensus,
        "splitbank": splitbank,
        "coordinate_history": coordinate_history,
        "saturating_history": saturating_history,
        "view_seconds": [float(item["total_seconds"]) for item in view_rows],
        "wall_seconds": time.perf_counter() - started,
        "peak_post_view_cuda_memory_bytes": int(
            torch.cuda.max_memory_allocated(device)
        ),
        "maximum_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "panel_probability_cache": {
            "entries": len(refinement_context.probability_cache),
            "bytes": sum(
                value.nbytes for value in refinement_context.probability_cache.values()
            ),
            "policy": f"LRU-{PANEL_CACHE_MAX_ENTRIES}",
        },
    }
    atomic_json(final_record, row)
    manifest.update(
        {
            "status": "complete",
            "completed_at": utc_now(),
            "wall_seconds": row["wall_seconds"],
            "final_prediction_sha256": output_sha256,
        }
    )
    atomic_json(work_dir / "run_manifest.json", manifest)
    (work_dir / "summary.md").write_text(
        f"# ConcordTree {quartet_model} inference\n\n"
        f"- Status: `complete`\n"
        f"- Quartet model / refinement scorer: `{quartet_model}` / `{scorer_label}`\n"
        f"- Quartet predictor: `{quartet_predictor}`\n"
        f"- Taxa / sites: `{n_taxa}` / `{alignment_length}`\n"
        f"- Wall seconds: `{row['wall_seconds']:.3f}`\n"
        f"- Coordinate moves: `{[item['moves'] for item in coordinate_history]}`\n"
        f"- Saturating moves: `{[item['moves'] for item in saturating_history]}`\n"
        f"- Refinement controls: `{refinement_controls}`\n"
        f"- Coordinate stop: `{row['refinement_stops']['coordinate']}`\n"
        f"- Saturation stop: `{row['refinement_stops']['saturation']}`\n"
        f"- Output SHA256: `{output_sha256}`\n"
        "- Reference topology access: `none`\n"
    )
    return row


def doctor(device_name: str = "cuda:0", *, verify_hashes: bool = True) -> dict[str, Any]:
    """Validate the packaged runtime without reading an MSA."""

    report: dict[str, Any] = {
        "version": __version__,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": device_name,
        "expected_assets": dict(ASSET_SHA256),
    }
    if verify_hashes:
        report["assets"] = verify_assets()
    if not torch.cuda.is_available():
        report["status"] = "blocked"
        report["error"] = "CUDA is unavailable"
        return report
    device = torch.device(device_name)
    torch.cuda.set_device(device)
    sequence, pattern = load_backends()
    panel_score = load_panel_score_backend()
    nni_plan = load_learned_nni_plan_backend()
    report["sequence_backend"] = str(sequence.__file__)
    report["pattern_backend"] = str(pattern.__file__)
    report["panel_score_backend"] = str(panel_score.__file__)
    report["learned_nni_plan_backend"] = str(nni_plan.__file__)
    report["gpu"] = torch.cuda.get_device_name(device)
    report["status"] = "ok"
    return report
