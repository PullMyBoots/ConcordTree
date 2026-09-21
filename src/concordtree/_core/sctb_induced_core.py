"""Shared-evidence induced-core scores for bounded attachment candidates."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable

import numpy as np

from concordtree._core.eapc_reachability import reattach_leaf
from concordtree._core.sctb_neural_router import Distance, attachment_specifications
from concordtree._core.sparse_nj import Edge, canonical_edge


Predictor = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class InducedCoreScores:
    candidates: tuple[Edge, ...]
    scores: np.ndarray
    quartets: np.ndarray
    probabilities: np.ndarray
    displayed_classes: np.ndarray


def attachment_pseudolikelihood_from_probabilities(
    query: int,
    candidates: list[Edge] | tuple[Edge, ...],
    adjacency: dict[int, set[int]],
    quartets: np.ndarray,
    probabilities: np.ndarray,
) -> InducedCoreScores:
    """Score candidates against one already-inferred shared quartet context."""

    ordered = tuple(sorted({canonical_edge(*edge) for edge in candidates}))
    quartets = np.asarray(quartets, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if not ordered:
        raise ValueError("candidate set is empty")
    if quartets.ndim != 2 or quartets.shape[1] != 4 or len(quartets) == 0:
        raise ValueError("quartets must have nonempty shape (m, 4)")
    if probabilities.shape != (len(quartets), 3):
        raise ValueError(f"probability shape mismatch: {probabilities.shape}")
    if not np.isfinite(probabilities).all():
        raise ValueError("nonfinite quartet probabilities")
    classes = []
    scores = []
    for edge in ordered:
        graph = reattach_leaf(adjacency, query, edge)
        displayed = displayed_quartet_classes(graph, quartets)
        classes.append(displayed)
        scores.append(
            float(
                np.mean(
                    np.log(
                        np.clip(
                            probabilities[np.arange(len(quartets)), displayed],
                            1e-8,
                            1.0,
                        )
                    )
                )
            )
        )
    return InducedCoreScores(
        candidates=ordered,
        scores=np.asarray(scores, dtype=np.float64),
        quartets=quartets,
        probabilities=probabilities,
        displayed_classes=np.stack(classes),
    )


def displayed_quartet_classes(
    adjacency: dict[int, set[int]], quartets: np.ndarray
) -> np.ndarray:
    """Return the three-class topology displayed by a binary tree."""

    quartets = np.asarray(quartets, dtype=np.int64)
    if quartets.ndim != 2 or quartets.shape[1] != 4:
        raise ValueError("quartets must have shape (m, 4)")
    taxa = sorted({int(value) for value in quartets.reshape(-1)})
    if any(taxon not in adjacency for taxon in taxa):
        raise ValueError("quartet taxon is absent from the candidate tree")
    position = {taxon: index for index, taxon in enumerate(taxa)}
    distances = np.empty((len(taxa), len(taxa)), dtype=np.int32)
    for row, source in enumerate(taxa):
        found = {source: 0}
        queue = deque([source])
        while queue:
            node = queue.popleft()
            for neighbor in adjacency[node]:
                if neighbor not in found:
                    found[neighbor] = found[node] + 1
                    queue.append(neighbor)
        distances[row] = [found[target] for target in taxa]
    # Map all quartet labels once and evaluate the three four-point sums in
    # vectorized NumPy.  This is exactly the former row loop, but contextual
    # 24-taxon panels call it for all C(24, 4)=10,626 quartets, where Python
    # iteration dominated the neural model's runtime.
    mapped = np.fromiter(
        (position[int(value)] for value in quartets.reshape(-1)),
        dtype=np.int32,
        count=quartets.size,
    ).reshape(-1, 4)
    a, b, c, d = mapped.T
    sums = np.column_stack(
        (
            distances[a, b] + distances[c, d],
            distances[a, c] + distances[b, d],
            distances[a, d] + distances[b, c],
        )
    )
    return np.argmin(sums, axis=1).astype(np.int8, copy=False)


def shared_attachment_pseudolikelihood(
    query: int,
    candidates: list[Edge],
    adjacency: dict[int, set[int]],
    distance: Distance,
    predict_probabilities: Predictor,
    n_taxa: int,
    current_taxa: int,
    representatives: int = 2,
) -> InducedCoreScores:
    """Score every candidate on one deduplicated union of quartet witnesses."""

    ordered = tuple(sorted({canonical_edge(*edge) for edge in candidates}))
    if not ordered:
        raise ValueError("candidate set is empty")
    specifications = attachment_specifications(
        query,
        list(ordered),
        adjacency,
        distance,
        n_taxa,
        current_taxa,
        representatives,
    )
    quartets = np.asarray(
        sorted({tuple(sorted(item[1])) for item in specifications}), dtype=np.int64
    )
    if quartets.size == 0:
        raise ValueError("candidate set produced no induced-core witnesses")
    probabilities = np.asarray(predict_probabilities(quartets), dtype=np.float64)
    return attachment_pseudolikelihood_from_probabilities(
        query, ordered, adjacency, quartets, probabilities
    )
