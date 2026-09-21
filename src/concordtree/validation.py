"""Topology-level validation helpers for ConcordTree releases."""

from __future__ import annotations

import hashlib
from pathlib import Path

from concordtree._core.eapc_reachability import split_bitmasks
from concordtree._core.learned_nni import tree_path_to_graph
from concordtree._core.scaleqf import validate_topology
from concordtree._core.sparse_treeformer_data import (
    load_sequential_phylip_sketch,
)


def canonical_split_set(msa: Path, tree: Path) -> frozenset[int]:
    names = list(
        load_sequential_phylip_sketch(
            msa.resolve(strict=True), max_sites=1, seed=1
        ).names
    )
    graph = tree_path_to_graph(tree.resolve(strict=True), names)
    validate_topology(graph, len(names))
    return split_bitmasks(graph, len(names))


def canonical_split_sha256(msa: Path, tree: Path) -> str:
    splits = sorted(canonical_split_set(msa, tree))
    payload = "\n".join(str(split) for split in splits).encode()
    return hashlib.sha256(payload).hexdigest()


def compare_topologies(msa: Path, expected: Path, actual: Path) -> dict[str, object]:
    expected_splits = canonical_split_set(msa, expected)
    actual_splits = canonical_split_set(msa, actual)
    missing = sorted(expected_splits - actual_splits)
    extra = sorted(actual_splits - expected_splits)
    return {
        "equivalent": not missing and not extra,
        "expected_splits": len(expected_splits),
        "actual_splits": len(actual_splits),
        "missing": missing,
        "extra": extra,
        "expected_split_sha256": canonical_split_sha256(msa, expected),
        "actual_split_sha256": canonical_split_sha256(msa, actual),
    }
