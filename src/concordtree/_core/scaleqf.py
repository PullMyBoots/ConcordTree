"""ScaleQF v0: bounded-sketch phylogenetic scaffold and local NNI repair.

The implementation deliberately avoids a global distance matrix.  It reads a
bounded, stratified column sketch from a sequential PHYLIP alignment, builds
constant-size Neighbor-Joining subtrees inside a recursive anchor partition,
and repairs internal edges using four-point scores on a few representatives
from each incident subtree.

This bounded-scaffold component is retained as part of the frozen ConcordTree path.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import numpy as np


DNA_LUT = np.full(256, 255, dtype=np.uint8)
for _base, _code in (("A", 0), ("C", 1), ("G", 2), ("T", 3), ("U", 3)):
    DNA_LUT[ord(_base)] = _code
    DNA_LUT[ord(_base.lower())] = _code


@dataclass
class TreeNode:
    """Small rooted representation used only while constructing a topology."""

    leaf: int | None = None
    children: list["TreeNode"] = field(default_factory=list)


@dataclass
class AlignmentSketch:
    names: list[str]
    states: np.ndarray
    positions: np.ndarray
    block_ids: np.ndarray
    alignment_length: int
    requested_sites: int

    @property
    def n_taxa(self) -> int:
        return len(self.names)

    @property
    def n_sites(self) -> int:
        return int(self.states.shape[1])


class BoundedDistanceOracle:
    """On-demand robust JC69 distances with bounded pair caching."""

    def __init__(
        self,
        sketch: AlignmentSketch,
        cache_size: int = 200_000,
        trim_fraction: float = 0.1,
    ) -> None:
        self.sketch = sketch
        self.cache_size = int(cache_size)
        self.trim_fraction = float(trim_fraction)
        self._cache: OrderedDict[tuple[int, int], float] = OrderedDict()
        self.calls = 0
        self.cache_hits = 0

    @staticmethod
    def _jc69(p: float) -> float:
        if not math.isfinite(p):
            return 10.0
        p = min(max(float(p), 0.0), 0.749999)
        return -0.75 * math.log(max(1.0 - (4.0 * p / 3.0), 1e-9))

    def __call__(self, i: int, j: int) -> float:
        if i == j:
            return 0.0
        key = (i, j) if i < j else (j, i)
        cached = self._cache.get(key)
        if cached is not None:
            self.cache_hits += 1
            self._cache.move_to_end(key)
            return cached

        self.calls += 1
        a = self.sketch.states[key[0]]
        b = self.sketch.states[key[1]]
        valid = (a < 4) & (b < 4)
        if not np.any(valid):
            value = 10.0
        else:
            per_block: list[tuple[float, int]] = []
            for block in np.unique(self.sketch.block_ids[valid]):
                mask = valid & (self.sketch.block_ids == block)
                count = int(mask.sum())
                if count >= 4:
                    p = float(np.count_nonzero(a[mask] != b[mask])) / count
                    per_block.append((self._jc69(p), count))

            if len(per_block) >= 4:
                per_block.sort(key=lambda item: item[0])
                trim = int(len(per_block) * self.trim_fraction)
                kept = per_block[trim : len(per_block) - trim] if trim else per_block
                values = np.asarray([x[0] for x in kept], dtype=np.float64)
                weights = np.asarray([x[1] for x in kept], dtype=np.float64)
                value = float(np.average(values, weights=weights))
            else:
                p = float(np.count_nonzero(a[valid] != b[valid])) / int(valid.sum())
                value = self._jc69(p)

        self._cache[key] = value
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return value

    @property
    def cache_entries(self) -> int:
        return len(self._cache)


def _stratified_positions(length: int, max_sites: int, seed: int) -> np.ndarray:
    if length <= max_sites:
        return np.arange(length, dtype=np.int64)
    rng = np.random.default_rng(seed)
    edges = np.linspace(0, length, max_sites + 1, dtype=np.int64)
    positions = np.empty(max_sites, dtype=np.int64)
    for idx in range(max_sites):
        low = int(edges[idx])
        high = max(int(edges[idx + 1]), low + 1)
        positions[idx] = rng.integers(low, high)
    positions.sort()
    return positions


def load_phylip_sketch(
    path: str | Path,
    max_sites: int = 8192,
    blocks: int = 32,
    seed: int = 20260828,
    progress_callback: Callable[[int, int], None] | None = None,
) -> AlignmentSketch:
    """Loads selected columns from a sequential, one-record-per-line PHYLIP file."""

    path = Path(path)
    with path.open("rb") as handle:
        header = handle.readline().split()
        if len(header) < 2:
            raise ValueError(f"Invalid PHYLIP header: {path}")
        n_taxa, length = int(header[0]), int(header[1])
        positions = _stratified_positions(length, max_sites, seed)
        states = np.empty((n_taxa, len(positions)), dtype=np.uint8)
        names: list[str] = []

        for row in range(n_taxa):
            line = handle.readline()
            while line and not line.strip():
                line = handle.readline()
            if not line:
                raise ValueError(f"Unexpected EOF after {row}/{n_taxa} taxa: {path}")
            parts = line.rstrip().split(maxsplit=1)
            if len(parts) != 2:
                raise ValueError(
                    "ScaleQF v0 expects sequential PHYLIP with one taxon per line; "
                    f"bad record {row + 1} in {path}"
                )
            name = parts[0].decode("utf-8")
            sequence = re.sub(rb"\s+", b"", parts[1])
            if len(sequence) != length:
                raise ValueError(
                    "ScaleQF v0 expects one complete aligned sequence per line; "
                    f"{name!r} has {len(sequence)} sites, expected {length}"
                )
            raw = np.frombuffer(sequence, dtype=np.uint8)
            states[row] = DNA_LUT[raw[positions]]
            names.append(name)
            if progress_callback is not None:
                progress_callback(row + 1, n_taxa)

    # Conserved and all-missing sampled columns add runtime but no topology signal.
    variable = np.zeros(states.shape[1], dtype=bool)
    for col in range(states.shape[1]):
        observed = states[:, col]
        observed = observed[observed < 4]
        variable[col] = observed.size >= 2 and np.any(observed != observed[0])
    if np.any(variable):
        states = states[:, variable]
        positions = positions[variable]
    block_ids = np.minimum((positions * blocks) // max(length, 1), blocks - 1).astype(
        np.int16
    )
    return AlignmentSketch(
        names=names,
        states=states,
        positions=positions,
        block_ids=block_ids,
        alignment_length=length,
        requested_sites=max_sites,
    )


def neighbor_joining(indices: list[int], distance: Callable[[int, int], float]) -> TreeNode:
    """Exact NJ inside one bounded leaf block."""

    if not indices:
        raise ValueError("NJ received an empty block")
    if len(indices) == 1:
        return TreeNode(leaf=indices[0])
    nodes = [TreeNode(leaf=i) for i in indices]
    matrix = np.asarray(
        [[distance(i, j) for j in indices] for i in indices], dtype=np.float64
    )

    while len(nodes) > 2:
        count = len(nodes)
        row_sums = matrix.sum(axis=1)
        q = (count - 2) * matrix - row_sums[:, None] - row_sums[None, :]
        np.fill_diagonal(q, np.inf)
        left, right = np.unravel_index(np.argmin(q), q.shape)
        if right < left:
            left, right = right, left
        merged = TreeNode(children=[nodes[left], nodes[right]])
        keep = [idx for idx in range(count) if idx not in (left, right)]
        new_dist = 0.5 * (
            matrix[left, keep] + matrix[right, keep] - matrix[left, right]
        )
        reduced = matrix[np.ix_(keep, keep)]
        matrix = np.block(
            [
                [reduced, new_dist[:, None]],
                [new_dist[None, :], np.zeros((1, 1), dtype=np.float64)],
            ]
        )
        nodes = [nodes[idx] for idx in keep] + [merged]
    return TreeNode(children=[nodes[0], nodes[1]])


def build_scaffold(
    indices: list[int],
    distance: Callable[[int, int], float],
    leaf_size: int = 24,
) -> TreeNode:
    """Anchor-bisect recursively, resolving constant-size leaves with NJ."""

    if len(indices) <= leaf_size:
        return neighbor_joining(indices, distance)

    ordered = sorted(indices)
    seed = ordered[0]
    anchor_a = max(ordered, key=lambda idx: (distance(seed, idx), -idx))
    anchor_b = max(ordered, key=lambda idx: (distance(anchor_a, idx), -idx))
    signed = [(distance(idx, anchor_a) - distance(idx, anchor_b), idx) for idx in ordered]
    left = [idx for delta, idx in signed if delta <= 0]
    right = [idx for delta, idx in signed if delta > 0]

    min_side = max(2, int(math.ceil(0.1 * len(ordered))))
    if len(left) < min_side or len(right) < min_side:
        signed.sort(key=lambda item: (item[0], item[1]))
        middle = len(signed) // 2
        left = [idx for _, idx in signed[:middle]]
        right = [idx for _, idx in signed[middle:]]

    if not left or not right:
        middle = len(ordered) // 2
        left, right = ordered[:middle], ordered[middle:]
    return TreeNode(
        children=[
            build_scaffold(left, distance, leaf_size),
            build_scaffold(right, distance, leaf_size),
        ]
    )


def tree_to_graph(root: TreeNode, n_taxa: int) -> dict[int, set[int]]:
    """Convert a rooted helper tree without imposing a recursion-depth bound."""

    adjacency: dict[int, set[int]] = {}
    next_internal = n_taxa

    root_id: int | None = None
    stack: list[tuple[TreeNode, int | None]] = [(root, None)]
    while stack:
        node, parent_id = stack.pop()
        if node.leaf is not None:
            node_id = node.leaf
        else:
            node_id = next_internal
            next_internal += 1
        adjacency.setdefault(node_id, set())
        if parent_id is None:
            root_id = node_id
        else:
            adjacency[parent_id].add(node_id)
            adjacency[node_id].add(parent_id)
        if node.leaf is None:
            # Reverse the push order so internal ids retain the recursive
            # preorder assignment consumed by deterministic downstream code.
            stack.extend((child, node_id) for child in reversed(node.children))
    if root_id is None:
        raise AssertionError("tree traversal did not visit a root")
    # A rooted binary Newick root has degree two.  Suppressing it yields the
    # standard unrooted binary topology and exposes the central NNI edge.
    if root_id >= n_taxa and len(adjacency[root_id]) == 2:
        left, right = sorted(adjacency[root_id])
        adjacency[left].remove(root_id)
        adjacency[right].remove(root_id)
        adjacency[left].add(right)
        adjacency[right].add(left)
        del adjacency[root_id]
    return adjacency


def _representatives(
    adjacency: dict[int, set[int]],
    start: int,
    blocked: int,
    n_taxa: int,
    limit: int,
) -> list[int]:
    stack = [(start, blocked)]
    leaves: list[int] = []
    while stack and len(leaves) < limit:
        node, parent = stack.pop()
        if node < n_taxa:
            leaves.append(node)
            continue
        children = sorted((x for x in adjacency[node] if x != parent), reverse=True)
        stack.extend((child, node) for child in children)
    return sorted(leaves)


def _between_group_distance(
    group_a: Iterable[int],
    group_b: Iterable[int],
    distance: Callable[[int, int], float],
) -> float:
    values = [distance(a, b) for a in group_a for b in group_b]
    return float(np.mean(values)) if values else math.inf


def refine_nni(
    adjacency: dict[int, set[int]],
    n_taxa: int,
    distance: Callable[[int, int], float],
    passes: int = 2,
    representatives: int = 3,
    relative_gain: float = 1e-3,
) -> dict[str, object]:
    """Greedy NNI using representative-averaged four-point scores."""

    moves_per_pass: list[int] = []
    gain_per_pass: list[float] = []
    for _ in range(passes):
        moves = 0
        total_gain = 0.0
        internal_edges = sorted(
            (u, v)
            for u in adjacency
            for v in adjacency[u]
            if u < v and u >= n_taxa and v >= n_taxa
        )
        for u, v in internal_edges:
            if len(adjacency[u]) != 3 or len(adjacency[v]) != 3:
                continue
            u_side = sorted(x for x in adjacency[u] if x != v)
            v_side = sorted(x for x in adjacency[v] if x != u)
            if len(u_side) != 2 or len(v_side) != 2:
                continue
            branch_nodes = [u_side[0], u_side[1], v_side[0], v_side[1]]
            owners = [u, u, v, v]
            groups = [
                _representatives(
                    adjacency, node, owner, n_taxa, max(1, representatives)
                )
                for node, owner in zip(branch_nodes, owners)
            ]
            if any(not group for group in groups):
                continue
            a, b, c, d = groups
            scores = [
                _between_group_distance(a, b, distance)
                + _between_group_distance(c, d, distance),
                _between_group_distance(a, c, distance)
                + _between_group_distance(b, d, distance),
                _between_group_distance(a, d, distance)
                + _between_group_distance(b, c, distance),
            ]
            best = int(np.argmin(scores))
            required = max(1e-9, abs(scores[0]) * relative_gain)
            if best == 0 or scores[0] - scores[best] <= required:
                continue

            displaced_u = u_side[1]
            displaced_v = v_side[0] if best == 1 else v_side[1]
            adjacency[u].remove(displaced_u)
            adjacency[displaced_u].remove(u)
            adjacency[v].remove(displaced_v)
            adjacency[displaced_v].remove(v)
            adjacency[u].add(displaced_v)
            adjacency[displaced_v].add(u)
            adjacency[v].add(displaced_u)
            adjacency[displaced_u].add(v)
            moves += 1
            total_gain += scores[0] - scores[best]
        moves_per_pass.append(moves)
        gain_per_pass.append(total_gain)
        if moves == 0:
            break
    return {"moves_per_pass": moves_per_pass, "gain_per_pass": gain_per_pass}


def _newick_label(name: str) -> str:
    # ETE3's default Newick reader treats single quotes as literal label
    # characters.  ``+`` is safe unquoted and occurs in empirical taxon names.
    if re.fullmatch(r"[A-Za-z0-9_.+\-]+", name):
        return name
    return "'" + name.replace("'", "''") + "'"


def graph_to_newick(
    adjacency: dict[int, set[int]], names: list[str], n_taxa: int
) -> str:
    degree_two = sorted(
        node for node in adjacency if node >= n_taxa and len(adjacency[node]) == 2
    )
    internal = sorted(node for node in adjacency if node >= n_taxa)
    if not internal:
        raise ValueError("Topology has no internal node")
    root = degree_two[0] if degree_two else internal[0]

    output: list[str] = []
    stack: list[tuple[str, int | str, int | None]] = [("node", root, None)]
    while stack:
        kind, value, parent = stack.pop()
        if kind == "text":
            output.append(str(value))
            continue
        node = int(value)
        if node < n_taxa:
            output.append(_newick_label(names[node]))
            continue
        children = sorted(x for x in adjacency[node] if x != parent)
        if not children:
            raise ValueError(f"Internal node {node} became empty")
        output.append("(")
        stack.append(("text", ")", None))
        for index in range(len(children) - 1, -1, -1):
            stack.append(("node", children[index], node))
            if index > 0:
                stack.append(("text", ",", None))
    return "".join(output) + ";"


def validate_topology(adjacency: dict[int, set[int]], n_taxa: int) -> None:
    if any(leaf not in adjacency or len(adjacency[leaf]) != 1 for leaf in range(n_taxa)):
        raise ValueError("Topology did not preserve all leaves at degree one")
    edge_count = sum(len(values) for values in adjacency.values()) // 2
    if edge_count != len(adjacency) - 1:
        raise ValueError("Topology is not a tree")
    visited: set[int] = set()
    stack = [0]
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        stack.extend(adjacency[node] - visited)
    if len(visited) != len(adjacency):
        raise ValueError("Topology is disconnected")


def infer_alignment(
    alignment_path: str | Path,
    max_sites: int = 8192,
    blocks: int = 32,
    leaf_size: int = 24,
    nni_passes: int = 2,
    representatives: int = 3,
    seed: int = 20260828,
) -> dict[str, object]:
    started = time.perf_counter()
    sketch = load_phylip_sketch(alignment_path, max_sites=max_sites, blocks=blocks, seed=seed)
    loaded_at = time.perf_counter()
    if sketch.n_taxa < 4:
        raise ValueError("ScaleQF requires at least four taxa")
    if sketch.n_sites == 0:
        raise ValueError("The sampled alignment contains no variable nucleotide sites")
    distance = BoundedDistanceOracle(sketch)
    scaffold_root = build_scaffold(list(range(sketch.n_taxa)), distance, leaf_size=leaf_size)
    scaffold_graph = tree_to_graph(scaffold_root, sketch.n_taxa)
    validate_topology(scaffold_graph, sketch.n_taxa)
    scaffold_newick = graph_to_newick(scaffold_graph, sketch.names, sketch.n_taxa)
    scaffold_at = time.perf_counter()
    repair = refine_nni(
        scaffold_graph,
        sketch.n_taxa,
        distance,
        passes=nni_passes,
        representatives=representatives,
    )
    validate_topology(scaffold_graph, sketch.n_taxa)
    refined_newick = graph_to_newick(scaffold_graph, sketch.names, sketch.n_taxa)
    finished = time.perf_counter()
    return {
        "scaffold_newick": scaffold_newick,
        "refined_newick": refined_newick,
        "metadata": {
            "alignment": str(Path(alignment_path).resolve()),
            "n_taxa": sketch.n_taxa,
            "alignment_length": sketch.alignment_length,
            "requested_sites": sketch.requested_sites,
            "variable_sketch_sites": sketch.n_sites,
            "blocks": blocks,
            "leaf_size": leaf_size,
            "nni_passes": nni_passes,
            "representatives": representatives,
            "seed": seed,
            "repair": repair,
            "distance_calls": distance.calls,
            "distance_cache_hits": distance.cache_hits,
            "distance_cache_entries": distance.cache_entries,
            "load_seconds": loaded_at - started,
            "scaffold_seconds": scaffold_at - loaded_at,
            "repair_seconds": finished - scaffold_at,
            "total_seconds": finished - started,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("alignment", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--max-sites", type=int, default=8192)
    parser.add_argument("--blocks", type=int, default=32)
    parser.add_argument("--leaf-size", type=int, default=24)
    parser.add_argument("--nni-passes", type=int, default=2)
    parser.add_argument("--representatives", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260828)
    args = parser.parse_args()
    result = infer_alignment(
        args.alignment,
        max_sites=args.max_sites,
        blocks=args.blocks,
        leaf_size=args.leaf_size,
        nni_passes=args.nni_passes,
        representatives=args.representatives,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(str(result["refined_newick"]) + "\n")
    args.output.with_suffix(args.output.suffix + ".scaffold.nwk").write_text(
        str(result["scaffold_newick"]) + "\n"
    )
    args.output.with_suffix(args.output.suffix + ".json").write_text(
        json.dumps(result["metadata"], indent=2) + "\n"
    )
    print(json.dumps(result["metadata"], sort_keys=True))


if __name__ == "__main__":
    main()
