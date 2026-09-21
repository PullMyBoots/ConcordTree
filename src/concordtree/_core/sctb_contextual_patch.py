"""Amortized contextual panels for bounded NNI patch refinement.

One fixed-size panel is shared by several nearby tree edges.  A contextual
quartet posterior tensor is scored against the current topology and both NNI
alternatives of every covered edge.  The implementation uses the exact fact
that an NNI changes only quartets containing one leaf from each of the four
incident subtrees, so candidate scoring never reconstructs a tree per action.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from itertools import combinations, product

import numpy as np

from concordtree._core.learned_nni import (
    EdgeEvidence,
    apply_independent_nni,
    class_for_group_pairing,
    near_edge_representatives,
)
from concordtree._core.graphrank_laminar import canonical_split
from concordtree._core.sctb_oracle_nni import directed_edge_leaf_masks
from concordtree._core.sctb_induced_core import displayed_quartet_classes
from concordtree._core.scaleqf import neighbor_joining, tree_to_graph


Edge = tuple[int, int]

_PAIR_CLASS_BY_RANK = np.asarray(
    (
        (-1, 0, 1, 2),
        (0, -1, 2, 1),
        (1, 2, -1, 0),
        (2, 1, 0, -1),
    ),
    dtype=np.int8,
)


@dataclass(frozen=True)
class ContextualPanel:
    cover: int
    panel_id: int
    taxa: tuple[int, ...]
    target_edges: tuple[Edge, ...]


@dataclass
class _PanelCoverWorkspace:
    """Tree-local topology facts shared by deterministic panel covers."""

    edges: tuple[Edge, ...]
    neighbors: dict[Edge, tuple[Edge, ...]]
    branches: dict[Edge, tuple[int, int, int, int]]
    representatives: dict[tuple[int, Edge], tuple[int, int, int, int]]


@dataclass(frozen=True)
class PanelEdgeScore:
    cover: int
    panel_id: int
    edge: Edge
    branch_nodes: tuple[int, int, int, int]
    scores: tuple[float, float, float]
    best: int
    gain: float
    discriminating_rows: int
    alternative_median_gains: tuple[float, float]


@dataclass(frozen=True)
class SparsePanelEdgePlan:
    """One edge's exact class gathers into a sparse panel row array."""

    edge: Edge
    branch_nodes: tuple[int, int, int, int]
    positions: np.ndarray
    current: np.ndarray
    alternative_one: np.ndarray
    alternative_two: np.ndarray


@dataclass(frozen=True)
class SparseContextualPanelPlan:
    """The union of quartet rows that can distinguish target-edge NNIs."""

    cover: int
    panel_id: int
    taxa: tuple[int, ...]
    row_indices: np.ndarray
    edges: tuple[SparsePanelEdgePlan, ...]
    emitted_rows: int


@dataclass(frozen=True)
class PanelScoreResult:
    edges: tuple[PanelEdgeScore, ...]
    base_score: float
    discriminating_rows: int
    emitted_rows: int
    predicted_classes: np.ndarray
    discriminating_mask: np.ndarray


@dataclass(frozen=True)
class VirtualPanelTree:
    """Compressed induced tree plus lift ports into the full topology.

    ``ports[(u, v)]`` is the immediate full-tree neighbor reached when walking
    from virtual node ``u`` toward virtual node ``v``.  A virtual NNI can
    therefore be lifted without materializing or traversing the full tree.
    """

    adjacency: dict[int, set[int]]
    ports: dict[tuple[int, int], int]
    taxa: tuple[int, ...]


@dataclass(frozen=True)
class PanelPatchMove:
    """One liftable NNI in a connected panel patch."""

    edge: Edge
    branch_nodes: tuple[int, int, int, int]
    best: int
    step_gain: float
    changed_rows: int


@dataclass(frozen=True)
class PanelPatch:
    """A locally optimized, globally liftable sequence of NNI edits."""

    cover: int
    panel_id: int
    moves: tuple[PanelPatchMove, ...]
    base_score: float
    final_score: float
    gain: float
    changed_rows: int
    evaluated_candidates: int


@dataclass
class _PatchState:
    adjacency: dict[int, set[int]]
    ports: dict[tuple[int, int], int]
    classes: np.ndarray
    score_sum: float
    moves: tuple[PanelPatchMove, ...]
    used_edges: frozenset[Edge]


def canonical_edge(left: int, right: int) -> Edge:
    return (left, right) if left < right else (right, left)


def internal_edges(adjacency: dict[int, set[int]], n_taxa: int) -> list[Edge]:
    return sorted(
        canonical_edge(left, right)
        for left, neighbors in adjacency.items()
        for right in neighbors
        if left < right and left >= n_taxa and right >= n_taxa
    )


def branch_nodes_for_edge(
    adjacency: dict[int, set[int]], edge: Edge
) -> tuple[int, int, int, int]:
    left, right = edge
    left_side = sorted(node for node in adjacency[left] if node != right)
    right_side = sorted(node for node in adjacency[right] if node != left)
    if len(left_side) != 2 or len(right_side) != 2:
        raise ValueError(f"edge {edge} is not an internal binary edge")
    return left_side[0], left_side[1], right_side[0], right_side[1]


def _representatives_for_edge(
    adjacency: dict[int, set[int]],
    edge: Edge,
    n_taxa: int,
    cover: int,
    tree_index: ContextualTreeIndex | None = None,
    branches: tuple[int, int, int, int] | None = None,
) -> tuple[int, int, int, int]:
    left, right = edge
    if branches is None:
        branches = branch_nodes_for_edge(adjacency, edge)
    owners = (left, left, right, right)
    selected: list[int] = []
    # Covers 0/1 retain the original nearest/second-nearest contract.  Further
    # covers consume the remaining fixed top-k messages rather than silently
    # duplicating cover 0.
    rank = cover % 4
    for branch, owner in zip(branches, owners):
        values = (
            list(tree_index.nearest[(branch, owner)])
            if tree_index is not None
            else near_edge_representatives(
                adjacency, branch, owner, n_taxa, limit=rank + 1
            )
        )
        selected.append(values[min(rank, len(values) - 1)])
    if len(set(selected)) != 4:
        raise AssertionError(f"edge representatives are not distinct: {edge} {selected}")
    return tuple(selected)


def _edge_neighbors(edges: list[Edge]) -> dict[Edge, list[Edge]]:
    incident: dict[int, list[Edge]] = {}
    for edge in edges:
        for node in edge:
            incident.setdefault(node, []).append(edge)
    result: dict[Edge, list[Edge]] = {}
    for edge in edges:
        result[edge] = sorted(
            {other for node in edge for other in incident[node] if other != edge}
        )
    return result


def _panel_cover_workspace(
    adjacency: dict[int, set[int]], n_taxa: int
) -> _PanelCoverWorkspace:
    edges = tuple(internal_edges(adjacency, n_taxa))
    neighbors = {
        edge: tuple(values) for edge, values in _edge_neighbors(list(edges)).items()
    }
    return _PanelCoverWorkspace(
        edges=edges,
        neighbors=neighbors,
        branches={
            edge: branch_nodes_for_edge(adjacency, edge) for edge in edges
        },
        representatives={},
    )


def _local_context_taxa(
    adjacency: dict[int, set[int]],
    target_edges: list[Edge],
    selected: set[int],
    n_taxa: int,
    cover: int,
) -> list[int]:
    """Return nearby leaves without materializing a dense taxon matrix."""

    starts = sorted({node for edge in target_edges for node in edge}, reverse=bool(cover % 2))
    queue = deque(starts)
    visited = set(starts)
    output: list[int] = []
    while queue and len(selected) + len(output) < min(24, n_taxa):
        node = queue.popleft()
        if node < n_taxa and node not in selected:
            output.append(node)
            continue
        neighbors = sorted(adjacency[node], reverse=bool(cover % 2))
        for neighbor in neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return output


def _indexed_local_context_taxa(
    adjacency: dict[int, set[int]],
    target_edges: list[Edge],
    selected: set[int],
    tree_index: ContextualTreeIndex,
    cover: int,
) -> list[int]:
    """Choose bounded nearby context from precomputed directed messages."""

    starts = sorted(
        {node for edge in target_edges for node in edge},
        reverse=bool(cover % 2),
    )
    candidates: set[int] = set()
    for node in starts:
        for neighbor in adjacency[node]:
            candidates.update(tree_index.nearest[(neighbor, node)])
    candidates.difference_update(selected)
    if not candidates:
        return []
    ordered_candidates = sorted(candidates)
    labels = [*starts, *ordered_candidates]
    distances = tree_index.distance_matrix(labels)
    nearest_distance = distances[: len(starts), len(starts) :].min(axis=0)
    distance_by_taxon = dict(zip(ordered_candidates, nearest_distance.tolist()))
    ranked = sorted(
        candidates,
        key=lambda taxon: (
            distance_by_taxon[taxon],
            -taxon if cover % 2 else taxon,
        ),
    )
    return ranked


def build_contextual_panel_cover(
    adjacency: dict[int, set[int]],
    n_taxa: int,
    *,
    cover: int,
    panel_size: int = 24,
    maximum_target_edges: int = 12,
    required_taxa_cap: int = 20,
    tree_index: ContextualTreeIndex | None = None,
    _workspace: _PanelCoverWorkspace | None = None,
) -> list[ContextualPanel]:
    """Cover every internal edge once with connected, fixed-size panels."""

    if n_taxa < panel_size:
        raise ValueError("the contextual model requires at least panel_size taxa")
    if required_taxa_cap > panel_size or required_taxa_cap < 4:
        raise ValueError("required_taxa_cap must lie in [4, panel_size]")
    workspace = (
        _workspace
        if _workspace is not None
        else _panel_cover_workspace(adjacency, n_taxa)
    )
    edges = list(workspace.edges)
    neighbors = workspace.neighbors
    remaining = set(edges)
    panels: list[ContextualPanel] = []
    reverse = bool(cover % 2)
    seed_order = tuple(reversed(edges)) if reverse else tuple(edges)
    seed_position = 0

    def representatives(edge: Edge) -> tuple[int, int, int, int]:
        key = (cover, edge)
        result = workspace.representatives.get(key)
        if result is None:
            result = _representatives_for_edge(
                adjacency,
                edge,
                n_taxa,
                cover,
                tree_index=tree_index,
                branches=workspace.branches[edge],
            )
            workspace.representatives[key] = result
        return result

    while remaining:
        while seed_order[seed_position] not in remaining:
            seed_position += 1
        seed = seed_order[seed_position]
        queue = deque([seed])
        queued = {seed}
        targets: list[Edge] = []
        required: set[int] = set()
        while queue and len(targets) < maximum_target_edges:
            edge = queue.popleft()
            if edge not in remaining:
                continue
            proposed = required | set(representatives(edge))
            if targets and len(proposed) > required_taxa_cap:
                continue
            targets.append(edge)
            required = proposed
            remaining.remove(edge)
            for other in sorted(neighbors[edge], reverse=reverse):
                if other in remaining and other not in queued:
                    queued.add(other)
                    queue.append(other)

        if not targets:
            raise AssertionError(f"failed to cover seed edge {seed}")

        selected = set(required)
        # Add alternate nearby representatives before generic local context.
        for edge in targets:
            left, right = edge
            branches = workspace.branches[edge]
            for branch, owner in zip(branches, (left, left, right, right)):
                values = (
                    list(tree_index.nearest[(branch, owner)])
                    if tree_index is not None
                    else near_edge_representatives(
                        adjacency, branch, owner, n_taxa, limit=4
                    )
                )
                values = list(reversed(values)) if reverse else values
                for taxon in values:
                    if len(selected) == panel_size:
                        break
                    selected.add(taxon)
            if len(selected) == panel_size:
                break
        if len(selected) < panel_size:
            context = (
                _indexed_local_context_taxa(
                    adjacency, targets, selected, tree_index, cover
                )
                if tree_index is not None
                else _local_context_taxa(
                    adjacency, targets, selected, n_taxa, cover
                )
            )
            for taxon in context:
                selected.add(taxon)
                if len(selected) == panel_size:
                    break
        if len(selected) < panel_size:
            fallback = range(n_taxa - 1, -1, -1) if reverse else range(n_taxa)
            for taxon in fallback:
                selected.add(int(taxon))
                if len(selected) == panel_size:
                    break
        if len(selected) != panel_size:
            raise AssertionError(f"panel has {len(selected)} taxa, expected {panel_size}")
        panels.append(
            ContextualPanel(
                cover=cover,
                panel_id=len(panels),
                taxa=tuple(sorted(selected)),
                target_edges=tuple(targets),
            )
        )

    observed = [edge for panel in panels for edge in panel.target_edges]
    if sorted(observed) != edges or len(observed) != len(set(observed)):
        raise AssertionError("panel cover is not an exact internal-edge partition")
    return panels


def build_contextual_panel_covers(
    adjacency: dict[int, set[int]],
    n_taxa: int,
    *,
    covers: tuple[int, ...] = (0, 1),
    panel_size: int = 24,
    maximum_target_edges: int = 12,
    required_taxa_cap: int = 20,
    tree_index: ContextualTreeIndex | None = None,
    plan_backend: object | None = None,
) -> list[ContextualPanel]:
    """Build several exact covers while sharing immutable tree-local facts."""

    if (
        tree_index is not None
        and plan_backend is not None
        and hasattr(plan_backend, "compile_contextual_panel_covers")
    ):
        nodes = sorted(adjacency)
        if nodes != list(range(len(nodes))):
            raise ValueError("native contextual covers require contiguous node ids")
        edges = tree_index.native_edges
        nearest_rows = tree_index.native_nearest_rows
        if edges is None or nearest_rows is None:
            edges = np.asarray(
                [
                    (left, right)
                    for left in nodes
                    for right in sorted(adjacency[left])
                    if left < right
                ],
                dtype=np.int32,
            ).reshape(-1, 2)
            representative_limit = max(
                len(values) for values in tree_index.nearest.values()
            )
            nearest_rows = np.full(
                (len(tree_index.nearest), 2 + representative_limit),
                -1,
                dtype=np.int32,
            )
            for row, ((branch, owner), values) in enumerate(
                sorted(tree_index.nearest.items())
            ):
                nearest_rows[row, 0] = branch
                nearest_rows[row, 1] = owner
                nearest_rows[row, 2 : 2 + len(values)] = values
        compiled = plan_backend.compile_contextual_panel_covers(
            edges,
            len(nodes),
            n_taxa,
            nearest_rows,
            np.asarray(covers, dtype=np.int32),
            panel_size,
            maximum_target_edges,
            required_taxa_cap,
        )
        return [
            ContextualPanel(
                cover=int(cover),
                panel_id=int(panel_id),
                taxa=tuple(int(taxon) for taxon in np.asarray(taxa)),
                target_edges=tuple(
                    (int(edge[0]), int(edge[1]))
                    for edge in np.asarray(target_edges).reshape(-1, 2)
                ),
            )
            for cover, panel_id, taxa, target_edges in compiled
        ]

    workspace = _panel_cover_workspace(adjacency, n_taxa)
    panels: list[ContextualPanel] = []
    for cover in covers:
        panels.extend(
            build_contextual_panel_cover(
                adjacency,
                n_taxa,
                cover=cover,
                panel_size=panel_size,
                maximum_target_edges=maximum_target_edges,
                required_taxa_cap=required_taxa_cap,
                tree_index=tree_index,
                _workspace=workspace,
            )
        )
    return panels


@dataclass(frozen=True)
class _RootedIndex:
    parent: dict[int, int | None]
    entered: dict[int, int]
    exited: dict[int, int]


@dataclass(frozen=True)
class ContextualTreeIndex:
    """One reusable topology index for all contextual panels on a tree."""

    rooted: _RootedIndex
    nodes: tuple[int, ...]
    position: dict[int, int]
    depth: np.ndarray
    ancestors: np.ndarray
    nearest: dict[tuple[int, int], tuple[int, ...]]
    native_edges: np.ndarray | None
    native_nearest_rows: np.ndarray | None

    def lca(self, left: int, right: int) -> int:
        """Return the lowest common ancestor in the fixed rooted view."""

        if left not in self.position or right not in self.position:
            raise ValueError("LCA node is absent from the indexed tree")
        entered = self.rooted.entered
        exited = self.rooted.exited

        def descendant(node: int, ancestor: int) -> bool:
            return entered[ancestor] <= entered[node] <= exited[ancestor]

        if descendant(left, right):
            return right
        if descendant(right, left):
            return left
        current = self.position[left]
        for level in range(self.ancestors.shape[0] - 1, -1, -1):
            candidate = int(self.ancestors[level, current])
            candidate_node = self.nodes[candidate]
            if not descendant(right, candidate_node):
                current = candidate
        return self.nodes[int(self.ancestors[0, current])]

    def first_step(
        self,
        adjacency: dict[int, set[int]],
        source: int,
        target: int,
    ) -> int:
        """Return the immediate neighbor of ``source`` on its path to target."""

        if source == target:
            raise ValueError("source and target must differ")
        for neighbor in sorted(adjacency[source]):
            if _in_directed_component(self.rooted, target, neighbor, source):
                return neighbor
        raise AssertionError(f"no path step from {source} to {target}")

    def distance_matrix(self, labels: tuple[int, ...] | list[int]) -> np.ndarray:
        """Return unweighted tree distances without traversing the tree."""

        positions = np.asarray([self.position[int(node)] for node in labels], dtype=np.int64)
        size = len(positions)
        left = np.repeat(positions, size)
        right = np.tile(positions, size)
        left_depth = self.depth[left]
        right_depth = self.depth[right]
        swap = left_depth < right_depth
        if np.any(swap):
            temporary = left[swap].copy()
            left[swap] = right[swap]
            right[swap] = temporary
            left_depth = self.depth[left]
            right_depth = self.depth[right]
        difference = left_depth - right_depth
        for level in range(self.ancestors.shape[0]):
            mask = ((difference >> level) & 1).astype(bool)
            if np.any(mask):
                left[mask] = self.ancestors[level, left[mask]]
        unequal = left != right
        for level in range(self.ancestors.shape[0] - 1, -1, -1):
            lifted_left = self.ancestors[level, left]
            lifted_right = self.ancestors[level, right]
            mask = unequal & (lifted_left != lifted_right)
            if np.any(mask):
                left[mask] = lifted_left[mask]
                right[mask] = lifted_right[mask]
        lca = left.copy()
        lca[unequal] = self.ancestors[0, left[unequal]]
        distances = left_depth + right_depth - 2 * self.depth[lca]
        return distances.reshape(size, size).astype(np.int32, copy=False)


def _rooted_index(adjacency: dict[int, set[int]]) -> _RootedIndex:
    root = min(adjacency)
    parent: dict[int, int | None] = {root: None}
    entered: dict[int, int] = {}
    exited: dict[int, int] = {}
    tick = 0
    stack: list[tuple[int, int | None, bool]] = [(root, None, False)]
    while stack:
        node, owner, closing = stack.pop()
        if closing:
            exited[node] = tick
            tick += 1
            continue
        if node in entered:
            raise ValueError("adjacency is not a tree")
        parent[node] = owner
        entered[node] = tick
        tick += 1
        stack.append((node, owner, True))
        for neighbor in sorted(adjacency[node], reverse=True):
            if neighbor != owner:
                stack.append((neighbor, node, False))
    if len(entered) != len(adjacency):
        raise ValueError("adjacency is disconnected")
    return _RootedIndex(parent=parent, entered=entered, exited=exited)


def directed_edge_nearest_representatives(
    adjacency: dict[int, set[int]],
    n_taxa: int,
    *,
    limit: int = 4,
    plan_backend: object | None = None,
) -> dict[tuple[int, int], tuple[int, ...]]:
    """Return the nearest fixed number of leaves on every directed edge side.

    A postorder/preorder reroot dynamic program computes every message in
    ``O(limit * n)`` work on a binary tree.  Distance and then taxon id define
    a topology-independent deterministic tie break.
    """

    if limit <= 0:
        raise ValueError("limit must be positive")
    if not adjacency:
        return {}
    if any(len(neighbors) > 3 for neighbors in adjacency.values()):
        raise ValueError("nearest messages require a binary tree")
    if plan_backend is not None and hasattr(
        plan_backend, "compile_directed_nearest_messages"
    ):
        nodes = sorted(adjacency)
        if nodes != list(range(len(nodes))):
            raise ValueError("native nearest messages require contiguous node ids")
        edges = np.asarray(
            [
                (left, right)
                for left in nodes
                for right in sorted(adjacency[left])
                if left < right
            ],
            dtype=np.int32,
        ).reshape(-1, 2)
        rows = np.asarray(
            plan_backend.compile_directed_nearest_messages(
                edges, len(nodes), n_taxa, limit
            ),
            dtype=np.int32,
        )
        result = {
            (int(row[0]), int(row[1])): tuple(
                int(value) for value in row[2:] if value >= 0
            )
            for row in rows
        }
        expected = sum(len(neighbors) for neighbors in adjacency.values())
        if len(result) != expected:
            raise AssertionError(
                f"missing native directed messages: {len(result)} != {expected}"
            )
        return result

    def closest(candidates: list[tuple[int, int]]) -> list[tuple[int, int]]:
        by_leaf: dict[int, int] = {}
        for distance, leaf in candidates:
            previous = by_leaf.get(leaf)
            if previous is None or distance < previous:
                by_leaf[leaf] = distance
        return sorted(
            ((distance, leaf) for leaf, distance in by_leaf.items()),
            key=lambda value: (value[0], value[1]),
        )[:limit]

    root = min(adjacency)
    parent: dict[int, int | None] = {root: None}
    order = [root]
    for node in order:
        for neighbor in sorted(adjacency[node]):
            if neighbor == parent[node]:
                continue
            if neighbor in parent:
                raise ValueError("adjacency is not a tree")
            parent[neighbor] = node
            order.append(neighbor)
    if len(parent) != len(adjacency):
        raise ValueError("adjacency is disconnected")

    down: dict[int, list[tuple[int, int]]] = {}
    for node in reversed(order):
        candidates: list[tuple[int, int]] = []
        if node < n_taxa:
            candidates.append((0, node))
        for child in adjacency[node]:
            if parent.get(child) == node:
                candidates.extend((distance + 1, leaf) for distance, leaf in down[child])
        down[node] = closest(candidates)
        if not down[node]:
            raise ValueError(f"subtree rooted at {node} contains no taxon")

    outside: dict[int, list[tuple[int, int]]] = {root: []}
    messages: dict[tuple[int, int], tuple[int, ...]] = {}
    for node in order:
        children = sorted(child for child in adjacency[node] if parent.get(child) == node)
        for child in children:
            candidates = list(outside[node])
            if node < n_taxa:
                candidates.append((0, node))
            for sibling in children:
                if sibling != child:
                    candidates.extend(
                        (distance + 1, leaf) for distance, leaf in down[sibling]
                    )
            side = closest(candidates)
            if not side:
                raise ValueError(f"directed side {(node, child)} contains no taxon")
            messages[(node, child)] = tuple(leaf for _distance, leaf in side)
            messages[(child, node)] = tuple(leaf for _distance, leaf in down[child])
            outside[child] = [(distance + 1, leaf) for distance, leaf in side]

    expected = sum(len(neighbors) for neighbors in adjacency.values())
    if len(messages) != expected:
        raise AssertionError(f"missing directed messages: {len(messages)} != {expected}")
    return messages


def build_contextual_tree_index(
    adjacency: dict[int, set[int]],
    n_taxa: int,
    *,
    representative_limit: int = 4,
    plan_backend: object | None = None,
) -> ContextualTreeIndex:
    """Build the single reusable index consumed by every panel."""

    rooted = _rooted_index(adjacency)
    nodes = tuple(sorted(adjacency))
    position = {node: offset for offset, node in enumerate(nodes)}
    depth_by_node: dict[int, int] = {}
    for node in sorted(nodes, key=rooted.entered.__getitem__):
        owner = rooted.parent[node]
        depth_by_node[node] = 0 if owner is None else depth_by_node[owner] + 1
    depth = np.asarray([depth_by_node[node] for node in nodes], dtype=np.int32)
    levels = max(1, len(nodes).bit_length())
    ancestors = np.empty((levels, len(nodes)), dtype=np.int32)
    for offset, node in enumerate(nodes):
        owner = rooted.parent[node]
        ancestors[0, offset] = offset if owner is None else position[owner]
    for level in range(1, levels):
        ancestors[level] = ancestors[level - 1, ancestors[level - 1]]
    native_edges: np.ndarray | None = None
    native_nearest_rows: np.ndarray | None = None
    if plan_backend is not None and hasattr(
        plan_backend, "compile_directed_nearest_messages"
    ):
        if list(nodes) != list(range(len(nodes))):
            raise ValueError("native nearest messages require contiguous node ids")
        native_edges = np.asarray(
            [
                (left, right)
                for left in nodes
                for right in sorted(adjacency[left])
                if left < right
            ],
            dtype=np.int32,
        ).reshape(-1, 2)
        native_nearest_rows = np.asarray(
            plan_backend.compile_directed_nearest_messages(
                native_edges, len(nodes), n_taxa, representative_limit
            ),
            dtype=np.int32,
        )
        nearest = {
            (int(row[0]), int(row[1])): tuple(
                int(value) for value in row[2:] if value >= 0
            )
            for row in native_nearest_rows
        }
        expected = sum(len(neighbors) for neighbors in adjacency.values())
        if len(nearest) != expected:
            raise AssertionError(
                f"missing native directed messages: {len(nearest)} != {expected}"
            )
    else:
        nearest = directed_edge_nearest_representatives(
            adjacency,
            n_taxa,
            limit=representative_limit,
            plan_backend=plan_backend,
        )
    return ContextualTreeIndex(
        rooted=rooted,
        nodes=nodes,
        position=position,
        depth=depth,
        ancestors=ancestors,
        nearest=nearest,
        native_edges=native_edges,
        native_nearest_rows=native_nearest_rows,
    )


def virtual_panel_tree(
    adjacency: dict[int, set[int]],
    taxa: tuple[int, ...] | list[int],
    tree_index: ContextualTreeIndex,
) -> VirtualPanelTree:
    """Extract the compressed tree induced by a fixed taxon panel.

    The standard Euler-order virtual-tree construction adds only adjacent
    LCAs, so it touches ``O(k)`` nodes and performs ``O(k log n)`` indexed
    work for ``k`` selected taxa.  Degree-two Steiner nodes are suppressed
    while retaining endpoint ports for exact global edit lifting.
    """

    selected = tuple(sorted({int(taxon) for taxon in taxa}))
    if len(selected) < 4:
        raise ValueError("a virtual panel tree requires at least four taxa")
    if len(selected) != len(taxa):
        raise ValueError("panel taxa must be distinct")
    if any(taxon not in tree_index.position for taxon in selected):
        raise ValueError("panel taxon is absent from the indexed tree")

    entered = tree_index.rooted.entered
    ordered_taxa = sorted(selected, key=entered.__getitem__)
    vertices = set(ordered_taxa)
    vertices.update(
        tree_index.lca(left, right)
        for left, right in zip(ordered_taxa, ordered_taxa[1:])
    )
    ordered_vertices = sorted(vertices, key=entered.__getitem__)
    virtual: dict[int, set[int]] = {node: set() for node in ordered_vertices}
    ports: dict[tuple[int, int], int] = {}
    stack: list[int] = []
    for node in ordered_vertices:
        while stack and not _is_descendant(tree_index.rooted, node, stack[-1]):
            stack.pop()
        if stack:
            owner = stack[-1]
            virtual[owner].add(node)
            virtual[node].add(owner)
            ports[(owner, node)] = tree_index.first_step(adjacency, owner, node)
            ports[(node, owner)] = tree_index.first_step(adjacency, node, owner)
        stack.append(node)

    selected_set = set(selected)
    queue = deque(
        sorted(
            node
            for node, neighbors in virtual.items()
            if node not in selected_set and len(neighbors) == 2
        )
    )
    while queue:
        node = queue.popleft()
        if node not in virtual or node in selected_set or len(virtual[node]) != 2:
            continue
        left, right = sorted(virtual[node])
        left_port = ports[(left, node)]
        right_port = ports[(right, node)]
        del ports[(left, node)], ports[(node, left)]
        del ports[(right, node)], ports[(node, right)]
        virtual[left].remove(node)
        virtual[right].remove(node)
        del virtual[node]
        virtual[left].add(right)
        virtual[right].add(left)
        ports[(left, right)] = left_port
        ports[(right, left)] = right_port
        for neighbor in (left, right):
            if neighbor not in selected_set and len(virtual[neighbor]) == 2:
                queue.append(neighbor)

    if any(len(virtual[taxon]) != 1 for taxon in selected):
        raise AssertionError("selected taxa are not leaves of the virtual tree")
    if any(
        node not in selected_set and len(neighbors) != 3
        for node, neighbors in virtual.items()
    ):
        raise AssertionError("virtual panel tree is not binary after suppression")
    if len(ports) != sum(len(neighbors) for neighbors in virtual.values()):
        raise AssertionError("virtual tree is missing directed lift ports")
    return VirtualPanelTree(adjacency=virtual, ports=ports, taxa=selected)


def _local_panel_groups(
    adjacency: dict[int, set[int]],
    taxa: tuple[int, ...],
    edge: Edge,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Partition panel positions among an edge's four incident components."""

    position = {taxon: offset for offset, taxon in enumerate(taxa)}
    left, right = edge
    branches = branch_nodes_for_edge(adjacency, edge)
    groups: list[tuple[int, ...]] = []
    for branch, owner in zip(branches, (left, left, right, right)):
        found: list[int] = []
        stack = [(branch, owner)]
        while stack:
            node, previous = stack.pop()
            if node in position:
                found.append(position[node])
            for neighbor in adjacency[node]:
                if neighbor != previous:
                    stack.append((neighbor, node))
        if not found:
            raise AssertionError(f"virtual edge branch is empty: {edge} {branch}")
        groups.append(tuple(sorted(found)))
    return tuple(groups)  # type: ignore[return-value]


def _apply_virtual_nni(
    adjacency: dict[int, set[int]],
    ports: dict[tuple[int, int], int],
    edge: Edge,
    best: int,
    *,
    step_gain: float,
    changed_rows: int,
) -> PanelPatchMove:
    """Apply one local NNI and update its full-tree lift ports in place."""

    if best not in (1, 2):
        raise ValueError("an NNI alternative must be 1 or 2")
    left, right = edge
    branches = branch_nodes_for_edge(adjacency, edge)
    global_branches = tuple(
        ports[(owner, branch)]
        for branch, owner in zip(branches, (left, left, right, right))
    )
    movable = branches[1]
    displaced = branches[2] if best == 1 else branches[3]
    movable_forward = ports[(left, movable)]
    movable_reverse = ports[(movable, left)]
    displaced_forward = ports[(right, displaced)]
    displaced_reverse = ports[(displaced, right)]

    adjacency[left].remove(movable)
    adjacency[movable].remove(left)
    adjacency[right].remove(displaced)
    adjacency[displaced].remove(right)
    del ports[(left, movable)], ports[(movable, left)]
    del ports[(right, displaced)], ports[(displaced, right)]

    adjacency[left].add(displaced)
    adjacency[displaced].add(left)
    adjacency[right].add(movable)
    adjacency[movable].add(right)
    ports[(left, displaced)] = displaced_forward
    ports[(displaced, left)] = displaced_reverse
    ports[(right, movable)] = movable_forward
    ports[(movable, right)] = movable_reverse
    return PanelPatchMove(
        edge=edge,
        branch_nodes=global_branches,
        best=best,
        step_gain=float(step_gain),
        changed_rows=int(changed_rows),
    )


def _apply_lifted_move_to_virtual(
    adjacency: dict[int, set[int]],
    ports: dict[tuple[int, int], int],
    move: PanelPatchMove,
) -> bool:
    """Replay a global move when it is visible in another induced panel."""

    left, right = move.edge
    if (
        left not in adjacency
        or right not in adjacency.get(left, set())
        or len(adjacency[left]) != 3
        or len(adjacency[right]) != 3
    ):
        return False
    left_by_port = {ports[(left, neighbor)]: neighbor for neighbor in adjacency[left]}
    right_by_port = {ports[(right, neighbor)]: neighbor for neighbor in adjacency[right]}
    one, two, three, four = move.branch_nodes
    displaced_port = three if move.best == 1 else four
    if two not in left_by_port or displaced_port not in right_by_port:
        return False
    movable = left_by_port[two]
    displaced = right_by_port[displaced_port]
    movable_forward = ports[(left, movable)]
    movable_reverse = ports[(movable, left)]
    displaced_forward = ports[(right, displaced)]
    displaced_reverse = ports[(displaced, right)]

    adjacency[left].remove(movable)
    adjacency[movable].remove(left)
    adjacency[right].remove(displaced)
    adjacency[displaced].remove(right)
    del ports[(left, movable)], ports[(movable, left)]
    del ports[(right, displaced)], ports[(displaced, right)]
    adjacency[left].add(displaced)
    adjacency[displaced].add(left)
    adjacency[right].add(movable)
    adjacency[movable].add(right)
    ports[(left, displaced)] = displaced_forward
    ports[(displaced, left)] = displaced_reverse
    ports[(right, movable)] = movable_forward
    ports[(movable, right)] = movable_reverse
    return True


def _state_nni_candidate(
    state: _PatchState,
    panel: ContextualPanel,
    edge: Edge,
    best: int,
    logp: np.ndarray,
    row_lookup: dict[tuple[int, int, int, int], int],
) -> _PatchState:
    """Create one exact child state using only rows changed by its NNI."""

    groups = _local_panel_groups(state.adjacency, panel.taxa, edge)
    changed_rows: list[int] = []
    alternative_classes: list[int] = []
    for one, two, three, four in product(*groups):
        quartet = tuple(sorted((one, two, three, four)))
        row = row_lookup[quartet]
        current = class_for_group_pairing(quartet, (one, two))
        if int(state.classes[row]) != current:
            raise AssertionError(f"beam class mismatch at {edge} for {quartet}")
        changed_rows.append(row)
        alternative_classes.append(
            class_for_group_pairing(
                quartet,
                (one, three) if best == 1 else (one, four),
            )
        )
    changed = np.asarray(changed_rows, dtype=np.int64)
    alternative = np.asarray(alternative_classes, dtype=np.int8)
    rows = np.arange(len(state.classes))
    old_sum = float(logp[changed, state.classes[changed]].sum())
    new_sum = float(logp[changed, alternative].sum())
    candidate_sum = state.score_sum - old_sum + new_sum
    candidate_classes = state.classes.copy()
    candidate_classes[changed] = alternative
    candidate_adjacency = {
        node: set(neighbors) for node, neighbors in state.adjacency.items()
    }
    candidate_ports = dict(state.ports)
    move = _apply_virtual_nni(
        candidate_adjacency,
        candidate_ports,
        edge,
        best,
        step_gain=(candidate_sum - state.score_sum) / len(rows),
        changed_rows=len(changed),
    )
    return _PatchState(
        adjacency=candidate_adjacency,
        ports=candidate_ports,
        classes=candidate_classes,
        score_sum=candidate_sum,
        moves=state.moves + (move,),
        used_edges=state.used_edges | {edge},
    )


def search_contextual_panel_patch(
    adjacency: dict[int, set[int]],
    panel: ContextualPanel,
    probabilities: np.ndarray,
    quartet_template: np.ndarray,
    tree_index: ContextualTreeIndex,
    *,
    maximum_depth: int = 2,
    beam_width: int = 8,
) -> PanelPatch:
    """Find a bounded connected NNI sequence under one full panel posterior."""

    if maximum_depth < 1 or beam_width < 1:
        raise ValueError("maximum_depth and beam_width must be positive")
    probabilities = np.asarray(probabilities, dtype=np.float64)
    template = np.asarray(quartet_template, dtype=np.int16)
    if probabilities.shape != (len(template), 3):
        raise ValueError("panel probability shape mismatch")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0):
        raise ValueError("probabilities must be finite and nonnegative")
    virtual = virtual_panel_tree(adjacency, panel.taxa, tree_index)
    for edge in panel.target_edges:
        if edge[1] not in virtual.adjacency.get(edge[0], set()):
            raise AssertionError(f"target edge {edge} was compressed away")
    current_classes = _indexed_displayed_quartet_classes(
        tree_index, panel, template
    )
    logp = np.log(np.clip(probabilities, 1e-8, 1.0))
    rows = np.arange(len(template))
    base_sum = float(logp[rows, current_classes].sum())
    initial = _PatchState(
        adjacency={node: set(neighbors) for node, neighbors in virtual.adjacency.items()},
        ports=dict(virtual.ports),
        classes=current_classes,
        score_sum=base_sum,
        moves=(),
        used_edges=frozenset(),
    )
    row_lookup = {
        tuple(int(value) for value in quartet): row
        for row, quartet in enumerate(template)
    }
    beam = [initial]
    evaluated = 0
    for _depth in range(maximum_depth):
        pool = list(beam)
        for state in beam:
            eligible = [
                edge
                for edge in panel.target_edges
                if edge not in state.used_edges
                and edge[1] in state.adjacency.get(edge[0], set())
                and len(state.adjacency[edge[0]]) == 3
                and len(state.adjacency[edge[1]]) == 3
                and (
                    not state.used_edges
                    or any(set(edge) & set(previous) for previous in state.used_edges)
                )
            ]
            for edge in eligible:
                for best in (1, 2):
                    pool.append(
                        _state_nni_candidate(
                            state, panel, edge, best, logp, row_lookup
                        )
                    )
                    evaluated += 1
        unique: dict[bytes, _PatchState] = {}
        for state in pool:
            signature = state.classes.tobytes()
            incumbent = unique.get(signature)
            state_key = tuple(
                (move.edge, move.branch_nodes, move.best) for move in state.moves
            )
            incumbent_key = (
                tuple((move.edge, move.branch_nodes, move.best) for move in incumbent.moves)
                if incumbent is not None
                else ()
            )
            if (
                incumbent is None
                or state.score_sum > incumbent.score_sum + 1e-12
                or (
                    abs(state.score_sum - incumbent.score_sum) <= 1e-12
                    and state_key < incumbent_key
                )
            ):
                unique[signature] = state
        beam = sorted(
            unique.values(),
            key=lambda state: (
                -state.score_sum,
                len(state.moves),
                tuple((move.edge, move.branch_nodes, move.best) for move in state.moves),
            ),
        )[:beam_width]
    best_state = beam[0]
    base_score = base_sum / len(template)
    final_score = best_state.score_sum / len(template)
    return PanelPatch(
        cover=panel.cover,
        panel_id=panel.panel_id,
        moves=best_state.moves,
        base_score=base_score,
        final_score=final_score,
        gain=final_score - base_score,
        changed_rows=int(np.count_nonzero(best_state.classes != current_classes)),
        evaluated_candidates=evaluated,
    )


def replay_panel_patch_score(
    adjacency: dict[int, set[int]],
    panel: ContextualPanel,
    probabilities: np.ndarray,
    quartet_template: np.ndarray,
    tree_index: ContextualTreeIndex,
    patch: PanelPatch,
) -> tuple[float, int, int]:
    """Return opposite-panel mean-log gain, visible moves, and changed rows."""

    probabilities = np.asarray(probabilities, dtype=np.float64)
    template = np.asarray(quartet_template, dtype=np.int16)
    virtual = virtual_panel_tree(adjacency, panel.taxa, tree_index)
    candidate_adjacency = {
        node: set(neighbors) for node, neighbors in virtual.adjacency.items()
    }
    candidate_ports = dict(virtual.ports)
    visible = 0
    for move in patch.moves:
        visible += int(
            _apply_lifted_move_to_virtual(
                candidate_adjacency, candidate_ports, move
            )
        )
    quartets = np.asarray(panel.taxa, dtype=np.int64)[template]
    base_classes = _indexed_displayed_quartet_classes(tree_index, panel, template)
    final_classes = displayed_quartet_classes(candidate_adjacency, quartets)
    rows = np.arange(len(template))
    logp = np.log(np.clip(probabilities, 1e-8, 1.0))
    gain = float(
        (logp[rows, final_classes].sum() - logp[rows, base_classes].sum())
        / len(template)
    )
    return gain, visible, int(np.count_nonzero(final_classes != base_classes))


def apply_panel_patch(
    adjacency: dict[int, set[int]],
    patch: PanelPatch,
) -> None:
    """Lift an accepted panel patch to the full tree in constant edit work."""

    for move in patch.moves:
        left, right = move.edge
        one, two, three, four = move.branch_nodes
        displaced = three if move.best == 1 else four
        if (
            right not in adjacency.get(left, set())
            or two not in adjacency.get(left, set())
            or displaced not in adjacency.get(right, set())
        ):
            raise ValueError(f"panel patch move is no longer liftable: {move}")
        adjacency[left].remove(two)
        adjacency[two].remove(left)
        adjacency[right].remove(displaced)
        adjacency[displaced].remove(right)
        adjacency[left].add(displaced)
        adjacency[displaced].add(left)
        adjacency[right].add(two)
        adjacency[two].add(right)


def _is_descendant(index: _RootedIndex, node: int, ancestor: int) -> bool:
    return index.entered[ancestor] <= index.entered[node] <= index.exited[ancestor]


def _in_directed_component(
    index: _RootedIndex, taxon: int, start: int, blocked: int
) -> bool:
    if index.parent.get(start) == blocked:
        return _is_descendant(index, taxon, start)
    if index.parent.get(blocked) == start:
        return not _is_descendant(index, taxon, blocked)
    raise ValueError(f"{start} and {blocked} are not adjacent")


def _panel_edge_groups(
    adjacency: dict[int, set[int]],
    panel: ContextualPanel,
    edge: Edge,
    index: _RootedIndex,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    left, right = edge
    branches = branch_nodes_for_edge(adjacency, edge)
    owners = (left, left, right, right)
    groups: list[list[int]] = [[] for _ in range(4)]
    for position, taxon in enumerate(panel.taxa):
        matches = [
            group
            for group, (branch, owner) in enumerate(zip(branches, owners))
            if _in_directed_component(index, taxon, branch, owner)
        ]
        if len(matches) != 1:
            raise AssertionError((edge, taxon, matches))
        groups[matches[0]].append(position)
    if any(not group for group in groups):
        raise AssertionError(f"panel does not distinguish edge {edge}: {groups}")
    return tuple(tuple(group) for group in groups)  # type: ignore[return-value]


def _native_panel_plan_inputs(
    adjacency: dict[int, set[int]],
    panel: ContextualPanel,
    index: _RootedIndex,
) -> tuple[np.ndarray, np.ndarray, tuple[tuple[int, int, int, int], ...]]:
    """Compile rooted component descriptors for the integer native kernel."""

    panel_entered = np.fromiter(
        (index.entered[taxon] for taxon in panel.taxa),
        dtype=np.int64,
        count=len(panel.taxa),
    )
    descriptors = np.empty((len(panel.target_edges), 4, 3), dtype=np.int64)
    branch_rows: list[tuple[int, int, int, int]] = []
    for edge_index, edge in enumerate(panel.target_edges):
        left, right = edge
        branches = branch_nodes_for_edge(adjacency, edge)
        branch_rows.append(branches)
        for group, (branch, owner) in enumerate(
            zip(branches, (left, left, right, right))
        ):
            if index.parent.get(branch) == owner:
                descriptors[edge_index, group] = (
                    index.entered[branch],
                    index.exited[branch],
                    0,
                )
            elif index.parent.get(owner) == branch:
                descriptors[edge_index, group] = (
                    index.entered[owner],
                    index.exited[owner],
                    1,
                )
            else:
                raise ValueError(f"{branch} and {owner} are not adjacent")
    return panel_entered, descriptors, tuple(branch_rows)


def _indexed_displayed_quartet_classes(
    tree_index: ContextualTreeIndex,
    panel: ContextualPanel,
    quartet_template: np.ndarray,
) -> np.ndarray:
    """Evaluate panel quartet classes from indexed leaf distances."""

    distances = tree_index.distance_matrix(panel.taxa)
    a, b, c, d = quartet_template.T
    sums = np.column_stack(
        (
            distances[a, b] + distances[c, d],
            distances[a, c] + distances[b, d],
            distances[a, d] + distances[b, c],
        )
    )
    return np.argmin(sums, axis=1).astype(np.int8, copy=False)


def _distinct_four_ranks(values: np.ndarray) -> np.ndarray:
    """Return ranks for rows of four distinct integers without sorting.

    The rank of one value is exactly the number of other values smaller than
    it.  This is the four-input comparison network underlying the three
    perfect-matching action, and avoids two general stable sorts per edge.
    """

    values = np.asarray(values)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError("values must have shape (rows, 4)")
    ranks = np.count_nonzero(
        values[:, :, None] > values[:, None, :], axis=2
    ).astype(np.int8, copy=False)
    return ranks


def compile_sparse_contextual_panel_plan(
    adjacency: dict[int, set[int]],
    panel: ContextualPanel,
    quartet_template: np.ndarray,
    *,
    tree_index: ContextualTreeIndex,
    row_lookup: np.ndarray,
    plan_backend: object,
) -> SparseContextualPanelPlan:
    """Compile only rows whose class changes under a requested edge NNI.

    Fast's MLP scores quartet rows independently.  Every row outside this
    union contributes the same log probability to all three edge topologies
    and therefore cancels from both the argmax and gain.  The QuartFormer path
    does not use this representation because its attention couples rows.
    """

    template = np.asarray(quartet_template, dtype=np.int16)
    if template.ndim != 2 or template.shape[1:] != (4,):
        raise ValueError("quartet_template must have shape (rows, 4)")
    if not panel.target_edges:
        raise ValueError("sparse panel plan requires at least one target edge")
    if not isinstance(row_lookup, np.ndarray):
        raise ValueError("sparse panel plan requires a dense row lookup")
    panel_entered, descriptors, branch_rows = _native_panel_plan_inputs(
        adjacency, panel, tree_index.rooted
    )
    raw_plans = plan_backend.compile_panel_edge_rows(
        panel_entered, descriptors, row_lookup
    )
    if len(raw_plans) != len(panel.target_edges):
        raise AssertionError("native sparse panel plan count mismatch")
    changed_rows = [
        np.asarray(raw[0], dtype=np.int64) for raw in raw_plans
    ]
    if any(rows.ndim != 1 or len(rows) == 0 for rows in changed_rows):
        raise AssertionError("sparse panel edge has invalid discriminating rows")
    row_indices = np.unique(np.concatenate(changed_rows))
    edge_plans: list[SparsePanelEdgePlan] = []
    for edge, branches, raw, changed in zip(
        panel.target_edges, branch_rows, raw_plans, changed_rows
    ):
        positions = np.searchsorted(row_indices, changed)
        if not np.array_equal(row_indices[positions], changed):
            raise AssertionError("sparse panel row remap is inconsistent")
        current, alternative_one, alternative_two = (
            np.asarray(values, dtype=np.int8) for values in raw[1:]
        )
        if any(
            values.shape != changed.shape
            for values in (current, alternative_one, alternative_two)
        ):
            raise AssertionError("sparse panel class vectors have invalid shapes")
        edge_plans.append(
            SparsePanelEdgePlan(
                edge=edge,
                branch_nodes=branches,
                positions=positions,
                current=current,
                alternative_one=alternative_one,
                alternative_two=alternative_two,
            )
        )
    return SparseContextualPanelPlan(
        cover=panel.cover,
        panel_id=panel.panel_id,
        taxa=panel.taxa,
        row_indices=row_indices,
        edges=tuple(edge_plans),
        emitted_rows=len(template),
    )


def score_sparse_contextual_panel_plan(
    plan: SparseContextualPanelPlan,
    probabilities: np.ndarray,
    *,
    plan_backend: object | None = None,
    compute_medians: bool = True,
) -> tuple[PanelEdgeScore, ...]:
    """Score an MLP panel from its discriminating rows only.

    Scores are translated so the current topology has value zero.  This
    removes a common additive term and leaves every argmax, gain and
    cross-cover consensus comparison mathematically unchanged.
    """

    values = np.asarray(probabilities, dtype=np.float64)
    if values.shape != (len(plan.row_indices), 3):
        raise ValueError(
            f"sparse probability shape {values.shape} does not match "
            f"({len(plan.row_indices)}, 3)"
        )
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("probabilities must be finite and nonnegative")
    logp = np.log(np.clip(values, 1e-8, 1.0))
    native_median_pair = (
        getattr(plan_backend, "median_pair", None)
        if plan_backend is not None
        else None
    )
    native_sum_scores = (
        getattr(plan_backend, "score_sparse_panel_sums", None)
        if plan_backend is not None and not compute_medians
        else None
    )
    if native_sum_scores is not None:
        reduced = np.asarray(
            native_sum_scores(
                np.asarray(probabilities, dtype=np.float32),
                [edge.positions for edge in plan.edges],
                [edge.current for edge in plan.edges],
                [edge.alternative_one for edge in plan.edges],
                [edge.alternative_two for edge in plan.edges],
                plan.emitted_rows,
            ),
            dtype=np.float64,
        )
        if reduced.shape != (len(plan.edges), 5):
            raise AssertionError("native sparse score reduction shape changed")
        return tuple(
            PanelEdgeScore(
                cover=plan.cover,
                panel_id=plan.panel_id,
                edge=edge_plan.edge,
                branch_nodes=edge_plan.branch_nodes,
                scores=(0.0, float(row[0]), float(row[1])),
                best=int(row[2]),
                gain=float(row[3]),
                discriminating_rows=int(row[4]),
                alternative_median_gains=(0.0, 0.0),
            )
            for edge_plan, row in zip(plan.edges, reduced)
        )
    result: list[PanelEdgeScore] = []
    for edge_plan in plan.edges:
        local = logp[edge_plan.positions]
        rows = np.arange(len(edge_plan.positions))
        current = local[rows, edge_plan.current]
        first_ratios = local[rows, edge_plan.alternative_one] - current
        second_ratios = local[rows, edge_plan.alternative_two] - current
        scores = np.asarray(
            [0.0, float(first_ratios.sum()), float(second_ratios.sum())],
            dtype=np.float64,
        ) / plan.emitted_rows
        best = int(np.argmax(scores))
        if compute_medians:
            medians = (
                tuple(
                    float(value)
                    for value in native_median_pair(first_ratios, second_ratios)
                )
                if native_median_pair is not None
                else (
                    float(np.median(first_ratios)),
                    float(np.median(second_ratios)),
                )
            )
        else:
            # SplitBank consumes only topology sums and discriminating counts.
            # Median gains are coordinate-only evidence and are deliberately
            # left unevaluated on that call path.
            medians = (0.0, 0.0)
        result.append(
            PanelEdgeScore(
                cover=plan.cover,
                panel_id=plan.panel_id,
                edge=edge_plan.edge,
                branch_nodes=edge_plan.branch_nodes,
                scores=tuple(float(value) for value in scores),
                best=best,
                gain=float(scores[best]),
                discriminating_rows=len(edge_plan.positions),
                alternative_median_gains=medians,
            )
        )
    return tuple(result)


def _score_contextual_panel_impl(
    adjacency: dict[int, set[int]],
    panel: ContextualPanel,
    probabilities: np.ndarray,
    quartet_template: np.ndarray | None = None,
    *,
    tree_index: ContextualTreeIndex | None = None,
    row_lookup: dict[tuple[int, int, int, int], int] | np.ndarray | None = None,
    plan_backend: object | None = None,
    collect_discriminating_mask: bool,
) -> tuple[tuple[PanelEdgeScore, ...], float, np.ndarray | None, int]:
    """Shared exact panel kernel with optional diagnostic materialization."""

    probabilities = np.asarray(probabilities, dtype=np.float64)
    if quartet_template is None:
        quartet_template = np.asarray(
            list(combinations(range(len(panel.taxa)), 4)), dtype=np.int16
        )
    else:
        quartet_template = np.asarray(quartet_template, dtype=np.int16)
    if probabilities.shape != (len(quartet_template), 3):
        raise ValueError(
            f"probability shape {probabilities.shape} does not match "
            f"{len(quartet_template)} quartet rows"
        )
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0):
        raise ValueError("probabilities must be finite and nonnegative")
    if tree_index is not None and plan_backend is not None:
        panel_positions = np.fromiter(
            (tree_index.position[taxon] for taxon in panel.taxa),
            dtype=np.int32,
            count=len(panel.taxa),
        )
        current_classes = np.asarray(
            plan_backend.compile_panel_current_classes(
                panel_positions,
                tree_index.depth,
                tree_index.ancestors,
                quartet_template,
            ),
            dtype=np.int8,
        )
        if current_classes.shape != (len(quartet_template),):
            raise AssertionError("native current-class shape mismatch")
    elif tree_index is not None:
        current_classes = _indexed_displayed_quartet_classes(
            tree_index, panel, quartet_template
        )
    else:
        global_quartets = np.asarray(panel.taxa, dtype=np.int64)[quartet_template]
        current_classes = displayed_quartet_classes(adjacency, global_quartets)
    logp = np.log(np.clip(probabilities, 1e-8, 1.0))
    rows = np.arange(len(quartet_template))
    current_values = logp[rows, current_classes]
    current_sum = float(current_values.sum())
    if row_lookup is None:
        row_lookup = {
            tuple(int(value) for value in quartet): row
            for row, quartet in enumerate(quartet_template)
        }
    rooted = tree_index.rooted if tree_index is not None else _rooted_index(adjacency)
    discriminating_mask = (
        np.zeros(len(quartet_template), dtype=bool)
        if collect_discriminating_mask
        else None
    )
    results: list[PanelEdgeScore] = []
    native_median_pair = (
        getattr(plan_backend, "median_pair", None)
        if plan_backend is not None
        else None
    )

    native_plans = None
    native_branch_rows: tuple[tuple[int, int, int, int], ...] | None = None
    if plan_backend is not None:
        if tree_index is None or not isinstance(row_lookup, np.ndarray):
            raise ValueError("native panel plans require a tree index and dense row lookup")
        panel_entered, descriptors, native_branch_rows = _native_panel_plan_inputs(
            adjacency, panel, rooted
        )
        native_plans = plan_backend.compile_panel_edge_rows(
            panel_entered, descriptors, row_lookup
        )
        if len(native_plans) != len(panel.target_edges):
            raise AssertionError("native panel plan count mismatch")

    for edge_index, edge in enumerate(panel.target_edges):
        if native_plans is not None:
            changed, current, alt1, alt2 = (
                np.asarray(value) for value in native_plans[edge_index]
            )
            changed = changed.astype(np.int64, copy=False)
            current = current.astype(np.int8, copy=False)
            alt1 = alt1.astype(np.int8, copy=False)
            alt2 = alt2.astype(np.int8, copy=False)
            if not np.array_equal(current_classes[changed], current):
                raise AssertionError(f"current class mismatch on edge {edge}")
        elif isinstance(row_lookup, np.ndarray):
            groups = _panel_edge_groups(adjacency, panel, edge, rooted)
            # Cartesian-product order is identical to itertools.product:
            # with indexing="ij" and C-order flattening, the rightmost group
            # varies fastest.  Preserving row order also preserves reduction
            # and median arithmetic while removing millions of Python calls.
            mesh = np.meshgrid(
                *(np.asarray(group, dtype=np.int16) for group in groups),
                indexing="ij",
                copy=False,
            )
            values = np.column_stack(tuple(axis.reshape(-1) for axis in mesh))
            changed = row_lookup[
                values[:, 0], values[:, 1], values[:, 2], values[:, 3]
            ].astype(np.int64, copy=False)
            if np.any(changed < 0):
                raise AssertionError(f"missing dense quartet row for edge {edge}")
            # Every row contains four distinct positions.  Their comparison
            # ranks index the exact finite action of the three perfect
            # matchings on four positions.
            ranks = _distinct_four_ranks(values)
            current = _PAIR_CLASS_BY_RANK[ranks[:, 0], ranks[:, 1]]
            alt1 = _PAIR_CLASS_BY_RANK[ranks[:, 0], ranks[:, 2]]
            alt2 = _PAIR_CLASS_BY_RANK[ranks[:, 0], ranks[:, 3]]
            if not np.array_equal(current_classes[changed], current):
                raise AssertionError(f"current class mismatch on edge {edge}")
        else:
            groups = _panel_edge_groups(adjacency, panel, edge, rooted)
            changed_rows: list[int] = []
            alt1_classes: list[int] = []
            alt2_classes: list[int] = []
            for one, two, three, four in product(*groups):
                quartet = tuple(sorted((one, two, three, four)))
                row = row_lookup[quartet]
                current_class = class_for_group_pairing(quartet, (one, two))
                if int(current_classes[row]) != current_class:
                    raise AssertionError(
                        f"current class mismatch on edge {edge}, quartet {quartet}"
                    )
                changed_rows.append(row)
                alt1_classes.append(
                    class_for_group_pairing(quartet, (one, three))
                )
                alt2_classes.append(
                    class_for_group_pairing(quartet, (one, four))
                )
            changed = np.asarray(changed_rows, dtype=np.int64)
            alt1 = np.asarray(alt1_classes, dtype=np.int8)
            alt2 = np.asarray(alt2_classes, dtype=np.int8)
        alt1_log_ratios = logp[changed, alt1] - current_values[changed]
        alt2_log_ratios = logp[changed, alt2] - current_values[changed]
        old = current_values[changed].sum()
        first_sum = current_sum - float(old) + float(logp[changed, alt1].sum())
        second_sum = current_sum - float(old) + float(logp[changed, alt2].sum())
        scores = np.asarray(
            [current_sum, first_sum, second_sum], dtype=np.float64
        ) / len(quartet_template)
        best = int(np.argmax(scores))
        gain = float(scores[best] - scores[0])
        alternative_medians = (
            tuple(
                float(value)
                for value in native_median_pair(
                    alt1_log_ratios, alt2_log_ratios
                )
            )
            if native_median_pair is not None
            else (
                float(np.median(alt1_log_ratios)),
                float(np.median(alt2_log_ratios)),
            )
        )
        if discriminating_mask is not None:
            discriminating_mask[changed] = True
        results.append(
            PanelEdgeScore(
                cover=panel.cover,
                panel_id=panel.panel_id,
                edge=edge,
                branch_nodes=(
                    native_branch_rows[edge_index]
                    if native_branch_rows is not None
                    else branch_nodes_for_edge(adjacency, edge)
                ),
                scores=tuple(float(value) for value in scores),
                best=best,
                gain=gain,
                discriminating_rows=len(changed),
                alternative_median_gains=alternative_medians,
            )
        )
    return tuple(results), current_sum, discriminating_mask, len(quartet_template)


def score_contextual_panel_edges(
    adjacency: dict[int, set[int]],
    panel: ContextualPanel,
    probabilities: np.ndarray,
    quartet_template: np.ndarray | None = None,
    *,
    tree_index: ContextualTreeIndex | None = None,
    row_lookup: dict[tuple[int, int, int, int], int] | np.ndarray | None = None,
    plan_backend: object | None = None,
) -> tuple[PanelEdgeScore, ...]:
    """Return only ordered edge decisions, omitting unused diagnostics."""

    edges, _, _, _ = _score_contextual_panel_impl(
        adjacency,
        panel,
        probabilities,
        quartet_template,
        tree_index=tree_index,
        row_lookup=row_lookup,
        plan_backend=plan_backend,
        collect_discriminating_mask=False,
    )
    return edges


def score_contextual_panel(
    adjacency: dict[int, set[int]],
    panel: ContextualPanel,
    probabilities: np.ndarray,
    quartet_template: np.ndarray | None = None,
    *,
    tree_index: ContextualTreeIndex | None = None,
    row_lookup: dict[tuple[int, int, int, int], int] | np.ndarray | None = None,
) -> PanelScoreResult:
    """Score current and both single-NNI alternatives with diagnostics."""

    edges, current_sum, discriminating_mask, emitted_rows = (
        _score_contextual_panel_impl(
            adjacency,
            panel,
            probabilities,
            quartet_template,
            tree_index=tree_index,
            row_lookup=row_lookup,
            plan_backend=None,
            collect_discriminating_mask=True,
        )
    )
    if discriminating_mask is None:
        raise AssertionError("diagnostic scorer did not construct its row mask")
    probabilities = np.asarray(probabilities, dtype=np.float64)
    return PanelScoreResult(
        edges=edges,
        base_score=current_sum / emitted_rows,
        discriminating_rows=int(np.count_nonzero(discriminating_mask)),
        emitted_rows=emitted_rows,
        predicted_classes=np.argmax(probabilities, axis=1).astype(np.int8),
        discriminating_mask=discriminating_mask,
    )


def expected_quartet_separation_distance(
    probabilities: np.ndarray,
    quartet_template: np.ndarray,
    n_taxa: int,
) -> np.ndarray:
    """Convert quartet posteriors to the weighted QDS pair-distance matrix."""

    probabilities = np.asarray(probabilities, dtype=np.float64)
    template = np.asarray(quartet_template, dtype=np.int64)
    if template.ndim != 2 or template.shape[1] != 4:
        raise ValueError("quartet_template must have shape (m, 4)")
    if probabilities.shape != (len(template), 3):
        raise ValueError("probability/template shape mismatch")
    matrix = np.zeros((n_taxa, n_taxa), dtype=np.float64)
    # Each pair receives one minus the posterior probability that it is paired
    # in the quartet.  The six updates are fully vectorized over quartet rows.
    for left_slot, right_slot, paired_class in (
        (0, 1, 0),
        (2, 3, 0),
        (0, 2, 1),
        (1, 3, 1),
        (0, 3, 2),
        (1, 2, 2),
    ):
        left = template[:, left_slot]
        right = template[:, right_slot]
        values = 1.0 - probabilities[:, paired_class]
        np.add.at(matrix, (left, right), values)
        np.add.at(matrix, (right, left), values)
    return matrix


def score_contextual_panel_projection(
    adjacency: dict[int, set[int]],
    panel: ContextualPanel,
    probabilities: np.ndarray,
    quartet_template: np.ndarray,
    *,
    tree_index: ContextualTreeIndex,
) -> PanelScoreResult:
    """Project one full posterior tensor through a coherent QDS/NJ local tree."""

    probabilities = np.asarray(probabilities, dtype=np.float64)
    template = np.asarray(quartet_template, dtype=np.int16)
    if probabilities.shape != (len(template), 3):
        raise ValueError("panel probability shape mismatch")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0):
        raise ValueError("probabilities must be finite and nonnegative")
    panel_size = len(panel.taxa)
    distance = expected_quartet_separation_distance(
        probabilities, template, panel_size
    )
    local_root = neighbor_joining(
        list(range(panel_size)),
        lambda left, right: float(distance[left, right]),
    )
    local_tree = tree_to_graph(local_root, panel_size)
    projected_classes = displayed_quartet_classes(local_tree, template)
    current_classes = _indexed_displayed_quartet_classes(
        tree_index, panel, template
    )
    row_lookup = {
        tuple(int(value) for value in quartet): row
        for row, quartet in enumerate(template)
    }
    rooted = tree_index.rooted
    all_discriminating: set[int] = set()
    results: list[PanelEdgeScore] = []
    for edge in panel.target_edges:
        groups = _panel_edge_groups(adjacency, panel, edge, rooted)
        rows: list[int] = []
        alternatives: list[tuple[int, int, int]] = []
        for one, two, three, four in product(*groups):
            quartet = tuple(sorted((one, two, three, four)))
            row = row_lookup[quartet]
            choices = (
                class_for_group_pairing(quartet, (one, two)),
                class_for_group_pairing(quartet, (one, three)),
                class_for_group_pairing(quartet, (one, four)),
            )
            if int(current_classes[row]) != choices[0]:
                raise AssertionError(
                    f"current class mismatch on projected edge {edge}"
                )
            rows.append(row)
            alternatives.append(choices)
        changed = np.asarray(rows, dtype=np.int64)
        choices_array = np.asarray(alternatives, dtype=np.int8)
        observed = projected_classes[changed, None]
        scores_array = np.mean(observed == choices_array, axis=0)
        best = int(np.argmax(scores_array))
        all_discriminating.update(rows)
        results.append(
            PanelEdgeScore(
                cover=panel.cover,
                panel_id=panel.panel_id,
                edge=edge,
                branch_nodes=branch_nodes_for_edge(adjacency, edge),
                scores=tuple(float(value) for value in scores_array),
                best=best,
                gain=float(scores_array[best] - scores_array[0]),
                discriminating_rows=len(rows),
                alternative_median_gains=(
                    float(scores_array[1] - scores_array[0]),
                    float(scores_array[2] - scores_array[0]),
                ),
            )
        )
    discriminating_mask = np.zeros(len(template), dtype=bool)
    if all_discriminating:
        discriminating_mask[np.fromiter(all_discriminating, dtype=np.int64)] = True
    return PanelScoreResult(
        edges=tuple(results),
        base_score=float(np.mean(projected_classes == current_classes)),
        discriminating_rows=len(all_discriminating),
        emitted_rows=len(template),
        predicted_classes=np.argmax(probabilities, axis=1).astype(np.int8),
        discriminating_mask=discriminating_mask,
    )


def consensus_evidence(
    scores: list[PanelEdgeScore],
    *,
    covers: int = 2,
    minimum_gain: float = 0.0,
) -> list[EdgeEvidence]:
    """Keep only edge actions selected independently by every panel cover."""

    by_edge: dict[Edge, list[PanelEdgeScore]] = {}
    for score in scores:
        by_edge.setdefault(score.edge, []).append(score)
    output: list[EdgeEvidence] = []
    for edge, values in sorted(by_edge.items()):
        values = sorted(values, key=lambda value: value.cover)
        if len(values) != covers or len({value.cover for value in values}) != covers:
            raise ValueError(f"edge {edge} does not have exactly {covers} cover scores")
        best = values[0].best
        if best == 0 or any(value.best != best for value in values):
            continue
        gain = min(value.gain for value in values)
        if gain <= minimum_gain:
            continue
        mean_scores = np.mean(np.asarray([value.scores for value in values]), axis=0)
        output.append(
            EdgeEvidence(
                edge=edge,
                branch_nodes=values[0].branch_nodes,
                scores=tuple(float(value) for value in mean_scores),
                best=best,
                margin=float(gain),
                quartet_count=sum(value.discriminating_rows for value in values),
            )
        )
    return output


def robust_consensus_evidence(
    scores: list[PanelEdgeScore],
    *,
    covers: int = 2,
) -> list[EdgeEvidence]:
    """Require positive mean and positive median row evidence in every cover.

    The ordinary score is a pseudo-log-likelihood mean over all discriminating
    quartet rows.  Its sign can be reversed by a minority of extreme posterior
    values.  Requiring the corresponding row-wise log-likelihood-ratio median
    to exceed the exact indifference point zero adds a parameter-free robust
    certificate without requesting another quartet or model evaluation.
    """

    by_edge: dict[Edge, list[PanelEdgeScore]] = {}
    for score in scores:
        by_edge.setdefault(score.edge, []).append(score)
    output: list[EdgeEvidence] = []
    for edge, values in sorted(by_edge.items()):
        values = sorted(values, key=lambda value: value.cover)
        if len(values) != covers or len({value.cover for value in values}) != covers:
            raise ValueError(f"edge {edge} does not have exactly {covers} cover scores")
        best = values[0].best
        if best == 0 or any(value.best != best for value in values):
            continue
        gain = min(value.gain for value in values)
        if gain <= 0.0:
            continue
        if any(value.alternative_median_gains[best - 1] <= 0.0 for value in values):
            continue
        mean_scores = np.mean(np.asarray([value.scores for value in values]), axis=0)
        output.append(
            EdgeEvidence(
                edge=edge,
                branch_nodes=values[0].branch_nodes,
                scores=tuple(float(value) for value in mean_scores),
                best=best,
                margin=float(gain),
                quartet_count=sum(value.discriminating_rows for value in values),
            )
        )
    return output


def agreement_consensus_evidence(
    primary_scores: list[PanelEdgeScore],
    verifier_scores: list[PanelEdgeScore],
    *,
    covers: int = 2,
) -> list[EdgeEvidence]:
    """Retain primary-model actions independently selected by a verifier."""

    primary = consensus_evidence(primary_scores, covers=covers, minimum_gain=0.0)
    verifier = consensus_evidence(verifier_scores, covers=covers, minimum_gain=0.0)
    verifier_actions = {(item.edge, item.best) for item in verifier}
    return [item for item in primary if (item.edge, item.best) in verifier_actions]


def edge_evidence_split(
    adjacency: dict[int, set[int]],
    evidence: EdgeEvidence,
    n_taxa: int,
    *,
    directed_masks: dict[tuple[int, int], int] | None = None,
) -> int:
    """Return the invariant taxon split inserted by one NNI proposal."""

    if evidence.best not in (1, 2):
        raise ValueError("an alternative NNI proposal must have best in {1, 2}")
    masks = (
        directed_masks
        if directed_masks is not None
        else directed_edge_leaf_masks(adjacency, n_taxa)
    )
    left, right = evidence.edge
    group_masks = [
        masks[(branch, owner)]
        for branch, owner in zip(
            evidence.branch_nodes, (left, left, right, right)
        )
    ]
    partner = 2 if evidence.best == 1 else 3
    split = canonical_split(group_masks[0] | group_masks[partner], n_taxa)
    if split is None:
        raise AssertionError("NNI proposal produced a trivial split")
    return split


def apply_consensus_nni(
    adjacency: dict[int, set[int]],
    scores: list[PanelEdgeScore],
    *,
    covers: int = 2,
    minimum_gain: float = 0.0,
) -> list[EdgeEvidence]:
    evidence = consensus_evidence(scores, covers=covers, minimum_gain=minimum_gain)
    return apply_independent_nni(adjacency, evidence, min_margin=minimum_gain)
