"""Active sparse quartet composite likelihood with persistent evidence.

At each sweep add the unchanged nearest24 discriminating queries, score each
quartet once, then take exact NNI gains for ALL accumulated observations. The
finite search state is (tree, queried quartets), not the tree alone. No median,
confidence gate, score blend, branch fit, or scientific sweep cap is present.

This is a heuristic active-query search: generic MSC scoring consistency,
decisiveness of its eventual query set, and global optimization are not proved.
"""
from __future__ import annotations
import time
import numpy as np
from .fixed_rank import scaffold_queries
from .core import compile_plan, select_moves, apply_move
from . import _msc_sparse
from search_progress import ExactRecurrence, EMERGENCY_SWEEPS

_ROW = np.dtype([(name, '<i4') for name in ['a','b','c','d']])


def keys(queries):
    return np.ascontiguousarray(queries, dtype='<i4').view(_ROW).reshape(-1)


def tree_edges(graph):
    return np.asarray([(u,v) for u in sorted(graph) for v in sorted(graph[u])
                       if u < v], dtype=np.int64)


class ActiveQuartetScorer:
    def __init__(self, base):
        self.base = base
        self.packed = base.packed
        self.queries = np.empty((0,4), dtype=np.int32)
        self.weights = np.empty((0,3), dtype=np.float64)
        self.timing = {}

    def score(self, graph):
        started = time.perf_counter()
        n = len(self.packed)
        proposed = np.asarray(scaffold_queries(graph, n), dtype=np.int32)
        oldkeys, proposedkeys = keys(self.queries), keys(proposed)
        if len(oldkeys):
            positions = np.searchsorted(oldkeys, proposedkeys)
            present = ((positions < len(oldkeys)) &
                       (oldkeys[np.minimum(positions, len(oldkeys)-1)] == proposedkeys))
            new = proposed[~present]
        else:
            new = proposed
        planning = time.perf_counter() - started
        phase = time.perf_counter()
        newweights = self.base.predict(new) if len(new) else np.empty((0,3))
        if newweights.shape != (len(new),3) or not np.isfinite(newweights).all():
            raise ArithmeticError('finite canonical quartet scores required')
        scoring = time.perf_counter() - phase
        phase = time.perf_counter()
        if len(new):
            merged = np.concatenate((self.queries, new))
            weights = np.concatenate((self.weights, newweights))
            order = np.argsort(keys(merged), kind='stable')
            is_new = order >= len(self.queries)
            self.queries = merged[order]
            self.weights = weights[order]
        else:
            is_new = np.zeros(len(self.queries), dtype=bool)
        plan = compile_plan(graph, n)
        result = _msc_sparse.evaluate(tree_edges(graph), n, self.queries, self.weights)
        if not np.array_equal(plan.edges, result['edges']):
            raise AssertionError('native exact NNI edge order changed')
        classes = np.asarray(result['classes'])
        new_value = float(self.weights[is_new, classes[is_new]].sum())
        arithmetic_bound = (512*np.finfo(np.float64).eps *
                            max(1., np.abs(self.weights).max(axis=1).sum()))
        scores = np.asarray(result['scores'])
        # Floating arithmetic guard only; no biological confidence threshold.
        scores[np.abs(scores) <= arithmetic_bound] = 0.
        self.timing = dict(objective=float(result['objective']),
                           new_evidence_objective=new_value,
                           new_quartets=len(new), proposed_quartets=len(proposed),
                           accumulated_quartets=len(self.queries),
                           evidence_bytes=self.queries.nbytes+self.weights.nbytes,
                           active_quartets=int(result['active_quartets']),
                           planning_seconds=planning, quartet_seconds=scoring,
                           exact_reduction_seconds=time.perf_counter()-phase,
                           arithmetic_gain_bound=float(arithmetic_bound),
                           elapsed_seconds=time.perf_counter()-started)
        return plan, scores


def refine_active(graph, scorer, *, stop_ratio=.005, callback=None):
    if not 0 <= stop_ratio < 1:
        raise ValueError('stop ratio must be in [0,1)')
    g = {u:set(v) for u,v in graph.items()}
    n = len(scorer.packed)
    history = []
    previous = None
    recurrence = ExactRecurrence()
    stop = None
    while True:
        if scorer.base.check is not None:
            scorer.base.check()
        plan, scores = scorer.score(g)
        value = scorer.timing['objective']
        if not np.isfinite(value) or not np.isfinite(scores).all():
            raise ArithmeticError('nonfinite accumulated MSC objective or NNI gain')
        # Evidence is append-only and frozen. Equal cardinality therefore
        # identifies the same evidence set; a larger set is a different state.
        # The already compiled central edges plus their four adjacent nodes
        # encode every edge of a binary tree; avoid another Python graph walk.
        if recurrence.repeated((plan.edges.tobytes(),plan.branches.tobytes(),len(scorer.queries))):
            stop='repeated_full_search_state'
            break
        if previous is not None:
            recovered = value-scorer.timing['new_evidence_objective']
            if not np.isclose(recovered, previous, rtol=1e-9, atol=1e-9):
                raise ArithmeticError('accumulated global objective delta failed')
        selected = select_moves(g, plan, scores)
        for edge, choice, gain in selected:
            apply_move(g, plan.edges[edge], plan.branches[edge], choice)
        gain = sum(move[2] for move in selected)
        previous = value+gain
        row = dict(sweep=len(history)+1, moves=len(selected),
                   move_ratio=len(selected)/(n-3), summed_objective_gain=gain,
                   **scorer.timing)
        history.append(row)
        if callback is not None:
            callback(g,row)
        if row['move_ratio'] <= stop_ratio:
            stop='requested_move_ratio'
            break
        if len(history)>=EMERGENCY_SWEEPS:
            raise RuntimeError('MSC refinement emergency sweep guard exhausted before convergence')
    return g, dict(history=history, stop=stop,
                           search_converged=stop=='requested_move_ratio',
                           warnings=[] if stop=='requested_move_ratio' else ['Repeated full MSC search state; result is not certified converged'],
                           emergency_sweep_guard=EMERGENCY_SWEEPS,
                           public_stop_ratio=stop_ratio,
                           branch_lengths_inferred=False,
                           objective='sum frozen quartet log scores over persistent accumulated nearest24 evidence',
                           exact_all_accumulated_nni_deltas=True,
                           scientific_round_cap=None,
                           general_msc_consistency_proved=False,
                           global_search_consistency_proved=False)
