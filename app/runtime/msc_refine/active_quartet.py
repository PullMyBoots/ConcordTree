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
from .core import select_moves, apply_move
from search_progress import ExactRecurrence, EMERGENCY_SWEEPS


def tree_edges(graph):
    return np.asarray([(u,v) for u in sorted(graph) for v in sorted(graph[u])
                       if u < v], dtype=np.int64)


class ActiveQuartetScorer:
    """Accumulated evidence state for StreamSumActiveScorer."""
    def __init__(self, base):
        self.base = base
        self.packed = base.packed
        self.queries = np.empty((0,4), dtype=np.int32)
        self.weights = np.empty((0,3), dtype=np.float64)
        self.timing = {}


def refine_active(graph, scorer, *, stop_ratio=.005, callback=None,max_rounds=None,min_gain=0.,backbone=None,names=None):
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
        candidate_scores=np.where(scores>min_gain,scores,0.) if min_gain else scores
        if backbone is not None:
            candidate_scores=candidate_scores.copy()
            allowed=np.asarray(backbone.allowed_edges(g,dict(enumerate(names)),plan.edges),dtype=bool)
            candidate_scores[~allowed,1:]=candidate_scores[~allowed,0,None]
        selected = select_moves(g, plan, candidate_scores)
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
        if max_rounds is not None and len(history)>=max_rounds:
            stop='round_budget';break
        if len(history)>=EMERGENCY_SWEEPS:
            raise RuntimeError('MSC refinement emergency sweep guard exhausted before convergence')
    return g, dict(history=history, stop=stop,
                           search_converged=stop=='requested_move_ratio',
                           warnings=[] if stop=='requested_move_ratio' else ['MSC search stopped at '+str(stop)+'; result is not certified converged'],
                           emergency_sweep_guard=EMERGENCY_SWEEPS,
                           public_stop_ratio=stop_ratio,
                           branch_lengths_inferred=False,
                           objective='sum frozen quartet log scores over persistent accumulated nearest24 evidence',
                           exact_all_accumulated_nni_deltas=True,
                           scientific_round_cap=max_rounds,minimum_objective_gain=min_gain,
                           general_msc_consistency_proved=False,
                           global_search_consistency_proved=False)
