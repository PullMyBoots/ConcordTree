"""Original MLP-NNI with a persistent sorted neighbor array.

Same native quartet plan, prediction batches, arithmetic reduction, NumPy tie
handling, margin/edge ordering, independent moves and move-count stopping.
Only graph representation and construction of transient evidence objects change.
"""
from time import perf_counter
import numpy as np
from search_progress import ExactRecurrence

def refine(adjacency,n_taxa,predict,*,plan_backend,nni,passes=10000,
           representatives=4,min_margin=.10,stop_moves_at_most=None,progress=None):
    from concordtree.assets import load_candidate_graph_backend
    backend=load_candidate_graph_backend()
    neighbors=nni._adjacency_neighbor_array(adjacency,n_taxa)
    recurrence=ExactRecurrence()
    recurrence.repeated(neighbors.tobytes())
    stop='emergency_pass_guard';converged=False
    history=[];timing={key:0. for key in ['plan_seconds','predict_seconds','compact_reduction_seconds','evidence_seconds','apply_seconds']}
    for pass_idx in range(passes):
        start=perf_counter()
        edges,branches,offsets,ordered,canonical=plan_backend.compile_nni_plan(neighbors,n_taxa,representatives)
        timing['plan_seconds']+=perf_counter()-start
        count=len(canonical);start=perf_counter()
        probabilities=np.empty((count,3),dtype=np.float64)
        for begin in range(0,count,65536):
            end=min(begin+65536,count);values=np.asarray(predict(canonical[begin:end]),dtype=np.float64)
            if values.shape!=(end-begin,3):raise ValueError('Probability shape mismatch')
            if (not np.isfinite(values).all() or np.any(values<0) or
                    np.any(values>1) or not np.allclose(values.sum(axis=1),1.,rtol=1e-5,atol=1e-7)):
                raise ArithmeticError('MLP probabilities must be finite normalized probabilities')
            probabilities[begin:end]=values
        timing['predict_seconds']+=perf_counter()-start;start=perf_counter()
        scores=np.asarray(plan_backend.reduce_nni_arithmetic(ordered,probabilities,offsets),dtype=np.float64)
        if not np.isfinite(scores).all():raise ArithmeticError('nonfinite MLP NNI scores')
        timing['compact_reduction_seconds']+=perf_counter()-start;start=perf_counter()
        best=np.asarray(np.argsort(scores,axis=1)[:,-1],dtype=np.int64)
        margins=np.zeros(len(scores),dtype=np.float64);rows=np.nonzero(best!=0)[0]
        margins[rows]=scores[rows,best[rows]]-scores[rows,0]
        timing['evidence_seconds']+=perf_counter()-start;start=perf_counter()
        selected=backend.apply_nni_moves(neighbors,edges,branches,best,margins,min_margin)
        timing['apply_seconds']+=perf_counter()-start
        record=dict(pass_=pass_idx+1,edges_scored=len(edges),quartets_scored=count,moves=len(selected),
                    selected=[dict(edge=edges[i].tolist(),scores=scores[i].tolist(),best=int(best[i]),margin=float(margins[i])) for i in selected])
        record['pass']=record.pop('pass_');history.append(record)
        if progress is not None:progress(record)
        if not len(selected) or (stop_moves_at_most is not None and len(selected)<=stop_moves_at_most):
            stop='requested_move_ratio';converged=True;break
        if recurrence.repeated(neighbors.tobytes()):
            stop='repeated_full_search_state';break
    for node,row in enumerate(neighbors):adjacency[node]=set(int(x) for x in row if x>=0)
    return dict(passes=history,moves=sum(x['moves'] for x in history),timing=timing,
                stop=stop,converged=converged,
                warnings=[] if converged else ['MLP warmstart stopped at '+stop+'; downstream refinement still required'])
