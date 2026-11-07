"""Uncalibrated final-branch evidence summaries. No sequence scoring or search.

Gene: normalized conditional three-topology likelihood weights, not aBayes.
Species: mean score advantage over each cached quartet's strongest alternative.
Neither number is a bootstrap percentage or a correctness probability.
"""
from __future__ import annotations
import hashlib
import math
from pathlib import Path
import time

import numpy as np

METHODS = {
    'gene': dict(id='gene_local_likelihood_weight', name='Local likelihood weight',
        interpretation='Normalized likelihood of the final split among three local NNI topologies, conditional on the final model and fixed surrounding subtrees; central lengths are approximately profiled.',
        limitations=['Not bootstrap, SH-aLRT, aBayes, or a calibrated posterior probability.',
                    'Conditional local comparison; does not measure model adequacy or global tree uncertainty.',
                    'More sites can give extreme weights even under systematic model error.']),
    'species': dict(id='mean_worst_quartet_margin', name='Quartet margin index',
        interpretation='Mean final-topology score minus the strongest alternative score for each informative cached quartet spanning the four incident subtrees; mapped monotonically to a display index.',
        ranking_field='mean_worst_margin',
        display_transform='50 + 50*m/(1+abs(m)); fixed unit scale, not fitted to correctness labels',
        limitations=['Not bootstrap, localPP, gCF, sCF, or a calibrated probability.',
                    'Quartets are adaptively selected, correlated and scored using the same alignment; their count is not an independent sample size.',
                    'The 0..100 index is not percentage agreement; legacy agreement is retained separately.',
                    'AUC-prioritized adoption does not imply improvement at every retained branch fraction.',
                    'The frozen quartet scorer has no general MSC consistency guarantee.']),
}


def gene_records(edges, branches, deltas, current_gain, sites):
    """O(N) normalization of local profile results already returned by a scan."""
    edges=np.asarray(edges,dtype=np.int64);alt=np.asarray(deltas,dtype=np.float64)
    base=np.asarray(current_gain,dtype=np.float64)
    if edges.ndim!=2 or edges.shape[1]!=2 or alt.shape!=(len(edges),2) or base.shape!=(len(edges),):
        raise ValueError('invalid local likelihood support dimensions')
    scores=np.column_stack((base,alt))
    if not np.isfinite(scores).all():raise ArithmeticError('nonfinite local support scores')
    shifted=scores-scores.max(axis=1,keepdims=True)
    weights=np.exp(shifted);weights/=weights.sum(axis=1,keepdims=True)
    result=[]
    for i,e in enumerate(edges):
        result.append(dict(edge=list(map(int,e)),value=float(100*weights[i,0]),status='available',
            topology_weights=weights[i].tolist(),
            relative_log_scores=(scores[i]-scores[i,0]).tolist(),
            local_log_likelihood_gap=float(scores[i,0]-max(scores[i,1:])),
            profiled_current_gain=float(base[i]),evaluated_sites=int(sites),
            adjacent_nodes=list(map(int,branches[i]))))
    return result


def species_margin_index(margin):
    """Fixed monotone display scale. The signed raw margin remains authoritative.

    Reciprocal form avoids cancellation near zero for strongly negative margins.
    This is a unit convention, not a learned correctness calibration.
    """
    value=np.asarray(margin,dtype=np.float64)
    if not np.isfinite(value).all():raise ArithmeticError('nonfinite quartet margin')
    distance=50./(1.+np.abs(value))
    return np.where(value>=0,100.-distance,distance)


def species_records(edges, codes, weights, *, block_size=262144,backend='native'):
    """O(Q+N), bounded-memory reduction; each unique quartet contributes once.

    The existing classifier encodes the current quartet topology in bits0..1,
    NNI alternative1 in bits2..3, and its unique discriminating edge in bits4+.
    Fully tied scores supply no resolution and are excluded. Two-way ties split
    one vote equally; tolerance is solely a floating-point roundoff allowance.
    """
    edges=np.asarray(edges,dtype=np.int64);codes=np.asarray(codes)
    weights=np.asarray(weights);m=len(edges)
    if edges.shape!=(m,2) or codes.ndim!=1 or weights.shape!=(len(codes),3):
        raise ValueError('invalid quartet support dimensions')
    if block_size<1:raise ValueError('positive block size required')
    if backend not in ('native','numpy'):raise ValueError('unknown support reduction backend')
    if backend=='native':
        from _branch_support_reduce import reduce_margins
        counts,votes,margin_sum=reduce_margins(codes,weights,m)
        totals,decisive,ties=counts.T
    else:
        totals=np.zeros(m,dtype=np.int64);decisive=totals.copy();ties=totals.copy()
        votes=np.zeros((m,3),dtype=np.float64)
        margin_sum=np.zeros(m,dtype=np.float64)
    for start in range(0,len(codes) if backend=='numpy' else 0,block_size):
        code=codes[start:start+block_size];w=weights[start:start+block_size]
        e=(code>>4)-1;state=code&3;alt=(code>>2)&3
        if np.any(code<0) or np.any(e>=m) or np.any(state>2):
            raise ValueError('invalid cached quartet topology code')
        active=e>=0
        if np.any(active & ((alt>2)|(alt==state))):raise ValueError('invalid alternative code')
        e=e[active];state=state[active];alt=alt[active];w=w[active]
        if not len(e):continue
        if not np.isfinite(w).all():raise ArithmeticError('nonfinite quartet support scores')
        tolerance=64*np.finfo(np.float64).eps*np.maximum(1.,np.abs(w).max(axis=1))
        best=w.max(axis=1,keepdims=True)
        winner=(best-w)<=tolerance[:,None];count=winner.sum(axis=1)
        informative=count<3
        totals+=np.bincount(e,minlength=m)
        decisive+=np.bincount(e[informative],minlength=m)
        ties+=np.bincount(e[count==2],minlength=m)
        rows=np.arange(len(e));other=3-state-alt
        gap=w[rows,state]-np.maximum(w[rows,alt],w[rows,other])
        margin_sum+=np.bincount(e[informative],weights=gap[informative],minlength=m)
        for j,choice in enumerate((state,alt,other)):
            vote=np.where(informative,winner[rows,choice]/count,0.)
            votes[:,j]+=np.bincount(e,weights=vote,minlength=m)
    result=[]
    for i,e in enumerate(edges):
        fraction=votes[i]/decisive[i] if decisive[i] else None
        margin=float(margin_sum[i]/decisive[i]) if decisive[i] else None
        result.append(dict(edge=list(map(int,e)),value=float(species_margin_index(margin)) if margin is not None else None,
            mean_worst_margin=margin,
            sampled_quartet_agreement=float(100*fraction[0]) if fraction is not None else None,
            status='available' if decisive[i] else 'no_informative_quartets',
            informative_quartets=int(decisive[i]),active_quartets=int(totals[i]),
            uninformative_quartets=int(totals[i]-decisive[i]),two_way_tied_quartets=int(ties[i]),
            topology_vote_fractions=fraction.tolist() if fraction is not None else None))
    return result


def split_identifiers(tree):
    """O(N) rerooting-invariant fingerprints for UI mapping, not proof keys.

    No per-branch leaf lists or growing N-bit split masks are materialized.
    Score assignment uses exact node pairs, never these probabilistic IDs.
    """
    root=next(u for u in tree.graph if u not in tree.names)
    parent={root:None};order=[root]
    for u in order:
        for v in sorted(tree.graph[u]):
            if v==parent[u]:continue
            if v in parent:raise ValueError('cycle in support tree')
            parent[v]=u;order.append(v)
    if len(order)!=len(tree.graph):raise ValueError('disconnected support tree')
    count={};token={}
    for u in reversed(order):
        count[u]=1 if u in tree.names else 0
        token[u]=int.from_bytes(hashlib.blake2b(tree.names[u].encode('utf-8'),digest_size=16,
                          person=b'CT-split-v1').digest(),'big') if u in tree.names else 0
        for v in tree.graph[u]:
            if parent.get(v)==u:count[u]+=count[v];token[u]^=token[v]
    result={};n=len(tree.names);full=token[root]
    for u in order:
        v=parent[u]
        if v is None or u in tree.names or v in tree.names:continue
        size,h=min((count[u],token[u]),(n-count[u],full^token[u]))
        result[tuple(sorted((u,v)))]=(f'{size}:{h:032x}',size)
    return result


def export_support(tree, task, raw, *, write_json, plain_prediction, serialize):
    """Write sidecar and annotated Newick without changing topology/lengths."""
    started=time.perf_counter();tree.require_binary()
    pairs={tuple(sorted((u,v))) for u in tree.graph for v in tree.graph[u]
           if u<v and u not in tree.names and v not in tree.names}
    records=raw['branches'];mapped={tuple(sorted(r['edge'])):r for r in records}
    if set(mapped)!=pairs or len(records)!=len(pairs):raise ValueError('support must cover final internal edges exactly')
    ids=split_identifiers(tree);annotations={};rows=[]
    for i,e in enumerate(sorted(pairs),1):
        row=dict(mapped[e]);bid=f'b{i:06d}';value=row['value']
        if value is not None and (not math.isfinite(value) or not 0<=value<=100):
            raise ArithmeticError('support outside0..100')
        # Standard numeric internal label, plus a stable sidecar row reference.
        label=format(value,'.17g' if task=='species' else '.6g') if value is not None else ''
        annotations[e]=label+f'[&support_id={bid}]'
        row.update(branch_id=bid,split_id=ids[e][0],smaller_side_taxa=ids[e][1]);rows.append(row)
    plain_prediction=Path(plain_prediction)
    plain=plain_prediction.with_name('prediction.unannotated.nwk')
    plain.write_bytes(plain_prediction.read_bytes())
    annotated=serialize(annotations)+'\n'
    plain_prediction.write_text(annotated)
    available=sum(r['value'] is not None for r in rows)
    report=dict(schema_version=1,task=task,method=METHODS[task],scale=[0,100],
        calibrated=False,is_bootstrap=False,is_posterior_probability=False,
        branches=rows,summary=dict(internal_branches=len(rows),available=available,
            unavailable=len(rows)-available),computation={k:v for k,v in raw.items() if k!='branches'},
        split_id_definition='128-bit XOR taxon-name fingerprint and smaller side count; display mapping only',
        annotated_tree=str(plain_prediction),unannotated_tree=str(plain))
    sidecar=plain_prediction.with_name('branch-support.json')
    report['computation']['export_seconds']=time.perf_counter()-started
    write_json(sidecar,report,indent=None)
    return dict(enabled=True,method=METHODS[task]['id'],label=METHODS[task]['name'],path=str(sidecar),
        annotated_tree=str(plain_prediction),unannotated_tree=str(plain),calibrated=False,
        **({'ranking_field':METHODS[task]['ranking_field']} if task=='species' else {}),
        **report['summary'],computation=report['computation'])
