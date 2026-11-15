"""Fast candidate-guided explicit likelihood placement. Inference has no truth IO."""
import time,heapq
import numpy as np
from taxon_placement import PlacementIndex,QuartetPredictor,attach,edge
from backbone import restricted_graph
from concordtree.treeio import Tree,newick_python
from .fast_likelihood import fit,EdgeCache

CONFIG=dict(top_edges=5,representatives=2,model='GTR+FO+I+G4',reference_iterations=64,
            candidate_iterations=40,context_limit=60,all_sites=True)

def shortlist(tree,alignment,predictor,query):
    names=alignment.names;lookup={n:i for i,n in enumerate(names)}
    index=PlacementIndex(tree,names,CONFIG['representatives'])
    val=predictor(index.queries(lookup[query]));scores=index.score_edges(index.potentials(val,'log'))
    candidates=heapq.nsmallest(CONFIG['top_edges'],scores,key=lambda e:(-scores[e],e))
    refs=set()
    for a,b in candidates:
        for u in (a,b):
            if u in tree.names:refs.add(tree.names[u])
            else:
                rows=index.triples[index.owners==index.node_index[u]]
                refs.update(names[int(row)] for row in rows.flat)
    if len(refs)>CONFIG['context_limit']:raise AssertionError('context bound exceeded')
    # For three-reference minimal backbone contexts, a quartet still defines
    # meaningful likelihood. No arbitrary minimum-four failure.
    g,labels=restricted_graph(tree.graph,tree.names,refs);context=Tree(g,labels,{})
    context.require_binary()
    assert all(a in g and b in g[a] for a,b in candidates),'candidate edge lost'
    local_names=[n for n in names if n in refs]
    rows=np.ascontiguousarray(alignment.packed[[lookup[n] for n in local_names+[query]]])
    pos=np.arange(alignment.original_sites)
    states=((rows[:,pos//2]>>((pos%2)*4))&15).astype(np.uint8)
    return context,local_names,states,candidates,scores

def lengths_after(reference,query,selected,three):
    out=attach(reference,query,selected);a,b=selected;leaf=max(reference.graph)+1;center=leaf+1
    out.lengths=dict(reference.lengths);out.lengths.pop(selected,None)
    for e,t in zip((edge(a,center),edge(b,center),edge(leaf,center)),three):out.lengths[e]=float(t)
    return out

def infer(tree,alignment,mlp,query,*,mode='cached',alternatives=False):
    start=time.perf_counter();predictor=QuartetPredictor(alignment.packed,alignment.names,alignment.original_sites,'gene',mlp=mlp)
    context,names,states,candidates,scores=shortlist(tree,alignment,predictor,query)
    mlp_seconds=time.perf_counter()-start;phase=time.perf_counter()
    fitted,meta=fit(context,names,states[:-1],iterations=CONFIG['reference_iterations'])
    reference_seconds=time.perf_counter()-phase
    valid=(states[:-1]<4)&(states[-1]<4);p=((states[:-1]!=states[-1])&valid).sum(1)/np.maximum(valid.sum(1),1)
    pendant=float(np.clip(-.75*np.log(max(1e-8,1-4*float(p.min())/3)),1e-6,2.))
    phase=time.perf_counter();compressed,counts=np.unique(states,axis=1,return_counts=True)
    cache=EdgeCache(fitted,names,compressed[:-1],meta,counts)
    trials=[cache.place(e,compressed[-1],pendant,CONFIG['candidate_iterations']) for e in candidates]
    cached_seconds=time.perf_counter()-phase
    results=dict(cached=trials);joint_seconds=0.
    if alternatives or mode=='joint':
        phase=time.perf_counter();joint=[]
        for e,trial in zip(candidates,trials):
            candidate=lengths_after(fitted,query,e,trial['insertion_lengths'])
            out,m=fit(candidate,names+[query],states,previous=meta,iterations=CONFIG['candidate_iterations'],model_iterations=False)
            joint.append(dict(edge=list(e),log_likelihood=m['log_likelihood'],model_fit=m,
                fitted_lengths=[[list(ed),float(t)] for ed,t in out.lengths.items()]))
        joint_seconds=time.perf_counter()-phase;results['joint']=joint
    selected=results[mode];winner=max(range(len(selected)),key=lambda i:(selected[i]['log_likelihood'],-i));chosen=candidates[winner]
    return attach(tree,query,chosen),dict(configuration=CONFIG,mode=mode,context_names=names,
        context_reference_newick=newick_python(fitted),reference_fit=meta,
        mlp_edge=list(candidates[0]),chosen_edge=list(chosen),chosen_rank=winner+1,trials=results,
        mlp_seconds=mlp_seconds,reference_seconds=reference_seconds,cached_seconds=cached_seconds,
        joint_seconds=joint_seconds,seconds=time.perf_counter()-start)
