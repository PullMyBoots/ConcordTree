"""Bounded groups share reference model fitting and exact conditional messages."""
import time
import numpy as np
from taxon_placement import QuartetPredictor,edge
from backbone import restricted_graph
from concordtree.treeio import Tree,newick_python
from .placement import shortlist,CONFIG
from .fast_likelihood import fit,EdgeCache

BATCH_CONFIG=dict(group_queries=8,reference_iterations=64,candidate_iterations=40,
                  model='GTR+FO+I+G4',query_order='alignment',joint_query_topology='sequential insertion with descendant-edge expansion')

def insert(tree,name,selected,leaf,three=None):
    a,b=selected;center=leaf+1
    if leaf in tree.graph or center in tree.graph:raise AssertionError('node ID collision')
    g={u:set(vs) for u,vs in tree.graph.items()};labels=dict(tree.names)
    g[a].remove(b);g[b].remove(a);g[a].add(center);g[b].add(center)
    g[center]={a,b,leaf};g[leaf]={center};labels[leaf]=name
    lengths=dict(tree.lengths);lengths.pop(selected,None)
    if three is not None:
        for e,t in zip((edge(a,center),edge(b,center),edge(leaf,center)),three):lengths[e]=float(t)
    return Tree(g,labels,lengths)

def complete(tree,alignment,mlp,missing):
    start=time.perf_counter();names=alignment.names;lookup={n:i for i,n in enumerate(names)}
    if len(set(names))!=len(names) or set(missing)!=set(names)-set(tree.names.values()):raise ValueError('taxon mismatch')
    predictor=QuartetPredictor(alignment.packed,names,alignment.original_sites,'gene',mlp=mlp)
    history=[];groups=[]
    for begin in range(0,len(missing),BATCH_CONFIG['group_queries']):
        queries=missing[begin:begin+BATCH_CONFIG['group_queries']];phase=time.perf_counter();refs=set();shortlists={}
        for q in queries:
            context,cn,states,candidates,scores=shortlist(tree,alignment,predictor,q)
            refs.update(cn);shortlists[q]=candidates
        mlp_seconds=time.perf_counter()-phase;phase=time.perf_counter()
        g,labels=restricted_graph(tree.graph,tree.names,refs);context=Tree(g,labels,{})
        local_names=[n for n in names if n in refs];nref=len(local_names)
        assert nref<=60*len(queries),'group reference bound'
        positions=np.arange(alignment.original_sites);rows=alignment.packed[[lookup[n] for n in local_names+queries]]
        states=((rows[:,positions//2]>>((positions%2)*4))&15).astype(np.uint8)
        fitted,meta=fit(context,local_names,states[:nref],iterations=BATCH_CONFIG['reference_iterations'])
        reference_lengths=[[list(e),t] for e,t in fitted.lengths.items()]
        reference_topology=newick_python(context)
        ref_seconds=time.perf_counter()-phase;phase=time.perf_counter()
        # Compress exact joint columns once for the known bounded query group.
        compressed,counts=np.unique(states,axis=1,return_counts=True)
        row_lookup={n:i for i,n in enumerate(local_names+queries)}
        descendants={e:{e} for q in queries for e in shortlists[q]}
        group_history=[]
        for q in queries:
            cache_states=compressed[[row_lookup[n] for n in local_names]]
            cache=EdgeCache(fitted,local_names,cache_states,meta,counts)
            candidates=sorted(set().union(*(descendants[e] for e in shortlists[q])))
            assert all(a in tree.graph and b in tree.graph[a] and a in fitted.graph and b in fitted.graph[a] for a,b in candidates)
            qi=row_lookup[q];query=compressed[qi]
            # A data-derived pendant starting point, not an alternate score.
            full_query=states[qi];ref_states=states[[row_lookup[n] for n in local_names]]
            valid=(ref_states<4)&(full_query<4);p=((ref_states!=full_query)&valid).sum(1)/np.maximum(valid.sum(1),1)
            pendant=float(np.clip(-.75*np.log(max(1e-8,1-4*float(p.min())/3)),1e-6,2.))
            trials=[cache.place(e,query,pendant,BATCH_CONFIG['candidate_iterations']) for e in candidates]
            winner=max(range(len(trials)),key=lambda i:(trials[i]['log_likelihood'],-i));chosen=candidates[winner];lengths=trials[winner]['insertion_lengths']
            leaf=max(tree.graph)+1;center=leaf+1
            tree=insert(tree,q,chosen,leaf);fitted=insert(fitted,q,chosen,leaf,lengths);local_names.append(q)
            # Later queries may form sisters to earlier queries, rather than
            # being forced independently onto the original reference backbone.
            new_edges={edge(chosen[0],center),edge(chosen[1],center),edge(leaf,center)}
            for initial,edges in descendants.items():
                if chosen in edges:edges.remove(chosen);edges.update(new_edges)
            r=dict(query=q,chosen_edge=list(chosen),candidates=len(candidates),trials=trials)
            history.append(r);group_history.append(r)
        groups.append(dict(queries=queries,reference_taxa=nref,mlp_seconds=mlp_seconds,
            reference_seconds=ref_seconds,query_seconds=time.perf_counter()-phase,
            reference_fit=meta,reference_names=[n for n in names if n in refs],
            reference_lengths=reference_lengths,reference_topology_newick=reference_topology,
            final_local_newick=newick_python(fitted),placements=group_history))
    return tree,dict(configuration=BATCH_CONFIG,groups=groups,placements=history,seconds=time.perf_counter()-start)
