"""Candidate-guided bounded local likelihood placement for one or many taxa."""
import heapq,time
import numpy as np
from taxon_placement import PlacementIndex,QuartetPredictor,edge
from backbone import restricted_graph
from concordtree.treeio import Tree,newick_python
from .profiles import reference_fit,PROFILES
from .incremental_cache import IncrementalCache
from .batch_placement import insert

CONFIG=dict(top_edges=5,mlp_representatives=2,context_representatives=4,
            group_limit=8,reference_profile=PROFILES['context32'],
            candidate_iterations=40,candidate_evaluations=160,
            grouping='connected components of overlapping candidate-edge sets',
            model='GTR+F+I+G4',sites='all',cache_update='one outward traversal; unaffected sides reused')

def complete(tree,alignment,mlp,missing,*,callback=None):
    start=time.perf_counter();names=alignment.names;lookup={n:i for i,n in enumerate(names)}
    if len(names)!=len(lookup) or len(set(missing))!=len(missing):raise ValueError('duplicate taxa')
    if set(missing)!=set(names)-set(tree.names.values()):raise ValueError('alignment/backbone taxon mismatch')
    tree.require_binary();history=[];groups=[]
    if not missing:return tree.copy(),dict(configuration=CONFIG,placements=[],groups=[],seconds=time.perf_counter()-start)
    predictor=QuartetPredictor(alignment.packed,names,alignment.original_sites,'gene',mlp=mlp)
    for start_q in range(0,len(missing),CONFIG['group_limit']):
        queries=missing[start_q:start_q+CONFIG['group_limit']];phase=time.perf_counter()
        index=PlacementIndex(tree,names,CONFIG['mlp_representatives'])
        context_index=PlacementIndex(tree,names,CONFIG['context_representatives'])
        offsets=np.r_[0,np.cumsum(context_index.counts)]
        owner_rows={u:context_index.triples[offsets[j]:offsets[j+1]] for u,j in context_index.node_index.items()}
        plans={};edge_queries={};parent={q:q for q in queries}
        def find(q):
            while parent[q]!=q:parent[q]=parent[parent[q]];q=parent[q]
            return q
        for q in queries:
            val=predictor(index.queries(lookup[q]));scores=index.score_edges(index.potentials(val,'log'))
            candidates=heapq.nsmallest(CONFIG['top_edges'],scores,key=lambda e:(-scores[e],e));refs=set()
            for a,b in candidates:
                for u in (a,b):
                    if u in tree.names:refs.add(tree.names[u])
                    else:refs.update(names[int(row)] for row in owner_rows[u].flat)
            for e in candidates:
                if e in edge_queries:parent[find(q)]=find(edge_queries[e])
                else:edge_queries[e]=q
            plans[q]=dict(candidates=candidates,refs=refs)
        components={}
        for q in queries:components.setdefault(find(q),[]).append(q)
        mlp_seconds=time.perf_counter()-phase
        for component in components.values():
            group_start=time.perf_counter();refs=set().union(*(plans[q]['refs'] for q in component))
            assert len(refs)<=120*len(component),'bounded context violated'
            g,labels=restricted_graph(tree.graph,tree.names,refs);context=Tree(g,labels,{})
            local_names=[n for n in names if n in refs];nref=len(local_names)
            rows=alignment.packed[[lookup[n] for n in local_names+component]];pos=np.arange(alignment.original_sites)
            states=((rows[:,pos//2]>>((pos%2)*4))&15).astype(np.uint8);row_lookup={n:i for i,n in enumerate(local_names+component)}
            phase=time.perf_counter();fitted,meta=reference_fit(context,local_names,states[:nref],CONFIG['reference_profile'])
            reference_seconds=time.perf_counter()-phase;reference_lengths=[[list(e),t] for e,t in fitted.lengths.items()]
            phase=time.perf_counter();compressed,counts=np.unique(states,axis=1,return_counts=True)
            cache=IncrementalCache(fitted,local_names,compressed[:nref],meta,counts)
            descendants={e:{e} for q in component for e in plans[q]['candidates']};group_history=[]
            for q in component:
                candidates=list(dict.fromkeys(e2 for e in plans[q]['candidates'] for e2 in sorted(descendants[e])))
                assert all(a in tree.graph and b in tree.graph[a] and a in fitted.graph and b in fitted.graph[a] for a,b in candidates)
                qi=row_lookup[q];query=compressed[qi];ref_states=states[[row_lookup[n] for n in local_names]]
                valid=(ref_states<4)&(states[qi]<4);p=((ref_states!=states[qi])&valid).sum(1)/np.maximum(valid.sum(1),1)
                pendant=float(np.clip(-.75*np.log(max(1e-8,1-4*float(p.min())/3)),1e-6,2.))
                trials=[cache.place(e,query,pendant,CONFIG['candidate_iterations']) for e in candidates]
                winner=max(range(len(trials)),key=lambda i:(trials[i]['log_likelihood'],-i));chosen=candidates[winner]
                leaf=max(tree.graph)+1;center=leaf+1;lengths=trials[winner]['insertion_lengths']
                tree=insert(tree,q,chosen,leaf);fitted=insert(fitted,q,chosen,leaf,lengths);local_names.append(q)
                cache.update(fitted,chosen,leaf,query)
                new_edges={edge(chosen[0],center),edge(chosen[1],center),edge(leaf,center)}
                for initial,edges in descendants.items():
                    if chosen in edges:edges.remove(chosen);edges.update(new_edges)
                row=dict(query=q,chosen_edge=list(chosen),leaf_id=leaf,trials=trials,original_candidates=[list(e) for e in plans[q]['candidates']],
                    has_observed_overlap=bool(np.any((states[qi]<4)&np.any(states[:nref]<4,axis=0))))
                history.append(row);group_history.append(row)
                if callback:callback(tree,row)
            groups.append(dict(queries=component,reference_taxa=nref,reference_names=[n for n in names if n in refs],
                reference_lengths=reference_lengths,reference_fit=meta,reference_topology_newick=newick_python(context),
                reference_seconds=reference_seconds,query_seconds=time.perf_counter()-phase,
                mlp_seconds=mlp_seconds*len(component)/len(queries),seconds=time.perf_counter()-group_start,
                placements=group_history,final_local_newick=newick_python(fitted)))
    return tree,dict(configuration=CONFIG,groups=groups,placements=history,seconds=time.perf_counter()-start)
