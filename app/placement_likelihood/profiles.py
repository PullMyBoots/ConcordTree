"""Predeclared lightweight reference profiles, same exact likelihood kernel."""
import time,heapq
import numpy as np
from scipy.optimize import minimize
from taxon_placement import PlacementIndex,QuartetPredictor,attach
from backbone import restricted_graph
from concordtree.treeio import Tree,newick_python
from .fast_likelihood import Plan,decode,transitions,BOUNDS,EdgeCache
from .accelerated import pullback

PROFILES={
    'native64':dict(reference_iterations=64,context_representatives=2,model_frequencies='optimized'),
    'empirical32':dict(reference_iterations=32,context_representatives=2,model_frequencies='empirical'),
    'context32':dict(reference_iterations=32,context_representatives=4,model_frequencies='empirical'),
}

def reference_fit(tree,names,states,profile):
    start=time.perf_counter();plan=Plan(tree,names,states);initial,m0=plan.seed()
    if profile['model_frequencies']=='empirical':
        counts=np.bincount(np.asarray(states).ravel(),minlength=5)[:4]+.5
        pi=counts/counts.sum();m0[5:8]=np.log(pi[:3]/pi[3]);free=[0,1,2,3,4,8,9]
    else:free=list(range(10))
    nb=plan.E;best=[np.inf,None];calls=0
    def objective(x):
        nonlocal calls
        t=(x[:nb]/2)**2;mx=m0.copy();mx[free]=x[nb:];model=decode(mx)
        p,dt=transitions(model,t,True);ll,dp,dpi,dw=plan.evaluate(p,model['pi'],model['weights'])
        grad=np.r_[np.einsum('ekij,ekij->e',dp,dt)*x[:nb]/2,pullback(model,t,p,dp,dpi,dw)[free]]
        value=-ll/plan.sites;g=-grad/plan.sites;calls+=1
        if not np.isfinite(value) or not np.isfinite(g).all():raise ArithmeticError('nonfinite profile objective')
        if value<best[0]:best[:]=[value,x.copy()]
        return value,g
    res=minimize(objective,np.r_[2*np.sqrt(initial),m0[free]],jac=True,method='L-BFGS-B',
        bounds=[(.002,2*np.sqrt(10.))]*nb+[BOUNDS[i] for i in free],
        options=dict(maxiter=profile['reference_iterations'],maxls=20,ftol=1e-9,gtol=1e-5,maxcor=10))
    x=best[1];t=(x[:nb]/2)**2;mx=m0.copy();mx[free]=x[nb:];model=decode(mx)
    out=tree.copy();out.lengths=dict(zip(plan.edges,map(float,t)))
    return out,dict(log_likelihood=-best[0]*plan.sites,model_vector=mx.tolist(),
        frequencies_acgt=model['pi'].tolist(),Q_acgt=model['q'].tolist(),
        category_rates=model['rates'].tolist(),category_weights=model['weights'].tolist(),
        gamma_alpha=model['alpha'],invariant_proportion=model['pinv'],
        converged=bool(res.success),message=str(res.message),iterations=int(res.nit),evaluations=calls,
        seconds=time.perf_counter()-start,profile=profile)

def infer(tree,alignment,mlp,query,profile_name):
    start=time.perf_counter();profile=PROFILES[profile_name];names=alignment.names;lookup={n:i for i,n in enumerate(names)}
    index=PlacementIndex(tree,names,2)
    predictor=QuartetPredictor(alignment.packed,names,alignment.original_sites,'gene',mlp=mlp)
    val=predictor(index.queries(lookup[query]));scores=index.score_edges(index.potentials(val,'log'))
    candidates=heapq.nsmallest(5,scores,key=lambda e:(-scores[e],e))
    context_index=index if profile['context_representatives']==2 else PlacementIndex(tree,names,profile['context_representatives'])
    refs=set()
    for a,b in candidates:
        for u in (a,b):
            if u in tree.names:refs.add(tree.names[u])
            else:
                rows=context_index.triples[context_index.owners==context_index.node_index[u]]
                refs.update(names[int(row)] for row in rows.flat)
    assert len(refs)<=120
    g,labels=restricted_graph(tree.graph,tree.names,refs);context=Tree(g,labels,{})
    assert all(a in g and b in g[a] for a,b in candidates)
    local_names=[n for n in names if n in refs];pos=np.arange(alignment.original_sites)
    rows=alignment.packed[[lookup[n] for n in local_names+[query]]]
    states=((rows[:,pos//2]>>((pos%2)*4))&15).astype(np.uint8)
    mlp_seconds=time.perf_counter()-start;phase=time.perf_counter()
    fitted,meta=reference_fit(context,local_names,states[:-1],profile);reference_seconds=time.perf_counter()-phase
    valid=(states[:-1]<4)&(states[-1]<4);p=((states[:-1]!=states[-1])&valid).sum(1)/np.maximum(valid.sum(1),1)
    pendant=float(np.clip(-.75*np.log(max(1e-8,1-4*float(p.min())/3)),1e-6,2.))
    phase=time.perf_counter();compressed,counts=np.unique(states,axis=1,return_counts=True)
    cache=EdgeCache(fitted,local_names,compressed[:-1],meta,counts)
    trials=[cache.place(e,compressed[-1],pendant,40) for e in candidates]
    cached_seconds=time.perf_counter()-phase
    winner=max(range(len(trials)),key=lambda i:(trials[i]['log_likelihood'],-i));chosen=candidates[winner]
    return attach(tree,query,chosen),dict(profile=profile_name,configuration=profile,context_names=local_names,
        reference_fit=meta,reference_lengths=[[list(e),t] for e,t in fitted.lengths.items()],
        context_reference_newick=newick_python(fitted),mlp_edge=list(candidates[0]),chosen_edge=list(chosen),
        chosen_rank=winner+1,trials=trials,mlp_seconds=mlp_seconds,reference_seconds=reference_seconds,
        cached_seconds=cached_seconds,seconds=time.perf_counter()-start)
