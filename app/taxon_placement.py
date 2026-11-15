"""Direct backbone placement by trichotomy evidence and exact edge rerooting.

One query is placed without AP-NJ, whole-tree NNI, or parameter fitting. All
backbone edges are considered; state is O(NR), with fixed representative width.
"""
from __future__ import annotations
import time
import heapq
import hashlib
import numpy as np


def edge(a,b):return (min(a,b),max(a,b))


class PlacementIndex:
    def __init__(self,tree,names,representatives=4):
        tree.require_binary()
        if not isinstance(representatives,int) or not 1<=representatives<=64:
            raise ValueError('representatives must be an integer in 1..64')
        self.graph=tree.graph;self.labels=tree.names
        lookup={name:i for i,name in enumerate(names)}
        if len(lookup)!=len(names):raise ValueError('duplicate alignment taxon names')
        self.rows={u:lookup[name] for u,name in tree.names.items()}
        root=min(tree.names,key=tree.names.get)
        parent={root:None};order=[root]
        for u in order:
            for v in sorted(tree.graph[u]):
                if v==parent[u]:continue
                if v in parent:raise ValueError('backbone is not a tree')
                parent[v]=u;order.append(v)
        if len(order)!=len(tree.graph):raise ValueError('disconnected backbone')
        self.parent=parent;self.order=order;self.root=root
        k=representatives
        def nearest(values):return sorted(values)[:k]
        down={}
        for u in reversed(order):
            values=[(0,self.rows[u])] if u in self.rows else []
            for v in tree.graph[u]:
                if parent.get(v)==u:values.extend((d+1,t) for d,t in down[v])
            down[u]=nearest(values)
        up={root:[]};messages={}
        for u in order:
            children=sorted(v for v in tree.graph[u] if parent.get(v)==u)
            for v in children:
                values=list(up[u])
                if u in self.rows:values.append((0,self.rows[u]))
                for sibling in children:
                    if sibling!=v:values.extend((d+1,t) for d,t in down[sibling])
                up[v]=nearest((d+1,t) for d,t in values)
                messages[u,v]=[t for d,t in down[v]]
                messages[v,u]=[t for d,t in up[v]]
        self.internal=[u for u in order if u not in tree.names]
        self.neighbors={u:sorted(tree.graph[u]) for u in self.internal}
        triples=[];owners=[]
        for j,u in enumerate(self.internal):
            groups=[messages[u,v] for v in self.neighbors[u]]
            seen=set()
            for r in range(k):
                triple=tuple(g[(r*(2*i+1))%len(g)] for i,g in enumerate(groups))
                if triple not in seen:triples.append(triple);owners.append(j);seen.add(triple)
        self.triples=np.asarray(triples,dtype=np.int64)
        self.owners=np.asarray(owners,dtype=np.int64)
        self.counts=np.bincount(self.owners,minlength=len(self.internal))
        self.node_index={u:j for j,u in enumerate(self.internal)}

    def queries(self,query):
        return np.column_stack((np.full(len(self.triples),query,dtype=np.int64),self.triples))

    def potentials(self,values,aggregation='log'):
        values=np.asarray(values,dtype=np.float64)
        if values.shape!=(len(self.triples),3) or not np.isfinite(values).all():
            raise ArithmeticError('finite three-way quartet evidence required')
        if aggregation in ('log','probability'):
            if np.any(values<0) or np.any(values>1) or not np.allclose(values.sum(axis=1),1.,rtol=1e-5,atol=1e-7):
                raise ArithmeticError('three normalized quartet probabilities required')
        if aggregation=='log':values=np.log(np.maximum(values,1e-12))
        elif aggregation not in ('probability','raw'):raise ValueError('unknown evidence aggregation')
        out=np.zeros((len(self.internal),3),dtype=np.float64)
        np.add.at(out,self.owners,values)
        out/=self.counts[:,None]
        out-=out.max(axis=1,keepdims=True)
        return out

    def score_edges(self,potentials):
        p=np.asarray(potentials,dtype=np.float64)
        if p.shape!=(len(self.internal),3) or not np.isfinite(p).all():
            raise ArithmeticError('finite three-way node potentials required')
        base=sum(p[j,self.neighbors[u].index(self.parent[u])] for j,u in enumerate(self.internal))
        scores={edge(self.root,next(iter(self.graph[self.root]))):float(base)}
        for u in self.order:
            if u in self.labels:continue
            j=self.node_index[u];incoming=scores[edge(u,self.parent[u])]
            old=p[j,self.neighbors[u].index(self.parent[u])]
            for v in self.graph[u]:
                if v==self.parent[u]:continue
                scores[edge(u,v)]=float(incoming+p[j,self.neighbors[u].index(v)]-old)
        if len(scores)!=len(self.graph)-1:raise AssertionError('incomplete placement edge scoring')
        return scores


def attach(tree,name,selected):
    from concordtree.treeio import Tree
    g={u:set(vs) for u,vs in tree.graph.items()};labels=dict(tree.names)
    a,b=selected;leaf=max(g)+1;center=leaf+1
    g[a].remove(b);g[b].remove(a)
    g[a].add(center);g[b].add(center);g[leaf]={center};g[center]={a,b,leaf}
    labels[leaf]=name
    # Topology-only output: do not fabricate new branch lengths or parameter fits.
    return Tree(g,labels,{})


def place(tree,names,missing,predict,*,representatives=4,aggregation='log',callback=None):
    history=[];started=time.perf_counter()
    names=list(names);missing=list(missing)
    if len(set(names))!=len(names) or len(set(missing))!=len(missing):raise ValueError('unique alignment and missing taxa required')
    if not set(tree.names.values()).issubset(names):raise ValueError('backbone taxa missing from alignment')
    if set(missing)!=set(names)-set(tree.names.values()):raise ValueError('missing taxa must equal alignment minus backbone')
    tree.require_binary()
    lookup={name:i for i,name in enumerate(names)}
    for name in missing:
        phase=time.perf_counter();index=PlacementIndex(tree,names,representatives)
        plan_seconds=time.perf_counter()-phase
        queries=index.queries(lookup[name]);phase=time.perf_counter()
        values=predict(queries);predict_seconds=time.perf_counter()-phase
        phase=time.perf_counter()
        potentials=index.potentials(values,aggregation)
        informative=bool(np.any(np.ptp(potentials,axis=1)>1e-12))
        scores=index.score_edges(potentials)
        ordered=heapq.nsmallest(2,scores,key=lambda e:(-scores[e],e))
        best=ordered[0];gap=float(scores[best]-scores[ordered[1]])
        tree=attach(tree,name,best)
        row=dict(taxon=name,edge=list(best),quartets=len(queries),score=float(scores[best]),score_gap=gap,informative=informative,
            plan_seconds=plan_seconds,predict_seconds=predict_seconds,decode_seconds=time.perf_counter()-phase)
        history.append(row)
        if callback:callback(tree,row)
    return tree,dict(method='direct trichotomy quartet placement with exact edge reroot DP',
        representatives=representatives,aggregation=aggregation,placements=history,
        placement_seconds=time.perf_counter()-started,quartets=sum(x['quartets'] for x in history),
        apnj_calls=0,whole_tree_refinement_calls=0,branch_lengths_inferred=False,
        evidence_interpretation='Uncalibrated differences of correlated quartet scores; not support percentages or placement probabilities.',
        warnings=['No discriminatory quartet evidence for '+r['taxon'] for r in history if not r['informative']])


class QuartetPredictor:
    def __init__(self,packed,names,sites,task,*,kind='mlp',loci=None,mlp=None):
        import torch
        started=time.perf_counter();self.kind=kind
        if kind=='mlp':
            from frontend import MLPWarmstart,TASK_PREDICTOR
            self.mlp=mlp or MLPWarmstart(TASK_PREDICTOR[task]);self.mlp.load()
            filtered,length=self.mlp.sequence.filter_conserved_packed(packed,int(sites),torch.get_num_threads())
            self.packed=torch.as_tensor(filtered,device='cuda:0');self.length=int(length)
        elif kind=='kernel':
            from msc_refine.barycenter_tree_tiled import TiledBarycenterScorer
            from msc_refine.observations import read_boundaries
            from runtime_config import HERE
            regions=read_boundaries(loci,sites) if loci else [(0,sites)]
            filename='observed_loci.npz' if loci else 'msa_only.npz'
            from engine import MSC_ASSETS
            mode='observed_loci' if loci else 'msa_only'
            asset=HERE/'assets/msc'/filename
            if hashlib.sha256(asset.read_bytes()).hexdigest()!=MSC_ASSETS[mode][1]:raise RuntimeError('MSC asset hash mismatch')
            self.kernel=TiledBarycenterScorer(packed,sites,regions,asset)
        else:raise ValueError('unknown placement predictor')
        torch.cuda.synchronize();self.setup_seconds=time.perf_counter()-started

    def __call__(self,ordered):
        import torch
        from concordtree._core.learned_nni import remap_group_order_probabilities
        canonical=np.sort(ordered,axis=1)
        if self.kind=='mlp':
            rows=[]
            with torch.inference_mode():
                for first in range(0,len(canonical),8192):
                    q=torch.as_tensor(canonical[first:first+8192],device='cuda:0',dtype=torch.long)
                    if self.length==0:
                        rows.append(np.full((len(q),3),1./3));continue
                    features=self.mlp.pattern.compute_pattern_frequencies_cuda_packed(self.packed,q,self.length).contiguous()
                    values=torch.softmax(self.mlp.model(features),dim=-1)
                    values=torch.where(features.any(dim=-1,keepdim=True),values,torch.full_like(values,1./3.))
                    rows.append(values.cpu().numpy())
            values=np.concatenate(rows) if rows else np.empty((0,3))
        else:values=self.kernel.predict(canonical)
        return remap_group_order_probabilities(ordered,values)
