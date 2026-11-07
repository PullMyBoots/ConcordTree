"""One exact integer Sankoff reconstruction as a cheap branch initializer.

Classical unit-cost parsimony is NOT the final estimator/model. Counts +1/2 per
edge seed the subsequent all-site joint GTR+Gamma optimization. No true lengths,
site sampling, model fitting, or data-dependent restart thresholds here.
"""
import time
import numpy as np
import torch
import triton
import triton.language as tl
from concordtree.treeio import rooted,edge


@triton.jit
def _leaves(COST,OBS,V,S:tl.constexpr,INF:tl.constexpr,B:tl.constexpr):
    row=tl.program_id(0);node=tl.load(V+row)
    s=tl.program_id(1)*B+tl.arange(0,B);mask=s<S
    code=tl.load(OBS+row*S+s,mask,other=4)
    offset=(node*S+s)*4
    for j in tl.static_range(4):
        tl.store(COST+offset+j,tl.where((code==j)|(code==4),0,INF),mask)


@triton.jit
def _up(COST,C,V,S:tl.constexpr,H:tl.constexpr,M:tl.constexpr,B:tl.constexpr):
    node=tl.load(V+tl.program_id(0));s=tl.program_id(1)*B+tl.arange(0,B);mask=s<S
    a=tl.full((B,),0,tl.int32);b=a;c=a;d=a
    for h in tl.static_range(H):
        child=tl.load(C+node*H+h)
        if child<M:
            off=(child*S+s)*4
            u0=tl.load(COST+off,mask,other=0);u1=tl.load(COST+off+1,mask,other=0)
            u2=tl.load(COST+off+2,mask,other=0);u3=tl.load(COST+off+3,mask,other=0)
            best=tl.minimum(tl.minimum(u0,u1),tl.minimum(u2,u3))+1
            a+=tl.minimum(u0,best);b+=tl.minimum(u1,best)
            c+=tl.minimum(u2,best);d+=tl.minimum(u3,best)
    off=(node*S+s)*4
    tl.store(COST+off,a,mask);tl.store(COST+off+1,b,mask)
    tl.store(COST+off+2,c,mask);tl.store(COST+off+3,d,mask)


@triton.jit
def _down(COST,ASSIGN,PAR,V,COUNT,PAIRS,S:tl.constexpr,B:tl.constexpr,STATS:tl.constexpr):
    node=tl.load(V+tl.program_id(0));parent=tl.load(PAR+node)
    s=tl.program_id(1)*B+tl.arange(0,B);mask=s<S
    pa=tl.load(ASSIGN+parent*S+s,mask,other=0)
    off=(node*S+s)*4
    best=tl.load(COST+off,mask,other=0)+(pa!=0).to(tl.int32)
    state=tl.full((B,),0,tl.int32)
    for j in tl.static_range(1,4):
        value=tl.load(COST+off+j,mask,other=0)+(pa!=j).to(tl.int32)
        state=tl.where(value<best,j,state);best=tl.minimum(best,value)
    tl.store(ASSIGN+node*S+s,state,mask)
    tl.atomic_add(COUNT+node,tl.sum(tl.where(mask&(state!=pa),1,0)))
    if STATS:
        lo=tl.minimum(pa,state);hi=tl.maximum(pa,state)
        slot=0
        for a in tl.static_range(4):
            for b in tl.static_range(a+1,4):
                count=tl.sum(tl.where(mask&(lo==a)&(hi==b),1,0)).to(tl.int64)
                tl.atomic_add(PAIRS+slot,count)
                slot+=1


@torch.no_grad()
def seed_parsimony(tree,names,states,*,device="cuda:0",block_sites=0,collect_model_statistics=False):
    start=time.perf_counter();tree.require_binary()
    if torch.device(device).type!="cuda":raise ValueError("CUDA initializer required")
    if len(names)!=len(set(names)) or set(names)!=set(tree.names.values()):raise ValueError("exact taxa match required")
    states=np.asarray(states)
    if states.ndim!=2 or states.shape[0]!=len(names) or states.shape[1]<1:raise ValueError("invalid alignment shape")
    if not np.issubdtype(states.dtype,np.integer) or states.min()<0 or states.max()>4 or states.min()==4:
        raise ValueError("states must be integer0..4 with observed nucleotides")
    if block_sites<0:raise ValueError("block_sites must be nonnegative")
    root,parent,order,depth=rooted(tree);m=len(order);ids={u:i for i,u in enumerate(order)}
    # One linear traversal groups levels; no repeated O(NH) node scan.
    groups=[[] for _ in range(max(depth.values())+1)];inner=[[] for _ in groups]
    leaves=[]
    for u in order:
        groups[depth[u]].append(ids[u])
        if u not in tree.names:inner[depth[u]].append(ids[u])
        else:leaves.append(u)
    h=max(map(len,tree.graph.values()));children=np.full((m+1,h),m,dtype=np.int64)
    for u in order:
        child=[ids[v] for v in sorted(tree.graph[u]) if parent.get(v)==u]
        children[ids[u],:len(child)]=child
    children=torch.as_tensor(children,device=device)
    parents=torch.tensor([ids[parent[u]] if u!=root else m for u in order],device=device)
    leaf_ids=torch.tensor([ids[u] for u in leaves],device=device)
    lookup={n:i for i,n in enumerate(names)}
    sequence=states[[lookup[tree.names[u]] for u in leaves]]
    if not block_sites:
        safe=max(1,(256*1024**2)//((m+1)*20));block_sites=2**int(np.log2(safe))
    b=min(block_sites,states.shape[1])
    cost=torch.zeros((m+1,b,4),device=device,dtype=torch.int32)
    assignment=torch.zeros((m+1,b),device=device,dtype=torch.int32)
    counts=torch.zeros(m+1,device=device,dtype=torch.int32)
    pairs=torch.zeros(6,device=device,dtype=torch.int64)
    up_levels=[torch.tensor(g,device=device) for g in reversed(inner) if g]
    down_levels=[torch.tensor(g,device=device) for g in groups[1:] if g]
    optimum=torch.zeros((),device=device,dtype=torch.int64)
    for pos in range(0,states.shape[1],b):
        n=min(b,states.shape[1]-pos)
        block=np.full((len(names),b),4,dtype=np.uint8);block[:,:n]=sequence[:,pos:pos+n]
        obs=torch.tensor(block,device=device)
        _leaves[(len(leaves),triton.cdiv(b,128))](cost,obs,leaf_ids,b,m+2,128)
        for g in up_levels:_up[(len(g),triton.cdiv(b,128))](cost,children,g,b,h,m,128)
        values,chosen=cost[ids[root]].min(-1)
        assignment[ids[root]]=chosen;optimum+=values.sum()
        for g in down_levels:_down[(len(g),triton.cdiv(b,128))](cost,assignment,parents,g,counts,pairs,b,128,collect_model_statistics)
    changes=counts.cpu().numpy();total=int(optimum.cpu())
    if int(changes.sum())!=total:raise RuntimeError("Sankoff cost/reconstruction disagreement")
    out=tree.copy()
    for u in order:
        if u!=root:out.lengths[edge(u,parent[u])]=float(np.clip((changes[ids[u]]+.5)/states.shape[1],1e-6,10.))
    result={"method":"unit-cost Sankoff single optimal reconstruction + half-count",
        "seconds":time.perf_counter()-start,"block_sites":b,"minimum_parsimony_changes":total,
        "taxa":len(names),"sites":states.shape[1],"warning":"initializer only, not substitutions/site estimate"}
    if collect_model_statistics:
        histogram=pairs.cpu().tolist()
        if sum(histogram)!=total:raise RuntimeError("replacement histogram/optimal cost disagreement")
        result["substitutions_ac_ag_at_cg_ct_gt"]=histogram
        result["seconds"]=time.perf_counter()-start
    return out,result
