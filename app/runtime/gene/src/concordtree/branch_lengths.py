"""Fixed-topology JC69 maximum likelihood using level-batched GPU pruning/EM.

Original invariant sites are essential. Lengths are substitutions/site, NOT time.
No GTR, Gamma, clock, or species-tree coalescent-length claim is made here.
"""
from __future__ import annotations
import math
import time
import numpy as np
import torch
from concordtree.treeio import rooted, edge

@torch.inference_mode()
def fit_jc69(tree, names, states, *, device="cuda:0", max_iter=32,
             block_sites=0, tolerance=1e-6, initial=0.05, backend="auto"):
    if max_iter<1 or block_sites<0 or tolerance<=0:
        raise ValueError("iterations, block size and tolerance must be positive")
    if set(names)!=set(tree.names.values()) or len(names)!=len(set(names)):
        raise ValueError("exact alignment/tree taxon match required")
    if states.shape[0]!=len(names) or states.ndim!=2 or states.shape[1]<1:
        raise ValueError("invalid alignment")
    if backend not in {"auto","torch","triton"}: raise ValueError("invalid JC69 backend")
    started=time.perf_counter()
    dev=torch.device(device); dtype=torch.float64
    root,parent,order,depth=rooted(tree)
    ids={u:i for i,u in enumerate(order)}; m=len(ids); root_i=ids[root]
    requested_block=block_sites
    if block_sites==0:
        # About 1 GiB for the two main messages, capped at 262144 sites.
        safe=max(256,(1024**3)//(2*(m+1)*4*8))
        block_sites=min(262144,2**int(math.log2(safe))) if dev.type=="cuda" else 2048
    block_sites=min(block_sites,states.shape[1])
    leaf_nodes=sorted(tree.names)
    leaf_ids=torch.tensor([ids[u] for u in leaf_nodes],device=dev)
    row_lookup={n:i for i,n in enumerate(names)}
    sequence=states[[row_lookup[tree.names[u]] for u in leaf_nodes]]
    # Only the site block lives on GPU; alignment host storage is O(NL).
    children=np.full((m, max(map(len,tree.graph.values()))),m,dtype=np.int64)
    for u in order:
        child=[v for v in sorted(tree.graph[u]) if parent.get(v)==u]
        children[ids[u],:len(child)]=[ids[v] for v in child]
    child_t=torch.as_tensor(children,device=dev)
    pars=torch.tensor([ids[parent[u]] if u!=root else m for u in order],device=dev)
    levels=[]
    for d in range(max(depth.values())+1):
        levels.append(torch.tensor([ids[u] for u in order if depth[u]==d],device=dev))
    internals=[]
    for level in levels:
        # CPU construction avoids synchronizing level selectors in inner loop.
        internals.append(torch.tensor([ids[u] for u in order if depth[u]==len(internals)
                                       and u not in tree.names],device=dev))
    nonroot=[level for level in levels[1:] if len(level)]
    lengths=torch.tensor([max(1e-6,min(10., tree.lengths.get(edge(u,parent[u])) or initial))
                          if u!=root else initial for u in order]+[initial],dtype=dtype,device=dev)
    history=[]; converged=False
    chosen_backend=("triton" if dev.type=="cuda" else "torch") if backend=="auto" else backend
    if chosen_backend=="triton" and dev.type!="cuda": raise ValueError("Triton requires CUDA")
    def expectation(lengths, need_counts):
        if chosen_backend=="triton":
            from concordtree.jc69_cuda import expectation as fused
            return fused(sequence,lengths,leaf_ids=leaf_ids,internals=internals,nonroot=nonroot,
                         children=child_t,parents=pars,block_sites=block_sites,need_counts=need_counts)
        x=torch.exp(-4*lengths/3)
        mismatch=torch.zeros(m+1,dtype=dtype,device=dev)
        loglike=torch.zeros((),dtype=dtype,device=dev)
        for start in range(0,sequence.shape[1],block_sites):
            block=torch.as_tensor(sequence[:,start:start+block_sites].astype(np.int64),device=dev)
            b=block.shape[1]
            up=torch.ones((m+1,b,4),device=dev,dtype=dtype)
            obs=torch.nn.functional.one_hot(block.clamp_max(3),4).to(dtype)
            obs=torch.where((block<4)[...,None],obs,torch.ones_like(obs))
            up[leaf_ids]=obs
            for level in reversed(internals):
                if not len(level): continue
                ci=child_t[level]; values=up[ci]
                transform=x[ci,None,None]*values + (1-x[ci,None,None])*values.sum(-1,keepdim=True)/4
                product=transform.prod(1)
                norm=product.sum(-1,keepdim=True).clamp_min(1e-300)
                up[level]=product/norm
                loglike+=norm.log().sum()
            loglike-=b*math.log(4.)
            if not need_counts: continue
            down=torch.ones_like(up)
            for level in nonroot:
                p=pars[level]; branch_x=x[level,None,None]
                subtree=up[level]
                msg=branch_x*subtree+(1-branch_x)*subtree.sum(-1,keepdim=True)/4
                a=down[p]*up[p]/msg.clamp_min(1e-300)
                a/=a.sum(-1,keepdim=True).clamp_min(1e-300)
                dot=(a*subtree).sum(-1)
                sums=a.sum(-1)*subtree.sum(-1)
                off=(1-x[level,None])*(sums-dot)/4
                denom=x[level,None]*dot+(1-x[level,None])*sums/4
                mismatch[level]+=(off/denom.clamp_min(1e-300)).sum(1)
                down[level]=branch_x*a+(1-branch_x)*a.sum(-1,keepdim=True)/4
        return float(loglike.cpu()),mismatch
    for iteration in range(max_iter):
        ll,counts=expectation(lengths,True)
        history.append(ll)
        if len(history)>1 and ll < history[-2]-max(1e-6,abs(ll)*1e-10):
            raise RuntimeError("JC69 EM likelihood decreased")
        if len(history)>1 and (ll-history[-2])/sequence.shape[1] < tolerance:
            converged=True; break
        expected=(counts/sequence.shape[1]).clamp(0,0.75-1e-8)
        lengths=(-.75*torch.log1p(-4*expected/3)).clamp(1e-6,10.)
        lengths[root_i]=initial; lengths[m]=initial
    final_ll,_=expectation(lengths,False)
    if final_ll<history[-1]-max(1e-6,abs(final_ll)*1e-10): raise RuntimeError("final likelihood decreased")
    if abs(final_ll-history[-1])>1e-8: history.append(final_ll)
    result=tree.copy()
    host=lengths.cpu().numpy()
    for u in order:
        if u!=root: result.lengths[edge(u,parent[u])]=float(host[ids[u]])
    if dev.type=="cuda": torch.cuda.synchronize(dev)
    return result,{"model":"JC69", "method":"level-batched pruning / endpoint EM",
                   "units":"expected substitutions per site", "device":str(dev), "backend":chosen_backend,
                   "sites":sequence.shape[1],"taxa":len(names),"max_iter":max_iter,
                   "converged":converged,"em_iterations":iteration+1,
                   "log_likelihood_history":history,"log_likelihood":final_ll,
                   "block_sites":block_sites,"requested_block_sites":requested_block,"tolerance_per_site":tolerance,
                   "seconds":time.perf_counter()-started,
                   "warning":"Fixed-topology, homogeneous JC69 only; not GTR+Gamma or dating."}

def fit_iqtree(tree, msa, output_dir, *, model="GTR+G4", threads=8, executable=None):
    """Optional mature CPU optimizer; only operates on copies in output_dir."""
    import os
    from pathlib import Path
    import re
    import shutil
    import subprocess
    from concordtree.treeio import newick,parse_newick
    from concordtree.dna_models import parse_model
    parse_model(model)
    if threads<1:
        raise ValueError("invalid IQ-TREE thread count")
    exe=executable or os.environ.get("CONCORDTREE_IQTREE") or shutil.which("iqtree3") or shutil.which("iqtree2") or shutil.which("iqtree")
    if not exe: raise ValueError("IQ-TREE not found; set CONCORDTREE_IQTREE")
    output_dir=Path(output_dir).resolve()
    from concordtree.output_paths import _output_path
    _output_path(output_dir,[Path(msa).resolve()])
    output_dir.mkdir(parents=True)
    copied=output_dir/"input.phy"; topology=output_dir/"topology.nwk"
    shutil.copyfile(msa,copied); topology.write_text(newick(tree)+"\n")
    command=[str(exe),"-s",str(copied),"-te",str(topology),"-m",model,
             "-nt",str(threads),"-pre",str(output_dir/"fit"),"--redo"]
    start=time.perf_counter()
    with (output_dir/"stdout.log").open("w") as f:
        subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=3600)
    fitted=parse_newick((output_dir/"fit.treefile").read_text())
    if set(fitted.names.values())!=set(tree.names.values()):
        raise RuntimeError("IQ-TREE returned a different taxon set")
    from concordtree.treeio import split_map
    order=sorted(tree.names.values())
    if set(split_map(tree,order).values())!=set(split_map(fitted,order).values()):
        raise RuntimeError("fixed topology changed")
    # Preserve node IDs for subsequent reliability annotations.
    old_splits=split_map(tree,order); new_splits=split_map(fitted,order)
    length_by_split={new_splits[e]:v for e,v in fitted.lengths.items() if e in new_splits}
    terminal={fitted.names[u]:fitted.lengths[edge(u,next(iter(fitted.graph[u])))] for u in fitted.names}
    out=tree.copy()
    for e,s in old_splits.items(): out.lengths[e]=length_by_split[s]
    for u,name in out.names.items(): out.lengths[edge(u,next(iter(out.graph[u])))]=terminal[name]
    text=(output_dir/"fit.iqtree").read_text()
    match=re.search(r"Log-likelihood of the tree:\s+([-\d.]+)",text)
    return out,{"model":model,"method":"IQ-TREE fixed-topology CPU ML","command":command,
                "units":"expected substitutions per site","seconds":time.perf_counter()-start,
                "log_likelihood":float(match.group(1)) if match else None}
