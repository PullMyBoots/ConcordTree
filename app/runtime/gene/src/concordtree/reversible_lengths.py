"""Fixed-topology general reversible DNA likelihood and joint GPU ML fitting.

Only endpoint sufficient statistics are differentiated, not the O(nodes*sites)
pruning tape. The observed-likelihood score equals the derivative of the complete
endpoint/category log likelihood at the current parameters (Fisher identity).
L-BFGS-B and Gamma quantiles are small host tasks; pruning/transitions/scores run
on the requested device. Site categories are shared along the ENTIRE site/tree.
"""
import math
import time
import numpy as np
import torch
from scipy.optimize import minimize
from concordtree.treeio import rooted, edge
from concordtree.dna_models import parse_model, ModelParameters, transition_matrices


class PruningPlan:
    def __init__(self, tree, names, states, device, categories, block_sites=0):
        if len(names)!=len(set(names)) or set(names)!=set(tree.names.values()):
            raise ValueError("exact alignment/tree taxon match required")
        states=np.asarray(states)
        if states.ndim!=2 or states.shape[0]!=len(names) or states.shape[1]<1:
            raise ValueError("invalid alignment shape")
        if not np.issubdtype(states.dtype,np.integer) or states.min()<0 or states.max()>4:
            raise ValueError("states must be integer A,C,G,T,missing codes 0..4")
        if states.min()==4: raise ValueError("alignment contains no observed nucleotides")
        if block_sites<0: raise ValueError("block_sites must be nonnegative")
        self.device=torch.device(device); self.categories=categories
        self.root,parent,self.order,depth=rooted(tree)
        ids={u:i for i,u in enumerate(self.order)}; self.ids=ids; self.m=len(ids)
        self.root_i=ids[self.root]
        self.edges=[edge(u,parent[u]) for u in self.order if u!=self.root]
        self.edge_nodes=[ids[u] for u in self.order if u!=self.root]
        leaf_nodes=sorted(tree.names)
        self.leaf_ids=torch.tensor([ids[u] for u in leaf_nodes],device=device)
        lookup={name:i for i,name in enumerate(names)}
        self.sequence=states[[lookup[tree.names[u]] for u in leaf_nodes]]
        h=max(map(len,tree.graph.values()))
        children=np.full((self.m,h),self.m,dtype=np.int64)
        for u in self.order:
            child=[v for v in sorted(tree.graph[u]) if parent.get(v)==u]
            children[ids[u],:len(child)]=[ids[v] for v in child]
        self.children=torch.as_tensor(children,device=device)
        self.parents=torch.tensor([ids[parent[u]] if u!=self.root else self.m for u in self.order],device=device)
        self.internals=[]; self.nonroot=[]
        for d in range(max(depth.values())+1):
            level=[ids[u] for u in self.order if depth[u]==d]
            self.internals.append(torch.tensor([ids[u] for u in self.order if depth[u]==d and u not in tree.names],device=device))
            if d: self.nonroot.append(torch.tensor(level,device=device))
        self.requested_block=block_sites
        if not block_sites:
            # ~256 MiB messages plus bounded sibling/endpoint temporaries.
            safe=max(1,(256*1024**2)//(2*(self.m+1)*categories*4*8))
            block_sites=min(65536,2**int(math.log2(safe))) if self.device.type=="cuda" else min(2048,safe)
        self.block_sites=min(block_sites,states.shape[1])

    @torch.no_grad()
    def expectation(self, transitions, pi, weights, *, need_counts=True, backend="torch"):
        """Return exact log likelihood and endpoint/root/category posteriors."""
        m=self.m; k=self.categories; dev=self.device; dtype=torch.float64
        counts=torch.zeros((m+1,k,4,4),device=dev,dtype=dtype)
        root_counts=torch.zeros(4,device=dev,dtype=dtype)
        category_counts=torch.zeros(k,device=dev,dtype=dtype)
        total=torch.zeros((),device=dev,dtype=dtype)
        # Sentinel represents an absent child: all-one likelihood, identity P.
        trans=torch.eye(4,device=dev,dtype=dtype).repeat(m+1,k,1,1)
        trans[self.edge_nodes]=transitions
        if backend=="triton":
            from concordtree.reversible_cuda import upward, downward
        for start in range(0,self.sequence.shape[1],self.block_sites):
            block=torch.as_tensor(self.sequence[:,start:start+self.block_sites].astype(np.int64),device=dev)
            b=block.shape[1]
            up=torch.ones((m+1,k,b,4),device=dev,dtype=dtype)
            obs=torch.nn.functional.one_hot(block.clamp_max(3),4).to(dtype)
            obs=torch.where((block<4)[...,None],obs,torch.ones_like(obs))
            up[self.leaf_ids]=obs[:,None,:,:]
            logscale=torch.zeros((k,b),device=dev,dtype=dtype)
            for level in reversed(self.internals):
                if not len(level): continue
                if backend=="triton":
                    logscale+=upward(up,trans,self.children,level)
                else:
                    ci=self.children[level]
                    values=up[ci]                       # level,child,k,site,state
                    message=torch.einsum("nhkij,nhksj->nhksi",trans[ci],values)
                    product=message.prod(1)
                    norm=product.sum(-1).clamp_min(1e-300)
                    up[level]=product/norm[...,None]
                    logscale+=norm.log().sum(0)
            root_joint=up[self.root_i]*pi
            root_prob=root_joint.sum(-1)
            catll=logscale+root_prob.log()+weights.log()[:,None]
            site_ll=torch.logsumexp(catll,0)
            total+=site_ll.sum()
            if not need_counts: continue
            posterior=(catll-site_ll[None,:]).exp()
            category_counts+=posterior.sum(1)
            root_counts+=(posterior[...,None]*root_joint/root_prob.clamp_min(1e-300)[...,None]).sum((0,1))
            down=torch.ones_like(up); down[self.root_i]=pi
            for level in self.nonroot:
                if backend=="triton":
                    downward(up,down,trans,self.children,self.parents,level,posterior,counts)
                    continue
                p=self.parents[level]
                # Explicit sibling product is necessary for the zero-rate class:
                # dividing the parent message by this child's message gives 0/0.
                ci=self.children[p]
                messages=torch.einsum("nhkij,nhksj->nhksi",trans[ci],up[ci])
                messages=torch.where((ci==level[:,None])[...,None,None,None],torch.ones_like(messages),messages)
                a=down[p]*messages.prod(1)
                a=a/a.sum(-1,keepdim=True).clamp_min(1e-300)
                subtree=up[level]
                msg=torch.einsum("nkij,nksj->nksi",trans[level],subtree)
                denominator=(a*msg).sum(-1).clamp_min(1e-300)
                # Sum sites without allocating [edge,category,site,4,4].
                endpoint=torch.einsum("nksi,nksj->nkij",a*(posterior[None,:,:]/denominator)[...,None],subtree)
                counts[level]+=endpoint*trans[level]
                down[level]=torch.einsum("nksi,nkij->nksj",a,trans[level])
        return total,counts[self.edge_nodes],root_counts,category_counts


def fit_reversible(tree,names,states,*,model="GTR+G4",device="cuda:0",max_iter=128,
                   block_sites=0,tolerance=1e-6,initial=.05,backend="auto",
                   frequencies=None,exchangeabilities=None,alpha=.5,pinv=.1,fixed_model=False,
                   warm_start=True,precondition=True):
    """Joint ML on a fixed topology. +F empirical (default); +FO optimized.

    Optional supplied parameters initialize the fit; fixed_model=True holds ALL
    model parameters fixed, but still optimizes branches. Zero lengths are bounded
    at 1e-6, lengths <=10; exchangeabilities relative GT in [1e-4,1e4]; alpha
    [.02,100]; invariant proportion [~1e-7,.95]. No automatic model selection.
    """
    if max_iter<1 or tolerance<=0 or not np.isfinite(tolerance) or not 0<initial<=10:
        raise ValueError("invalid optimization controls")
    tree.require_binary()
    started=time.perf_counter(); spec=parse_model(model)
    k=spec.gamma_categories+int(spec.invariant)
    plan=PruningPlan(tree,names,states,device,k,block_sites)
    parameters=ModelParameters(spec,states,frequencies=frequencies,exchangeabilities=exchangeabilities,
                               alpha=alpha,pinv=pinv,fixed=fixed_model)
    chosen=("triton" if plan.device.type=="cuda" else "torch") if backend=="auto" else backend
    if chosen not in {"torch","triton"} or (chosen=="triton" and plan.device.type!="cuda"):
        raise ValueError("backend must be auto, torch, or triton (CUDA only)")
    e=len(plan.edges); sites=plan.sequence.shape[1]
    initialization={"method":"provided/constant lengths","seconds":0.}
    if warm_start and not any(tree.lengths.get(ed) is not None for ed in plan.edges):
        # The cheap incumbent is a numerical initializer, not a fixed model or
        # substitution for the subsequent general likelihood/parameter fit.
        from concordtree.branch_lengths import fit_jc69
        tree,seed_metrics=fit_jc69(tree,names,states,device=device,max_iter=32,
                                  block_sites=block_sites,initial=initial,backend=chosen)
        initialization={"method":"JC69 endpoint EM (32 cap)","seconds":seed_metrics["seconds"],
                        "converged":seed_metrics["converged"],"log_likelihood":seed_metrics["log_likelihood"]}
    length0=[np.log(max(1e-6,min(10.,tree.lengths.get(ed) if tree.lengths.get(ed) is not None else initial))) for ed in plan.edges]
    x0=np.array(length0+parameters.initial)
    bounds=[(np.log(1e-6),np.log(10.))]*e+parameters.bounds
    # L-BFGS-B projects its initial point. Evaluate that same projected point,
    # especially when a nucleotide is absent and initial ML logits are extreme.
    x0=np.clip(x0,np.array(bounds)[:,0],np.array(bounds)[:,1])
    # Mutation-count diagonal conditioning: short-branch log lengths have much
    # smaller curvature, while global model parameters accumulate all branches.
    # This changes only optimizer coordinates, not the likelihood or constraints.
    origin=x0.copy(); scale=np.ones_like(x0)
    if precondition:
        scale[:e]=1/np.sqrt(np.maximum(np.exp(origin[:e]),1e-3))
        scale[e:]=1/math.sqrt(max(1.,float(np.exp(origin[:e]).sum())))
    bounds=[((lo-origin[j])/scale[j],(hi-origin[j])/scale[j]) for j,(lo,hi) in enumerate(bounds)]
    x0=np.zeros_like(x0)
    cache={}; evaluations=0; history=[]

    def evaluate(vector):
        nonlocal evaluations
        # Rebuild only the small matrix tape; pruning runs entirely without tape.
        x=torch.tensor(origin+scale*vector,device=plan.device,dtype=torch.float64,requires_grad=True)
        pi,rates,q,site_rates,weights,_,_=parameters.decode(x[e:])
        p=transition_matrices(q,x[:e].exp(),site_rates)
        ll,counts,root_counts,category_counts=plan.expectation(p.detach(),pi.detach(),weights.detach(),backend=chosen)
        surrogate=(counts*p.clamp_min(1e-300).log()).sum()+(root_counts*pi.log()).sum()+(category_counts*weights.log()).sum()
        gradient=torch.autograd.grad(-surrogate/sites,x)[0]
        value=-float(ll.cpu())/sites
        grad=gradient.detach().cpu().numpy()*scale
        if not np.isfinite(value) or not np.all(np.isfinite(grad)):
            raise RuntimeError("nonfinite reversible-model likelihood or score")
        cache.update(x=np.array(vector,copy=True),value=value,gradient=grad)
        evaluations+=1
        return value,grad

    first,_=evaluate(x0); history.append(-first*sites)
    def objective(vector):
        if np.array_equal(vector,cache.get("x")): return cache["value"],cache["gradient"]
        return evaluate(vector)
    def accepted(vector):
        value,_=objective(vector); ll=-value*sites
        if ll<history[-1]-max(1e-6,abs(ll)*1e-10): raise RuntimeError("accepted ML likelihood decreased")
        history.append(ll)
    result=minimize(objective,x0,jac=True,method="L-BFGS-B",bounds=bounds,callback=accepted,
                    options={"maxiter":max_iter,"maxls":30,"maxcor":10,"ftol":1e-12,"gtol":tolerance})
    final_value,gradient=objective(result.x)
    fitted_parameters=origin+scale*result.x
    final_ll=-final_value*sites
    if final_ll<history[-1]-max(1e-6,abs(final_ll)*1e-10): raise RuntimeError("final ML likelihood decreased")
    if final_ll!=history[-1]: history.append(final_ll)
    out=tree.copy()
    for ed,length in zip(plan.edges,np.exp(fitted_parameters[:e])): out.lengths[ed]=float(length)
    with torch.no_grad():
        x=torch.tensor(fitted_parameters[e:],device=plan.device,dtype=torch.float64)
        pi,rates,q,site_rates,weights,alpha_fit,pinv_fit=parameters.decode(x)
    def host(t): return t.detach().cpu().tolist()
    projected=gradient.copy()/scale
    for j,(lo,hi) in enumerate(bounds):
        if (result.x[j]<=lo+1e-10 and projected[j]>0) or (result.x[j]>=hi-1e-10 and projected[j]<0): projected[j]=0
    return out,{"model":spec.name,"method":"scaled category pruning / analytic endpoint score / L-BFGS-B",
        "device":str(plan.device),"backend":chosen,"precision":"float64","units":"expected substitutions per site",
        "taxa":len(names),"sites":sites,"max_iter":max_iter,"iterations":int(result.nit),
        "converged":bool(result.success),"optimizer_status":int(result.status),"optimizer_message":str(result.message),
        "likelihood_evaluations":evaluations,"log_likelihood":final_ll,"log_likelihood_history":history,
        "projected_score_inf_per_site":float(np.abs(projected).max()),
        "projected_conditioned_score_inf_per_site":float(np.abs(projected*scale).max()),
        "conditioned_score_tolerance_per_site":tolerance,
        "block_sites":plan.block_sites,"requested_block_sites":block_sites,"frequency_mode":spec.frequencies,
        "frequencies_acgt":host(pi),"exchangeabilities_ac_ag_at_cg_ct_gt":host(rates),"Q_acgt":host(q),
        "gamma_alpha":float(alpha_fit.cpu()) if spec.gamma_categories>1 else None,
        "invariant_proportion":float(pinv_fit.cpu()),"category_rates":host(site_rates),"category_weights":host(weights),
        "model_parameters_fixed":fixed_model,"seconds":time.perf_counter()-started,
        "initialization":initialization,"optimizer_preconditioned":precondition,
        "parameter_bounds":{"lengths":[1e-6,10.],"exchangeabilities_relative_gt":[1e-4,1e4],"alpha":[.02,100.],"pinv":[1.12535162055095e-7,.95]},
        "warning":"Local fixed-topology stationary reversible DNA ML; no global optimum, model selection, protein/codon, partition, clock or dating guarantee."}
