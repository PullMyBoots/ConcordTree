# Isolated no-ordinary-iteration-limit copy; original fitter/release untouched.
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
from scipy.optimize import minimize, OptimizeResult
from concordtree.treeio import rooted, edge
from concordtree.dna_models import parse_model, ModelParameters, transition_matrices


from resident import ResidentPlan
from search_progress import ExactRecurrence, MAX_PARAMETER_CONTINUATIONS


def fit_reversible_reference_optimizer(tree,names,states,*,model="GTR+G4",device="cuda:0",max_iter=128,
                   block_sites=0,tolerance=1e-6,initial=.05,backend="auto",
                   frequencies=None,exchangeabilities=None,alpha=.5,pinv=.1,fixed_model=False,
                   warm_start=True,precondition=True,precision="float64",graph=True,
                   transition_backend="matrix_exp",length_coordinate="log",seed_method=None,
                   plan_backend="resident",model_seed="none",coordinate_metric="incumbent",
                   stopping_policy="optimizer",model_backend="device",score_backend="autograd",optimizer_backend="lbfgsb",coupled_linear_solver="woodbury"):
    """Joint ML on a fixed topology. +F empirical (default); +FO optimized.

    Optional supplied parameters initialize the fit; fixed_model=True holds ALL
    model parameters fixed, but still optimizes branches. Zero lengths are bounded
    at 1e-6, lengths <=10; exchangeabilities relative GT in [1e-4,1e4]; alpha
    [.02,100]; invariant proportion [~1e-7,.95]. No automatic model selection.
    """
    if max_iter<1 or tolerance<=0 or not np.isfinite(tolerance) or not 0<initial<=10:
        raise ValueError("invalid optimization controls")
    if transition_backend not in {"matrix_exp","spectral","spectral-cuda"}:
        raise ValueError("invalid transition_backend")
    if length_coordinate not in {"log","sqrt"}:
        raise ValueError("length_coordinate must be log or sqrt")
    if seed_method not in {None,"parsimony"}:
        raise ValueError("seed_method must be None or parsimony")
    if plan_backend not in {"resident","compact","fused","repeats","repeat-native","repeat-generic","repeat-segmented","repeat-flat","repeat-category","repeat-category-bounded","repeat-radix","repeat-matrix","repeat-matrix-factorized","repeat-matrix-concurrent"}:
        raise ValueError("invalid plan_backend")
    if model_seed not in {"none","parsimony-rates"}:
        raise ValueError("model_seed must be none or parsimony-rates")
    if coordinate_metric not in {"incumbent","complete-fisher"}:
        raise ValueError("invalid coordinate_metric")
    if coordinate_metric=="complete-fisher" and not precondition:
        raise ValueError("complete-fisher requires preconditioning")
    if stopping_policy not in {"optimizer","physical"}:
        raise ValueError("invalid stopping_policy")
    if model_backend not in {"device","cpu"}:
        raise ValueError("model_backend must be device or cpu")
    if score_backend not in {"autograd","analytic-cuda"}:
        raise ValueError("invalid score_backend")
    if score_backend=="analytic-cuda" and (model_backend!="device" or transition_backend!="spectral-cuda" or torch.device(device).type!="cuda"):
        raise ValueError("analytic-cuda requires CUDA, spectral-cuda transitions and model_backend=device")
    if optimizer_backend not in {"lbfgsb","coupled-gram"}:
        raise ValueError("invalid optimizer_backend")
    if coupled_linear_solver not in {"woodbury","bordered","native"} or (coupled_linear_solver!="woodbury" and optimizer_backend!="coupled-gram"):
        raise ValueError("bordered solver requires coupled-gram optimizer")
    if optimizer_backend=="coupled-gram" and (score_backend!="analytic-cuda" or stopping_policy!="physical" or coordinate_metric!="incumbent"):
        raise ValueError("coupled-gram requires analytic-cuda, physical stopping and incumbent affine chart")
    tree.require_binary()
    started=time.perf_counter(); spec=parse_model(model)
    if model_seed!="none" and (seed_method!="parsimony" or spec.base!="GTR" or fixed_model or exchangeabilities is not None):
        raise ValueError("parsimony model seed requires parsimony branches, free GTR, and no supplied rates")
    k=spec.gamma_categories+int(spec.invariant)
    plan_class=ResidentPlan
    plan_options={}
    if plan_backend in {"compact","fused"}:
        from compact import CompactPlan
        plan_class=CompactPlan
        plan_options["contraction"]="fused" if plan_backend=="fused" else "reference"
    elif plan_backend in {"repeats","repeat-native","repeat-generic","repeat-segmented","repeat-flat","repeat-category","repeat-category-bounded","repeat-radix","repeat-matrix","repeat-matrix-factorized","repeat-matrix-concurrent"}:
        from frontier_repeats import RepeatPlan
        plan_class=RepeatPlan
        plan_options["builder"]="radix" if plan_backend in {"repeat-radix","repeat-matrix","repeat-matrix-factorized","repeat-matrix-concurrent"} else "flat" if plan_backend in {"repeat-flat","repeat-category","repeat-category-bounded"} else ("numpy" if plan_backend=="repeats" else "native")
        plan_options["kernel_schedule"]="matrix-factorized" if plan_backend=="repeat-matrix-factorized" else "matrix" if plan_backend in {"repeat-matrix","repeat-matrix-concurrent"} else "category-bounded" if plan_backend=="repeat-category-bounded" else "category" if plan_backend=="repeat-category" else "segmented" if plan_backend=="repeat-segmented" else ("generic" if plan_backend in {"repeat-generic","repeat-flat","repeat-radix","repeat-matrix","repeat-matrix-factorized","repeat-matrix-concurrent"} else "specialized")
    if plan_backend=="repeat-matrix-concurrent":plan_options["down_schedule"]="depth-concurrent"
    plan=plan_class(tree,names,states,device,k,block_sites,precision=precision,graph=graph,**plan_options)
    parameters=ModelParameters(spec,states,frequencies=frequencies,exchangeabilities=exchangeabilities,
                               alpha=alpha,pinv=pinv,fixed=fixed_model)
    chosen=("triton" if plan.device.type=="cuda" else "torch") if backend=="auto" else backend
    if chosen not in {"torch","triton"} or (chosen=="triton" and plan.device.type!="cuda"):
        raise ValueError("backend must be auto, torch, or triton (CUDA only)")
    e=len(plan.edges); sites=plan.sequence.shape[1]
    initialization={"method":"provided/constant lengths","seconds":0.}
    absent_lengths=not any(tree.lengths.get(ed) is not None for ed in plan.edges)
    if model_seed!="none" and not absent_lengths:
        raise ValueError("parsimony model seed requires an uninitialized tree")
    if seed_method=="parsimony" and absent_lengths:
        from parsimony_seed import seed_parsimony
        tree,seed_metrics=seed_parsimony(tree,names,states,device=device,block_sites=block_sites,
            collect_model_statistics=model_seed=="parsimony-rates")
        initialization=seed_metrics
        if model_seed=="parsimony-rates":
            a,b=np.triu_indices(4,1)
            raw=(np.array(seed_metrics["substitutions_ac_ag_at_cg_ct_gt"])+.5)/(parameters.pi[a]*parameters.pi[b])
            seed_rates=np.clip(raw/raw[-1],1e-4,1e4)
            parameters=ModelParameters(spec,states,frequencies=parameters.pi,exchangeabilities=seed_rates,
                alpha=alpha,pinv=pinv,fixed=False)
            initialization["model_seed"]={"method":"short-edge stationary replacement-count ratio; initializer only",
                "exchangeabilities_ac_ag_at_cg_ct_gt":seed_rates.tolist(),"alpha_unchanged":alpha}
    elif warm_start and absent_lengths:
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
    log_bounds=np.array(bounds,copy=True)
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
    if length_coordinate=="sqrt":
        # Rare-substitution Fisher-Rao coordinate: s=2sqrt(t) has constant
        # Poisson(L*t) information L. Unlike static log preconditioning, this
        # metric adapts along the entire trajectory. Same likelihood/bounds.
        origin[:e]=2*np.exp(origin[:e]/2)
        scale[:e]=1.
        bounds=[(2*np.exp(lo/2),2*np.exp(hi/2)) if j<e else (lo,hi)
                for j,(lo,hi) in enumerate(bounds)]
    metric_setup={"method":"incumbent static heuristic","seconds":0.}
    if coordinate_metric=="complete-fisher":
        from fisher_diagonal import complete_fisher_diagonal
        _,scale,metric_setup=complete_fisher_diagonal(parameters,x0[e:],np.exp(x0[:e]),
            device=plan.device,coordinate=length_coordinate)
    bounds=[((lo-origin[j])/scale[j],(hi-origin[j])/scale[j]) for j,(lo,hi) in enumerate(bounds)]
    x0=np.zeros_like(x0)
    cache={}; evaluations=0; history=[]

    def evaluate(vector):
        nonlocal evaluations
        if score_backend=="analytic-cuda":
            if optimizer_backend=="coupled-gram":
                from coupled_optimizer import evaluate_with_state
                value,gradient,state=evaluate_with_state(plan,parameters,origin+scale*vector,coordinate=length_coordinate,backend=chosen)
                cache['coupled_state']=state
            else:
                from analytic_score import evaluate as analytic_evaluate
                value,gradient=analytic_evaluate(plan,parameters,origin+scale*vector,coordinate=length_coordinate,backend=chosen)
            grad=gradient*scale
            if not np.isfinite(value) or not np.all(np.isfinite(grad)):
                raise RuntimeError("nonfinite reversible-model likelihood or score")
            cache.update(x=np.array(vector,copy=True),value=value,gradient=grad)
            evaluations+=1
            return value,grad
        # Rebuild only the small matrix tape; pruning runs entirely without tape.
        values=origin+scale*vector
        raw=torch.tensor(values[:e] if model_backend=="cpu" else values,device=plan.device,dtype=torch.float64,requires_grad=True)
        x=(torch.cat((2*(raw[:e]/2).log(),raw[e:])) if length_coordinate=="sqrt" else raw)
        if model_backend=="cpu":
            from host_model import decode_host_model
            model_raw=torch.tensor(values[e:],dtype=torch.float64,requires_grad=True)
            pi,rates,q,site_rates,weights,_,_=decode_host_model(parameters,model_raw,plan.device)
        else:
            pi,rates,q,site_rates,weights,_,_=parameters.decode(x[e:])
        if transition_backend=="spectral-cuda":
            from spectral_cuda import spectral_cuda_transitions
            p=spectral_cuda_transitions(q,x[:e].exp(),site_rates,pi)
        elif transition_backend=="spectral":
            from spectral import spectral_transitions
            p=spectral_transitions(q,x[:e].exp(),site_rates,pi)
        else:
            p=transition_matrices(q,x[:e].exp(),site_rates)
        ll,counts,root_counts,category_counts=plan.expectation(p.detach(),pi.detach(),weights.detach(),backend=chosen)
        surrogate=(counts*p.clamp_min(1e-300).log()).sum()+(root_counts*pi.log()).sum()+(category_counts*weights.log()).sum()
        if model_backend=="cpu":
            branch_grad,model_grad=torch.autograd.grad(-surrogate/sites,(raw,model_raw),allow_unused=True)
        else:
            gradient=torch.autograd.grad(-surrogate/sites,raw)[0]
        value=-float(ll.cpu())/sites
        if model_backend=="cpu":
            # Fixed/zero-dimensional model vectors legitimately have no tape.
            grad=np.r_[branch_grad.detach().cpu().numpy(),
                model_grad.detach().numpy() if model_grad is not None else np.zeros(len(model_raw))]*scale
        else:
            grad=gradient.detach().cpu().numpy()*scale
        if not np.isfinite(value) or not np.all(np.isfinite(grad)):
            raise RuntimeError("nonfinite reversible-model likelihood or score")
        cache.update(x=np.array(vector,copy=True),value=value,gradient=grad)
        evaluations+=1
        return value,grad

    first,first_gradient=evaluate(x0); history.append(-first*sites)
    from physical_stop import projected_log_score, physical_convergence, INTERNAL_TARGET, INDEPENDENT_TARGET
    def physical_score(vector,grad):
        return float(np.abs(projected_log_score(vector,grad,origin,scale,e,length_coordinate,log_bounds)).max())
    physical_history=[]; callback_stopped=False; initial_stopped=False
    if stopping_policy=="physical":physical_history.append(physical_score(x0,first_gradient))
    def objective(vector):
        if np.array_equal(vector,cache.get("x")): return cache["value"],cache["gradient"]
        return evaluate(vector)
    def accepted(vector):
        nonlocal callback_stopped
        value,grad=objective(vector); ll=-value*sites
        if ll<history[-1]-max(1e-6,abs(ll)*1e-10): raise RuntimeError("accepted ML likelihood decreased")
        history.append(ll)
        if stopping_policy=="physical":
            score=physical_score(vector,grad); physical_history.append(score)
            if score<=INTERNAL_TARGET:
                callback_stopped=True
                raise StopIteration
    if stopping_policy=="physical" and physical_history[0]<=INTERNAL_TARGET:
        initial_stopped=True
        result=OptimizeResult(x=x0,nit=0,success=True,status=0,message="physical initial score target met; optimizer not called")
    else:
        if optimizer_backend=="coupled-gram":
            from coupled_optimizer import minimize_coupled,metric_from_state
            def current_metric(vector):
                assert np.array_equal(vector,cache['x']),"metric must use current all-site counts"
                state=cache['coupled_state']
                assert np.array_equal(state['values'],origin+scale*vector)
                return metric_from_state(parameters,state,sites,coordinate=length_coordinate,scale=scale,device=plan.device)
            result=minimize_coupled(objective,x0,bounds=bounds,callback=accepted,metric_callback=current_metric,
                options={"maxiter":max_iter,"maxls":30,"maxcor":10,"ftol":0.,"gtol":0.},linear_solver=coupled_linear_solver)
        else:
            result=minimize(objective,x0,jac=True,method="L-BFGS-B",bounds=bounds,callback=accepted,
                            options={"maxiter":max_iter,"maxls":30,"maxcor":10,
                                "ftol":0. if stopping_policy=="physical" else 1e-12,
                                "gtol":0. if stopping_policy=="physical" else tolerance,
                                "maxfun":np.iinfo(np.int32).max})
    # A local solve can converge within a wrong discrete-rate scale basin.
    # Profile only scales implied by the model, then continue the same optimizer.
    # Likelihood-only upward passes reuse this plan; no extra ancestral counts.
    profile_history=[];total_iterations=int(result.nit);continuation_stop=None
    if spec.gamma_categories>1 and score_backend=="analytic-cuda":
        from rate_mode_profile import profile_modes
        from analytic_score import likelihood_only
        recurrence=ExactRecurrence()
        while True:
            if recurrence.repeated(np.asarray(result.x).tobytes()):
                continuation_stop='repeated_parameter_state';break
            current_value,_=objective(result.x)
            profile_start=time.perf_counter()
            candidates=profile_modes(result.x,origin=origin,scale=scale,edges=e,
                coordinate=length_coordinate,parameters=parameters,
                objective=lambda point:likelihood_only(plan,parameters,origin+scale*point,
                    coordinate=length_coordinate))
            winner=min(candidates,key=lambda row:row['value'])
            roundoff=64*np.finfo(float).eps*max(1.,abs(current_value))
            improved=bool(winner['value']<current_value-roundoff)
            profile_history.append(dict(seconds=time.perf_counter()-profile_start,
                incumbent_loss=current_value,selected_ratio=winner['ratio'],accepted=improved,
                trials=[dict(ratio=r['ratio'],loss=r['value']) for r in candidates]))
            if not improved:break
            if len(profile_history)>=MAX_PARAMETER_CONTINUATIONS:
                continuation_stop='parameter_continuation_guard';break
            # Synchronize the full derivative cache with the chosen observed
            # likelihood state before continuing local shape/model optimization.
            value,grad=evaluate(winner['point'])
            history.append(-value*sites)
            callback_stopped=False;initial_stopped=False
            result=minimize(objective,winner['point'],jac=True,method="L-BFGS-B",bounds=bounds,
                callback=accepted,options={"maxiter":max_iter,"maxls":30,"maxcor":10,
                    "ftol":0. if stopping_policy=="physical" else 1e-12,
                    "gtol":0. if stopping_policy=="physical" else tolerance,
                    "maxfun":np.iinfo(np.int32).max})
            total_iterations+=int(result.nit)
    final_value,gradient=objective(result.x)
    final_physical_score=physical_score(result.x,gradient)
    if stopping_policy=="physical":
        physical_history.append(final_physical_score)
        converged,convergence_reason=physical_convergence(result,final_physical_score,callback_stopped,initial_stopped)
    else:
        converged=bool(result.success);convergence_reason="optimizer-status"
    if continuation_stop is not None:
        converged=False;convergence_reason=continuation_stop
    fitted_parameters=origin+scale*result.x
    if length_coordinate=="sqrt":
        fitted_parameters[:e]=2*np.log(fitted_parameters[:e]/2)
    final_ll=-final_value*sites
    if final_ll<history[-1]-max(1e-6,abs(final_ll)*1e-10): raise RuntimeError("final ML likelihood decreased")
    if final_ll!=history[-1]: history.append(final_ll)
    out=tree.copy()
    for ed,length in zip(plan.edges,np.exp(fitted_parameters[:e])): out.lengths[ed]=float(length)
    with torch.no_grad():
        x=torch.tensor(fitted_parameters[e:],device="cpu" if model_backend=="cpu" else plan.device,dtype=torch.float64)
        pi,rates,q,site_rates,weights,alpha_fit,pinv_fit=parameters.decode(x)
    def host(t): return t.detach().cpu().tolist()
    projected=gradient.copy()/scale
    for j,(lo,hi) in enumerate(bounds):
        if (result.x[j]<=lo+1e-10 and projected[j]>0) or (result.x[j]>=hi-1e-10 and projected[j]<0): projected[j]=0
    return out,{"model":spec.name,"method":"scaled category pruning / analytic endpoint score / coupled-Gram box-QP BFGS" if optimizer_backend=="coupled-gram" else "scaled category pruning / analytic endpoint score / L-BFGS-B",
        "device":str(plan.device),"backend":chosen,"plan_backend":plan_backend,"precision":precision,"transition_backend":transition_backend,"cuda_graph":graph,"resident_setup":plan.setup_metrics,"units":"expected substitutions per site",
        "taxa":len(names),"sites":sites,"max_iter":max_iter,"iterations":total_iterations,"rate_mode_profiles":profile_history,
        "converged":converged,"optimizer_success":bool(result.success),"optimizer_status":int(result.status),"optimizer_message":str(result.message),
        "stopping_policy":stopping_policy,"convergence_reason":convergence_reason,
        "continuation_stop":continuation_stop,"maximum_parameter_continuations":MAX_PARAMETER_CONTINUATIONS,
        "physical_projected_log_score_inf_per_site":final_physical_score,
        "physical_score_history_per_site":physical_history,
        "physical_internal_target_per_site":INTERNAL_TARGET if stopping_policy=="physical" else None,
        "physical_independent_target_per_site":INDEPENDENT_TARGET if stopping_policy=="physical" else None,
        "physical_callback_stopped":callback_stopped,"physical_initial_stopped":initial_stopped,
        "likelihood_evaluations":evaluations,"log_likelihood":final_ll,"log_likelihood_history":history,
        "projected_score_inf_per_site":float(np.abs(projected).max()),
        "projected_conditioned_score_inf_per_site":float(np.abs(projected*scale).max()),
        "conditioned_score_tolerance_per_site":tolerance,
        "block_sites":plan.block_sites,"requested_block_sites":block_sites,"frequency_mode":spec.frequencies,
        "frequencies_acgt":host(pi),"exchangeabilities_ac_ag_at_cg_ct_gt":host(rates),"Q_acgt":host(q),
        "gamma_alpha":float(alpha_fit.cpu()) if spec.gamma_categories>1 else None,
        "invariant_proportion":float(pinv_fit.cpu()),"category_rates":host(site_rates),"category_weights":host(weights),
        "model_parameters_fixed":fixed_model,"seconds":time.perf_counter()-started,
        "initialization":initialization,"model_seed":model_seed,"optimizer_preconditioned":precondition,"length_coordinate":length_coordinate,
        "coordinate_metric":coordinate_metric,"coordinate_metric_setup":metric_setup,
        "model_backend":"numpy-host" if score_backend=="analytic-cuda" else model_backend,
        "requested_model_backend":model_backend,"score_backend":score_backend,
        "parameter_bounds":{"lengths":[1e-6,10.],"exchangeabilities_relative_gt":[1e-4,1e4],"alpha":[.02,100.],"pinv":[1.12535162055095e-7,.95]},
        "warning":"Local fixed-topology stationary reversible DNA ML; no global optimum, model selection, protein/codon, partition, clock or dating guarantee.",
        **({"optimizer_backend":optimizer_backend,"coupled_optimizer":getattr(result,"coupled",{"method":"initial physical target;optimizer not called"})} if optimizer_backend=="coupled-gram" else {})}


def fit_hybrid(tree,names,states,*,model="GTR+G4",max_iter=128,**kwargs):
    """FP32 prefit then fresh FP64 optimization of the SAME exact objective.

    FP32 optimizer success is not final convergence. Every phase and its full
    allocation/capture/initialization cost is retained in total seconds.
    """
    started=time.perf_counter()
    seed,first=fit_reversible_reference_optimizer(tree,names,states,model=model,
                max_iter=max_iter,precision="float32",transition_backend="spectral",**kwargs)
    result,final=fit_reversible_reference_optimizer(seed,names,states,model=model,
                max_iter=max_iter,precision="float64",transition_backend="spectral",
                frequencies=first["frequencies_acgt"],exchangeabilities=first["exchangeabilities_ac_ag_at_cg_ct_gt"],
                alpha=first["gamma_alpha"] or .5,pinv=max(first["invariant_proportion"],1e-6),**kwargs)
    final["phase_seconds"]={"float32_prefit":first["seconds"],"float64_refinement":final["seconds"]}
    final["prefit_metrics"]=first
    final["seconds"]=time.perf_counter()-started
    final["method"]="resident spectral FP32 prefit + exact FP64 joint refinement"
    return result,final
