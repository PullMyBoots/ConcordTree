"""Budgeted block-coordinate DNA likelihood inference, no neural scoring.

Supplied initializer -> joint model/length update -> full-site NNI -> acceptance.
The unified engine supplies the current full-site bounded AP-NJ + MLP initializer.
One homogeneous sweep is the unit of user budget. A finite valid result is
different from a strict optimization certificate. This is a local search,
not a global ML algorithm or an MSC-consistent species-tree estimator.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from concordtree.dna_models import parse_model
from concordtree.treeio import newick
from invariant_site_patterns import InvariantPatterns
from stable_patterns import StablePatterns
from frontier_fitter import fit_reversible_reference_optimizer
from nni_moves import apply_proposals,independent_subset
from global_compression import compress_globally
from pruned_weighted_likelihood_eager import PrunedWeightedLikelihood
from flat_uniformized_likelihood import FlatInvariantLikelihood
from shared_mixture import SharedInvariantReplay as InvariantReplay, SharedVariableReplay as VariableReplay
from local_profile import CurvatureInvariant,CurvatureVariable
from profile_graph import GraphInvariant,GraphVariable
from exact_topology_key import topology_key
from message_workspace import MessageWorkspace
from search_progress import NumericalPlateau, EMERGENCY_SWEEPS


def write(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def missing_counts(packed,sites):
    byte=np.arange(256,dtype=np.uint8)
    lut=((byte&15)==4).astype(np.uint8)+((byte>>4)==4)
    counts=np.zeros(packed.shape[0],dtype=np.int64)
    width=max(1,(64<<20)//packed.shape[0])
    for pos in range(0,packed.shape[1],width):
        counts+=lut[packed[:,pos:pos+width]].sum(axis=1,dtype=np.int64)
    if sites%2:counts-=(packed[:,-1]>>4)==4
    return counts


def fit(tree,names,sample,model,previous=None,*,max_iter=32):
    """Observed-likelihood fit; profile invariant mass when present in the model."""
    spec=parse_model(model)
    if spec.invariant:
        from profile_invariant_fitter import fit_profiled
        return fit_profiled(tree,names,sample,model,previous,max_iter=max_iter)
    warm={} if previous is None else dict(frequencies=previous['frequencies_acgt'],
        exchangeabilities=previous['exchangeabilities_ac_ag_at_cg_ct_gt'],
        alpha=previous['gamma_alpha'] or .5,pinv=max(previous['invariant_proportion'],1e-7))
    return fit_reversible_reference_optimizer(tree,names,sample,model=model,device='cuda:0',
        max_iter=max_iter,precision='binary64',length_coordinate='sqrt',
        seed_method='parsimony' if previous is None else None,
        model_seed='parsimony-rates' if previous is None and spec.base=='GTR' else 'none',
        block_sites=0,plan_backend='repeat-matrix-factorized',transition_backend='spectral-cuda',
        coordinate_metric='incumbent',stopping_policy='physical',model_backend='device',
        score_backend='analytic-cuda',optimizer_backend='lbfgsb',graph=True,**warm)


def infer(msa,out,*,model='GTR+FO+I+G4',sample_fraction=None,views=1,
          max_rounds=None,stop_ratio=.005,missing_data_model='standard',progress=None,initialize_fn=None,support_sink=None,
          initializer_uses_all_sites=False,optimizer_iterations=32,min_gain=.001,checkpoint_fn=None):
    if initialize_fn is None:raise ValueError('use UnifiedEngine to supply the release AP-NJ + MLP initializer')
    if views<1 or (max_rounds is not None and max_rounds<1):raise ValueError('positive views/rounds required')
    if sample_fraction is not None and not 0<sample_fraction<=1:raise ValueError('sample fraction in (0,1]')
    if not 0<=stop_ratio<1:raise ValueError('stopping ratio in [0,1)')
    if missing_data_model not in ('standard','coverage-aware'):raise ValueError('unknown missingness model')
    spec=parse_model(model)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter();phases={};warnings=[];parameter_interventions=[]
    def emit(**value):
        value['elapsed_seconds']=time.perf_counter()-started
        write(out/'progress.json',value)
        if progress:progress(value)
    alignment=StablePatterns(msa);names=alignment.names
    missing=missing_counts(alignment.packed,alignment.original_sites)
    if np.any(missing==alignment.original_sites):raise ValueError('entirely unobserved taxon')
    cap=min(alignment.original_sites,16384) if sample_fraction is None else max(1,math.ceil(alignment.original_sites*sample_fraction))
    if initializer_uses_all_sites:
        # The full-site initializer reads existing packed observations. Do not extract
        # an unused dense initialization sample or repeat it for later views.
        sample,indices=None,None
    else:sample,indices=alignment.parameter_sample(cap=cap,seed=20261022)
    # Initial sampling controls AP-NJ alone. Continuous parameters and topology
    # use the same complete observed alignment and one likelihood objective.
    if cap==alignment.original_sites and not initializer_uses_all_sites:fit_sample,fit_indices=sample,indices
    else:fit_sample,fit_indices=alignment.parameter_sample(cap=alignment.original_sites,seed=20261022)
    if indices is not None:np.save(out/'initialization-sites.npy',indices)
    np.save(out/'parameter-sites.npy',fit_indices)
    diagnostic=dict(missing_fraction=float(missing.sum())/(len(names)*alignment.original_sites),
        missing_per_taxon=missing.tolist(),likelihood='unknown-state marginalization',
        initializer='full-site injected initializer' if initializer_uses_all_sites else missing_data_model,
        no_reference_or_dataset_route=True)
    # Bounded packed-mask CSI measurement, never an automatic route or a truth signal.
    ti=np.linspace(0,len(names)-1,min(128,len(names)),dtype=np.int64)
    si=np.linspace(0,alignment.original_sites-1,min(65536,alignment.original_sites),dtype=np.int64)
    raw=alignment.packed[ti[:,None],si[None,:]//2]
    masks=((raw>>((si[None,:]%2)*4))&15)<4
    overlap=[];jaccard=[]
    for i in range(len(ti)):
        for j in range(i):
            shared=int(np.count_nonzero(masks[i]&masks[j]));union=int(np.count_nonzero(masks[i]|masks[j]))
            overlap.append(shared/len(si));jaccard.append(shared/union if union else 1.)
    mo=float(np.median(overlap));mj=float(np.median(jaccard))
    diagnostic.update(median_pairwise_shared_coverage=mo,median_coverage_mask_jaccard=mj,
        coherent_sparsity_index=(1-mo)*mj,sampled_taxa=len(ti),sampled_sites=len(si),
        guidance='diagnostic only; explicit user choice; no validated universal final-tree threshold')
    write(out/'missing-diagnostic.json',diagnostic)
    phases['read_diagnose_sample']=time.perf_counter()-started
    t=time.perf_counter();compression=alignment.compress()
    write(out/'compression.json',compression);phases['compress_all_sites']=time.perf_counter()-t
    k=spec.gamma_categories
    # Engineering workspace is fixed independently of dataset identity.
    workspace=512<<20;arena=MessageWorkspace('cuda:0');patterns=max(x.shape[1] for x,_ in alignment.chunks)
    def refit(tree, previous=None):
        # End the previous conditional topology block before fitting. The same
        # allocator storage can now serve the repeat-pattern fit workspace.
        arena.clear()
        emit(phase='parameter_fit', sites=fit_sample.shape[1])
        result=fit(tree,names,fit_sample,model,previous,max_iter=optimizer_iterations)
        if result[1].get('continuation_stop'):
            parameter_interventions.append(dict(fit=len(parameter_interventions)+1,
                reason=result[1]['continuation_stop']))
        torch.cuda.empty_cache()
        return result
    state_version=0;scan_state=None;scan_evidence=None
    def score(tree,parameters,nni=False,propose=True):
        nonlocal scan_state,scan_evidence
        rows=(3*len(names)-6) if nni else (2*len(names)-3)
        bound=min(65536,patterns,workspace//max(1,rows*k*24))
        if bound<1:raise MemoryError('one message site exceeds workspace')
        width=1<<(int(bound).bit_length()-1)
        cls=(GraphInvariant if spec.invariant else GraphVariable) if nni else (InvariantReplay if spec.invariant else VariableReplay)
        plan=cls(tree,names,parameters,block_sites=width,precision='float32',workspace_bytes=workspace,nni=nni,message_workspace=arena)
        values=plan.score_patterns(alignment,nni=nni)
        if nni and values['max_baseline_error']>.01+1e-6*alignment.original_sites:
            raise ArithmeticError('local and full-tree likelihood disagree')
        if nni and support_sink is not None:
            # Retain only O(N) host aggregates; never keep site-level messages.
            scan_state=state_version
            scan_evidence=(plan.internal_edges,plan.branches,values['deltas'],values['profiled_baseline_gain'])
        proposals=plan.proposals(values,min_gain=min_gain) if nni and propose else []
        return values,proposals
    best=None;init_records=[]
    for view in range(views):
        t=time.perf_counter()
        if view and not initializer_uses_all_sites:
            # Same user fraction, independently seeded sample of original sites.
            view_sample,view_indices=alignment.parameter_sample(cap=cap,seed=20261022+view)
            np.save(out/f'initialization-sites-{view:02d}.npy',view_indices)
        else:view_sample=sample
        tree,info=initialize_fn(names,view_sample,20260934+view,missing_data_model,packed_data=(alignment.packed,names,alignment.original_sites))
        (out/f'initial-{view:02d}.nwk').write_text(newick(tree)+'\n')
        tree,parameters=refit(tree)
        write(out/f'fit-initial-{view:02d}.json',parameters)
        value,_=score(tree,parameters)
        init_records.append(dict(view=view,seconds=time.perf_counter()-t,ll=value['log_likelihood'],**info))
        if best is None or value['log_likelihood']>best[0]:best=(value['log_likelihood'],tree,parameters)
        emit(phase='initialization',view=view,log_likelihood=value['log_likelihood'])
    ll,tree,parameters=best
    fitted_topology=topology_key(tree)
    del sample,view_sample  # initialization-only dense buffers no longer live
    alignment.release_packed()
    arena.clear()
    global_compression=compress_globally(alignment)
    phases['global_pattern_equality']=global_compression['seconds']
    compression.update(global_equality=global_compression,
        patterns=global_compression['global_patterns'],
        host_pattern_bytes=global_compression['host_pattern_bytes'])
    write(out/'compression.json',compression)
    patterns=max(x.shape[1] for x,_ in alignment.chunks)
    write(out/'initialization.json',init_records)
    history=[];stop=None;round_index=0;pending_moves=0
    plateau=NumericalPlateau(.01+1e-6*alignment.original_sites)
    while True:
        t=time.perf_counter();before=ll
        # Alternate a topology block (to the user's stopping ratio) and a
        # bounded parameter block. Rebuilding/refitting after every NNI sweep
        # dominated the broad rc2 pilot's cost.
        update=None
        values,proposals=score(tree,parameters,nni=True)
        # Use the authoritative scan LL for same-topology comparisons.
        scan_ll=values['log_likelihood']
        if abs(scan_ll-ll)>.01+1e-6*alignment.original_sites:raise ArithmeticError('scan baseline drift')
        ll=scan_ll;chosen=independent_subset(tree,proposals);trials=[]
        while chosen:
            candidate=apply_proposals(tree,chosen);verified,_=score(candidate,parameters)
            trials.append(dict(moves=len(chosen),ll=verified['log_likelihood']))
            if verified['log_likelihood']>ll+min_gain:
                tree,ll=candidate,verified['log_likelihood'];state_version+=1;break
            chosen=chosen[:len(chosen)//2]
        ratio=len(chosen)/max(1,len(names)-3)
        pending_moves+=len(chosen)
        round_index+=1
        if max_rounds is not None and round_index>=max_rounds:stop='round-budget'
        elif ratio<=stop_ratio:
            if pending_moves:
                candidate,updated=refit(tree,parameters)
                write(out/f'fit-round-{round_index:03d}.json',updated)
                checked,_=score(candidate,updated)
                accepted=checked['log_likelihood']>=ll
                update=dict(ll=checked['log_likelihood'],accepted=accepted,converged=updated['converged'],
                    seconds=updated['seconds'],evaluations=updated['likelihood_evaluations'])
                if accepted:
                    tree,parameters,ll=candidate,updated,checked['log_likelihood']
                    state_version+=1
                    fitted_topology=topology_key(tree)
                else:stop='accepted-move-ratio;parameter-update-did-not-improve-full-data'
                pending_moves=0
                # Changed parameters change the conditional topology objective;
                # rescan before calling its move ratio stable.
            else:stop='accepted-move-ratio'
        if stop is None and plateau.observe(before,ll,lambda:topology_key(tree),
                lambda:(tree.copy(),parameters,fitted_topology)):
            tree,parameters,fitted_topology=plateau.best_payload
            ll=plateau.best_value;state_version+=1
            stop='repeated_topology_numerical_plateau'
            warnings.append('Repeated topologies with negligible cumulative likelihood improvement; retained best verified state, not certified search convergence')
        if stop is None and round_index>=EMERGENCY_SWEEPS:
            raise RuntimeError('gene refinement emergency sweep guard exhausted before convergence')
        record=dict(round=round_index,before_ll=before,after_ll=ll,moves=len(chosen),move_ratio=ratio,
            update=update,trials=trials,stop=stop,seconds=time.perf_counter()-t,
            max_baseline_error=values['max_baseline_error'])
        history.append(record);write(out/'history.json',history)
        (out/'tree.partial.nwk').write_text(newick(tree)+'\n')
        if checkpoint_fn:checkpoint_fn(tree,dict(stage='likelihood',round=round_index))
        emit(phase='refinement',**record)
        if stop:break
        # Acyclic message workspaces are released by reference counting. Do not
        # sweep the entire interpreter heap after every accepted topology step.
    tree.require_binary()
    if set(tree.names.values())!=set(names):raise RuntimeError('lost taxa')
    if not all(math.isfinite(x) and x>0 for x in tree.lengths.values()):raise ArithmeticError('invalid branch lengths')
    # Reuse the last search scan if it describes this exact final state. Budget
    # stops may accept moves/refits after that scan: then upgrade the mandatory
    # final verification to a local NNI evaluation, without proposing/searching.
    upgrade=support_sink is not None and scan_state!=state_version
    support_started=time.perf_counter()
    checked,_=score(tree,parameters,nni=upgrade,propose=False)
    if abs(checked['log_likelihood']-ll)>.01+1e-6*alignment.original_sites:raise ArithmeticError('final LL disagreement')
    if support_sink is not None:
        from branch_support import gene_records
        support_sink.update(branches=gene_records(*scan_evidence,alignment.original_sites),
            source='upgraded_final_verification' if upgrade else 'last_search_scan',
            additional_inference_runs=0,additional_parameter_fits=0,
            additional_nni_evaluations=int(upgrade),final_state_verified=True,
            final_verification_and_summary_seconds=time.perf_counter()-support_started)
    fit_matches_final=fitted_topology==topology_key(tree)
    strict_converged=bool(parameters['converged'] and fit_matches_final)
    if not strict_converged:warnings.append('final topology parameters have not met the parameter-coordinate score target')
    if all(x.get('uninformative') for x in init_records):warnings.append('initialization sample has no variable sites; topology is unidentifiable from this sample')
    warnings.append('local search; no global maximum or raw-parameter stationarity certificate')
    (out/'prediction.nwk').write_text(newick(tree)+'\n');write(out/'final-model.json',parameters)
    details=dict(algorithm='exact-pattern-profile-ML',version='0.5.0dev11-fused-scan-profile',model=spec.name,
        valid_tree=True,strict_parameter_converged=strict_converged,parameter_fit_matches_final_topology=fit_matches_final,stop=stop,warnings=warnings,
        search_converged=stop.startswith('accepted-move-ratio'),emergency_sweep_guard=EMERGENCY_SWEEPS,
        numerical_plateau_window=8,numerical_plateau_ll_tolerance=plateau.tolerance,
        parameter_continuation_stop=parameters.get('continuation_stop'),
        parameter_interventions=parameter_interventions,
        raw_sites=alignment.original_sites,parameter_sites=len(fit_indices),
        initialization_sites=alignment.original_sites if initializer_uses_all_sites else len(indices),
        all_sites_used_for_topology_scoring=True,all_sites_used_for_parameter_fitting=True,supplied_neural_tree_or_fit=False,views=views,
        max_rounds=max_rounds,stop_ratio=stop_ratio,rounds=round_index,phases=phases,history=history,
        log_likelihood=checked['log_likelihood'],seconds=time.perf_counter()-started,
        missing_data_model=missing_data_model,local_optimizer_iterations_per_solve=optimizer_iterations,minimum_likelihood_gain=min_gain,
        invariant_mass_optimization='conditional concave likelihood profile' if spec.invariant else 'not in supplied model',
        update_schedule='topology block to requested ratio; parameter block; rescan',
        units='expected substitutions per site',precision='FP64 upward fit / FP32 downward and topology messages / int32 binary exponents / FP64 sums',
        scope='stationary reversible unpartitioned DNA gene likelihood; not MSC-consistent species inference')
    write(out/'statistical.json',details)
    return tree,details
