"""Explicit, task-checked composition of AP-NJ and refinement modules."""
import copy
import time
import shutil

def module_options(configuration, id):
    return next((s for s in configuration['stages'] if s['id']==id),None) if configuration else None

def canonical_route(configuration):
    """Keep the original execution order, IDs and numerics for default workflows."""
    active=[s['id'] for s in configuration['stages'] if s['enabled']]
    terminal='likelihood' if configuration['task']=='gene' else 'kernel'
    return bool(active and active[-1]==terminal and active==[s for s in ['distance','mlp',terminal] if s in active])

def save_supplied(tree,info,out):
    from concordtree.treeio import newick_python
    from engine import write_json
    out.mkdir(parents=True,exist_ok=True)
    value=copy.deepcopy(info);value['supplied_pipeline_tree']=True
    (out/'supplied.nwk').write_text(newick_python(tree)+'\n')
    (out/'mlp.nwk').write_text(newick_python(tree)+'\n')
    write_json(out/'initialization.json',value)
    return tree.copy(),value

def relabel_tree(tree,names):
    from concordtree.treeio import Tree,edge
    lookup={name:i for i,name in enumerate(names)}
    if set(tree.names.values())!=set(names):raise ValueError('Pipeline stage changed the input taxa.')
    mapping={u:lookup[name] for u,name in tree.names.items()}
    mapping.update({u:len(names)+j for j,u in enumerate(sorted(set(tree.graph)-set(tree.names)))})
    return Tree({mapping[u]:{mapping[v] for v in vs} for u,vs in tree.graph.items()},
        dict(enumerate(names)),{edge(mapping[a],mapping[b]):v for (a,b),v in tree.lengths.items()})

def distance_refine(tree,names,packed,sites,configuration,options,cpu_threads,checkpoint=None,backbone=None):
    from concordtree._core.learned_nni import _adjacency_neighbor_array
    from concordtree.treeio import Tree
    import _ct_apnj_packed as native
    import _ct_apnj_topology as topology
    from apnj_cleanup import refine
    tree=relabel_tree(tree,names);neighbors=_adjacency_neighbor_array(tree.graph,len(names))
    state=native.State(packed,int(sites),min(16,cpu_threads));model=configuration['apnj']['distance_model']
    alpha=state.gamma_shape() if model>=2 else 100.
    def publish(adj,event):
        if checkpoint:checkpoint(Tree(topology.export_graph(adj,len(names),True),dict(enumerate(names)),{}),event)
    result=refine(neighbors,len(names),state,model,True,alpha,checkpoint=publish if checkpoint else None,
        max_rounds=options['max_rounds'],stop_ratio=options['stop_ratio'],backbone=backbone,names=names,**options['advanced'])
    result['gamma_shape']=alpha
    return Tree(topology.export_graph(neighbors,len(names),True),dict(enumerate(names)),{}),result

def execute(engine,msa,out,task,configuration,loci,model,fraction,views,missing,progress,branch_support):
    import torch
    from exact_site_patterns import PackedPatterns
    from concordtree.treeio import newick_python,parse_newick
    from engine import write_json
    if views!=1:raise ValueError('Custom module ordering uses one AP-NJ initialization.')
    alignment=PackedPatterns(msa);names=alignment.names;sites=alignment.original_sites
    packed=alignment.packed
    active=[s for s in configuration['stages'] if s['enabled']]
    # The leading distance/MLP steps share the original constructor's exact
    # pre-canonical cleanup path, preserving existing outputs when steps are disabled.
    prefix=[]
    for s in active:
        if s['id']=='distance' and not prefix:prefix.append(s)
        elif s['id']=='mlp' and (not prefix or prefix[-1]['id']=='distance'):prefix.append(s)
        else:break
    initial_config=copy.deepcopy(configuration)
    for s in initial_config['stages']:s['enabled']=s['id'] in {x['id'] for x in prefix}
    tree,info=engine._initialize(task,msa,out/'initial',names,None,configuration['apnj']['seed'],missing,
        (packed,names,sites),progress=progress,configuration=initial_config)
    stages=[dict(id='apnj',seconds=info['apnj_seconds']-info['local_cleanup']['seconds'],stop='complete')]
    if any(s['id']=='distance' for s in prefix):stages.append(dict(id='distance',**info['local_cleanup']))
    if any(s['id']=='mlp' for s in prefix):
        stages.append(dict(id='mlp',seconds=info['mlp_seconds'],stop=info['mlp_stop'],converged=info['mlp_converged']))
    support=dict(enabled=False,requested=branch_support,reason='The final module does not produce branch support.')
    posterior=dict(stop='direct_output',search_converged=True,input_mode='observed_loci' if loci else 'msa_only')
    branch_lengths=False
    for index,s in enumerate(active[len(prefix):],len(prefix)):
        id=s['id'];directory=out/'modules'/f'{index+1:02d}-{id}';directory.mkdir(parents=True)
        progress(dict(stage='module_started',module=id));started=time.perf_counter()
        if id=='distance':
            tree,record=distance_refine(tree,names,packed,sites,configuration,s,engine.runtime['cpu_threads'],engine._checkpoint,backbone=engine._backbone)
            branch_lengths=False;support=dict(enabled=False,requested=branch_support,reason='Final distance NNI has no branch support.')
        elif id=='mlp':
            tree,record=engine.predictors[task].refine(tree,names,msa,packed_data=(packed,names,sites),
                max_rounds=s['max_rounds'],stop_ratio=s['stop_ratio'],checkpoint=engine._checkpoint,backbone=engine._backbone,**s['advanced'],
                progress=lambda row:progress(dict(stage='mlp',**row)))
            info.update(mlp_seconds=record['seconds'],mlp_stop=record['stop'],mlp_converged=record['converged'],
                mlp_predictor=engine.predictors[task].predictor,mlp_checkpoint_sha256=record['checkpoint_sha256'])
            branch_lengths=False;support=dict(enabled=False,requested=branch_support,reason='Final MLP NNI has no branch support.')
        else:
            final=index==len(active)-1
            if id=='likelihood':
                details=engine._gene(msa,directory,model,fraction,1,s['max_rounds'],s['stop_ratio'],missing,progress,
                    branch_support and final,configuration=configuration,supplied_tree=tree,supplied_info=info)
            elif id=='kernel':
                details=engine._species(msa,directory,loci,fraction,s['stop_ratio'],missing,progress,
                    branch_support and final,configuration=configuration,supplied_tree=tree,supplied_info=info)
            posterior=details['posterior'];record=dict(posterior)
            plain=directory/'prediction.unannotated.nwk'
            tree=relabel_tree(parse_newick((plain if plain.exists() else directory/'prediction.nwk').read_text()),names)
            branch_lengths=id=='likelihood'
            support=details.get('branch_support',dict(enabled=False,requested=branch_support))
            if final:shutil.copyfile(directory/'prediction.nwk',out/'prediction.nwk')
            if final and plain.exists():shutil.copyfile(plain,out/'prediction.unannotated.nwk')
        torch.cuda.synchronize();record.update(id=id,seconds=time.perf_counter()-started)
        stages.append(record);write_json(directory/'stage.json',record)
        (directory/'tree.nwk').write_text(newick_python(tree)+'\n')
        progress(dict(stage='module_complete',module=id,seconds=record['seconds']))
    if not active or active[-1]['id'] in ('distance','mlp'):
        (out/'prediction.nwk').write_text(newick_python(tree)+'\n')
    issues=[r for r in stages if r.get('converged') is False or r.get('search_converged') is False]
    if issues:posterior=dict(posterior,search_converged=False,stop=';'.join(r['id']+':'+r['stop'] for r in issues))
    alignment.release_packed()
    return dict(initialization=[info],posterior=posterior,stages=stages,branch_support=support,branch_lengths_inferred=branch_lengths)
