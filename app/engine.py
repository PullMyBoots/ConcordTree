"""Unified gene/species inference with a shared bounded AP-NJ / MLP-NNI frontend.

One engine may serve serial requests and retains both MLP checkpoints. The
numerical backends use one logical CUDA device. Concurrent calls are serialized.
"""
from pathlib import Path
from threading import RLock
import hashlib
import json
import shutil
import time

from runtime_config import HERE, configure_runtime, owned_path

_LOCK = RLock()
MSC_ASSETS = {
    'msa_only': ('msa_only.npz', '861d790d0f6c9510c182367342c7473fb49ef21449ccd4897ed9579df8318491'),
    'observed_loci': ('observed_loci.npz', 'c2925749b22eebc5e9525a9415cb5101d655abcf69c2d57172e81c4dae80a23f'),
}

def write_json(path, value, *, indent=2):
    path = Path(path); temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=indent, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)

class UnifiedEngine:
    def __init__(self, *, gpu=None, cpu_threads=16):
        self._checkpoint=None
        self.runtime = configure_runtime(gpu=gpu, cpu_threads=cpu_threads)
        import torch
        if not torch.cuda.is_available():raise RuntimeError('this release requires a CUDA GPU')
        torch.set_num_threads(self.runtime['cpu_threads'])
        if torch.get_num_interop_threads() != 1:torch.set_num_interop_threads(1)
        torch.cuda.set_device(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        from frontend import MLPWarmstart, TASK_PREDICTOR
        self.predictors = {task: MLPWarmstart(name) for task, name in TASK_PREDICTOR.items()}

    def _initialize(self, task, msa, out, names, states, seed, missing, packed_data, *, view=0, progress=None,configuration=None):
        import torch
        from apnj_frontend import initialize as build_apnj
        from concordtree.treeio import newick_python
        from exact_topology_key import topology_key
        out.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        if packed_data is None:raise ValueError('AP-NJ requires the original packed alignment')
        if progress:progress(dict(stage='apnj_running', view=view, sites=int(packed_data[2])))
        from pipeline_runtime import module_options
        distance=module_options(configuration,'distance');mlp=module_options(configuration,'mlp')
        options=configuration['apnj'] if configuration else None
        if options:seed=(options['seed']+view)%(2**32)
        cleanup=dict(max_rounds=distance['max_rounds'],stop_ratio=distance['stop_ratio'],**distance['advanced']) if distance and distance['enabled'] else (dict(max_rounds=0) if distance else None)
        tree, info = build_apnj(names, packed_data, task=task, seed=seed, cpu_threads=self.runtime['cpu_threads'],options=options,cleanup_options=cleanup,checkpoint=self._checkpoint)
        torch.cuda.synchronize(); apnj_seconds = time.perf_counter() - started
        suffix = '' if view == 0 else f'-{view:02d}'
        (out / f'apnj{suffix}.nwk').write_text(newick_python(tree) + '\n')
        named, structure = topology_key(tree)
        identity = json.dumps(named, ensure_ascii=True, separators=(',', ':')).encode()
        info['initial_split_hash'] = hashlib.sha256(identity + b'\0' + structure).hexdigest()
        info['requested_missing_data_model'] = missing
        if progress and distance and distance['enabled']:
            progress(dict(stage='module_complete', module='distance', seconds=info['local_cleanup']['seconds']))
        if mlp is None or mlp['enabled']:
            if progress:progress(dict(stage='module_started', module='mlp'))
            tree, neural = self.predictors[task].refine(tree, names, msa, stop_ratio=mlp['stop_ratio'] if mlp else .005,
                packed_data=packed_data, progress=(lambda row: progress(dict(stage='mlp', view=view, **row))) if progress else None,
                checkpoint=self._checkpoint,**(dict(max_rounds=mlp['max_rounds'],**mlp['advanced']) if mlp else {}))
        else:
            neural=dict(seconds=0.,passes=[],moves=0,stop='disabled',converged=True,warnings=[],checkpoint_sha256=None)
        if progress and (mlp is None or mlp['enabled']):
            progress(dict(stage='module_complete', module='mlp', seconds=neural['seconds']))
        (out / f'mlp{suffix}.nwk').write_text(newick_python(tree) + '\n')
        info.update(apnj_seconds=apnj_seconds, mlp_seconds=neural['seconds'],
                    mlp_stop=neural.get('stop'),mlp_converged=neural.get('converged',True),
                    mlp_warnings=neural.get('warnings',[]),
                    mlp_predictor=self.predictors[task].predictor if neural['checkpoint_sha256'] else None,
                    mlp_checkpoint_sha256=neural['checkpoint_sha256'],
                    frontend_seconds=time.perf_counter() - started)
        write_json(out / f'mlp{suffix}.json', neural)
        write_json(out / f'initialization{suffix}.json', info)
        return tree, info

    def infer(self, msa, out, *, task, model=None, loci=None, sample_fraction=None,
              views=1, max_rounds=None, stop_ratio=.005, missing_data_model='standard', progress=None,branch_support=True,configuration=None,pause_request=None):
        with _LOCK:
            return self._infer(msa, out, task=task, model=model, loci=loci,
                sample_fraction=sample_fraction, views=views, max_rounds=max_rounds,
                stop_ratio=stop_ratio, missing_data_model=missing_data_model, progress=progress,branch_support=branch_support,configuration=configuration,pause_request=pause_request)

    def _infer(self, msa, out, *, task, model, loci, sample_fraction, views, max_rounds, stop_ratio, missing_data_model, progress,branch_support,configuration,pause_request):
        if not isinstance(branch_support,bool):raise ValueError('branch_support must be a boolean')
        if task not in self.predictors:raise ValueError('task must be gene or species')
        if sample_fraction is not None and not 0 < sample_fraction <= 1:raise ValueError('sample fraction must be in (0,1]')
        if not 0 <= stop_ratio < 1:raise ValueError('stop ratio must be in [0,1)')
        if not isinstance(views, int) or views < 1:raise ValueError('positive integer views required')
        if max_rounds is not None and (not isinstance(max_rounds,int) or max_rounds < 1):raise ValueError('positive integer rounds required')
        if missing_data_model not in ('standard','coverage-aware'):raise ValueError('unknown missing-data model')
        if task == 'gene' and loci is not None:raise ValueError('--loci is supported for species trees only')
        if task == 'species' and (model is not None or views != 1 or max_rounds is not None):
            raise ValueError('species inference uses one view and stop-ratio; likelihood model/rounds apply to gene trees only')
        if task == 'gene':
            from concordtree.dna_models import parse_model
            parse_model(model or 'GTR+FO+I+G4')
        from pipeline_config import normalize
        from pipeline_runtime import canonical_route,module_options
        explicit_configuration=configuration is not None
        configuration=normalize(configuration,task)
        if not explicit_configuration:
            posterior_options=module_options(configuration,'likelihood' if task=='gene' else 'kernel')
            posterior_options['stop_ratio']=stop_ratio
            if task=='gene':
                posterior_options['max_rounds']=max_rounds
                if model:posterior_options['advanced']['model']=model
        else:
            posterior_options=module_options(configuration,'likelihood' if task=='gene' else 'kernel')
            stop_ratio=posterior_options['stop_ratio'];max_rounds=posterior_options['max_rounds']
        if task=='gene':model=posterior_options['advanced']['model']
        msa = Path(msa).resolve(strict=True)
        loci = Path(loci).resolve(strict=True) if loci is not None else None
        out = owned_path(out, inputs=[msa] + ([loci] if loci else []))
        if out.exists():raise ValueError('fresh output directory required')
        out.mkdir(parents=True)
        from checkpoints import Checkpoints
        checkpoints=Checkpoints(out,pause_request)
        started = time.perf_counter()
        def emit(value):
            event = dict(value)
            if 'elapsed_seconds' in event:event['stage_elapsed_seconds'] = event['elapsed_seconds']
            event.update(task=task, elapsed_seconds=time.perf_counter()-started)
            from progress_events import append
            append(out/'events.jsonl', event)
            write_json(out/'progress.json', event)
            if progress:progress(event)
        def checkpoint(tree, event=None):
            event = event or {}
            if event.get('stage') == 'apnj':
                emit(dict(stage='module_complete', module='apnj'))
            elif event.get('stage') == 'distance':
                emit(event)
            checkpoints.publish(tree, event)
        self._checkpoint = checkpoint
        write_json(out/'runtime.json', dict(version='1.0.2', **self.runtime))
        write_json(out/'configuration.json',configuration)
        try:
            from inputs import prepare_input
            actual = prepare_input(msa, out)
            emit(dict(stage='input_ready'))
            if not canonical_route(configuration):
                from pipeline_runtime import execute
                details=execute(self,actual,out,task,configuration,loci,model or 'GTR+FO+I+G4',sample_fraction,views,missing_data_model,emit,branch_support)
            elif task == 'gene':
                details = self._gene(actual, out, model or 'GTR+FO+I+G4', sample_fraction,
                                     views, max_rounds, stop_ratio, missing_data_model, emit,branch_support,configuration=configuration)
            else:
                details = self._species(actual, out, loci, sample_fraction, stop_ratio, missing_data_model, emit,branch_support,configuration=configuration)
            support_details=details.pop('branch_support',dict(enabled=False))
            posterior=details['posterior']
            interventions=[]
            for item in details['initialization']:
                if not item.get('mlp_converged',True):
                    interventions.append(dict(stage='mlp',reason=item.get('mlp_stop')))
            search_converged=posterior.get('search_converged',True)
            if not search_converged:
                interventions.append(dict(stage='posterior',reason=posterior.get('stop')))
            for item in posterior.get('parameter_interventions',[]):
                interventions.append(dict(stage='parameter_fit',**item))
            if posterior.get('parameter_continuation_stop') and not posterior.get('parameter_interventions'):
                interventions.append(dict(stage='parameter_fit',reason=posterior['parameter_continuation_stop']))
            initialization_warnings=[message for item in details['initialization'] for message in item.get('health',{}).get('warnings',[])]
            quality=dict(search_converged=search_converged,guard_interventions=interventions,
                parameter_converged=posterior.get('strict_parameter_converged') if task=='gene' else None,
                initialization_informative=all(item.get('health',{}).get('informative',True) for item in details['initialization']),
                initialization_warnings=initialization_warnings,
                warnings=[f"{r['stage']}: {r['reason']}" for r in interventions]+initialization_warnings)
            result = dict(status='complete', version='1.0.2', task=task,
                input=str(msa), prediction=str(out/'prediction.nwk'),
                mlp_predictor=self.predictors[task].predictor if module_options(configuration,'mlp')['enabled'] else None,
                mlp_checkpoint_sha256=getattr(self.predictors[task],'checkpoint_sha256',None) if module_options(configuration,'mlp')['enabled'] else None,
                pipeline=' -> '.join(['AP-NJ']+[s['id'] for s in configuration['stages'] if s['enabled']]),configuration=configuration,
                file_to_tree_seconds=time.perf_counter()-started-checkpoints.paused_seconds,paused_seconds=checkpoints.paused_seconds,
                branch_lengths_inferred=details.get('branch_lengths_inferred',task=='gene'), details=details,branch_support=support_details,quality=quality)
            write_json(out/'RESULT.json',result); emit(dict(stage='complete'))
            return result
        except Exception as exc:
            write_json(out/'RESULT.json',dict(status='failed',task=task,version='1.0.2',error=str(exc)))
            emit(dict(stage='failed',error=str(exc)))
            raise

    def _gene(self, msa, out, model, fraction, views, rounds, stop_ratio, missing, progress,branch_support,configuration=None,supplied_tree=None,supplied_info=None):
        import algorithm
        from pipeline_runtime import module_options
        posterior = out/'gene'
        def initialize(names, states, seed, mode, packed_data=None):
            view = len(records)
            if supplied_tree is None:
                tree, info = self._initialize('gene', msa, out/'initial', names, states, seed,
                                              mode, packed_data, view=view, progress=progress,configuration=configuration)
            else:
                from pipeline_runtime import save_supplied
                tree,info=save_supplied(supplied_tree,supplied_info,out/'initial')
            records.append(info)
            return tree, info
        records = []
        evidence={} if branch_support else None
        tree, details = algorithm.infer(msa, posterior, model=model, sample_fraction=fraction,
            views=views, max_rounds=rounds, stop_ratio=stop_ratio, missing_data_model=missing,
            initialize_fn=initialize, initializer_uses_all_sites=True,
            progress=lambda row: progress(dict(stage='gene_refinement', detail=row)),support_sink=evidence,checkpoint_fn=self._checkpoint,
            **({k:v for k,v in module_options(configuration,'likelihood')['advanced'].items() if k!='model'} if configuration else {}))
        details.update(supplied_neural_tree_or_fit=True,
                       initialization_provider='supplied pipeline tree' if supplied_tree is not None else 'configured AP-NJ frontend')
        write_json(posterior/'statistical.json', details)
        shutil.copyfile(posterior/'prediction.nwk', out/'prediction.nwk')
        output=dict(initialization=records, posterior=details)
        if branch_support:
            from branch_support import export_support
            from concordtree.treeio import newick
            progress(dict(stage='branch_support'))
            output['branch_support']=export_support(tree,'gene',evidence,write_json=write_json,
                plain_prediction=out/'prediction.nwk',serialize=lambda a:newick(tree,a))
        return output

    def _species(self, msa, out, loci, fraction, stop_ratio, missing, progress,branch_support,configuration=None,supplied_tree=None,supplied_info=None):
        import torch
        from exact_site_patterns import PackedPatterns
        from concordtree._core.learned_nni import tree_path_to_graph
        from concordtree._core.scaleqf import validate_topology
        from msc_refine.observations import read_boundaries
        from msc_refine.barycenter_tree_tiled import TiledBarycenterScorer
        from msc_refine.active_streamsum import StreamSumActiveScorer
        from msc_refine.active_quartet import refine_active
        from species_treeio import graph_to_newick
        alignment = PackedPatterns(msa)
        names = alignment.names; length = alignment.original_sites
        if len(names) < 24:raise ValueError('the frozen MSC nearest24 refinement requires at least 24 taxa')
        regions = read_boundaries(loci, length) if loci else [(0,length)]
        mode = 'observed_loci' if loci else 'msa_only'
        filename, expected = MSC_ASSETS[mode]; asset = HERE/'assets/msc'/filename
        if hashlib.sha256(asset.read_bytes()).hexdigest() != expected:raise RuntimeError('MSC asset hash mismatch')
        initial = out/'initial'; initial.mkdir()
        if supplied_tree is None:
            tree, info = self._initialize('species', msa, initial, names, None, 20260934,
                 missing, (alignment.packed,names,length), progress=progress,configuration=configuration)
        else:
            from pipeline_runtime import save_supplied
            tree,info=save_supplied(supplied_tree,supplied_info,initial)
        # Match the standalone MSC input convention, including internal node IDs.
        graph = tree_path_to_graph(initial/'mlp.nwk', names); validate_topology(graph,len(names))
        del tree
        phase = time.perf_counter()
        from pipeline_runtime import module_options
        options=module_options(configuration,'kernel')
        base = TiledBarycenterScorer(alignment.packed,length,regions,asset,**(dict(batch_size=options['advanced']['batch_size']) if options else {}))
        scorer = StreamSumActiveScorer(base,collect_support=branch_support)
        alignment.release_packed()
        torch.cuda.synchronize(); setup_seconds = time.perf_counter()-phase
        phase = time.perf_counter()
        def kernel_progress(graph,row):
            from concordtree.treeio import Tree
            self._checkpoint(Tree(graph,dict(enumerate(names)),{}),dict(stage='kernel',round=row['sweep']))
            progress(dict(stage='species_refinement',**row))
        final, details = refine_active(graph,scorer,stop_ratio=stop_ratio,
            callback=kernel_progress,
            **(dict(max_rounds=options['max_rounds'],min_gain=options['advanced']['min_gain']) if options else {}))
        torch.cuda.synchronize(); search_seconds=time.perf_counter()-phase
        validate_topology(final,len(names))
        (out/'prediction.nwk').write_text(graph_to_newick(final,names,len(names))+'\n')
        posterior = dict(input_mode=mode, asset_sha256=expected, setup_seconds=setup_seconds,
                         search_seconds=search_seconds, scorer_metadata=base.metadata, **details)
        write_json(out/'species.json',posterior)
        output=dict(initialization=[info], posterior=posterior)
        if branch_support:
            from branch_support import export_support
            from concordtree.treeio import Tree
            progress(dict(stage='branch_support'))
            evidence=scorer.branch_support(final)
            tree=Tree(final,dict(enumerate(names)),{})
            output['branch_support']=export_support(tree,'species',evidence,write_json=write_json,
                plain_prediction=out/'prediction.nwk',serialize=lambda a:graph_to_newick(final,names,len(names),a))
        return output
