"""Offline validation only: predictions are frozen before reference trees are opened."""
from pathlib import Path
import collections, csv, hashlib, importlib.util, json, math, re, sys

H = Path(__file__).resolve().parent
W = H.parents[1]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p, x): p.write_text(json.dumps(x, indent=2, allow_nan=False)+'\n')

def main():
    inputs = {r['id']: r for r in json.loads((H/'INPUTS.json').read_text())}
    old = {r['id']: r for r in json.loads((H/'BASELINES.json').read_text()) if r['method']=='DIPPER GPU'}
    historical_accuracy={r['id']:float(r['nrf']) for r in csv.DictReader((W/'subprojects-main/concordtree-linux-distribution/app/technical-report/data/records.csv').open()) if r['method']=='DIPPER GPU' and r['nrf']}
    rows = []
    for task, count in [('gene',200), ('species',700)]:
        terminal = json.loads((H/f'{task}-TERMINAL.json').read_text())
        rr = json.loads((H/f'{task}-RUNS.json').read_text())
        assert terminal['complete']==count==len(rr)
        rows += rr
    assert len({r['id'] for r in rows})==900 and set(old)=={r['id'] for r in rows}
    spec=importlib.util.spec_from_file_location('timing_treeio',W/'subprojects-main/concordtree-linux-distribution/app/runtime/gene/src/concordtree/treeio.py')
    treeio=importlib.util.module_from_spec(spec);sys.modules[spec.name]=treeio;spec.loader.exec_module(treeio)
    frozen=[]
    # Freeze and validate outputs before reading any reference.
    for r in rows:
        case=inputs[r['id']]
        assert r['status']=='complete' and r['source_sha256']==case['source_sha256']
        assert r['original_seconds']==old[r['id']]['seconds']
        assert r['tree_sha256']==sha(r['tree'])
        assert math.isfinite(r['seconds']) and r['seconds']>0
        assert abs(r['seconds']-r['conversion_seconds']-r['native_file_to_tree_seconds'])<1e-12
        tree=treeio.parse_newick(re.sub(r':[^,();\s]+','',Path(r['tree']).read_text()))
        tree.require_binary();assert len(tree.names)==case['taxa']
        command=json.loads((Path(r['tree']).parent/'COMMAND.json').read_text())['command']
        assert command[command.index('-m')+1]==('0' if r['task']=='gene' else '1')
        assert command[command.index('-d')+1]=='2' and '--force-binary' in command
        if r['task']=='species':assert command[command.index('-K')+1]=='10'
        frozen.append({'id':r['id'],'tree_sha256':r['tree_sha256'],'seconds':r['seconds']})
    write(H/'PREDICTION_BARRIER.json',{'status':'complete','predictions':len(frozen),'reference_opened':False,'records':frozen})
    evaluated=[]
    for r in rows:
        case=inputs[r['id']];assert sha(case['truth'])==case['truth_sha256']
        actual=treeio.parse_newick(re.sub(r':[^,();\s]+','',Path(r['tree']).read_text()))
        truth=treeio.parse_newick(re.sub(r':[^,();\s]+','',Path(case['truth']).read_text()))
        assert set(actual.names.values())==set(truth.names.values())
        evaluated.append({'id':r['id'],'task':r['task'],'taxa':r['taxa'],'family':r['family'],'tree_sha256':r['tree_sha256'],'rerun_nrf':treeio.nrf(actual,truth),'historical_nrf':historical_accuracy[r['id']]})
    write(H/'RERUN_ACCURACY.json',{'policy':'Independent offline check; report accuracy remains attached to historical predictions. Not substituted into historical accuracy figures.','records':evaluated})
    summary=[]
    import statistics
    for task in ['gene','species']:
        rr=[r for r in rows if r['task']==task]
        for taxa in [None]+sorted({r['taxa'] for r in rr}):
            group=[r for r in rr if taxa is None or r['taxa']==taxa]
            er=[r for r in evaluated if r['task']==task and (taxa is None or r['taxa']==taxa)]
            summary.append({'task':task,'taxa':taxa,'n':len(group),'old_mean_seconds':statistics.mean(r['original_seconds'] for r in group),'new_mean_seconds':statistics.mean(r['seconds'] for r in group),'old_median_seconds':statistics.median(r['original_seconds'] for r in group),'new_median_seconds':statistics.median(r['seconds'] for r in group),'mean_excluded_initialization_seconds':statistics.mean(r['excluded_initialization_seconds'] for r in group),'mean_process_wall_seconds':statistics.mean(r['process_wall_seconds'] for r in group),'historical_mean_nrf':statistics.mean(r['historical_nrf'] for r in er),'rerun_mean_nrf':statistics.mean(r['rerun_nrf'] for r in er)})
    write(H/'TIMING_SUMMARY.json',summary)
    write(H/'VERIFICATION.json',{'status':'passed','records':900,'gene':200,'species':700,'checks':['same immutable input hashes recorded by worker','same original algorithm options','positive synchronized native time plus format conversion','all tree hashes','all binary topologies','exact reference taxon sets','reference hashes','all predictions frozen before offline RF evaluation'],'historical_accuracy_preserved':True,'timing_summary':summary})
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
