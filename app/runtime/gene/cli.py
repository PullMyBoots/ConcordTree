"""Isolated release candidate CLI. Historical defaults remain unchanged."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path[:0]=[str(HERE/'src'),str(HERE/'runtime'),str(HERE)]

def main():
    p=argparse.ArgumentParser(description='ConcordTree statistical DNA likelihood candidate')
    p.add_argument('--version',action='version',version='ConcordTree gene 0.5.0dev11 full-site profiled likelihood (experimental)')
    p.add_argument('--msa',type=Path,required=True);p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--sample-fraction',type=float,default=None,help='Initialization fraction; default min(1,16384/L). All sites fit parameters and score topology.')
    p.add_argument('--views',type=int,default=1)
    budget=p.add_mutually_exclusive_group()
    budget.add_argument('--max-rounds',type=int,help='Maximum complete NNI topology scans')
    budget.add_argument('--stop-ratio',type=float,default=.005,help='Accepted NNI moves / internal edges')
    p.add_argument('--model',choices=['JC69','HKY+G4','GTR+G4','GTR+FO+I+G4'],default='GTR+FO+I+G4')
    p.add_argument('--missing-data-model',choices=['standard','coverage-aware'],default='standard')
    p.add_argument('--threads',type=int,default=8,help='CPU execution threads, not a scientific parameter')
    a=p.parse_args()
    if a.views<1 or a.threads<1 or (a.max_rounds is not None and a.max_rounds<1):p.error('positive counts required')
    if a.sample_fraction is not None and not 0<a.sample_fraction<=1:p.error('sample fraction in (0,1]')
    if not 0<=a.stop_ratio<1:p.error('stop ratio in [0,1)')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0':p.error('set CUDA_VISIBLE_DEVICES=0; only GPU0 is authorized for this gene-tree campaign')
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[key]=str(a.threads)
    # All caches belong to this project; never to inputs or linked targets.
    cache=HERE/'cache'/('gpu'+os.environ['CUDA_VISIBLE_DEVICES'])
    for key,name in [('TRITON_CACHE_DIR','triton'),('TORCH_EXTENSIONS_DIR','extensions'),('CUDA_CACHE_PATH','cuda'),
                     ('TORCH_HOME','torch'),('XDG_CACHE_HOME','xdg'),('TMPDIR','tmp'),('NUMBA_CACHE_DIR','numba'),('MPLCONFIGDIR','mpl')]:
        folder=cache/name;folder.mkdir(parents=True,exist_ok=True);os.environ[key]=str(folder)
    import torch
    from algorithm import infer,write
    from concordtree.output_paths import ensure_output_safe
    source=a.msa.resolve(strict=True);out=ensure_output_safe(a.out_dir,[source])
    if not out.is_relative_to(ROOT) or out.is_relative_to(ROOT/'.ResearchProject'):p.error('output must be project-owned outside data/ and memory')
    if out.exists():p.error('fresh output directory required')
    out.mkdir(parents=True);start=time.perf_counter()
    status=dict(status='running',physical_gpu=int(os.environ['CUDA_VISIBLE_DEVICES']),input=str(source),
        configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()})
    write(out/'status.json',status)
    try:
        torch.set_num_threads(a.threads)
        if not getattr(main,'interop_initialized',False):
            torch.set_num_interop_threads(1);main.interop_initialized=True
        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats()
        digest=hashlib.sha256()
        with source.open('rb') as handle:
            for b in iter(lambda:handle.read(8<<20),b''):digest.update(b)
        status['source_sha256']=digest.hexdigest()
        actual=source;decode=0.
        if source.suffix=='.zst':
            t=time.perf_counter();actual=out/'input.phy'
            with actual.open('xb') as handle:subprocess.run(['zstd','-q','-dc',str(source)],stdout=handle,check=True)
            decode=time.perf_counter()-t
        _,details=infer(actual,out,model=a.model,sample_fraction=a.sample_fraction,views=a.views,
            max_rounds=a.max_rounds,stop_ratio=a.stop_ratio,missing_data_model=a.missing_data_model,
            progress=lambda x:print(json.dumps(x),flush=True))
        torch.cuda.synchronize()
        status.update(status='complete',valid_tree=True,strict_parameter_converged=details['strict_parameter_converged'],
            warnings=details['warnings'],stop=details['stop'],decode_seconds=decode,
            torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),torch_peak_reserved_bytes=torch.cuda.max_memory_reserved())
        files={}
        for module in list(sys.modules.values()):
            f=getattr(module,'__file__',None)
            if f:
                f=Path(f).resolve()
                if f.is_relative_to(HERE) and f.is_file() and f.suffix in ('.py','.so'):
                    files[str(f.relative_to(HERE))]=hashlib.sha256(f.read_bytes()).hexdigest()
        write(out/'runtime-manifest.json',files)
        if actual!=source:actual.unlink() # delete ONLY our decompressed derived working file
    except Exception as exc:
        status.update(status='failed',error=repr(exc),traceback=traceback.format_exc())
    status.update(request_seconds=time.perf_counter()-start,peak_self_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    write(out/'status.json',status);print(json.dumps(status),flush=True)
    return 0 if status['status']=='complete' else 1
if __name__=='__main__':raise SystemExit(main())
