"""Explicit project-local build, never run automatically during inference."""
import argparse,hashlib,json,subprocess,sysconfig,time
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('--replace',action='store_true');a=p.parse_args();h=Path(__file__).resolve().parent
source=h/'repeat_builder_invariant.cpp';out=h/('_repeat_builder_invariant'+sysconfig.get_config_var('EXT_SUFFIX'))
if out.exists() and not a.replace:raise FileExistsError('existing project binary; pass --replace only for an explicit rebuild')
cmd=['c++','-O3','-std=c++17','-shared','-fPIC','-fopenmp','-I'+sysconfig.get_path('include'),'-I'+str(Path(torch.__file__).parent/'include'),str(source),'-o',str(out)]
t=time.perf_counter();subprocess.run(cmd,check=True);report=dict(command=cmd,seconds=time.perf_counter()-t,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),binary_sha256=hashlib.sha256(out.read_bytes()).hexdigest());(h/'repeat-invariant-build.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
