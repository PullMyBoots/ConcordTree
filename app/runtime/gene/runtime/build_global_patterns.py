"""Explicit build only; no compiler is invoked during a timed inference."""
from pathlib import Path
import argparse,hashlib,json,subprocess,sysconfig,time
import torch
p=argparse.ArgumentParser();p.add_argument('--replace',action='store_true');a=p.parse_args()
here=Path(__file__).resolve().parent;source=here/'global_patterns.cpp'
out=here/('_global_patterns'+sysconfig.get_config_var('EXT_SUFFIX'))
if out.exists() and not a.replace:raise FileExistsError('use --replace to rebuild this project-owned binary')
cmd=['c++','-O3','-std=c++17','-shared','-fPIC','-fopenmp','-I'+sysconfig.get_path('include'),'-I'+str(Path(torch.__file__).parent/'include'),str(source),'-o',str(out)]
t=time.perf_counter();subprocess.run(cmd,check=True)
report=dict(command=cmd,seconds=time.perf_counter()-t,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),binary_sha256=hashlib.sha256(out.read_bytes()).hexdigest())
(here/'global-pattern-build.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
