"""Build this release's bounded AP-NJ closure with the active CPython ABI."""
from pathlib import Path
import argparse,json,subprocess,sysconfig
import pybind11
p=Path(__file__).resolve().parent;root=p.parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--arch',default='89');args=parser.parse_args()
if not args.arch.isdecimal():raise ValueError('CUDA architecture must be numeric, e.g.89 for L40S')
suffix=sysconfig.get_config_var('EXT_SUFFIX')
includes=['-I'+sysconfig.get_path('include'),'-I'+pybind11.get_include()]
commands=[]
for source,module in [('graph.cpp','graph'),('topology.cpp','topology'),('local.cpp','local'),('terminal.cpp','terminal'),('packed.cu','packed')]:
    target=root/('_ct_apnj_'+module+suffix);temporary=target.with_suffix(target.suffix+'.new')
    if source.endswith('.cu'):
        command=['/usr/local/cuda/bin/nvcc','-O3','--shared','-std=c++17','-Xcompiler=-fPIC,-fopenmp',f'-gencode=arch=compute_{args.arch},code=sm_{args.arch}',*includes,str(p/source),'-o',str(temporary)]
    else:command=['g++','-O3','-std=c++17','-shared','-fPIC','-fopenmp',*includes,str(p/source),'-o',str(temporary)]
    commands.append(command)
    with (p/('build-'+module+'.log')).open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    temporary.replace(target);print('BUILT',module,flush=True)
(p/'BUILD_COMMANDS.json').write_text(json.dumps(commands,indent=2)+'\n')
