"""Explicit CPU-only build; never invoked during inference."""
from pathlib import Path
import argparse,sysconfig,subprocess,hashlib,json
import pybind11
p=argparse.ArgumentParser();p.add_argument('--replace',action='store_true');a=p.parse_args()
here=Path(__file__).resolve().parent;src=here/'branch_support_reduce.cpp'
out=here/('_branch_support_reduce'+sysconfig.get_config_var('EXT_SUFFIX'))
if out.exists() and not a.replace:raise FileExistsError(out)
cmd=['g++','-O3','-std=c++17','-fPIC','-shared','-I'+pybind11.get_include(),
     '-I'+sysconfig.get_paths()['include'],str(src),'-o',str(out)]
subprocess.run(cmd,check=True)
print(json.dumps(dict(command=cmd,source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
    binary_sha256=hashlib.sha256(out.read_bytes()).hexdigest())))
