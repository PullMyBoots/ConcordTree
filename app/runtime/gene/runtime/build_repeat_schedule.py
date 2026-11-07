"""Explicit local rebuild of the active exact repeat scheduling extension."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sysconfig
import time
import torch

def main():
    p=argparse.ArgumentParser();p.add_argument('--replace',action='store_true');a=p.parse_args()
    home=Path(__file__).resolve().parent
    source=home/'repeat_builder_schedule.cpp';header=home/'repeat_tuple_keys.h'
    out=home/('_repeat_builder_schedule'+sysconfig.get_config_var('EXT_SUFFIX'))
    if out.exists() and not a.replace:raise FileExistsError('pass --replace for an explicit audited rebuild')
    temporary=out.with_suffix(out.suffix+'.tmp')
    command=['c++','-O3','-std=c++17','-shared','-fPIC','-fopenmp',
             '-I'+sysconfig.get_path('include'),'-I'+str(Path(torch.__file__).parent/'include'),
             str(source),'-o',str(temporary)]
    started=time.perf_counter()
    try:
        subprocess.run(command,check=True)
        temporary.replace(out)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps(dict(command=command,seconds=time.perf_counter()-started,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        header_sha256=hashlib.sha256(header.read_bytes()).hexdigest(),
        binary_sha256=hashlib.sha256(out.read_bytes()).hexdigest())))

if __name__=='__main__':main()
