"""Atomic build of our separate exact native uniform-query compiler."""
from pathlib import Path
import subprocess,sysconfig
import pybind11
p=Path(__file__).resolve().parent
out=p/('_uniform_plan'+sysconfig.get_config_var('EXT_SUFFIX'));temporary=out.with_name(out.name+'.build')
subprocess.run(['g++','-O3','-DNDEBUG','-std=c++17','-fPIC','-shared',str(p/'uniform_plan_backend.cpp'),
 '-I'+pybind11.get_include(),'-I'+sysconfig.get_paths()['include'],'-o',str(temporary)],check=True)
temporary.replace(out);print(out)
