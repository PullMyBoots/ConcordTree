"""Build only our exact native tree-plan compiler, outside all data inputs."""
from pathlib import Path
import subprocess
import sysconfig
import pybind11

p=Path(__file__).resolve().parent
out=p/('_msc_plan'+sysconfig.get_config_var('EXT_SUFFIX'))
temporary=out.with_name(out.name+'.build')
subprocess.run(['g++','-O3','-DNDEBUG','-std=c++17','-fPIC','-shared',str(p/'plan_backend.cpp'),
               '-I'+pybind11.get_include(),'-I'+sysconfig.get_paths()['include'],'-o',str(temporary)],check=True)
temporary.replace(out)  # Never truncate a shared library mapped by a live process.
print(out)
