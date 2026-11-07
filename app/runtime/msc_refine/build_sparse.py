"""Build our independent fixed-quartet NNI objective evaluator."""
from pathlib import Path
import subprocess,sysconfig
import pybind11
p=Path(__file__).parent
out=p/('_msc_sparse'+sysconfig.get_config_var('EXT_SUFFIX'))
subprocess.run(['g++','-O3','-std=c++17','-fPIC','-shared',str(p/'sparse_backend.cpp'),
               '-I'+pybind11.get_include(),'-I'+sysconfig.get_paths()['include'],'-o',str(out)],check=True)
print(out)
