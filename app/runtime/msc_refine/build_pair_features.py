"""Build isolated CPU polynomial feature extension;no data directory writes."""
import subprocess,sysconfig
from pathlib import Path
import pybind11
base=Path(__file__).resolve().parent
command=['g++','-O3','-std=c++17','-shared','-fPIC','-fopenmp','-I'+pybind11.get_include(),
 '-I'+sysconfig.get_paths()['include'],str(base/'pair_feature_backend.cpp'),
 '-o',str(base/('_pair_features'+sysconfig.get_config_var('EXT_SUFFIX')))]
subprocess.run(command,check=True)
