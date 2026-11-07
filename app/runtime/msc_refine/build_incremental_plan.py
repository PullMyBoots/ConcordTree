"""Build a separate backend; never overwrite the frozen incumbent module."""
from pathlib import Path
import subprocess
import sysconfig
import pybind11

p = Path(__file__).resolve().parent
out = p / ('_msc_incremental_plan' + sysconfig.get_config_var('EXT_SUFFIX'))
subprocess.run(['g++', '-O3', '-std=c++17', '-fPIC', '-shared', '-fopenmp',
                str(p / 'active_incremental_plan_backend.cpp'), '-I' + pybind11.get_include(),
                '-I' + sysconfig.get_paths()['include'], '-o', str(out)], check=True)
print(out)
