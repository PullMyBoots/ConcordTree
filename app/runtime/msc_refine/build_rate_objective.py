"""Build the CPU-only independent global-objective auditing kernel."""
from pathlib import Path
import subprocess
import sysconfig
import pybind11

if __name__ == '__main__':
    here = Path(__file__).resolve().parent
    out = here/('_msc_rate_objective'+sysconfig.get_config_var('EXT_SUFFIX'))
    subprocess.run(['c++','-O3','-std=c++17','-shared','-fPIC','-fopenmp','-ffp-contract=off',
                    '-I'+pybind11.get_include(),'-I'+sysconfig.get_paths()['include'],
                    str(here/'rate_objective_backend.cpp'),'-o',str(out)],check=True)
    print(out)
