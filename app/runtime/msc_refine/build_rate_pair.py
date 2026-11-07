from pathlib import Path
import subprocess,sysconfig
import pybind11
if __name__=='__main__':
    here=Path(__file__).resolve().parent
    out=here/('_msc_rate_pair'+sysconfig.get_config_var('EXT_SUFFIX'))
    subprocess.run(['c++','-O3','-std=c++17','-shared','-fPIC','-fopenmp',
        '-I'+pybind11.get_include(),'-I'+sysconfig.get_paths()['include'],str(here/'rate_pair_backend.cpp'),'-o',str(out)],check=True)
    print(out)
