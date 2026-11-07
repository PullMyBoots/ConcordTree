from pathlib import Path
import subprocess,sysconfig,pybind11
p=Path(__file__).resolve().parent
out=p/('_msc_active_gpu'+sysconfig.get_config_var('EXT_SUFFIX'))
subprocess.run(['g++','-O3','-std=c++17','-fPIC','-shared',str(p/'active_gpu_backend.cpp'),'-I'+pybind11.get_include(),'-I'+sysconfig.get_paths()['include'],'-o',str(out)],check=True)
print(out)
