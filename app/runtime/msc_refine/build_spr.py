"""Build isolated CPU exact-SPR extension, without touching neural backends."""
from pathlib import Path
import os,subprocess,sys,sysconfig
import pybind11
def main():
    here=Path(__file__).resolve().parent
    output=here/('_msc_spr'+sysconfig.get_config_var('EXT_SUFFIX'))
    command=['c++','-O3','-shared','-std=c++17','-fPIC',
        '-I'+pybind11.get_include(),'-I'+sysconfig.get_paths()['include'],
        str(here/'spr_backend.cpp'),'-o',str(output)]
    subprocess.run(command,check=True);print(output)
if __name__=='__main__':main()
