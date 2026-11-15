"""Rebuild the four small native CPU libraries on the target host."""
import os,subprocess,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent
for source in ('native','model_gradient','insertion_newton','repeat_prune'):
    fd,name=tempfile.mkstemp(prefix=source+'-',suffix='.so',dir=ROOT);os.close(fd);temporary=Path(name)
    try:
        subprocess.run(['g++','-O3','-march=native','-std=c++17','-shared','-fPIC',str(ROOT/(source+'.cpp')),'-o',str(temporary)],check=True)
        temporary.replace(ROOT/(source+'.so'))
    finally:
        if temporary.exists():temporary.unlink()
