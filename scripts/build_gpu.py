"""Rebuild only the AP-NJ CUDA extension, preserving its source and math flags."""
from pathlib import Path
import hashlib
import json
import subprocess
import sysconfig
import pybind11

ROOT = Path(__file__).resolve().parents[1]
app = ROOT / 'app'
source = app / 'native/apnj/packed.cu'
output = app / ('_ct_apnj_packed' + sysconfig.get_config_var('EXT_SUFFIX'))
arches = ['70', '75', '80', '86', '89', '90']
command = ['/usr/local/cuda-12.5/bin/nvcc', '-O3', '--shared', '-std=c++17',
           '-Xcompiler=-fPIC,-fopenmp', '-I' + sysconfig.get_path('include'),
           '-I' + pybind11.get_include()]
for arch in arches:
    command += ['-gencode', f'arch=compute_{arch},code=sm_{arch}']
command += ['-gencode', 'arch=compute_70,code=compute_70',
            str(source), '-o', str(output) + '.new']
before = hashlib.sha256(output.read_bytes()).hexdigest()
subprocess.run(command, check=True)
Path(str(output) + '.new').replace(output)
(ROOT / 'evidence/gpu-build.json').write_text(json.dumps(dict(
    command=command, cubin_architectures=arches, forward_ptx='compute_70',
    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    before_sha256=before, after_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
    numerical_source_changed=False), indent=2) + '\n')
print('Built AP-NJ CUDA fat binary:', ', '.join(arches), '+ PTX70', flush=True)
