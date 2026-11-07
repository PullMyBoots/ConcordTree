"""Explicit native build for the current server's L40S (SM89); never runs at inference."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse
import json
import os
import subprocess
import sysconfig
import pybind11

HERE = Path(__file__).resolve().parent
RELEASE = HERE.parents[1]
PROJECT = next(p for p in HERE.parents if (p / '.ResearchProject').is_dir())

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir', type=Path, default=PROJECT / 'output/dipper-integration-20261106/build')
    parser.add_argument('--arch', default='89')
    args = parser.parse_args()
    from sys import path
    path.insert(0, str(RELEASE))
    from runtime_config import owned_path
    build = owned_path(args.build_dir); build.mkdir(parents=True, exist_ok=True)
    cuda = Path('/usr/local/cuda')
    sources = ['MSA.cu', 'placement_close_k.cu', 'tree.cpp']
    commands = []
    for name in sources:
        source = HERE / 'upstream' / name
        obj = build / (name + '.o')
        if source.suffix == '.cu':
            command = [str(cuda / 'bin/nvcc'), '-O3', '-std=c++17', '--extended-lambda',
                       '-gencode', f'arch=compute_{args.arch},code=sm_{args.arch}',
                       '-gencode', f'arch=compute_{args.arch},code=compute_{args.arch}',
                       '-Xcompiler=-fPIC', '-c', str(source), '-o', str(obj)]
        else:
            command = ['g++', '-O3', '-std=c++17', '-fPIC', '-c', str(source), '-o', str(obj)]
        commands.append(command)
    bridge = ['g++', '-O3', '-std=c++17', '-fPIC', '-fopenmp',
              '-I' + sysconfig.get_paths()['include'], '-I' + pybind11.get_include(),
              '-I' + str(cuda / 'include'), '-c', str(HERE / 'bridge.cpp'), '-o', str(build / 'bridge.o')]
    commands.append(bridge)
    def compile_one(command):
        with (build / (Path(command[-1]).name + '.log')).open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    # Keep object output and logs outside the release and immutable input trees.
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(compile_one, commands))
    output = RELEASE / ('_dipper_native' + sysconfig.get_config_var('EXT_SUFFIX'))
    temporary = build / output.name
    link = ['g++', '-shared', '-fopenmp', *[str(build / (s + '.o')) for s in sources],
            str(build / 'bridge.o'), '-L' + str(cuda / 'lib64'), '-Wl,-rpath,' + str(cuda / 'lib64'),
            '-lcudart', '-ltbb', '-o', str(temporary)]
    subprocess.run(link, check=True)
    os.replace(temporary, output)
    (build / 'commands.json').write_text(json.dumps(commands + [link], indent=2) + '\n')
    print(output, flush=True)

if __name__ == '__main__':
    main()
