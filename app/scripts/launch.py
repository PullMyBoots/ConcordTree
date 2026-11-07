"""Portable application entry; the default action starts the local Web UI."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import platform
import shutil
import socket
import stat
import subprocess
import sys
import threading
import time
import webbrowser

ROOT = Path(__file__).resolve().parents[1]


def preflight(gpu):
    if platform.system() != 'Linux' or platform.machine() != 'x86_64':
        raise RuntimeError('This download requires Linux x86_64. ARM builds require a separate artifact.')
    if not shutil.which('nvidia-smi'):
        raise RuntimeError('NVIDIA driver not detected. Install a compatible NVIDIA driver first.')
    if not (os.environ.get('CC') or shutil.which('gcc') or shutil.which('clang')):
        raise RuntimeError('A C compiler is required for first-run GPU kernels. Install gcc and libc development headers (Ubuntu/Debian: sudo apt install gcc libc6-dev), or use the container download.')
    driver = subprocess.check_output(['nvidia-smi', '-i', str(gpu),
        '--query-gpu=driver_version,name,compute_cap', '--format=csv,noheader'], text=True).strip()
    version, name, capability = [v.strip() for v in driver.split(',')]
    cc = tuple(int(v) for v in capability.split('.'))
    if cc < (7, 0):
        raise RuntimeError(f'{name} (compute capability {capability}) is below this runtime\'s CC 7.0 minimum.')
    if cc >= (10, 0):
        raise RuntimeError('This CUDA 12.1 runtime does not support Blackwell. A separately validated modern runtime is required.')
    if tuple(int(v) for v in version.split('.')) < (525, 60, 13):
        raise RuntimeError(f'NVIDIA driver {version} is too old for this CUDA 12.1 runtime; use >=525.60.13.')
    libc, libc_version = platform.libc_ver()
    if libc != 'glibc' or tuple(int(v) for v in libc_version.split('.')) < (2, 34):
        raise RuntimeError('The portable download requires glibc >=2.34. Use the container on older distributions.')
    return dict(name=name, driver=version, compute_capability=capability,
                machine=platform.machine(), libc=libc_version)


def open_when_ready(url):
    import urllib.request
    for _ in range(100):
        try:
            with urllib.request.urlopen(url + '/api/config', timeout=1):
                webbrowser.open(url)
                return
        except OSError:
            time.sleep(.2)


def compiler_paths():
    # Triton 3.1 invokes ptxas through a shell without quoting its paths.
    # A private alias keeps all actual cache data in the installation while
    # giving that unchanged compiler a path without whitespace/metacharacters.
    parent = Path('/tmp') / f'concordtree-paths-{os.getuid()}'
    parent.mkdir(mode=0o700, exist_ok=True)
    info = parent.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise RuntimeError(f'Expected a private, owned directory: {parent}')
    alias = parent / hashlib.sha256(os.fsencode(ROOT)).hexdigest()[:24]
    try:
        alias.symlink_to(ROOT, target_is_directory=True)
    except FileExistsError:
        if not alias.is_symlink() or alias.resolve() != ROOT:
            raise RuntimeError(f'Unexpected compiler path alias: {alias}')
    os.environ['TRITON_PTXAS_PATH'] = str(alias / '.conda/env/lib/python3.10/site-packages/triton/backends/nvidia/bin/ptxas')
    os.environ['CONCORDTREE_SAFE_TMPDIR'] = str(alias / '.runtime/cache/tmp')
    os.environ['TMPDIR'] = os.environ['CONCORDTREE_SAFE_TMPDIR']


def main():
    parser = argparse.ArgumentParser(description='ConcordTree: start the Web interface by default.')
    parser.add_argument('action', nargs='?', choices=['web', 'doctor', 'cli'], default='web')
    parser.add_argument('--gpu', type=int, default=int(os.environ.get('CONCORDTREE_GPU', '0')))
    parser.add_argument('--cpu-threads', type=int, default=int(os.environ.get('CONCORDTREE_CPU_THREADS', '16')))
    parser.add_argument('--port', type=int, default=int(os.environ.get('CONCORDTREE_PORT', '8765')))
    parser.add_argument('--no-browser', action='store_true')
    args, extra = parser.parse_known_args()
    if args.gpu < 0 or not 1 <= args.cpu_threads <= 256 or not 1 <= args.port <= 65535:
        parser.error('Invalid GPU, CPU thread count or TCP port.')
    try:
        hardware = preflight(args.gpu)
    except (RuntimeError, OSError, subprocess.SubprocessError, ValueError) as exc:
        parser.exit(2, f'ConcordTree: {exc}\n')
    threads = min(args.cpu_threads, len(os.sched_getaffinity(0)), 64)
    os.environ['CONCORDTREE_GPU'] = str(args.gpu)
    os.environ['CONCORDTREE_CPU_THREADS'] = str(threads)
    for key, tail in [('TRITON_CACHE_DIR', 'triton'), ('CUDA_CACHE_PATH', 'cuda'),
                      ('TMPDIR', 'tmp'), ('XDG_CACHE_HOME', 'xdg')]:
        folder = ROOT / '.runtime/cache' / tail
        folder.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(folder)
    compiler_paths()
    if args.action == 'doctor':
        print(json.dumps(hardware, indent=2), flush=True)
        os.execv('/usr/bin/bash', ['bash', str(ROOT / 'scripts/run_local.sh'),
                  'doctor', '--gpu', str(args.gpu), *extra])
    if args.action == 'cli':
        os.execv('/usr/bin/bash', ['bash', str(ROOT / 'scripts/run_local.sh'),
                  '--gpu', str(args.gpu), '--cpu-threads', str(threads), *extra])
    if extra:
        parser.error('Unknown Web options: ' + ' '.join(extra))
    host = os.environ.get('CONCORDTREE_BIND', '127.0.0.1')
    with socket.socket() as probe:
        try:
            probe.bind((host, args.port))
        except OSError:
            parser.exit(2, f'ConcordTree: port {args.port} is already in use; choose another with --port.\n')
    url = f'http://127.0.0.1:{args.port}'
    print(f'ConcordTree 1.0.2 | {hardware["name"]} | {threads} CPU threads\n'
          f'Open {url}\nJobs and results: {ROOT / "runs/web"}\nPress Ctrl+C to stop.', flush=True)
    if not args.no_browser and (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
        threading.Thread(target=open_when_ready, args=(url,), daemon=True).start()
    # runpy keeps the readiness/browser thread alive while the Web server runs.
    sys.argv = [str(ROOT / 'scripts/serve.py'), '--port', str(args.port), '--host', host]
    import runpy
    runpy.run_path(str(ROOT / 'scripts/serve.py'), run_name='__main__')


if __name__ == '__main__':
    main()
