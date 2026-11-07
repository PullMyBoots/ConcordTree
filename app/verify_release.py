"""Verify the frozen release files without importing CUDA or writing outputs."""
from pathlib import Path
import hashlib
import json
import os
import sys

LOCAL_STATE = {'.conda', '.web-python', '.venv-web', '.runtime', '.standalone-migration', 'runs'}

def delivered_files(here):
    """Enumerate the delivered code/assets, without walking local environments."""
    for directory, dirs, files in os.walk(here):
        if Path(directory) == here:
            dirs[:] = [name for name in dirs if name not in LOCAL_STATE]
        dirs[:] = [name for name in dirs if name != '__pycache__']
        for name in files:
            path = Path(directory) / name
            if path.suffix != '.pyc' and name != 'MANIFEST.json':
                yield path

def main():
    here=Path(__file__).resolve().parent
    manifest=json.loads((here/'MANIFEST.json').read_text())
    failures=[]
    for relative,expected in manifest['files'].items():
        path=here/relative
        if not path.is_file():failures.append(relative+': missing');continue
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected['sha256']:
            failures.append(relative+': hash/type mismatch')
    actual={str(p.relative_to(here)) for p in delivered_files(here)}
    failures.extend(relative+': unlisted file' for relative in sorted(actual-set(manifest['files'])))
    if failures:
        print('\n'.join(failures),file=sys.stderr);return 1
    print(f"Verified ConcordTree {manifest['version']}: {len(manifest['files'])} files")
    return 0

if __name__=='__main__':sys.exit(main())
