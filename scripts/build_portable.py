"""Build a relocatable, checksummed application archive; never include local jobs."""
from pathlib import Path
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / 'dist'
DIST.mkdir(exist_ok=True)
name = 'concordtree-1.0.2-linux-x86_64-cuda121'
archive = DIST / (name + '.tar.gz')
temporary = archive.with_suffix('.partial')
subprocess.run(['python3', str(ROOT / 'scripts/refresh_manifest.py')], check=True)
subprocess.run(['tar', '--sort=name', '--mtime=2026-11-07 00:00:00Z',
    '--owner=0', '--group=0', '--numeric-owner',
    '--exclude=app/runs', '--exclude=app/.runtime', '--exclude=__pycache__',
    '--exclude=*.pyc', '--exclude=app/.standalone-migration',
    '--transform=s,^app,' + name + ',',
    '-I', 'pigz -p 4 -6', '-cf', str(temporary), '-C', str(ROOT), 'app'], check=True)
temporary.replace(archive)
h = hashlib.sha256()
with archive.open('rb') as f:
    for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
        h.update(chunk)
(DIST / (archive.name + '.sha256')).write_text(h.hexdigest() + '  ' + archive.name + '\n')
(ROOT / 'evidence/portable-artifact.json').write_text(json.dumps(dict(
    file=archive.name, bytes=archive.stat().st_size, sha256=h.hexdigest(),
    excludes=['runs', '.runtime', '__pycache__', '*.pyc', '.standalone-migration']), indent=2) + '\n')
print(archive)
subprocess.run(['split', '-b', '1024M', '-d', '-a', '3', str(archive), str(archive) + '.part-'], check=True)
lines = []
for part in sorted(DIST.glob(archive.name + '.part-*')):
    h = hashlib.sha256()
    with part.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    lines.append(h.hexdigest() + '  ' + part.name)
(DIST / 'SHA256SUMS').write_text('\n'.join(lines) + '\n')
