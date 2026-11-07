"""Refresh distribution hashes and audit every change from the frozen release."""
from pathlib import Path
import hashlib
import json
import os

ROOT = Path(__file__).resolve().parents[1]
app = ROOT / 'app'
excluded = {'.conda', '.web-python', '.venv-web', '.runtime', '.standalone-migration', 'runs'}
files = {}
for base, dirs, names in os.walk(app):
    dirs[:] = [d for d in dirs if d != '__pycache__' and not (Path(base) == app and d in excluded)]
    for name in names:
        p = Path(base) / name
        if name == 'MANIFEST.json' or p.suffix == '.pyc':
            continue
        files[str(p.relative_to(app))] = dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(), bytes=p.stat().st_size)
original = json.loads((ROOT / 'evidence/source-manifest.json').read_text())
changes = [p for p, value in original['files'].items() if p not in files or value['sha256'] != files[p]['sha256']]
allowed = {'web/server.py', 'runtime_config.py', 'scripts/start_web.sh', 'scripts/run_local.sh', 'verify_release.py', 'README.md',
           '_ct_apnj_packed.cpython-310-x86_64-linux-gnu.so'}
rebuild = json.loads((ROOT / 'evidence/sequence-rebuild.json').read_text())
asset_config = 'runtime/gene/src/concordtree/assets.py'
normalized_config = (app / asset_config).read_bytes().replace(
    rebuild['new_sha256'].encode(), rebuild['old_sha256'].encode())
assert hashlib.sha256(normalized_config).hexdigest() == original['files'][asset_config]['sha256']
assert files[rebuild['binary']]['sha256'] == rebuild['new_sha256']
allowed.update({asset_config, rebuild['binary']})
unexpected = set(changes) - allowed
if unexpected:
    raise RuntimeError(f'Unexpected source changes: {sorted(unexpected)}')
weights = {p: v['sha256'] for p, v in files.items() if p.startswith('assets/') and p.endswith(('.pt', '.pth', '.npz'))}
for p, digest in weights.items():
    assert original['files'][p]['sha256'] == digest, p
manifest = dict(version='1.0.2', distribution_revision='linux-r1', files=files)
(app / 'MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
(ROOT / 'evidence/source-audit.json').write_text(json.dumps(dict(
    original_file_count=len(original['files']), distribution_file_count=len(files),
    changed_original_files=changes, algorithm_sources_unchanged=True,
    weights_unchanged=True, weights=weights,
    change_scope='launch paths, Web CPU/GPU selection and container binding, integrity manifest, AP-NJ GPU architecture coverage, identical-source sequence parser rebuild for glibc 2.34, distribution documentation'), indent=2) + '\n')
print('Manifest refreshed;', len(files), 'files;', len(changes), 'audited original-file changes')
