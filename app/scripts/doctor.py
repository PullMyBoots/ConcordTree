"""Check V1.0.2's local environment and loaded runtime file origins."""
from pathlib import Path
import argparse
import importlib
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--output', type=Path)
args = parser.parse_args()
if Path(sys.prefix).resolve() != ROOT / '.conda/env':
    parser.error('use this release\'s .conda/env/bin/python')

from engine import UnifiedEngine
engine = UnifiedEngine(gpu=args.gpu, cpu_threads=4)
checks = {task: model.validate_features() for task, model in engine.predictors.items()}
for name in ('_ct_apnj_graph', '_ct_apnj_local', '_ct_apnj_terminal',
             '_ct_apnj_topology', '_ct_apnj_packed'):
    importlib.import_module(name)
modules = {}
for name in ('torch', 'triton', 'numpy', 'scipy', 'ete3', 'concordtree',
             'taxon_placement', 'msc_refine.barycenter_tree_tiled'):
    module = importlib.import_module(name)
    modules[name] = dict(file=str(Path(module.__file__).resolve()),
                         version=getattr(module, '__version__', None))
outside = []
for name, module in list(sys.modules.items()):
    filename = getattr(module, '__file__', None)
    if filename and Path(filename).is_absolute():
        path = Path(filename).resolve()
        if str(path).startswith('/home/') and not path.is_relative_to(ROOT):
            outside.append(dict(module=name, file=str(path)))
libraries = sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                    if line.split()[-1].startswith('/')})
outside_libraries = [path for path in libraries
                     if path.startswith('/home/') and not Path(path).is_relative_to(ROOT)]
outside_links = []
for directory, dirs, files in os.walk(ROOT):
    for name in dirs + files:
        path = Path(directory) / name
        if path.is_symlink() and not path.resolve().is_relative_to(ROOT):
            outside_links.append(dict(file=str(path), target=str(path.resolve())))
report = dict(version='1.0.2', python=sys.executable, prefix=sys.prefix,
              modules=modules, model_feature_checks=checks, runtime=engine.runtime,
              external_home_modules=outside, external_home_libraries=outside_libraries,
              external_symlinks=outside_links, loaded_library_files=libraries,
              status='ok' if not (outside or outside_libraries or outside_links) else 'failed')
text = json.dumps(report, indent=2) + '\n'
if args.output:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text)
print(text)
raise SystemExit(report['status'] != 'ok')
