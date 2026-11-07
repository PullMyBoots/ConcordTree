"""A persistent, isolated science process controlled through JSON lines."""
from pathlib import Path
import argparse
import contextlib
import ctypes
import hashlib
import json
import os
import re
import sys
import time
import traceback

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, required=True)
p.add_argument('--gpu', type=int, default=1)
p.add_argument('--threads', type=int, default=16)
p.add_argument('--align-main-cpu', action='store_true')
a = p.parse_args()
root = a.root.resolve()
cpu_affinity = sorted(os.sched_getaffinity(0))
getcpu = ctypes.CDLL(None).sched_getcpu
sys.path.insert(0, str(root))


def emit(data):
    print('CONCORDTREE_RESULT ' + json.dumps(data, allow_nan=False), flush=True)


def topology(text):
    from ete3 import Tree
    text = re.sub(r'\[[^\]]*\]', '', text).strip().split(';')[0] + ';'
    tree = Tree(text, format=1)
    leaves = set(tree.get_leaf_names())
    if len(leaves) != len(tree):
        raise ValueError('Duplicate taxon names in tree')
    splits = set()
    descendants = {}
    for node in tree.traverse('postorder'):
        subset = {node.name} if node.is_leaf() else set().union(*(descendants[c] for c in node.children))
        descendants[node] = subset
        other = leaves - subset
        if len(subset) >= 2 and len(other) >= 2:
            splits.add(min(tuple(sorted(subset)), tuple(sorted(other)), key=lambda v: (len(v), v)))
    return leaves, splits


try:
    started = time.perf_counter()
    with contextlib.redirect_stdout(sys.stderr):
        from engine import UnifiedEngine
        engine = UnifiedEngine(gpu=a.gpu, cpu_threads=a.threads)
    provenance = {}
    for file in [root / '_ct_apnj_packed.cpython-310-x86_64-linux-gnu.so',
                 root / 'runtime/gene/src/concordtree/assets.py',
                 *sorted((root / 'runtime/gene/src/concordtree/assets/backends').glob('*.so')),
                 *sorted((root / 'assets').rglob('*.pt'))]:
        provenance[str(file.relative_to(root))] = hashlib.sha256(file.read_bytes()).hexdigest()
    emit({'ready': True, 'startup_seconds': time.perf_counter() - started,
          'artifact_sha256': provenance})
    for line in sys.stdin:
        message = json.loads(line)
        if message.get('stop'):
            break
        case = message['case']
        out = root / 'runs/regression' / message['run'] / message['relative_output']
        options = dict(task=case['task'], branch_support=case.get('support', True))
        if case.get('loci'):
            options['loci'] = case['loci']
        if case.get('model'):
            options['model'] = case['model']
        if a.align_main_cpu:
            os.sched_setaffinity(0, {cpu_affinity[0]})
            os.sched_setaffinity(0, cpu_affinity)
        main_cpu_start = getcpu()
        started = time.perf_counter()
        with contextlib.redirect_stdout(sys.stderr):
            if case.get('profile'):
                import cProfile
                profiler = cProfile.Profile()
                profiler.enable()
                try:
                    engine.infer(case['source'], out, **options)
                finally:
                    profiler.disable()
                    profiler.dump_stats(str(out / 'profile.pstats'))
            else:
                engine.infer(case['source'], out, **options)
        wall = time.perf_counter() - started
        main_cpu_end = getcpu()
        result = json.loads((out / 'RESULT.json').read_text())
        assert result['status'] == 'complete', result
        artifacts = {}
        for pattern in ('prediction*.nwk', 'initial/*.nwk', 'gene/prediction.nwk'):
            for file in out.glob(pattern):
                artifacts[str(file.relative_to(out))] = hashlib.sha256(file.read_bytes()).hexdigest()
        predicted = (out / 'prediction.nwk').read_text()
        leaves, splits = topology(predicted)
        accuracy = {'taxa': len(leaves), 'splits': len(splits)}
        if case.get('truth'):
            try:
                ref_leaves, ref_splits = topology(Path(case['truth']).read_text())
                if leaves != ref_leaves:
                    raise ValueError(f'Reference leaf mismatch: predicted {len(leaves)}, reference {len(ref_leaves)}, common {len(leaves & ref_leaves)}')
                rf = len(splits ^ ref_splits)
                maximum = len(splits) + len(ref_splits)
                accuracy.update(rf=rf, max_rf=maximum, normalized_rf=rf / maximum if maximum else 0.)
            except Exception as exc:
                accuracy['reference_error'] = str(exc)
        emit(dict(ok=True, case=case['id'], output=str(out), artifacts=artifacts,
                  main_cpu_start=main_cpu_start, main_cpu_end=main_cpu_end,
                  file_to_tree_seconds=result['file_to_tree_seconds'], wall_seconds=wall,
                  accuracy=accuracy, quality=result.get('quality')))
except BaseException as exc:
    traceback.print_exc(file=sys.stderr)
    emit({'ok': False, 'error': str(exc)})
    raise
