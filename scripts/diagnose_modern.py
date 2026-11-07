"""Measure a candidate runtime without weakening the stable artifact gate.

Run under the frozen scientific Python for its existing ete3 parser. Candidate
and baseline inference remain isolated in their own persistent processes.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_regression import ROOT, Worker, atomic


def tree_data(path):
    from ete3 import Tree
    text = re.sub(r'\[[^\]]*\]', '', path.read_text()).strip()
    tree = Tree(text, format=1)
    leaves = frozenset(tree.get_leaf_names())
    if len(leaves) != len(tree):
        raise ValueError(f'Duplicate taxa: {path}')
    descendants, branches, supports, internal = {}, {}, {}, set()
    for node in tree.traverse('postorder'):
        subset = frozenset([node.name]) if node.is_leaf() else frozenset().union(*(descendants[c] for c in node.children))
        descendants[node] = subset
        if node.is_root():
            continue
        other = leaves - subset
        split = min(tuple(sorted(subset)), tuple(sorted(other)), key=lambda s: (len(s), s))
        branches[split] = branches.get(split, 0.) + node.dist
        if len(subset) >= 2 and len(other) >= 2:
            internal.add(split)
            if node.name:
                supports.setdefault(split, []).append(float(node.name))
    return leaves, internal, branches, {k: sorted(v) for k, v in supports.items()}


def compare_trees(baseline, candidate):
    left, right = baseline['artifacts'], candidate['artifacts']
    rows = {}
    for name in sorted(left.keys() & right.keys()):
        a = tree_data(Path(baseline['output']) / name)
        b = tree_data(Path(candidate['output']) / name)
        shared = a[2].keys() & b[2].keys()
        support_shared = a[3].keys() & b[3].keys()
        support_errors = [abs(x-y) for key in support_shared
                          for x, y in zip(a[3][key], b[3][key])]
        rows[name] = dict(
            byte_identical=left[name] == right[name],
            taxa_identical=a[0] == b[0], topology_identical=a[:2] == b[:2],
            tree_to_tree_rf=len(a[1] ^ b[1]) if a[0] == b[0] else None,
            branch_keys_identical=a[2].keys() == b[2].keys(),
            max_branch_absolute_delta=max((abs(a[2][key]-b[2][key]) for key in shared), default=0.),
            support_identical=a[3] == b[3],
            max_shared_support_absolute_delta=max(support_errors, default=0.))
    return dict(file_sets_identical=left.keys() == right.keys(),
                all_bytes_identical=left == right,
                all_topologies_identical=left.keys() == right.keys() and all(x['topology_identical'] for x in rows.values()),
                reference_accuracy_identical=baseline['accuracy'] == candidate['accuracy'], trees=rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--candidate', type=Path, default=ROOT/'app-modern')
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--threads', type=int, default=16)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    output = ROOT/'output'/args.run
    output.mkdir(parents=True, exist_ok=False)
    cases = json.loads((ROOT/'evidence/regression-cases.json').read_text())
    if args.limit:
        cases = cases[:args.limit]
    cpus = sorted(os.sched_getaffinity(0))[:args.threads]
    os.sched_setaffinity(0, cpus)
    protocol = dict(run=args.run, candidate=str(args.candidate.resolve()), gpu=args.gpu,
                    cpu_threads=args.threads, cpus=cpus, cases=cases,
                    warmup='one complete unmeasured run per case and runtime',
                    measured_order='alternating paired repeats, original 3 simulated / 1 empirical',
                    main_thread_alignment=True, admission='diagnostic only; stable strict byte gate unchanged',
                    input_hashes_verified=True)
    for case in cases:
        for field in ['source', 'truth', 'loci']:
            if case.get(field) and case.get(field+'_sha256'):
                assert hashlib.sha256(Path(case[field]).read_bytes()).hexdigest() == case[field+'_sha256'], (case['id'], field)
    atomic(output/'protocol.json', protocol)
    workers, records = {}, []
    status = dict(state='starting', cases=len(cases), completed_cases=0, results=[])
    atomic(output/'status.json', status)
    try:
        for name, app in [('baseline', ROOT/'baseline'), ('candidate', args.candidate.resolve())]:
            workers[name] = Worker(name, app, args.gpu, args.threads, output, align_main_cpu=True)
        atomic(output/'startup.json', {k: v.ready for k, v in workers.items()})
        for index, case in enumerate(cases):
            print(f'CASE {index+1}/{len(cases)} {case["id"]}', flush=True)
            status.update(state='running', current_case=case['id'])
            atomic(output/'status.json', status)
            measured = {'baseline': [], 'candidate': []}
            comparisons, first, deterministic = [], {}, {'baseline': True, 'candidate': True}
            for repeat in range(-1, case['repeats']):
                order = ['baseline', 'candidate'] if repeat % 2 == 0 else ['candidate', 'baseline']
                pair = {}
                for name in order:
                    relative = f'{index:03d}/' + ('warmup' if repeat == -1 else f'repeat-{repeat:02d}')
                    result = workers[name].infer(case, args.run, relative)
                    first.setdefault(name, result['artifacts'])
                    deterministic[name] &= result['artifacts'] == first[name]
                    pair[name] = result
                    if repeat >= 0:
                        measured[name].append(result['file_to_tree_seconds'])
                comparison = compare_trees(pair['baseline'], pair['candidate'])
                comparisons.append(comparison)
                records.append(dict(case=case['id'], repeat=repeat, runtimes=pair, comparison=comparison))
                atomic(output/'records.json', records)
            base = statistics.median(measured['baseline'])
            candidate = statistics.median(measured['candidate'])
            trees = [tree for comparison in comparisons for tree in comparison['trees'].values()]
            row = dict(case=case['id'], group=case['group'], repeats=case['repeats'],
                       baseline_median_seconds=base, candidate_median_seconds=candidate,
                       candidate_over_baseline=candidate/base,
                       all_bytes_identical=all(x['all_bytes_identical'] for x in comparisons),
                       all_topologies_identical=all(x['all_topologies_identical'] for x in comparisons),
                       reference_accuracy_identical=all(x['reference_accuracy_identical'] for x in comparisons),
                       all_support_identical=all(x['support_identical'] for x in trees),
                       max_branch_absolute_delta=max(x['max_branch_absolute_delta'] for x in trees),
                       max_shared_support_absolute_delta=max(x['max_shared_support_absolute_delta'] for x in trees),
                       deterministic=deterministic)
            status['results'].append(row)
            status.update(completed_cases=index+1)
            atomic(output/'status.json', status)
            print(json.dumps(row), flush=True)
        status.update(state='complete',
                      all_bytes_identical=all(x['all_bytes_identical'] for x in status['results']),
                      all_topologies_identical=all(x['all_topologies_identical'] for x in status['results']),
                      all_reference_accuracy_identical=all(x['reference_accuracy_identical'] for x in status['results']),
                      median_case_runtime_ratio=statistics.median(x['candidate_over_baseline'] for x in status['results']),
                      aggregate_runtime_ratio=sum(x['candidate_median_seconds'] for x in status['results']) /
                                              sum(x['baseline_median_seconds'] for x in status['results']))
        atomic(output/'status.json', status)
    except BaseException as exc:
        status.update(state='failed', error=str(exc))
        atomic(output/'status.json', status)
        raise
    finally:
        for worker in workers.values():
            worker.close()


if __name__ == '__main__':
    main()
