"""Run matched, alternating baseline/package trials on one physical GPU."""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import os
import selectors
import statistics
import subprocess
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]


def atomic(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


class Worker:
    def __init__(self, name, app, gpu, threads, output, align_main_cpu=False):
        self.name = name
        self.log = (output / (name + '.log')).open('w')
        env = os.environ.copy()
        for key in ('PYTHONPATH', 'PYTHONHOME', 'LD_PRELOAD'):
            env.pop(key, None)
        packages = app / '.conda/env/lib/python3.10/site-packages'
        env['LD_LIBRARY_PATH'] = ':'.join(map(str, [packages / 'torch/lib', app / '.conda/env/lib', *packages.glob('nvidia/*/lib')]))
        env['PATH'] = str(app / '.conda/env/bin') + ':/usr/bin:/bin'
        env['PYTHONDONTWRITEBYTECODE'] = '1'
        command = [str(app / '.conda/env/bin/python'), '-I', '-B',
            str(ROOT / 'scripts/regression_worker.py'), '--root', str(app), '--gpu', str(gpu), '--threads', str(threads)]
        command += ['--align-main-cpu'] if align_main_cpu else []
        self.process = subprocess.Popen(command,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, text=True, bufsize=1, env=env)
        self.ready = self.read(180)
        if not self.ready.get('ready'):
            raise RuntimeError(f'{name} failed startup: {self.ready}')

    def read(self, timeout):
        # stdout contains only protocol lines; science diagnostics go to stderr.
        selector = selectors.DefaultSelector()
        selector.register(self.process.stdout, selectors.EVENT_READ)
        try:
            if not selector.select(timeout):
                raise TimeoutError(f'{self.name}: response exceeded {timeout}s')
            line = self.process.stdout.readline()
        finally:
            selector.close()
        if not line.startswith('CONCORDTREE_RESULT '):
            raise RuntimeError(f'{self.name} stopped: {line!r}; see worker log')
        result = json.loads(line.split(' ', 1)[1])
        if result.get('ok') is False:
            raise RuntimeError(f'{self.name}: {result}')
        return result

    def infer(self, case, run, relative):
        self.process.stdin.write(json.dumps(dict(case=case, run=run, relative_output=relative)) + '\n')
        self.process.stdin.flush()
        return self.read(3600)

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.write('{"stop":true}\n')
            self.process.stdin.flush()
            self.process.wait(timeout=30)
        self.log.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--threads', type=int, default=16)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--repeats', type=int)
    parser.add_argument('--run')
    parser.add_argument('--align-main-cpu', action='store_true')
    parser.add_argument('--package', type=Path, default=ROOT / 'app')
    parser.add_argument('--case-manifest', type=Path, default=ROOT / 'evidence/regression-cases.json')
    args = parser.parse_args()
    run = args.run or 'local-' + datetime.datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d-%H%M%S')
    output = ROOT / 'output' / run
    output.mkdir(parents=True, exist_ok=False)
    cases = json.loads(args.case_manifest.read_text())
    if args.limit:
        cases = cases[:args.limit]
    if args.repeats:
        for case in cases:
            case['repeats'] = args.repeats
    allowed = sorted(os.sched_getaffinity(0))
    # Fix both processes to the same CPU set. Record rather than infer NUMA performance.
    cpus = allowed[:args.threads]
    os.sched_setaffinity(0, cpus)
    atomic(output / 'protocol.json', dict(run=run, gpu=args.gpu, cpu_threads=args.threads,
        cpus=cpus, case_count=len(cases), cases=cases, warmup='one complete unmeasured run per case and artifact',
        measured_order='baseline/package alternates every repeat',
        accuracy_gate='byte-identical final and initial Newicks, unchanged RF',
        speed_investigation_threshold=1.05))
    if args.align_main_cpu:
        protocol = json.loads((output / 'protocol.json').read_text())
        protocol['main_thread_alignment'] = 'move both main threads to first allowed CPU before timing, then restore full affinity for parallel workers'
        atomic(output / 'protocol.json', protocol)
    workers = {}
    records = []
    status = dict(state='starting', run=run, cases=len(cases), completed_cases=0)
    atomic(output / 'status.json', status)
    try:
        for name, app in [('baseline', ROOT / 'baseline'), ('package', args.package.resolve())]:
            workers[name] = Worker(name, app, args.gpu, args.threads, output, args.align_main_cpu)
        atomic(output / 'startup.json', {k: v.ready for k, v in workers.items()})
        for index, case in enumerate(cases):
            print(f'CASE {index + 1}/{len(cases)} {case["id"]}', flush=True)
            status.update(state='running', current_case=case['id'], completed_cases=index)
            atomic(output / 'status.json', status)
            expected = None
            measured = {'baseline': [], 'package': []}
            for repeat in range(-1, case['repeats']):
                order = ['baseline', 'package'] if repeat % 2 == 0 else ['package', 'baseline']
                pair = {}
                for name in order:
                    relative = f'{index:03d}/' + ('warmup' if repeat == -1 else f'repeat-{repeat:02d}')
                    result = workers[name].infer(case, run, relative)
                    if expected is None:
                        expected = result['artifacts']
                    result['artifacts_match_first_run'] = result['artifacts'] == expected
                    records.append(dict(artifact=name, repeat=repeat, **result))
                    atomic(output / 'records.json', records)
                    pair[name] = result
                    if repeat >= 0:
                        measured[name].append(result['file_to_tree_seconds'])
                if pair['baseline']['artifacts'] != pair['package']['artifacts'] or pair['baseline']['accuracy'] != pair['package']['accuracy']:
                    raise RuntimeError(f'Accuracy regression: {case["id"]}, repeat={repeat}')
            base = statistics.median(measured['baseline'])
            packaged = statistics.median(measured['package'])
            row = dict(case=case['id'], group=case['group'], baseline_median_seconds=base,
                package_median_seconds=packaged, package_over_baseline=packaged / base,
                accuracy=pair['package']['accuracy'], repeats=case['repeats'], artifacts_match=True)
            status.setdefault('results', []).append(row)
            status.update(completed_cases=index + 1)
            atomic(output / 'status.json', status)
            print(json.dumps(row), flush=True)
        ratios = [r['package_over_baseline'] for r in status['results']]
        status.update(state='complete', median_paired_runtime_ratio=statistics.median(ratios),
            aggregate_median_runtime_ratio=sum(r['package_median_seconds'] for r in status['results']) /
                sum(r['baseline_median_seconds'] for r in status['results']),
            all_artifacts_match=True)
        atomic(output / 'status.json', status)
    except BaseException as exc:
        status.update(state='failed', error=str(exc))
        atomic(output / 'status.json', status)
        raise
    finally:
        for worker in workers.values():
            worker.close()


if __name__ == '__main__':
    main()
