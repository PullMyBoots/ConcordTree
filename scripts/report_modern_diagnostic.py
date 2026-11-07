"""Export candidate diagnostics while preserving the stable admission boundary."""
from pathlib import Path
import argparse
import json

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument('--run', default='modern-diagnostic-48-v1')
a = p.parse_args()
status = json.loads((ROOT/'output'/a.run/'status.json').read_text())
protocol = json.loads((ROOT/'output'/a.run/'protocol.json').read_text())
records = json.loads((ROOT/'output'/a.run/'records.json').read_text())
provenance = json.loads((ROOT/'evidence/modern-runtime-provenance.json').read_text())
compilation = json.loads((ROOT/'evidence/modern-dispatch-compilation.json').read_text())
assert status['state'] in ['complete', 'failed'], status['state']
assert provenance['weights_identical'] is True and len(provenance['weight_sha256']) == 4
assert len(compilation['results']) == 32 and all(x['status'] == 'compiled' for x in compilation['results'])
rows = status['results']
trials = []
latest = {}
for record in records:
    pair = record['runtimes']
    latest[record['case']] = pair
    trials.append(dict(case=record['case'], repeat=record['repeat'],
                       baseline_seconds=pair['baseline']['file_to_tree_seconds'],
                       candidate_seconds=pair['candidate']['file_to_tree_seconds'],
                       comparison=record['comparison']))
for row in rows:
    row['baseline_accuracy'] = latest[row['case']]['baseline']['accuracy']
    row['candidate_accuracy'] = latest[row['case']]['candidate']['accuracy']
    assert 'reference_error' not in row['baseline_accuracy']
    assert 'reference_error' not in row['candidate_accuracy']
summary = dict(run=a.run, status=status, stable_admission='not_admitted',
               hardware='NVIDIA L40S', blackwell_hardware_executed=False,
               modern_provenance=provenance, compilation=compilation, trials=trials,
               measurement=dict(gpu=protocol['gpu'], cpu_threads=protocol['cpu_threads'],
                                cpus=protocol['cpus'], warmup=protocol['warmup'],
                                measured_order=protocol['measured_order'], main_thread_alignment=True),
               inputs=[{k:v for k,v in case.items() if k not in ['source', 'truth', 'loci']}
                       for case in protocol['cases']])
(ROOT/'dist/modern-regression-results.json').write_text(json.dumps(summary, indent=2)+'\n')
topology = sum(x['all_topologies_identical'] for x in rows)
rf = sum(x['reference_accuracy_identical'] for x in rows)
byte = sum(x['all_bytes_identical'] for x in rows)
slower = [x['case'] for x in rows if x['candidate_over_baseline'] > 1.05]
lines = ['# Experimental modern runtime diagnostic', '',
         '**Not admitted to the stable distribution. No Blackwell hardware execution has been performed.**', '',
         f"Run `{a.run}`: **{status['state']}**, {status['completed_cases']}/{status['cases']} cases completed. The stable scientific runtime, archive and image remain unchanged.", '',
         f"Candidate: Torch {provenance['torch']}, CUDA user libraries {provenance['torch_cuda']}, Triton {provenance['triton']}, NumPy {provenance['numpy']}, SciPy {provenance['scipy']}. All four MLP/MSC weight/configuration files match the stable artifact SHA256. Source changes are the pointer-type constexpr API adaptation, an identical-source C++ ABI backend rebuild/hash, and a startup temporary-alias difference. See the JSON provenance for exact hashes.", '',
         '## Evidence contract', '',
         'This diagnostic continues after byte differences to locate their extent. It does not relax the stable byte-identity gate. Unrooted bipartitions are compared directly for every initial/final Newick, independently of reference RF. Branch lengths are compared on shared canonical edge splits; two edges introduced by a binary root are summed. Support labels are compared by the corresponding split. Reordered child lists alone do not count as a topology change. A separate comparator check covers identical/reordered trees, branch/support differences and different topologies with equal reference RF. The JSON includes actual per-case reference accuracy, per-trial timings and per-tree comparisons.', '',
         'The original 48-case manifest and input hashes are retained. Baseline and candidate run alternately on one L40S with 16 CPU threads and the same CPU affinity. Each case receives an unmeasured complete warm-up; simulated cases have three measured pairs and empirical cases one. Both main threads start on the same CPU before restoring full affinity. Warm file-to-tree timings exclude startup and first JIT compilation. Cases outside 5% are observations requiring investigation, not an admission test passed by averaging.', '',
         '## Results', '',
         f'Among completed cases: {topology}/{len(rows)} have identical initial/final topology sets; {rf}/{len(rows)} have identical reference accuracy measurements; {byte}/{len(rows)} have byte-identical initial/final artifacts.', '']
if status['state'] == 'complete':
    lines += [f"Aggregate warm runtime ratio: **{status['aggregate_runtime_ratio']:.6f}**; median case ratio: **{status['median_case_runtime_ratio']:.6f}** (candidate / frozen baseline).", '']
if status.get('error'):
    lines += [f"Terminal error: `{status['error']}`. No full-suite claim is supported.", '']
slower_label = 'case is' if len(slower) == 1 else 'cases are'
lines += [f'{len(slower)} completed {slower_label} more than 5% slower in this run. Raw measurements, including all mismatches, are retained. These measurements do not certify Blackwell speed or correctness.', '',
          '| Case | Topology same | Reference metrics same | Bytes same | Max branch delta | Max shared support delta | Runtime ratio |',
          '| --- | --- | --- | --- | ---: | ---: | ---: |']
for row in rows:
    lines.append(f"| {row['case']} | {row['all_topologies_identical']} | {row['reference_accuracy_identical']} | {row['all_bytes_identical']} | {row['max_branch_absolute_delta']:.6g} | {row['max_shared_support_absolute_delta']:.6g} | {row['candidate_over_baseline']:.4f} |")
lines += ['', '## Compilation and remaining boundary', '',
          'The four representative active likelihood/count/species Triton kernels, plus the adapted downward-dispatch kernel in fp32/fp64 with int32/int64 indices, compile for SM100/103/120/121: 32 successful variant/architecture combinations. The original 16 PTX hashes remain identical when adding the dispatch checks. This is compilation coverage, not exhaustive kernel coverage or execution on those GPUs. The actual candidate execution device is L40S. The native AP-NJ backend carries forward PTX70 and the rebuilt Torch pattern backend carries PTX90. PyTorch compiled architecture flags are recorded in the JSON. [NVIDIA explains PTX forward compatibility and the required execution check](https://docs.nvidia.com/cuda/archive/12.8.0/blackwell-compatibility-guide/index.html).', '',
          'A modern artifact cannot be advertised as preserving the stable output or as validated on every Linux/NVIDIA machine from these diagnostics. Differences and missing hardware evidence remain explicit. The supported stable artifact continues to require Linux x86_64 and CC7.x–9.x; Blackwell/ARM are not admitted.', '']
(ROOT/'dist/MODERN_VALIDATION.md').write_text('\n'.join(lines))
print(json.dumps(dict(state=status['state'], completed=len(rows), topology_identical=topology,
                      reference_identical=rf, bytes_identical=byte, slower_cases=slower), indent=2))
