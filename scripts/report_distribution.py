"""Summarize completed distribution checks without hiding failed candidates."""
from pathlib import Path
import hashlib
import json
import statistics
import subprocess

ROOT=Path(__file__).resolve().parents[1]
def read(relative): return json.loads((ROOT/relative).read_text())
primary=read('output/paired-48-v2/status.json')
followup=read('output/speed-followup-v1/status.json')
aligned=read('output/speed-followup-aligned-v1/status.json')
assert primary['state']==followup['state']==aligned['state']=='complete'
records=read('output/paired-48-v2/records.json')
assert all(x['artifacts_match_first_run'] for x in records)
assert primary['all_artifacts_match'] and followup['all_artifacts_match']
artifact=read('evidence/portable-artifact.json')
web={name:read('evidence/'+name+'/report.json') for name in
     ['container-final-web','installed-final-web','workstation-web','relocated-web-v3']}
assert all(x['status']=='ok' for x in web.values())
assert read('evidence/container/doctor-final.json')['status']=='ok'
assert read('evidence/workstation-doctor.json')['status']=='ok'
assert all(x['topology_identical'] for x in read('evidence/workstation-real-data-equality.json'))
assert all(x['status']=='compiled' for x in read('evidence/architecture-compilation.json')['results'])
speed_ok=all(x['package_over_baseline']<=1.05 for x in aligned['results'])
routing=read('evidence/installer-routing.json')
installer_ok=routing['status']=='passed' and all(x['status']=='passed' for x in routing['checks'].values())
for name, expected in routing['launcher_sha256'].items():
    for path in (ROOT/name, ROOT/'dist'/name):
        assert hashlib.sha256(path.read_bytes()).hexdigest()==expected, path
summary=dict(primary=primary,followup=followup,aligned_followup=aligned,
             isolated_49=read('output/speed-isolation-v1/status.json'),
             main_cpu_aligned_49=read('output/speed-core-aligned-v1/status.json'),artifact=artifact,
             cross_device=read('evidence/workstation-real-data-equality.json'),
             modern_candidate=read('evidence/modern-admission.json'),
             installer_routing=routing)
# RESULT fields can include server-local input paths. Release the summary rows,
# input hashes and full accuracy measurements without embedding local paths.
summary['cross_device']=[{k:v for k,v in row.items() if k!='result'} for row in summary['cross_device']]
modern_diagnostic=None
if (ROOT/'dist/modern-regression-results.json').exists():
    modern_diagnostic=read('dist/modern-regression-results.json')
    assert modern_diagnostic['stable_admission']=='not_admitted'
    diagnostic_status=modern_diagnostic['status']
    summary['modern_diagnostic']=dict(
        run=modern_diagnostic['run'],state=diagnostic_status['state'],
        completed_cases=diagnostic_status['completed_cases'],cases=diagnostic_status['cases'],
        topology_identical_cases=sum(x['all_topologies_identical'] for x in diagnostic_status['results']),
        reference_identical_cases=sum(x['reference_accuracy_identical'] for x in diagnostic_status['results']),
        byte_identical_cases=sum(x['all_bytes_identical'] for x in diagnostic_status['results']),
        stable_admission='not_admitted',blackwell_hardware_executed=False,
        details='modern-regression-results.json')
(ROOT/'dist/regression-results.json').write_text(json.dumps(summary,indent=2)+'\n')
trial_runs={}
for run in ['paired-48-v2','speed-followup-v1','speed-isolation-v1',
            'speed-core-aligned-v1','speed-followup-aligned-v1']:
    protocol=read('output/'+run+'/protocol.json')
    run_records=read('output/'+run+'/records.json')
    trial_runs[run]=dict(
        protocol={k:v for k,v in protocol.items() if k!='cases'},
        inputs=[{k:v for k,v in case.items() if k not in ['source','truth','loci']}
                for case in protocol['cases']],
        trials=[{k:v for k,v in record.items() if k in {
            'artifact','repeat','case','artifacts','accuracy','file_to_tree_seconds',
            'wall_seconds','main_cpu_start','main_cpu_end','artifacts_match_first_run'}}
            for record in run_records])
(ROOT/'dist/regression-trials.json').write_text(json.dumps(trial_runs,indent=2,allow_nan=False)+'\n')
lines=['# ConcordTree 1.0.2 Linux distribution validation','',
    '## Artifacts','',f"Portable archive: `{artifact['file']}` ({artifact['bytes']:,} bytes).",
    f"SHA256: `{artifact['sha256']}`.",'',
    'The portable archive and container include the scientific and Web runtimes, weights and native libraries. The unified installer automatically selects portable when glibc/compiler prerequisites are met, otherwise an existing accessible Docker installation. It verifies five ordered parts before extraction/import. Portable extraction additionally verifies 1121 static files; Docker import checks the exact image ID. Outputs and caches are excluded from release archives.','',
    '## Matched regression','',
    'Frozen V1.0.2 vs final CUDA 12.1 distribution, same L40S GPU, CPU affinity 0–15, 16 threads. 48 inputs: 10 gene simulations (256–4096 taxa), 28 species configurations (32–1024 taxa, observed loci and MSA-only), and all 10 retained empirical inputs. Each artifact receives a complete unmeasured warm-up per case. Simulations have three alternating paired measurements; empirical cases have one pair after warm-up. Inputs and references were hashed and read without modification.','',
    '**48/48 initial and final Newick sets match byte for byte, including branches and support labels; all reference RF values match. Repeated outputs are deterministic for these runs.** Empirical reference trees are comparison references, not established biological truth.','',
    f"Primary median of case runtime ratios: **{primary['median_paired_runtime_ratio']:.4f}**; ratio of summed per-case medians: **{primary['aggregate_median_runtime_ratio']:.4f}** (package / baseline; lower is faster).",'',
    'All simulated cases differing by more than 5% in either direction receive a separately reported nine-repeat follow-up. Primary observations remain in the report. CPU work for packaging and compiler probes was assigned separate affinity sets; some early primary measurements overlapped environment/build preparation. The follow-up runs after the main build work. A 5% threshold is an investigation rule, not proof of universal performance equivalence.','',
    'The nine-repeat follow-up retained an 8.17% (12.8 ms) delay on one 0.16-second case. A fresh 49-repeat run measured 6.12% (9.0 ms). Profiling showed most difference outside construction and scoring, in CPU input processing; unchanged Python regex/loop work also differed. Aligning both main threads to the same starting CPU before timing, then restoring all 16 allowed CPUs for parallel workers, reduced the 49-repeat delay to 1.73% (2.5 ms). This supports sensitivity to CPU scheduling/placement, not a guarantee that every future run will stay below 5%. All seven flagged cases then receive 17 paired repeats with this additional control. Earlier measurements are retained.','',
    '| Case | Primary ratio | Nine repeats | Aligned 17 repeats |','| --- | ---: | ---: | ---: |']
follow_by={x['case']:x for x in followup['results']}
aligned_by={x['case']:x for x in aligned['results']}
for row in primary['results']:
    f=follow_by.get(row['case'])
    aligned_row=aligned_by.get(row['case'])
    lines.append(f"| {row['case']} | {row['package_over_baseline']:.4f} | {f['package_over_baseline']:.4f} | {aligned_row['package_over_baseline']:.4f} |" if f else f"| {row['case']} | {row['package_over_baseline']:.4f} | — | — |")
lines += ['',f"Follow-up investigation gate (no case >5% slower): **{'passed' if speed_ok else 'unresolved'}**.",'',
    'Warm timings use persistent scientific processes, exclude process/model startup and initial JIT compilation, and measure file-to-tree inference. Web tests exercise separate scientific processes per job, so their turnaround includes upload validation, queueing, startup, model loading and kernel compilation. First-run latency must not be described as warm inference time. No statement here guarantees equal wall time on different GPUs.','',
    '## Relocation, UI and hardware','',
    '- Unified installer: automatic portable selection and reuse passed. A fresh-cache localhost HTTP installation downloaded and verified all five portable parts, extracted atomically and verified 1121 static files. A clean Ubuntu 22.04 CUDA base container without a C compiler selected Docker successfully using ordinary UID/GID and an existing host Docker daemon. Explicit Docker installation verified all five parts and streamed image import using a fixed installer snapshot. A fresh-cache localhost HTTP Docker installation downloaded and verified the launcher, reusing the exact image without downloading its parts again.',
    '- Latest Docker launcher: GPU index 1, four CPU threads, ordinary UID/GID, persistent state, custom loopback port 8882 mapped identically inside/outside. Real Gene and Species upload/inference/download succeeded and both trees matched the final container outputs byte for byte; invalid upload returned 422. Public GitHub download URLs have not yet been exercised. Earlier shell-edit-during-test failure is retained as an invalidated attempt; the fixed-snapshot rerun passed.',
    '- Complete final archive installed from verified parts into a new local directory; all static hashes verified. A directory containing spaces also completed actual Gene and Species Web inference after startup-path adaptation.',
    '- Final container: Ubuntu 22.04 userspace, NVIDIA L40S, ordinary UID/GID, persistent results/cache mounts; actual Gene/Species upload, inference, download and invalid input rejection passed. Image save/load passed.',
    '- Workstation: Ubuntu 22.04 WSL, RTX 4090, driver 610.88. Archive copied to `D:\\serve-backup\\ConcordTree\\distribution-test`, SHA256 verified, extracted independently. Environment doctor, two Web examples, and two existing regression inputs completed.',
    '- Across L40S and RTX 4090 the two retained regression inputs have identical topology (tree-to-tree RF=0). Species output is byte identical; Gene branch lengths differ by at most 1.425e-8. Cross-device byte identity is not claimed.',
    '- Browser checks cover real upload, both task types, preview, download, refresh recovery and mobile rendering. The original Web test suite has 26 passing tests.',
    '- SM70/75/80/86/89/90 native fatbinary coverage and four representative active Triton kernels compile on every listed architecture. This is build evidence, not full hardware correctness/performance certification. L40S and RTX 4090 are the actual tested devices.','',
    '## Support boundary','',
    'Portable: Linux x86_64, glibc >=2.34, driver >=525.60.13, compute capability 7.x–9.x; GCC/Clang and C headers are needed for first-run Triton helpers. Docker includes compiler prerequisites and requires Docker plus NVIDIA Container Toolkit on the host. No host CUDA Toolkit or Conda setup is needed. Available RAM/VRAM limits input sizes. Web uploads are limited to 100 MiB; larger alignments use CLI. ARM, CPU-only, pre-Volta and Blackwell are not admitted by this stable runtime.','',
    'The experimental Torch 2.11/CUDA 12.8/Triton 3.6 candidate required a type-specialization API adaptation. Its first Gene case retained topology and RF but changed branch values (max 2.041e-8), failing the strict artifact gate. It is not shipped as stable Blackwell support; no Blackwell hardware was available.','',
    '## Provenance','',
    'Scientific source and checkpoints remain frozen. Changes are limited to startup/configuration, Web GPU/CPU/bind selection, integrity/documentation, identical-source sequence-parser rebuild for glibc 2.34, and AP-NJ fatbinary architecture coverage. The original and expanded AP-NJ SM89 builds have 5960 identical SASS instructions. Scientific versions remain Torch 2.5.1+cu121, Triton 3.1.0, NumPy 2.1.2 and SciPy 1.15.3. `regression-trials.json` exports input hashes, measured/warm-up timings, RF measurements and output hashes for the primary and follow-up runs without embedding server-local input/output paths. Full outputs and original records remain in the Main-owned distribution workspace.','',
    'Official compatibility references: [NVIDIA Blackwell guide](https://docs.nvidia.com/cuda/blackwell-compatibility-guide/index.html), [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html), [PyTorch version matrix](https://pytorch.org/get-started/previous-versions/).','']
if modern_diagnostic is not None:
    d=summary['modern_diagnostic']
    lines += ['## Expanded modern-runtime diagnostic','',
        f"A subsequent diagnostic completed {d['completed_cases']}/{d['cases']} cases (state: {d['state']}). Among completed cases, {d['topology_identical_cases']} have identical initial/final topology sets, {d['reference_identical_cases']} have identical reference accuracy measurements, and {d['byte_identical_cases']} have byte-identical artifacts. This diagnostic continues after differences to measure their extent; the stable byte-identity gate is unchanged. See [MODERN_VALIDATION.md](MODERN_VALIDATION.md) and `modern-regression-results.json` for timings, numerical changes and source/weight provenance. The adapted dispatch and representative kernels also compile in 32 variant/architecture combinations for SM100/103/120/121. These are compilation checks; no Blackwell hardware execution or stable admission is claimed.",'']
(ROOT/'dist/VALIDATION.md').write_text('\n'.join(lines))
ready=dict(stable_runtime_ready=speed_ok and installer_ok,public_repo='PullMyBoots/ConcordTree',
           public_release_published=False,modern_runtime_admitted=False,
           accuracy_cases_passed=48,speed_followup_passed=speed_ok,
           installer_routing_ready=installer_ok,launcher_sha256=routing['launcher_sha256'],
           source_checkout='container/public-repo',
           source_commit=subprocess.check_output(['git','-C',str(ROOT/'container/public-repo'),'rev-parse','HEAD'],text=True).strip(),
           pending_local_checks=[],
           remaining=['GitHub Release API login and publication', 'Blackwell/ARM require separate validated artifacts'],
           portable_sha256=artifact['sha256'])
(ROOT/'evidence/release-readiness.json').write_text(json.dumps(ready,indent=2)+'\n')
print(json.dumps(ready,indent=2))
