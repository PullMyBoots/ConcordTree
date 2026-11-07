"""Compile representative active kernels; this is not GPU hardware validation."""
from pathlib import Path
import hashlib
import inspect
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
APP = Path(os.environ.get('CONCORDTREE_COMPILE_APP', ROOT / 'app')).resolve()
os.environ['TRITON_CACHE_DIR'] = os.environ.get('CONCORDTREE_COMPILE_CACHE', str(ROOT / 'output/architecture-cache'))
sys.path[:0] = [str(APP), str(APP / 'runtime/gene/src'),
               str(APP / 'runtime/gene/runtime'), str(APP / 'runtime')]

def compile_arch(arch):
    import triton
    from triton.compiler import ASTSource
    from triton.backends.compiler import GPUTarget
    from repeats_generic_cuda import _up
    from repeats_fused_counts import _down
    from msc_refine.kernel_transposed import _transposed_runtime
    variants = []
    for dtype in ('fp32', 'fp64'):
        variants.append(('gene_up_' + dtype, _up,
            dict(U='*'+dtype, LOG='*fp64', P='*'+dtype, C='*i32', MAP='*i32', NODE='*i32', OFF='*i64', FIRST='i32', LAST='i32', NI='i32', M='i32'),
            dict(K=4, B=128, FLOOR=1e-30 if dtype=='fp32' else 1e-300)))
    variants.append(('gene_counts', _down,
        dict(U='*fp32', THETA='*fp32', PARTIAL='*fp64', P='*fp32', OFF='*i64', LINK='*i32', REC='*i32', BLOCK='*i64', NI='i32'),
        dict(K=4,W=4,T=32,FLOOR=1e-30)))
    variants.append(('species_score', _transposed_runtime,
        dict(X='*fp32', P='*i64', Z='*fp32', C='*fp32', OUT='*fp32', N='i32'),
        dict(M=1024,T=8,SCALE=.5,BQ=16,BZ=128,RG=0)))
    if os.environ.get('CONCORDTREE_COMPILE_DISPATCH') == '1':
        import triton.language as tl
        from depth_downward_cuda import _dispatch
        for dtype in ('fp32','fp64'):
            for bits in (32,64):
                index = bits == 64 if 'constexprs' in inspect.signature(ASTSource).parameters else tl.pointer_type(tl.int64 if bits == 64 else tl.int32)
                variants.append((f'gene_dispatch_{dtype}_index{bits}',_dispatch,
                    dict(U='*'+dtype,THETA='*'+dtype,PARTIAL='*fp64',P='*'+dtype,OFF='*i64',LINK='*i32',WORK='*i64',NI='i32'),
                    dict(K=4,MAXPOWER=7,PRESENT=255,FLOOR=1e-30 if dtype == 'fp32' else 1e-300,INDEX=index)))
    rows=[]
    for name,fn,types,constants in variants:
        if 'constexprs' in inspect.signature(ASTSource).parameters:
            # Triton 3.6 compilation harness API; scientific kernels are unchanged.
            source=ASTSource(fn,{key:types[key] for key in fn.arg_names if key not in constants},constexprs=constants)
        else:
            sig={i:types[key] for i,key in enumerate(fn.arg_names) if key not in constants}
            const={i:constants[key] for i,key in enumerate(fn.arg_names) if key in constants}
            source=ASTSource(fn,sig,constants=const)
        compiled=triton.compile(source,target=GPUTarget('cuda',arch,32),options={'num_warps':4})
        rows.append(dict(arch=arch,kernel=name,status='compiled',shared_bytes=compiled.metadata.shared,
                         ptx_sha256=hashlib.sha256(compiled.asm['ptx'].encode()).hexdigest()))
    print(json.dumps(rows))

if len(sys.argv)>1:
    compile_arch(int(sys.argv[1]))
else:
    rows=[]
    for arch in map(int,os.environ.get('CONCORDTREE_COMPILE_TARGETS','70,75,80,86,89,90').split(',')):
        proc=subprocess.run([sys.executable,__file__,str(arch)],capture_output=True,text=True,timeout=180)
        if proc.returncode:
            rows.append(dict(arch=arch,status='failed',stderr=proc.stderr[-8000:]))
        else:
            rows.extend(json.loads(proc.stdout.strip().splitlines()[-1]))
        print(arch,proc.returncode,flush=True)
    report=Path(os.environ.get('CONCORDTREE_COMPILE_REPORT',ROOT/'evidence/architecture-compilation.json'))
    scope='four representative active likelihood/count/species kernels, not exhaustive kernel/input coverage'
    if os.environ.get('CONCORDTREE_COMPILE_DISPATCH') == '1':
        scope += '; plus low-width downward dispatch in fp32/fp64 with int32/int64 indices'
    report.write_text(json.dumps(dict(
        hardware_execution=False,scope=scope,results=rows),indent=2)+'\n')
