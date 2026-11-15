"""Owned outputs and release-local caches, configured before CUDA imports."""
from functools import lru_cache
from pathlib import Path
import os
import sys

HERE = Path(__file__).resolve().parent
PROJECT = next((p for p in HERE.parents if (p / '.ResearchProject').is_dir()), HERE)

@lru_cache(maxsize=1)
def protected_targets():
    targets = [PROJECT / 'data', PROJECT / '.ResearchProject']
    for root, dirs, files in os.walk(PROJECT / 'data', followlinks=False):
        for name in dirs + files:
            link = Path(root) / name
            if link.is_symlink():
                # Reserve missing targets too: an unrelated dangling input link
                # must not block valid jobs, and its target is still read-only.
                targets.append(link.resolve(strict=False))
    return tuple(targets)

def owned_path(path, *, inputs=()):
    requested = Path(path).absolute()
    if any(requested.is_relative_to(PROJECT / name) for name in ('data', '.ResearchProject')):
        raise ValueError('outputs and caches must be outside immutable data and research memory')
    path = requested.resolve()
    if path == PROJECT or not path.is_relative_to(PROJECT):
        raise ValueError('outputs and caches must be inside this project')
    if any(path == root or path.is_relative_to(root) for root in protected_targets()):
        raise ValueError('outputs and caches must be outside immutable data and research memory')
    for source in inputs:
        source = Path(source).resolve(strict=True)
        if path == source or path == source.parent or path.is_relative_to(source):
            raise ValueError('output cannot replace an input or its directory')
    return path

def configure_runtime(*, gpu=None, cpu_threads=16):
    # Historical releases use the same top-level module names. Fail explicitly
    # if a host interpreter has already imported one of those runtimes.
    generic = {p.stem for folder in (HERE/'runtime/gene', HERE/'runtime/gene/runtime')
               for p in folder.glob('*.py')}
    generic.update(('concordtree', 'msc_refine', 'frontend', 'array_mlp_nni','branch_support','search_progress',
                    'apnj_frontend','apnj_core','apnj_cleanup','backbone','taxon_placement','_ct_apnj_graph',
                    '_ct_apnj_topology','_ct_apnj_local','_ct_apnj_terminal','_ct_apnj_packed'))
    for name, module in tuple(sys.modules.items()):
        origin = getattr(module, '__file__', None)
        if name.split('.')[0] in generic and origin and not Path(origin).resolve().is_relative_to(HERE):
            raise RuntimeError('incompatible runtime already imported: ' + name + '; use a fresh Python process')
    if not isinstance(cpu_threads, int) or not 1 <= cpu_threads <= 256:
        raise ValueError('cpu_threads must be between 1 and 256')
    requested_cpu_threads=cpu_threads
    # Inherited packed extraction/global compression extensions support <=64.
    # Accept the public request, but use one consistent supported worker count.
    cpu_threads=min(cpu_threads,64)
    if gpu is not None:
        if not isinstance(gpu, int) or gpu < 0:
            raise ValueError('gpu must be a nonnegative physical device index')
        if 'torch' in sys.modules and sys.modules['torch'].cuda.is_initialized():
            if os.environ.get('CUDA_VISIBLE_DEVICES') != str(gpu):
                raise RuntimeError('select the GPU before initializing CUDA')
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    caches = {}
    for key, tail in [('TRITON_CACHE_DIR','triton'), ('TORCH_EXTENSIONS_DIR','extensions'),
                      ('CUDA_CACHE_PATH','cuda'), ('TMPDIR','tmp'), ('XDG_CACHE_HOME','xdg'),
                      ('TORCHINDUCTOR_CACHE_DIR','inductor'), ('TORCH_HOME','torch')]:
        default = HERE / '.runtime/cache' / tail
        try:
            cache = owned_path(os.environ.get(key, default))
            if not cache.is_relative_to(HERE):
                raise ValueError('runtime caches must stay inside this release')
        except ValueError:
            # Host-wide temporary/cache defaults are not job output locations.
            cache = owned_path(default)
        cache.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(cache); caches[key] = str(cache)
    for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:
        os.environ[key] = str(cpu_threads)
    os.environ.setdefault('OMP_WAIT_POLICY', 'PASSIVE')
    paths = [HERE, HERE/'runtime/gene/src', HERE/'runtime/gene/runtime', HERE/'runtime/gene', HERE/'runtime']
    for path in reversed(paths):
        if str(path) not in sys.path:sys.path.insert(0, str(path))
    return dict(cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES','all'),
                logical_device='cuda:0', cpu_threads=cpu_threads,
                requested_cpu_threads=requested_cpu_threads,caches=caches)
