"""Integrity-checked native statistical kernels only; no neural checkpoints."""
from pathlib import Path
import importlib.util
import platform
import os
from functools import lru_cache
from typing import Any
import hashlib
PACKAGE_ROOT=Path(__file__).resolve().parent
EXPECTED={
    'sequence_processor_backend':'148cdb73b8eb1eeb6c28d213d4395408e490a873ecee168e5f77be384c94d54d',
    'pattern_freq_cuda_backend':'90c70c0f53a618abb536024c9544bf6b43d329ffa0391929c68a1b3e7915498d',
    'learned_nni_plan_backend':'cb48c4ce4e7d2465c78cf4aa28a6f4ffa41852a7d853ca7a15ca291f425eae0b',
    'panel_score_backend':'ce1ec2c040507461739cff9d92b1d2a530d0c2fae24d071d102601464b494fe5',
    'candidate_graph_backend':'3a20444e0db387fe6f09afaec07a61f75fa99b1873f2a49ca0b2d8d7bf3957ec',
}
def asset_path(relative):
    p=PACKAGE_ROOT/'assets'/relative
    if not p.is_file():raise FileNotFoundError(p)
    return p
def _load_extension(name: str, path: Path) -> Any:
    if platform.python_version_tuple()[:2] != ("3", "10"):
        raise RuntimeError(
            "the bundled binary backends require CPython 3.10 on Linux x86_64"
        )
    if name in EXPECTED and hashlib.sha256(path.read_bytes()).hexdigest()!=EXPECTED[name]:
        raise RuntimeError("native asset hash mismatch: "+name)
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load binary extension: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def load_backends() -> tuple[Any, Any]:
    sequence = _load_extension(
        "sequence_processor_backend",
        asset_path(
            "backends/sequence_processor_backend.cpython-310-x86_64-linux-gnu.so"
        ),
    )
    pattern = _load_extension('pattern_freq_cuda_backend', asset_path('backends/pattern_freq_cuda_backend.cpython-310-x86_64-linux-gnu.so'))
    return sequence, pattern



@lru_cache(maxsize=1)
def load_candidate_graph_backend():
    return _load_extension('candidate_graph_backend',asset_path('backends/candidate_graph_backend.cpython-310-x86_64-linux-gnu.so'))

@lru_cache(maxsize=1)
def load_panel_score_backend():
    return _load_extension('panel_score_backend', asset_path('backends/panel_score_backend.cpython-310-x86_64-linux-gnu.so'))

@lru_cache(maxsize=1)
def load_learned_nni_plan_backend():
    return _load_extension('learned_nni_plan_backend', asset_path('backends/learned_nni_plan_backend.cpython-310-x86_64-linux-gnu.so'))
