"""Optional local rebuild for the ConcordTree extension modules.

The release wheel already contains tested CPython 3.10/Linux x86_64 binaries.
This script is included so that the bundled binaries are traceable to source.
It deliberately does not run as part of the main wheel build.
"""

import sys
import os
from pathlib import Path

from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


_source_root = Path(__file__).resolve().parent
_prefix_maps = [
    f"-ffile-prefix-map={Path(sys.prefix).resolve()}=__PYTHON_ENV__",
    f"-ffile-prefix-map={_source_root}=__CONCORDTREE_NATIVE_SRC__",
]


ext_modules = [
    Pybind11Extension(
        name="candidate_graph_backend",
        sources=["candidate_graph_backend.cpp"],
        cxx_std=17,
        extra_compile_args=["-O3", "-fopenmp", "-ffp-contract=off", *_prefix_maps],
        extra_link_args=["-fopenmp"],
    ),
    Pybind11Extension(
        name="panel_score_backend",
        sources=["panel_score_backend.cpp"],
        cxx_std=17,
        extra_compile_args=["-O3", *_prefix_maps],
    ),
    Pybind11Extension(
        name="learned_nni_plan_backend",
        sources=["learned_nni_plan_backend.cpp"],
        cxx_std=17,
        extra_compile_args=["-O3", *_prefix_maps],
    ),
    Pybind11Extension(
        name="split_compat_backend",
        sources=["split_compat_backend.cpp"],
        cxx_std=17,
        extra_compile_args=["-O3", *_prefix_maps],
    ),
    Pybind11Extension(
        name="sequence_processor_backend",
        sources=["sequence_processor_backend.cpp"],
        cxx_std=17,
        extra_compile_args=["-O3", *_prefix_maps],
    ),
    CUDAExtension(
        name="pattern_freq_cuda_backend",
        sources=[
            "pattern_freq_cuda_backend_host.cpp",
            "pattern_freq_cuda_backend.cu",
        ],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17", *_prefix_maps],
            "nvcc": [
                "-O3",
                "--use_fast_math",
                *(f"-Xcompiler={flag}" for flag in _prefix_maps),
            ],
        },
    ),
]

_requested_extension = os.environ.get("CONCORDTREE_BUILD_EXTENSION")
if _requested_extension:
    ext_modules = [
        extension for extension in ext_modules
        if extension.name == _requested_extension
    ]
    if not ext_modules:
        raise RuntimeError(f"unknown CONCORDTREE_BUILD_EXTENSION={_requested_extension!r}")


setup(
    name="concordtree-native-extensions",
    version="0.1.2",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
