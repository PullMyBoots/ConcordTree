"""CUDA capture for explicitly owned, warmed inference workspaces.

Unlike the general-purpose torch.cuda.graph context, this does not sweep the
whole Python heap or flush the process CUDA allocator on every tree score.
Callers own all input/output buffers, warm the exact workload before entry,
and establish stream dependencies. This scope changes no numerical operation.
"""
from contextlib import contextmanager
import gc
import torch


@contextmanager
def capture_scope(graph, stream):
    # Finish this workload's warmup, not all work in the CUDA process/device.
    # Suppress cyclic GC only while capture is active: an unrelated finalizer
    # must not call cudaFree during stream capture. Refcount cleanup is intact.
    stream.synchronize()
    enabled = gc.isenabled()
    gc.disable()
    try:
        with torch.cuda.stream(stream):
            graph.capture_begin(capture_error_mode='global')
            try:
                yield
            finally:
                graph.capture_end()
    finally:
        if enabled:
            gc.enable()
