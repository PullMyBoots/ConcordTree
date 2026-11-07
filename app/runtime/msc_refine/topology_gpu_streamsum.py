"""Exact integer quartet classification on GPU; FP64 reduction stays on CPU."""
import numpy as np
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice
from . import _msc_slim_index as _msc_active_gpu

from .topology_gpu import _classify

@torch.inference_mode()
def classify_stream(tree,n,queries,*,device='cuda:0'):
    idx=_msc_active_gpu.export_index(tree,n)
    dev=torch.device(device)
    arrays={key:torch.as_tensor(idx[key],device=dev,dtype=torch.int32) for key in ['nodes','rmq','edges','branches']}
    q=np.ascontiguousarray(queries,dtype=np.int32)
    if q.ndim!=2 or q.shape[1]!=4:
        raise ValueError('canonical Qx4 queries required')
    if len(q) and (q.min()<0 or q.max()>=n):
        raise ValueError('query out of range')
    cpu=np.empty(len(q),dtype=np.int32)
    # Execution storage cap: one Mi-query is never required to be resident.
    # This limit does not drop or resample any observation.
    for start in range(0,len(q),262144):
        block=q[start:start+262144]
        query=torch.as_tensor(block,device=dev)
        codes=torch.empty((len(block),),device=dev,dtype=torch.int32)
        _classify[(triton.cdiv(len(block),256),)](query,arrays['nodes'],arrays['rmq'],arrays['edges'],arrays['branches'],codes,len(block),2*n-2,arrays['rmq'].shape[1],n,256)
        cpu[start:start+len(block)]=codes.cpu().numpy()
    return idx,cpu

@torch.inference_mode()
def evaluate_stream(tree,n,queries,weights,is_new,*,device='cuda:0',return_classes=False):
    from . import _msc_active_streamsum
    idx,cpu=classify_stream(tree,n,queries,device=device)
    result=_msc_active_streamsum.reduce(cpu,weights,is_new,n-3,np.getbufsize())
    result['edges']=idx['edges']
    if return_classes:result['classes']=cpu
    return result
