"""Lossless message-DAG pruning with eager kernels of the retained scorer.

No message toward a leaf can feed an internal-edge NNI, so none is allocated or
computed. Root-only checks need upward messages only. Same edges, K, width,
precision, contractions, local alternatives and reduction order as incumbent.
"""
import math
import time
import numpy as np
import torch
import triton
from concordtree.treeio import edge, rooted
from concordtree.dna_models import transition_matrices
from streamed_likelihood import _leaf, _send
from weighted_likelihood import WeightedLikelihood, _weighted_root, _weighted_nni


class PrunedWeightedLikelihood(WeightedLikelihood):
    @torch.no_grad()
    def __init__(self, tree, names, metrics, *, device='cuda:0', block_sites=256,
                 precision='float32', workspace_bytes=512 << 20, nni=True, message_workspace=None, scale_dtype=torch.float64):
        start = time.perf_counter()
        if precision != 'float32' or torch.device(device).type != 'cuda':
            raise ValueError('FP32 CLV CUDA mode required')
        from _tree_plan_native import build
        meta=build(tree.graph,tree.names,tree.lengths,list(names),nni)
        self.tree,self.names,self.device=tree,tuple(names),torch.device(device)
        self.k,self.dtype,self.nni=len(metrics['category_rates']),torch.float32,nni
        self.root,self.depth=meta['root'],meta['depth']
        self.edges,self.internal_edges,self.branches=meta['edges'],meta['internal_edges'],meta['branches']
        lengths=meta['lengths'];self.directed_rows=meta['rows']
        old_bytes=(4*len(names)-6)*self.k*24
        self.width=min(block_sites,workspace_bytes//max(1,old_bytes))
        if self.width<1:raise MemoryError('workspace cannot hold original site width')
        # Power-of-two frontier capacities are an execution shape, not a
        # search budget. Sentinel rows launch no work and never write a CLV.
        def padded(rows):
            n=1<<max(0,int(len(rows)-1).bit_length())
            return np.pad(rows,(0,n-len(rows)),constant_values=-1)
        depth_capacity=1<<max(0,int(self.depth-1).bit_length())
        up_levels=[padded(a) for a in meta['up']]
        down_levels=[padded(a) for a in meta['down']]
        empty=np.array([-1],dtype=np.int32)
        up_levels=[empty]*(depth_capacity-len(up_levels))+up_levels
        down_levels=down_levels+[empty]*(depth_capacity-len(down_levels)) if nni else []
        arrays=meta['arrays']+up_levels+down_levels
        self.message_workspace=message_workspace
        # One contiguous transfer for the whole message DAG, not one blocking
        # host->device copy for every tree-depth frontier. All integer indices,
        # frontier ordering and per-kernel grids are exactly the old ones.
        sizes = [a.size for a in arrays]
        metadata = torch.as_tensor(np.concatenate([a.reshape(-1) for a in arrays]),device=self.device)
        self.metadata=metadata
        offsets = np.cumsum([0]+sizes)
        views = [metadata[int(lo):int(hi)].view(a.shape) for lo,hi,a in zip(offsets,offsets[1:],arrays)]
        (self.deps,self.rowedge,self.leafrows,self.leafindex,self.rootrows,
         self.score_edges,self.incoming) = views[:7]
        self.up_levels = views[7:7+len(up_levels)]
        self.levels = self.up_levels + views[7+len(up_levels):]
        from fixed_transitions import fixed_transitions
        self.p = fixed_transitions(metrics['Q_acgt'], metrics['frequencies_acgt'],
            lengths, metrics['category_rates'], self.device).to(self.dtype)
        self.pi = torch.tensor(metrics['frequencies_acgt'], device=self.device, dtype=torch.float64).to(self.dtype)
        self.logweights = torch.tensor(np.log(metrics['category_weights']), device=self.device)
        if message_workspace is None:
            self.u = torch.empty((self.directed_rows,self.k,self.width,4), device=self.device, dtype=self.dtype)
            self.logs = torch.empty((self.directed_rows,self.k,self.width), device=self.device, dtype=scale_dtype)
        else:
            self.u, self.logs = message_workspace.borrow(self.directed_rows, self.k, self.width, scale_dtype=scale_dtype)
        self.parts = triton.cdiv(self.width,128)
        self.partial = torch.empty((len(self.internal_edges),3,self.parts), device=self.device, dtype=torch.float64)
        self.rootpartial = torch.empty(self.parts, device=self.device, dtype=torch.float64)
        self.message_buffer_bytes = self.u.numel()*4 + self.logs.numel()*self.logs.element_size()
        torch.cuda.synchronize(self.device)
        self.setup_seconds = time.perf_counter() - start

    def _kernels(self, slab, weights):
        k,w,parts = self.k,self.width,self.parts
        _leaf[(len(self.leafrows),k,parts)](self.u,self.logs,self.p,slab,self.leafrows,self.leafindex,self.rowedge,k,w,w,128,1e-30,enable_fp_fusion=False)
        for level in self.levels:
            _send[(len(level),k,parts)](self.u,self.logs,self.p,level,self.deps,self.rowedge,k,w,w,128,1e-30,enable_fp_fusion=False)
        _weighted_root[(parts,)](self.u,self.logs,self.pi,self.logweights,self.rootrows,self.rootpartial,weights,k,w,w,128,enable_fp_fusion=False)
        if self.nni:
            _weighted_nni[(len(self.internal_edges),parts)](self.u,self.logs,self.p,self.pi,self.logweights,self.score_edges,self.incoming,self.partial,weights,k,w,w,128,parts,enable_fp_fusion=False)

    @torch.no_grad()
    def score_patterns(self, alignment, *, nni=True):
        if nni != self.nni:
            raise ValueError('plan construction must match score mode')
        start = time.perf_counter()
        totals = torch.zeros((len(self.internal_edges),3), device=self.device, dtype=torch.float64)
        root_ll = torch.zeros((), device=self.device, dtype=torch.float64)
        host = np.full((len(self.names),self.width),4,np.uint8)
        counts = np.zeros(self.width,np.int64)
        slab = torch.tensor(host,device=self.device)
        weights = torch.tensor(counts,device=self.device)
        blocks = sites = patterns = 0
        for obs,multiplicity in alignment.blocks(self.width):
            if obs.shape[0] != len(self.names) or obs.shape[1] != len(multiplicity):
                raise ValueError('pattern dimensions')
            if not np.issubdtype(multiplicity.dtype,np.integer) or (multiplicity <= 0).any():
                raise ValueError('positive integer multiplicity required')
            if obs.min() < 0 or obs.max() > 4:
                raise ValueError('DNA states must be0..4')
            host.fill(4); counts.fill(0)
            host[:,:obs.shape[1]] = obs
            counts[:len(multiplicity)] = multiplicity
            slab.copy_(torch.from_numpy(host))
            weights.copy_(torch.from_numpy(counts))
            self._kernels(slab,weights)
            # Retain the incumbent's exact reduction shape and order.
            if nni:
                totals += self.partial.sum(-1)
            root_ll += self.rootpartial.sum()
            blocks += 1; sites += int(multiplicity.sum()); patterns += obs.shape[1]
        root,output = float(root_ll),totals.cpu().numpy()
        if not math.isfinite(root) or not np.isfinite(output).all():
            raise RuntimeError('nonfinite weighted likelihood')
        if sites != alignment.original_sites:
            raise ValueError('lost/duplicated site multiplicity')
        return dict(log_likelihood=root,deltas=output[:,:2] if nni else None,
            max_baseline_error=float(np.max(np.abs(output[:,2]-root))) if nni else None,
            seconds=time.perf_counter()-start,site_blocks=blocks,block_sites=self.width,
            depth=self.depth,message_buffer_bytes=self.message_buffer_bytes,precision=str(self.dtype),
            all_sites=sites,evaluated_patterns=patterns,dependency_pruned=True,
            graph_replay=False,directed_rows=self.directed_rows)
