"""Resident scaled expectation, retaining V-0.2.0 Fisher-score interface.

Never changes the model or samples sites. The graph captures the entire exact
expectation at static shapes. Inputs are copied into fixed addresses on replay.
Release primitives/optimizer remain the numerical oracle, not modified here.
"""
import time
import numpy as np
import torch
from concordtree.reversible_lengths import PruningPlan
from resident_cuda import upward, downward


class ResidentPlan(PruningPlan):
    def __init__(self, tree, names, states, device, categories, block_sites=0,
                 *, precision="float64", graph=True):
        started=time.perf_counter()
        if torch.device(device).type!="cuda":
            raise ValueError("experimental resident backend requires CUDA")
        if precision not in {"float64","float32","mixed"}:
            raise ValueError("precision must be float64, float32 or mixed")
        # Accurate FP64 observed likelihood, optionally FP32 downward score.
        # This avoids the false relative-function convergence seen with FP32 up.
        dtype=torch.float64 if precision=="mixed" else getattr(torch,precision)
        down_dtype=torch.float32 if precision=="mixed" else dtype
        # Choose bounded scratch from a conservative budget, not free GPU memory
        # at evaluation time. Explicit blocks remain valid for equivalence tests.
        if not block_sites:
            m=len(tree.graph)
            per_site=(m+1)*categories*(4*(torch.empty((),dtype=dtype).element_size()+torch.empty((),dtype=down_dtype).element_size())+8)
            cap=max(1,(10*1024**3)//per_site)
            block_sites=min(states.shape[1],2**int(np.log2(cap)))
        super().__init__(tree,names,states,device,categories,block_sites)
        self.graph_enabled=graph; self.graph=None; self.dtype=dtype
        dev=self.device; m=self.m; k=categories; b=self.block_sites
        self.edge_index=torch.tensor(self.edge_nodes,device=dev,dtype=torch.int64)
        # Only one scratch block; observations cached on GPU once.
        self.up=torch.ones((m+1,k,b,4),device=dev,dtype=dtype)
        self.down=torch.ones(self.up.shape,device=dev,dtype=down_dtype)
        self.logs=torch.zeros((m+1,k,b),device=dev,dtype=torch.float64)
        self.trans=torch.eye(4,device=dev,dtype=dtype).repeat(m+1,k,1,1)
        self.pi=torch.full((4,),.25,device=dev,dtype=torch.float64)
        self.weights=torch.full((k,),1/k,device=dev,dtype=torch.float64)
        self.counts=torch.zeros((m+1,k,4,4),device=dev,dtype=torch.float64)
        self.observations=[]; self.masks=[]
        for start in range(0,self.sequence.shape[1],b):
            n=min(b,self.sequence.shape[1]-start)
            block=np.full((len(names),b),4,dtype=np.int64)
            block[:,:n]=self.sequence[:,start:start+n]
            block=torch.as_tensor(block,device=dev)
            obs=torch.nn.functional.one_hot(block.clamp_max(3),4).to(dtype)
            obs=torch.where((block<4)[...,None],obs,torch.ones_like(obs))
            self.observations.append(obs[:,None,:,:])
            self.masks.append(torch.arange(b,device=dev)<n)
        self.up[self.leaf_ids]=self.observations[0]
        self.nonempty_up=[level for level in reversed(self.internals) if len(level)]
        self.setup_metrics={"allocation_seconds":time.perf_counter()-started,
                            "capture_seconds":0.,"calls":0,
                            "scratch_bytes":self.up.numel()*self.up.element_size()+self.down.numel()*self.down.element_size()+self.logs.numel()*8,
                            "observation_bytes":sum(o.numel()*o.element_size() for o in self.observations)}

    @torch.no_grad()
    def _run(self):
        self.counts.zero_()
        total=self.pi.new_zeros(())
        root_counts=self.pi.new_zeros(4)
        category_counts=self.weights.new_zeros(len(self.weights))
        for obs,mask in zip(self.observations,self.masks):
            if len(self.observations)>1:
                self.up[self.leaf_ids]=obs
            # All internal messages/logs overwritten by dependency order. Leaf
            # and sentinel log-scales remain zero; no per-level reductions.
            for level in self.nonempty_up:
                upward(self.up,self.trans,self.children,level,self.logs)
            logscale=self.logs.sum(0)
            root_joint=self.up[self.root_i].to(torch.float64)*self.pi
            root_prob=root_joint.sum(-1)
            catll=logscale+root_prob.clamp_min(1e-300).log()+self.weights.log()[:,None]
            site_ll=torch.logsumexp(catll,0)
            total+=(site_ll*mask).sum()
            posterior=(catll-site_ll[None,:]).exp()*mask[None,:]
            category_counts+=posterior.sum(1)
            root_counts+=(posterior[...,None]*root_joint/root_prob.clamp_min(1e-300)[...,None]).sum((0,1))
            self.down[self.root_i]=self.pi
            for level in self.nonroot:
                downward(self.up,self.down,self.trans,self.children,self.parents,level,posterior,self.counts)
        return total,self.counts.index_select(0,self.edge_index),root_counts,category_counts

    @torch.no_grad()
    def expectation(self, transitions, pi, weights, *, need_counts=True, backend="triton"):
        self.trans[self.edge_index]=transitions.to(self.dtype)
        self.pi.copy_(pi); self.weights.copy_(weights)
        self.setup_metrics["calls"]+=1
        if not self.graph_enabled:
            return self._run()
        if self.graph is None:
            started=time.perf_counter()
            # Warm allocator and JIT on a side stream, before capture. Input
            # stream ordering is explicit; no CPU data reads inside the graph.
            current=torch.cuda.current_stream(self.device)
            stream=torch.cuda.Stream(device=self.device)
            stream.wait_stream(current)
            with torch.cuda.stream(stream):
                for _ in range(2): self._run()
            current.wait_stream(stream)
            self.graph=torch.cuda.CUDAGraph()
            from capture_scope import capture_scope
            with capture_scope(self.graph,stream):
                self.outputs=self._run()
            torch.cuda.synchronize(self.device)
            self.setup_metrics["capture_seconds"]=time.perf_counter()-started
        self.graph.replay()
        # These output buffers are overwritten on next replay; the caller must
        # consume them before evaluating a different parameter vector.
        return self.outputs
