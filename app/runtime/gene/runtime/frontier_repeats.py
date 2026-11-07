"""Exact site-repeat DAG with NumPy oracle or native direct-table/hash/radix setup.

Site repeats: Kobert, Stamatakis & Flouri (2017), doi:10.1093/sysbio/syw075.
Raw-site weights, shared Gamma classes and all endpoint statistics retained.
"""
import time
import math
import numpy as np
import torch
from concordtree.treeio import rooted,edge
from resident import ResidentPlan
from repeats_cuda import upward,downward,reduce_counts


class RepeatPlan(ResidentPlan):
    @torch.no_grad()
    def _likelihood_only(self):
        """Same observed objective, without endpoint posterior/count work."""
        for first,last in self.up_levels:self.upward(self,first,last)
        n=self.root_patterns
        root_prob=(self.up[:n].double()*self.pi).sum(-1)
        scales=self.logs[:n].double()*(math.log(2.) if self.binary_scaling else 1.)
        catll=torch.where(root_prob>0,scales+root_prob.clamp_min(1e-300).log()+self.weights.log(),-torch.inf)
        return (torch.logsumexp(catll,1)*self.root_frequency).sum()

    @torch.no_grad()
    def likelihood(self,transitions,pi,weights):
        self.trans[self.edge_index]=transitions.to(self.dtype)
        self.pi.copy_(pi);self.weights.copy_(weights)
        if not self.graph_enabled:return self._likelihood_only()
        if not hasattr(self,'likelihood_graph'):
            current=torch.cuda.current_stream(self.device)
            stream=torch.cuda.Stream(device=self.device);stream.wait_stream(current)
            with torch.cuda.stream(stream):
                for _ in range(2):self._likelihood_only()
            current.wait_stream(stream)
            self.likelihood_graph=torch.cuda.CUDAGraph()
            from capture_scope import capture_scope
            with capture_scope(self.likelihood_graph,stream):
                self.likelihood_output=self._likelihood_only()
            torch.cuda.synchronize(self.device)
        self.likelihood_graph.replay()
        return self.likelihood_output

    def __init__(self,tree,names,states,device,categories,block_sites=0,*,precision="float64",graph=True,builder="numpy",kernel_schedule="specialized",down_schedule="sequential"):
        start=time.perf_counter();tree.require_binary();states=np.asarray(states)
        if torch.device(device).type!="cuda":raise ValueError("repeat prototype requires CUDA")
        if precision not in {"float64","float32","mixed","binary64"}:raise ValueError("invalid precision")
        if builder != "radix":raise ValueError("frontier child requires audited radix builder")
        if kernel_schedule not in {"specialized","generic","segmented","category","category-bounded","matrix","matrix-factorized"}:raise ValueError("invalid kernel schedule")
        if down_schedule not in {"sequential","depth-concurrent"}:raise ValueError("invalid down schedule")
        if down_schedule=="depth-concurrent" and kernel_schedule!="matrix":raise ValueError("depth-concurrent requires full-J matrix kernels")
        self.upward,self.downward,self.reduce_counts=upward,downward,reduce_counts
        if kernel_schedule=="generic":
            from repeats_generic_cuda import upward as gu,downward as gd,reduce_counts as gr
            self.upward,self.downward,self.reduce_counts=gu,gd,gr
        elif kernel_schedule=="matrix":
            from repeats_matrix_cuda import upward as mu,downward as md,reduce_counts as mr
            self.upward,self.downward,self.reduce_counts=mu,md,mr
        elif kernel_schedule=="matrix-factorized":
            from repeats_fused_counts import upward as fu,downward as fd,reduce_counts as fr
            self.upward,self.downward,self.reduce_counts=fu,fd,fr
        elif kernel_schedule=="segmented":
            from repeats_segmented_cuda import upward as su,downward as sd,reduce_counts as sr
            self.upward,self.downward,self.reduce_counts=su,sd,sr
        elif kernel_schedule=="category":
            from repeats_category_cuda import upward as cu,downward as cd,reduce_counts as cr
            self.upward,self.downward,self.reduce_counts=cu,cd,cr
        elif kernel_schedule=="category-bounded":
            from repeats_category_bounded_cuda import upward as bu,downward as bd,reduce_counts as br
            self.upward,self.downward,self.reduce_counts=bu,bd,br
        if len(names)!=len(set(names)) or set(names)!=set(tree.names.values()):raise ValueError("exact taxa match required")
        if states.ndim!=2 or states.shape[0]!=len(names) or states.shape[1]<1:raise ValueError("invalid alignment")
        if not np.issubdtype(states.dtype,np.integer) or states.min()<0 or states.max()>4 or states.min()==4:raise ValueError("invalid states")
        l=states.shape[1]
        if block_sites not in {0,l}:raise ValueError("repeat prototype supports only one all-site block")
        if kernel_schedule!="matrix-factorized" and max(l+1,5)**3>=np.iinfo(np.int64).max:
            raise ValueError("legacy repeat schedule exceeds int64 tuple encoding; use matrix-factorized")
        self.binary_scaling=precision=="binary64"
        if self.binary_scaling:
            from repeats_binary64_cuda import upward as binary_upward
            if len(names)*4096>=2**31:raise OverflowError('int32 exponent capacity exceeded')
            self.upward=binary_upward
        self.device=torch.device(device);dev=self.device;k=categories;self.categories=k
        self.dtype=torch.float64 if precision=="mixed" else torch.float64 if self.binary_scaling else getattr(torch,precision)
        self.ddtype=torch.float32 if precision in {"mixed","binary64"} else self.dtype
        self.graph_enabled=graph;self.graph=None
        self.sequence=states;self.requested_block=block_sites;self.block_sites=l
        self.root,parent,self.order,depth=rooted(tree)
        internal=[u for u in self.order if u not in tree.names]
        leaves=[u for u in self.order if u in tree.names]
        ids={u:i for i,u in enumerate(internal+leaves)};self.ids=ids
        self.ni=len(internal);self.m=len(ids);self.root_i=ids[self.root]
        # Live postorder IDs replace the complete node×site scratch grid.
        # Persistent pattern/CSR and GPU scratch guards remain unchanged.
        self.edges=[edge(u,parent[u]) for u in self.order if u!=self.root]
        self.edge_nodes=[ids[u] for u in self.order if u!=self.root]
        self.edge_index=torch.tensor(self.edge_nodes,device=dev)
        lookup={n:i for i,n in enumerate(names)}
        sequence=np.ascontiguousarray(states[[lookup[tree.names[u]] for u in leaves]],dtype=np.uint8)
        children=np.full((self.m+1,3),self.m,dtype=np.int64)
        for u in internal:
            cc=[ids[v] for v in sorted(tree.graph[u]) if parent.get(v)==u]
            children[ids[u],:len(cc)]=cc
        depths=np.array([depth[u] for u in internal],dtype=np.int64)
        t=time.perf_counter()
        if builder=="radix":
            if kernel_schedule=="matrix-factorized":
                from _repeat_builder_schedule import build
            else:
                from _repeat_builder_invariant import build
            meta=build(sequence,children,depths,torch.get_num_threads())
        elif builder=="flat":
            from _repeat_builder_flat import build
            meta=build(sequence,children,depths,torch.get_num_threads())
        elif builder=="native":
            from _repeat_builder import build
            meta=build(sequence,children,depths,torch.get_num_threads())
        else:
            from repeats_metadata import numpy_metadata
            meta=numpy_metadata(children,sequence,depths)
        builder_seconds=time.perf_counter()-t
        offset=meta["offsets"];self.r=int(offset[self.ni]);self.all_r=int(offset[-1])
        up_bytes=self.r*k*(4*torch.empty((),dtype=self.dtype).element_size()+(4 if self.binary_scaling else 8))
        down_bytes=self.r*k*4*torch.empty((),dtype=self.ddtype).element_size()
        endpoint_width=4 if kernel_schedule=="matrix-factorized" else 16
        joint_bytes=0 if kernel_schedule in {"segmented","matrix-factorized"} else self.all_r*k*endpoint_width*torch.empty((),dtype=self.ddtype).element_size()
        scratch=up_bytes+down_bytes+joint_bytes
        # Allocations are accounted in setup_metrics; no obsolete 10GiB research guard.
        self.offsets=torch.tensor(offset,device=dev);self.children=torch.tensor(children,device=dev)
        self.pattern_nodes=torch.tensor(meta["pattern_nodes"],device=dev)
        self.mapping=torch.tensor(meta["mapping"],device=dev);self.links=torch.tensor(meta["links"],device=dev)
        self.buckets=[];segment_blocks=0
        schedule_audit={}
        scheduled=meta["buckets"]
        if kernel_schedule=="matrix-factorized":
            scheduled=meta["scheduled_buckets"]
            schedule_audit=meta["terminal_schedule_audit"]
            count_blocks=meta["endpoint_blocks"]
            count_ranges=meta["endpoint_ranges"]
            count_capacity=meta["endpoint_capacity"]
            count_block_count=meta["endpoint_block_count"]
        for bucket_index,(_,power,rec) in enumerate(scheduled):
            width=2**int(power);data=torch.tensor(rec,device=dev)
            if kernel_schedule=="segmented":
                # Builders concatenate node groups; split each group into tiles
                # without padding/copying the much larger raw record matrix.
                cuts=np.r_[0,np.flatnonzero(rec[1:,0]!=rec[:-1,0])+1,len(rec)]
                starts=cuts[:-1];sizes=np.diff(cuts);tile=max(1,min(128,128//width))
                blocks=(sizes+tile-1)//tile
                origin=np.repeat(starts,blocks)
                local=np.arange(int(blocks.sum()))-np.repeat(np.cumsum(blocks)-blocks,blocks)
                first=origin+local*tile
                count=np.minimum(tile,np.repeat(sizes,blocks)-local*tile)
                descriptor=np.stack((first,count,rec[first,0]),axis=1)
                segment_blocks+=len(descriptor)
                data=(data,torch.tensor(descriptor,device=dev))
            elif kernel_schedule=="matrix-factorized":
                descriptor=torch.tensor(count_blocks[bucket_index],device=dev)
                data=(data,descriptor)
                scratch+=descriptor.numel()*descriptor.element_size()
            self.buckets.append((width,data))
        self.chunks=None if kernel_schedule in {"segmented","matrix-factorized"} else torch.tensor(meta["chunks"],device=dev)
        if kernel_schedule=="matrix-factorized":
            self.count_ranges=torch.tensor(count_ranges,device=dev)
            self.count_capacity=count_capacity
            self.count_partials=torch.empty((count_block_count,k,16),device=dev,dtype=torch.float64)
            scratch+=self.count_partials.numel()*8+self.count_ranges.numel()*8
        self.up_levels=[tuple(map(int,p)) for p in meta["up_levels"]]
        self.root_patterns=int(meta["sizes"][self.root_i])
        self.root_frequency=torch.tensor(meta["root_frequency"],device=dev,dtype=torch.float64)
        self.root_invariant_masks=torch.tensor(meta["root_invariant_masks"],device=dev,dtype=torch.uint8)
        self.up=torch.ones((self.r,k,4),device=dev,dtype=self.dtype)
        self.logs=torch.zeros((self.r,k),device=dev,dtype=torch.int32 if self.binary_scaling else torch.float64)
        self.theta=torch.zeros((self.r,k,4),device=dev,dtype=self.ddtype)
        self.joint=None if kernel_schedule in {"segmented","matrix-factorized"} else torch.zeros((self.all_r,k,endpoint_width),device=dev,dtype=self.ddtype)
        self.trans=torch.eye(4,device=dev,dtype=self.dtype).repeat(self.m+1,k,1,1)
        self.pi=torch.full((4,),.25,device=dev,dtype=torch.float64)
        self.weights=torch.full((k,),1/k,device=dev,dtype=torch.float64)
        self.counts=torch.zeros((self.m+1,k,4,4),device=dev,dtype=torch.float64)
        self.setup_metrics=dict(allocation_seconds=time.perf_counter()-start,capture_seconds=0.,calls=0,
            layout="exact subtree-pattern DAG/posterior CSR",builder=builder,kernel_schedule=kernel_schedule,builder_seconds=builder_seconds,
            scratch_bytes=scratch,internal_patterns=self.r,raw_internal_sites=self.ni*l,
            internal_pattern_fraction=self.r/(self.ni*l),csr_links=len(meta["links"]),
            reverse_buckets=len(self.buckets),segment_blocks=segment_blocks,root_patterns=self.root_patterns)
        self.setup_metrics.update(meta["frontier_audit"])
        self.setup_metrics.update(index_bits=meta.get("index_bits",64), index_representation="capacity-safe structural indices; unchanged scientific arithmetic")
        self.setup_metrics.update(schedule_audit)
        if kernel_schedule=="matrix-factorized":
            self.setup_metrics.update(endpoint_reduction='deterministic fused matrix tiles',
                endpoint_count_blocks=count_block_count,endpoint_count_capacity=count_capacity,
                endpoint_factor_bytes=0)
        if kernel_schedule=="matrix-factorized":
            from depth_downward_cuda import DepthDownward
            self.depth_down=DepthDownward(self,tree)
            self.setup_metrics.update(self.depth_down.metrics)
        if down_schedule=="depth-concurrent":
            from concurrent_buckets import ConcurrentBuckets
            setup_start=time.perf_counter()
            self.concurrent=ConcurrentBuckets([d for d,_,_ in meta["buckets"]],self.buckets,dev)
            self.setup_metrics.update(self.concurrent.metrics,down_schedule=down_schedule,
                concurrency_setup_seconds=time.perf_counter()-setup_start,allocation_seconds=time.perf_counter()-start)

    @torch.no_grad()
    def _run(self):
        self.counts.zero_()
        for first,last in self.up_levels:self.upward(self,first,last)
        n=self.root_patterns
        root_joint=self.up[:n].to(torch.float64)*self.pi
        root_prob=root_joint.sum(-1)
        if self.binary_scaling:
            # Zero invariant/underflow classes stay zero; never manufacture
            # positive probability from a log floor after integer scaling.
            catll=torch.where(root_prob>0,
                self.logs[:n].double()*math.log(2.)+root_prob.clamp_min(1e-300).log()+self.weights.log()[None,:],
                torch.full_like(root_prob,-torch.inf))
        else:catll=self.logs[:n]+root_prob.clamp_min(1e-300).log()+self.weights.log()[None,:]
        site_ll=torch.logsumexp(catll,1)
        posterior=(catll-site_ll[:,None]).exp()*self.root_frequency[:,None]
        theta=posterior[...,None]*root_joint/root_prob.clamp_min(1e-300)[...,None]
        self.theta[:n]=theta
        if hasattr(self,"depth_down"):
            self.depth_down.run(self)
        elif hasattr(self,"concurrent"):
            self.concurrent.run(self)
        else:
            for width,records in self.buckets:self.downward(self,width,records)
        self.reduce_counts(self)
        return (site_ll*self.root_frequency).sum(),self.counts.index_select(0,self.edge_index),theta.sum((0,1)),posterior.sum(0)
