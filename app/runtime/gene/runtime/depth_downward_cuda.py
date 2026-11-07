"""Depth-batched dispatch of unchanged deterministic posterior/count tiles."""
import numpy as np
import torch
import triton
import triton.language as tl
from concordtree.treeio import rooted
from repeats_fused_counts import _down


@triton.jit
def _dispatch(U,THETA,PARTIAL,P,OFF,LINK,WORK,K:tl.constexpr,NI,
              MAXPOWER:tl.constexpr,PRESENT:tl.constexpr,FLOOR:tl.constexpr,INDEX:tl.constexpr):
    block=tl.program_id(0)
    records=tl.load(WORK+block*4).to(INDEX)
    blocks=tl.load(WORK+block*4+1).to(INDEX)
    local=tl.load(WORK+block*4+2)
    power=tl.load(WORK+block*4+3)
    # Generated static width cases for the native builder L<=2,000,000 bound.
    # Explicit constexpr tile shapes keep the unchanged reduction geometry.
    if PRESENT & 1:
      if power==0:
        _down(U,THETA,PARTIAL,P,OFF,LINK,records,blocks+(local-block)*4,
              K,NI,1,128,FLOOR)
    if PRESENT & 2:
      if power==1:
        _down(U,THETA,PARTIAL,P,OFF,LINK,records,blocks+(local-block)*4,
              K,NI,2,64,FLOOR)
    if PRESENT & 4:
      if power==2:
        _down(U,THETA,PARTIAL,P,OFF,LINK,records,blocks+(local-block)*4,
              K,NI,4,32,FLOOR)
    if PRESENT & 8:
      if power==3:
        _down(U,THETA,PARTIAL,P,OFF,LINK,records,blocks+(local-block)*4,
              K,NI,8,16,FLOOR)
    if PRESENT & 16:
      if power==4:
        _down(U,THETA,PARTIAL,P,OFF,LINK,records,blocks+(local-block)*4,
              K,NI,16,8,FLOOR)
    if PRESENT & 32:
      if power==5:
        _down(U,THETA,PARTIAL,P,OFF,LINK,records,blocks+(local-block)*4,
              K,NI,32,4,FLOOR)
    if PRESENT & 64:
      if power==6:
        _down(U,THETA,PARTIAL,P,OFF,LINK,records,blocks+(local-block)*4,
              K,NI,64,2,FLOOR)
    if PRESENT & 128:
      if power==7:
        _down(U,THETA,PARTIAL,P,OFF,LINK,records,blocks+(local-block)*4,
              K,NI,128,1,FLOOR)
    if PRESENT & 256:
      if power==8:
        _down(U,THETA,PARTIAL,P,OFF,LINK,records,blocks+(local-block)*4,
              K,NI,256,1,FLOOR)
    if PRESENT & 512:
      if power==9:
        _down(U,THETA,PARTIAL,P,OFF,LINK,records,blocks+(local-block)*4,
              K,NI,512,1,FLOOR)
    if PRESENT & 1024:
      if power==10:
        _down(U,THETA,PARTIAL,P,OFF,LINK,records,blocks+(local-block)*4,
              K,NI,1024,1,FLOOR)
    if PRESENT & 2048:
      if power==11:
        _down(U,THETA,PARTIAL,P,OFF,LINK,records,blocks+(local-block)*4,
              K,NI,2048,1,FLOOR)
    if PRESENT & 4096:
      if power==12:
        _down(U,THETA,PARTIAL,P,OFF,LINK,records,blocks+(local-block)*4,
              K,NI,4096,1,FLOOR)
    if PRESENT & 8192:
      if power==13:
        _down(U,THETA,PARTIAL,P,OFF,LINK,records,blocks+(local-block)*4,
              K,NI,8192,1,FLOOR)
    if PRESENT & 16384:
      if power==14:
        _down(U,THETA,PARTIAL,P,OFF,LINK,records,blocks+(local-block)*4,
              K,NI,16384,1,FLOOR)
    if PRESENT & 32768:
      if power==15:
        _down(U,THETA,PARTIAL,P,OFF,LINK,records,blocks+(local-block)*4,
              K,NI,32768,1,FLOOR)
    if PRESENT & 65536:
      if power==16:
        _down(U,THETA,PARTIAL,P,OFF,LINK,records,blocks+(local-block)*4,
              K,NI,65536,1,FLOOR)
    if PRESENT & 131072:
      if power==17:
        _down(U,THETA,PARTIAL,P,OFF,LINK,records,blocks+(local-block)*4,
              K,NI,131072,1,FLOOR)
    if PRESENT & 262144:
      if power==18:
        _down(U,THETA,PARTIAL,P,OFF,LINK,records,blocks+(local-block)*4,
              K,NI,262144,1,FLOOR)
    if PRESENT & 524288:
      if power==19:
        _down(U,THETA,PARTIAL,P,OFF,LINK,records,blocks+(local-block)*4,
              K,NI,524288,1,FLOOR)
    if PRESENT & 1048576:
      if power==20:
        _down(U,THETA,PARTIAL,P,OFF,LINK,records,blocks+(local-block)*4,
              K,NI,1048576,1,FLOOR)
    if PRESENT & 2097152:
      if power==21:
        _down(U,THETA,PARTIAL,P,OFF,LINK,records,blocks+(local-block)*4,
              K,NI,2097152,1,FLOOR)


class DepthDownward:
    def __init__(self,plan,tree):
        _,_,_,depth=rooted(tree)
        byid={plan.ids[u]:d for u,d in depth.items()}
        last=max(depth.values())+1
        groups={}
        for width,(records,blocks) in plan.buckets:
            node=int(records[0,0].item())
            d=byid[node] if node<plan.ni else last
            power=width.bit_length()-1
            work=np.empty((len(blocks),4),dtype=np.int64)
            work[:,0]=records.data_ptr();work[:,1]=blocks.data_ptr()
            work[:,2]=np.arange(len(blocks));work[:,3]=power
            groups.setdefault(d,[]).append((power,work))
        # T*W<=128 is the inherited tile budget, not a new scientific control.
        # Compile one complete low-width dispatch for every tree shape.
        self.groups=[]
        self.high={}
        self.index=tl.pointer_type(tl.int32 if plan.buckets[0][1][0].dtype==torch.int32 else tl.int64)
        for width,data in plan.buckets:
            if width<=128:continue
            node=int(data[0][0,0].item())
            d=byid[node] if node<plan.ni else last
            self.high.setdefault(d,[]).append((width,data))
        for d,parts in sorted(groups.items()):
            low=[w for power,w in parts if power<=7]
            work=torch.tensor(np.concatenate(low),device=plan.device) if low else None
            self.groups.append((d,work))
        self.metrics=dict(original_launches=len(plan.buckets),
                          depth_launches=sum(w is not None for _,w in self.groups)+sum(len(v) for v in self.high.values()),
                          extra_descriptor_bytes=sum(w.numel()*w.element_size() for _,w in self.groups if w is not None),
                          compile_strategy='one invariant low-width dispatch; original kernels beyond row-tile budget')

    def run(self,plan):
        floor=1e-300 if plan.ddtype==torch.float64 else 1e-30
        from repeats_fused_counts import downward
        for d,work in self.groups:
            if work is not None:
                _dispatch[(len(work),plan.categories)](plan.up,plan.theta,plan.count_partials,
                    plan.trans,plan.offsets,plan.links,work,plan.categories,plan.ni,7,255,floor,self.index)
            for width,data in self.high.get(d,[]):
                downward(plan,width,data)
