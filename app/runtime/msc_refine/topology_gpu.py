"""Exact integer quartet classification on GPU; FP64 reduction stays on CPU."""
import numpy as np
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice
from . import _msc_active_gpu

@triton.jit
def _lca(a,b,A,R,V:tl.constexpr,W:tl.constexpr):
    fa=tl.load(A+4*V+a);fb=tl.load(A+4*V+b)
    left=tl.minimum(fa,fb);right=tl.maximum(fa,fb)
    level=31-libdevice.clz(right-left+1)
    u=tl.load(R+level*W+left)
    v=tl.load(R+level*W+right-(1<<level)+1)
    du=tl.load(A+3*V+u);dv=tl.load(A+3*V+v)
    return tl.where(du<=dv,u,v)

@triton.jit
def _ancestor(a,b,A,V:tl.constexpr):
    left=tl.load(A+a);right=tl.load(A+V+a);tip=tl.load(A+b)
    return (left<=tip)&(tip<right)

@triton.jit
def _pair(a,b):
    lo=tl.minimum(a,b);hi=tl.maximum(a,b)
    return tl.where(((lo==0)&(hi==1))|((lo==2)&(hi==3)),0,
                    tl.where(((lo==0)&(hi==2))|((lo==1)&(hi==3)),1,2))

@triton.jit
def _classify(Q,A,R,E,B,O,N,V:tl.constexpr,W:tl.constexpr,TAXA:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);mask=i<N
    a=tl.load(Q+i*4,mask,other=0);b=tl.load(Q+i*4+1,mask,other=1)
    c=tl.load(Q+i*4+2,mask,other=2);d=tl.load(Q+i*4+3,mask,other=3)
    valid=(a>=0)&(d<TAXA)&(a<b)&(b<c)&(c<d)
    ta=tl.load(A+a);tb=tl.load(A+b);tc=tl.load(A+c);td=tl.load(A+d)
    p0=tl.full((BLOCK,),0,tl.int32);p1=tl.full((BLOCK,),1,tl.int32)
    p2=tl.full((BLOCK,),2,tl.int32);p3=tl.full((BLOCK,),3,tl.int32)
    swap=ta>tb
    aa=tl.where(swap,b,a);bb=tl.where(swap,a,b);pp0=tl.where(swap,p1,p0);pp1=tl.where(swap,p0,p1)
    a=aa;b=bb;p0=pp0;p1=pp1;ta=tl.minimum(ta,tb);tb=tl.maximum(tl.load(A+aa),tl.load(A+bb))
    swap=tc>td
    cc=tl.where(swap,d,c);dd=tl.where(swap,c,d);pp2=tl.where(swap,p3,p2);pp3=tl.where(swap,p2,p3)
    c=cc;d=dd;p2=pp2;p3=pp3;tc=tl.load(A+c);td=tl.load(A+d)
    swap=ta>tc
    aa=tl.where(swap,c,a);cc=tl.where(swap,a,c);pp0=tl.where(swap,p2,p0);pp2=tl.where(swap,p0,p2)
    a=aa;c=cc;p0=pp0;p2=pp2;ta=tl.load(A+a);tc=tl.load(A+c)
    swap=tb>td
    bb=tl.where(swap,d,b);dd=tl.where(swap,b,d);pp1=tl.where(swap,p3,p1);pp3=tl.where(swap,p1,p3)
    b=bb;d=dd;p1=pp1;p3=pp3;tb=tl.load(A+b);td=tl.load(A+d)
    swap=tb>tc
    bb=tl.where(swap,c,b);cc=tl.where(swap,b,c);pp1=tl.where(swap,p2,p1);pp2=tl.where(swap,p1,p2)
    b=bb;c=cc;p1=pp1;p2=pp2
    h0=_lca(a,b,A,R,V,W);h1=_lca(b,c,A,R,V,W);h2=_lca(c,d,A,R,V,W)
    dh0=tl.load(A+3*V+h0);dh1=tl.load(A+3*V+h1);dh2=tl.load(A+3*V+h2)
    u=tl.where(dh0<dh1,h1,h0);v=tl.where(dh0<dh1,h0,h1)
    v=tl.where(dh2>tl.minimum(dh0,dh1),h2,v)
    cross=(dh1>dh0)&(dh1>dh2)
    tip0=tl.where(cross,b,a);tip2=tl.where(cross,a,c)
    pp0=tl.where(cross,p1,p0);pp1=tl.where(cross,p2,p1)
    pp2=tl.where(cross,p0,p2);pp3=p3
    p0=pp0;p1=pp1;p2=pp2;p3=pp3
    state=_pair(p0,p1)
    pu=tl.load(A+2*V+u);pv=tl.load(A+2*V+v)
    active=(pu==v)|(pv==u)
    uu=tl.minimum(u,v);vv=tl.maximum(u,v)
    child=tl.where(pu==v,u,v)
    e=tl.load(A+7*V+child)
    safe=tl.maximum(e,0)
    center=tl.where(_ancestor(child,tip0,A,V),child,tl.load(A+2*V+child))
    offset=tl.where(center==uu,0,2)
    branch=tl.load(B+safe*4+offset)
    parent=tl.load(A+2*V+center)
    in_first=tl.where(branch==parent,~_ancestor(center,tip0,A,V),_ancestor(branch,tip0,A,V))
    group0=offset+tl.where(in_first,0,1)
    center2=tl.where(center==uu,vv,uu);offset2=tl.where(center2==uu,0,2)
    branch2=tl.load(B+safe*4+offset2)
    parent2=tl.load(A+2*V+center2)
    in_second=tl.where(branch2==parent2,~_ancestor(center2,tip2,A,V),_ancestor(branch2,tip2,A,V))
    group2=offset2+tl.where(in_second,0,1)
    at0=tl.where(group0==0,p0,tl.where(group0==1,p1,tl.where(group2==0,p2,p3)))
    at2=tl.where(group0==2,p0,tl.where(group0==3,p1,tl.where(group2==2,p2,p3)))
    alt=_pair(at0,at2)
    code=state|tl.where(active,(alt<<2)|((e+1)<<4),0)
    code=tl.where(valid&(~active|((e>=0)&(alt!=state))),code,-1)
    tl.store(O+i,code,mask)

@torch.inference_mode()
def evaluate_gpu(tree,n,queries,weights,*,device='cuda:0'):
    idx=_msc_active_gpu.export_index(tree,n)
    dev=torch.device(device)
    arrays={key:torch.as_tensor(idx[key],device=dev,dtype=torch.int32) for key in ['nodes','rmq','edges','branches']}
    q=np.ascontiguousarray(queries,dtype=np.int32)
    if q.ndim!=2 or q.shape[1]!=4:
        raise ValueError('canonical Qx4 queries required')
    if len(q) and (q.min()<0 or q.max()>=n):
        raise ValueError('query out of range')
    query=torch.as_tensor(q,device=dev)
    codes=torch.empty((len(q),),device=dev,dtype=torch.int32)
    _classify[(triton.cdiv(len(q),256),)](query,arrays['nodes'],arrays['rmq'],arrays['edges'],arrays['branches'],codes,len(q),2*n-2,arrays['rmq'].shape[1],n,256)
    cpu=codes.cpu().numpy()
    result=_msc_active_gpu.reduce(cpu,weights,n-3)
    result['edges']=idx['edges']
    return result
