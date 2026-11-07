"""Same pruning arithmetic, pack multiple short-site rows into one CUDA CTA.

For width<128, the incumbent wastes most lanes and launches one CTA per row and
category. Flattening the independent depth-frontier rows preserves the DAG and
per-message operation order, without enlarging the fixed message workspace.
"""
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice as ld
from concordtree.jc69_cuda import _read4,_write4
from streamed_likelihood import _p4

@triton.jit
def _flat_leaf(U,LOG,P,OBS,LEAF_ROWS,LEAF_INDEX,ROW_EDGE,M,K:tl.constexpr,W:tl.constexpr,T:tl.constexpr,FLOOR:tl.constexpr):
    pos=tl.program_id(0)*T+tl.arange(0,T);j=pos//W;s=pos%W;ok=j<M;k=tl.program_id(1)
    row=tl.load(LEAF_ROWS+j,ok,other=0);tax=tl.load(LEAF_INDEX+j,ok,other=0)
    code=tl.load(OBS+tax*W+s,ok,other=4);e=tl.load(ROW_EDGE+row,ok,other=0)
    v0,v1,v2,v3=_p4(P,e,k,((code==0)|(code==4)).to(U.dtype.element_ty),
        ((code==1)|(code==4)).to(U.dtype.element_ty),((code==2)|(code==4)).to(U.dtype.element_ty),
        ((code==3)|(code==4)).to(U.dtype.element_ty),K)
    norm=tl.maximum(tl.maximum(tl.maximum(v0,v1),tl.maximum(v2,v3)),FLOOR)
    _write4(U,row,k*W+s,K*W,v0/norm,v1/norm,v2/norm,v3/norm,ok)
    tl.store(LOG+(row*K+k)*W+s,ld.log(norm.to(tl.float64)),ok)

@triton.jit
def _flat_send(U,LOG,P,ROWS,DEPEND,ROW_EDGE,M,K:tl.constexpr,W:tl.constexpr,T:tl.constexpr,FLOOR:tl.constexpr):
    pos=tl.program_id(0)*T+tl.arange(0,T);j=pos//W;s=pos%W;ok=j<M;k=tl.program_id(1)
    row=tl.load(ROWS+j,ok,other=0);r0=tl.load(DEPEND+row*2,ok,other=0);r1=tl.load(DEPEND+row*2+1,ok,other=0)
    e=tl.load(ROW_EDGE+row,ok,other=0)
    a,b,c,d=_read4(U,r0,k*W+s,K*W,ok);x,y,z,w=_read4(U,r1,k*W+s,K*W,ok)
    a*=x;b*=y;c*=z;d*=w
    norm=tl.maximum(tl.maximum(tl.maximum(a,b),tl.maximum(c,d)),FLOOR)
    x,y,z,w=_p4(P,e,k,a/norm,b/norm,c/norm,d/norm,K)
    norm2=tl.maximum(tl.maximum(tl.maximum(x,y),tl.maximum(z,w)),FLOOR)
    _write4(U,row,k*W+s,K*W,x/norm2,y/norm2,z/norm2,w/norm2,ok)
    scale=tl.load(LOG+(r0*K+k)*W+s,ok,other=0.)+tl.load(LOG+(r1*K+k)*W+s,ok,other=0.)
    tl.store(LOG+(row*K+k)*W+s,scale+ld.log(norm.to(tl.float64))+ld.log(norm2.to(tl.float64)),ok)
