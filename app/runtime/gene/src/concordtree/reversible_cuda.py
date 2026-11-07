"""Fused category-aware float64 DNA pruning and endpoint-count kernels."""
import torch
import triton
import triton.language as tl
from concordtree.jc69_cuda import _read4, _write4


@triton.jit
def _transform(P,node,k,a,b,c,d,K:tl.constexpr):
    off=(node*K+k)*16
    y0=tl.load(P+off)*a+tl.load(P+off+1)*b+tl.load(P+off+2)*c+tl.load(P+off+3)*d
    y1=tl.load(P+off+4)*a+tl.load(P+off+5)*b+tl.load(P+off+6)*c+tl.load(P+off+7)*d
    y2=tl.load(P+off+8)*a+tl.load(P+off+9)*b+tl.load(P+off+10)*c+tl.load(P+off+11)*d
    y3=tl.load(P+off+12)*a+tl.load(P+off+13)*b+tl.load(P+off+14)*c+tl.load(P+off+15)*d
    return y0,y1,y2,y3


@triton.jit
def _up(U,P,C,V,LOG,K:tl.constexpr,S:tl.constexpr,H:tl.constexpr,B:tl.constexpr):
    pos=tl.program_id(0); node=tl.load(V+pos); k=tl.program_id(1)
    s=tl.program_id(2)*B+tl.arange(0,B); valid=s<S
    a=tl.full((B,),1.,tl.float64); b=a; c=a; d=a
    for h in tl.static_range(H):
        child=tl.load(C+node*H+h)
        u0,u1,u2,u3=_read4(U,child,k*S+s,K*S,valid)
        y0,y1,y2,y3=_transform(P,child,k,u0,u1,u2,u3,K)
        a*=y0; b*=y1; c*=y2; d*=y3
    norm=tl.maximum(a+b+c+d,1e-300)
    _write4(U,node,k*S+s,K*S,a/norm,b/norm,c/norm,d/norm,valid)
    tl.store(LOG+(pos*K+k)*S+s,tl.log(norm),valid)


@triton.jit
def _down(U,D,P,C,PAR,V,POST,COUNT,K:tl.constexpr,S:tl.constexpr,H:tl.constexpr,B:tl.constexpr):
    node=tl.load(V+tl.program_id(0)); parent=tl.load(PAR+node); k=tl.program_id(1)
    s=tl.program_id(2)*B+tl.arange(0,B); valid=s<S
    a0,a1,a2,a3=_read4(D,parent,k*S+s,K*S,valid)
    for h in tl.static_range(H):
        child=tl.load(C+parent*H+h)
        u0,u1,u2,u3=_read4(U,child,k*S+s,K*S,valid)
        y0,y1,y2,y3=_transform(P,child,k,u0,u1,u2,u3,K)
        a0*=tl.where(child==node,1.,y0); a1*=tl.where(child==node,1.,y1)
        a2*=tl.where(child==node,1.,y2); a3*=tl.where(child==node,1.,y3)
    norm=tl.maximum(a0+a1+a2+a3,1e-300)
    a0/=norm; a1/=norm; a2/=norm; a3/=norm
    u0,u1,u2,u3=_read4(U,node,k*S+s,K*S,valid)
    y0,y1,y2,y3=_transform(P,node,k,u0,u1,u2,u3,K)
    den=tl.maximum(a0*y0+a1*y1+a2*y2+a3*y3,1e-300)
    weight=tl.load(POST+k*S+s,valid,other=0.)/den
    off=(node*K+k)*16
    p00=tl.load(P+off+0)
    tl.atomic_add(COUNT+off+0,tl.sum(tl.where(valid,weight*a0*p00*u0,0.)))
    p01=tl.load(P+off+1)
    tl.atomic_add(COUNT+off+1,tl.sum(tl.where(valid,weight*a0*p01*u1,0.)))
    p02=tl.load(P+off+2)
    tl.atomic_add(COUNT+off+2,tl.sum(tl.where(valid,weight*a0*p02*u2,0.)))
    p03=tl.load(P+off+3)
    tl.atomic_add(COUNT+off+3,tl.sum(tl.where(valid,weight*a0*p03*u3,0.)))
    p10=tl.load(P+off+4)
    tl.atomic_add(COUNT+off+4,tl.sum(tl.where(valid,weight*a1*p10*u0,0.)))
    p11=tl.load(P+off+5)
    tl.atomic_add(COUNT+off+5,tl.sum(tl.where(valid,weight*a1*p11*u1,0.)))
    p12=tl.load(P+off+6)
    tl.atomic_add(COUNT+off+6,tl.sum(tl.where(valid,weight*a1*p12*u2,0.)))
    p13=tl.load(P+off+7)
    tl.atomic_add(COUNT+off+7,tl.sum(tl.where(valid,weight*a1*p13*u3,0.)))
    p20=tl.load(P+off+8)
    tl.atomic_add(COUNT+off+8,tl.sum(tl.where(valid,weight*a2*p20*u0,0.)))
    p21=tl.load(P+off+9)
    tl.atomic_add(COUNT+off+9,tl.sum(tl.where(valid,weight*a2*p21*u1,0.)))
    p22=tl.load(P+off+10)
    tl.atomic_add(COUNT+off+10,tl.sum(tl.where(valid,weight*a2*p22*u2,0.)))
    p23=tl.load(P+off+11)
    tl.atomic_add(COUNT+off+11,tl.sum(tl.where(valid,weight*a2*p23*u3,0.)))
    p30=tl.load(P+off+12)
    tl.atomic_add(COUNT+off+12,tl.sum(tl.where(valid,weight*a3*p30*u0,0.)))
    p31=tl.load(P+off+13)
    tl.atomic_add(COUNT+off+13,tl.sum(tl.where(valid,weight*a3*p31*u1,0.)))
    p32=tl.load(P+off+14)
    tl.atomic_add(COUNT+off+14,tl.sum(tl.where(valid,weight*a3*p32*u2,0.)))
    p33=tl.load(P+off+15)
    tl.atomic_add(COUNT+off+15,tl.sum(tl.where(valid,weight*a3*p33*u3,0.)))
    d0=a0*p00+a1*p10+a2*p20+a3*p30
    d1=a0*p01+a1*p11+a2*p21+a3*p31
    d2=a0*p02+a1*p12+a2*p22+a3*p32
    d3=a0*p03+a1*p13+a2*p23+a3*p33
    _write4(D,node,k*S+s,K*S,d0,d1,d2,d3,valid)


def upward(up, transitions, children, level):
    k,s=up.shape[1:3]
    logs=torch.empty((len(level),k,s),device=up.device,dtype=torch.float64)
    _up[(len(level),k,triton.cdiv(s,128))](up,transitions,children,level,logs,k,s,children.shape[1],128)
    return logs.sum(0)


def downward(up, down, transitions, children, parents, level, posterior, counts):
    k,s=up.shape[1:3]
    _down[(len(level),k,triton.cdiv(s,128))](up,down,transitions,children,parents,level,posterior,counts,
                                        k,s,children.shape[1],128)
