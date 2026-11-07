"""Same repeat DAG kernels; taxon/node counts are runtime scalars, not JIT keys.

Reference specialized kernels remain unchanged in repeats_cuda.py.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _partial(U,OFF,node,local,k,valid,K:tl.constexpr,NI,DT:tl.constexpr):
    inner=node<NI
    row=tl.load(OFF+node).to(tl.int64)+local
    off=(row*K+k)*4
    a=tl.load(U+off,valid&inner,other=0.).to(DT)
    b=tl.load(U+off+1,valid&inner,other=0.).to(DT)
    c=tl.load(U+off+2,valid&inner,other=0.).to(DT)
    d=tl.load(U+off+3,valid&inner,other=0.).to(DT)
    a=tl.where(inner,a,((local==0)|(local==4)).to(DT))
    b=tl.where(inner,b,((local==1)|(local==4)).to(DT))
    c=tl.where(inner,c,((local==2)|(local==4)).to(DT))
    d=tl.where(inner,d,((local==3)|(local==4)).to(DT))
    return a,b,c,d,row


@triton.jit
def _up(U,LOG,P,C,MAP,NODE,OFF,FIRST,LAST,K:tl.constexpr,NI,M,B:tl.constexpr,FLOOR:tl.constexpr):
    row=FIRST+tl.program_id(0)*B+tl.arange(0,B);k=tl.program_id(1);valid=row<LAST
    node=tl.load(NODE+row,valid,other=M)
    a=tl.full((B,),1.,U.dtype.element_ty);b=a;c=a;d=a
    logs=tl.full((B,),0.,tl.float64)
    for h in tl.static_range(3):
        child=tl.load(C+node*3+h,valid,other=M)
        local=tl.load(MAP+row*3+h,valid,other=0)
        mask=valid&(child<M)
        u0,u1,u2,u3,cp=_partial(U,OFF,child,local,k,mask,K,NI,U.dtype.element_ty)
        off=(child*K+k)*16
        y0=tl.load(P+off,mask,other=0.)*u0+tl.load(P+off+1,mask,other=0.)*u1+tl.load(P+off+2,mask,other=0.)*u2+tl.load(P+off+3,mask,other=0.)*u3
        y1=tl.load(P+off+4,mask,other=0.)*u0+tl.load(P+off+5,mask,other=0.)*u1+tl.load(P+off+6,mask,other=0.)*u2+tl.load(P+off+7,mask,other=0.)*u3
        y2=tl.load(P+off+8,mask,other=0.)*u0+tl.load(P+off+9,mask,other=0.)*u1+tl.load(P+off+10,mask,other=0.)*u2+tl.load(P+off+11,mask,other=0.)*u3
        y3=tl.load(P+off+12,mask,other=0.)*u0+tl.load(P+off+13,mask,other=0.)*u1+tl.load(P+off+14,mask,other=0.)*u2+tl.load(P+off+15,mask,other=0.)*u3
        a*=tl.where(child<M,y0,1.);b*=tl.where(child<M,y1,1.)
        c*=tl.where(child<M,y2,1.);d*=tl.where(child<M,y3,1.)
        logs+=tl.load(LOG+cp*K+k,mask&(child<NI),other=0.)
    norm=tl.maximum(a+b+c+d,FLOOR)
    off=(row*K+k)*4
    tl.store(U+off,a/norm,valid);tl.store(U+off+1,b/norm,valid)
    tl.store(U+off+2,c/norm,valid);tl.store(U+off+3,d/norm,valid)
    tl.store(LOG+row*K+k,logs+tl.log(norm.to(tl.float64)),valid)


@triton.jit
def _down(U,THETA,J,P,OFF,LINK,REC,R,K:tl.constexpr,NI,W:tl.constexpr,T:tl.constexpr,FLOOR:tl.constexpr):
    rr=tl.program_id(0)*T+tl.arange(0,T);k=tl.program_id(1);valid=rr<R
    node=tl.load(REC+rr*5,valid,other=NI)
    row=tl.load(REC+rr*5+1,valid,other=0)
    local=tl.load(REC+rr*5+2,valid,other=0)
    start=tl.load(REC+rr*5+3,valid,other=0)
    degree=tl.load(REC+rr*5+4,valid,other=0)
    w=tl.arange(0,W)
    parent=tl.load(LINK+start[:,None]+w[None,:],valid[:,None]&(w[None,:]<degree[:,None]),other=0)
    ptr=(parent*K+k)*4
    mask=valid[:,None]&(w[None,:]<degree[:,None])
    a0=tl.sum(tl.load(THETA+ptr,mask,other=0.),1)
    a1=tl.sum(tl.load(THETA+ptr+1,mask,other=0.),1)
    a2=tl.sum(tl.load(THETA+ptr+2,mask,other=0.),1)
    a3=tl.sum(tl.load(THETA+ptr+3,mask,other=0.),1)
    u0,u1,u2,u3,_=_partial(U,OFF,node,local,k,valid,K,NI,THETA.dtype.element_ty)
    off=(node*K+k)*16
    p00=tl.load(P+off,valid,other=0.).to(THETA.dtype.element_ty)
    p01=tl.load(P+off+1,valid,other=0.).to(THETA.dtype.element_ty)
    p02=tl.load(P+off+2,valid,other=0.).to(THETA.dtype.element_ty)
    p03=tl.load(P+off+3,valid,other=0.).to(THETA.dtype.element_ty)
    p10=tl.load(P+off+4,valid,other=0.).to(THETA.dtype.element_ty)
    p11=tl.load(P+off+5,valid,other=0.).to(THETA.dtype.element_ty)
    p12=tl.load(P+off+6,valid,other=0.).to(THETA.dtype.element_ty)
    p13=tl.load(P+off+7,valid,other=0.).to(THETA.dtype.element_ty)
    p20=tl.load(P+off+8,valid,other=0.).to(THETA.dtype.element_ty)
    p21=tl.load(P+off+9,valid,other=0.).to(THETA.dtype.element_ty)
    p22=tl.load(P+off+10,valid,other=0.).to(THETA.dtype.element_ty)
    p23=tl.load(P+off+11,valid,other=0.).to(THETA.dtype.element_ty)
    p30=tl.load(P+off+12,valid,other=0.).to(THETA.dtype.element_ty)
    p31=tl.load(P+off+13,valid,other=0.).to(THETA.dtype.element_ty)
    p32=tl.load(P+off+14,valid,other=0.).to(THETA.dtype.element_ty)
    p33=tl.load(P+off+15,valid,other=0.).to(THETA.dtype.element_ty)
    a0/=tl.maximum(p00*u0+p01*u1+p02*u2+p03*u3,FLOOR)
    a1/=tl.maximum(p10*u0+p11*u1+p12*u2+p13*u3,FLOOR)
    a2/=tl.maximum(p20*u0+p21*u1+p22*u2+p23*u3,FLOOR)
    a3/=tl.maximum(p30*u0+p31*u1+p32*u2+p33*u3,FLOOR)
    off=(row*K+k)*16
    tl.store(J+off,a0*p00*u0,valid);tl.store(J+off+1,a0*p01*u1,valid)
    tl.store(J+off+2,a0*p02*u2,valid);tl.store(J+off+3,a0*p03*u3,valid)
    tl.store(J+off+4,a1*p10*u0,valid);tl.store(J+off+5,a1*p11*u1,valid)
    tl.store(J+off+6,a1*p12*u2,valid);tl.store(J+off+7,a1*p13*u3,valid)
    tl.store(J+off+8,a2*p20*u0,valid);tl.store(J+off+9,a2*p21*u1,valid)
    tl.store(J+off+10,a2*p22*u2,valid);tl.store(J+off+11,a2*p23*u3,valid)
    tl.store(J+off+12,a3*p30*u0,valid);tl.store(J+off+13,a3*p31*u1,valid)
    tl.store(J+off+14,a3*p32*u2,valid);tl.store(J+off+15,a3*p33*u3,valid)
    off=(row*K+k)*4
    tl.store(THETA+off,(a0*p00+a1*p10+a2*p20+a3*p30)*u0,valid&(node<NI))
    tl.store(THETA+off+1,(a0*p01+a1*p11+a2*p21+a3*p31)*u1,valid&(node<NI))
    tl.store(THETA+off+2,(a0*p02+a1*p12+a2*p22+a3*p32)*u2,valid&(node<NI))
    tl.store(THETA+off+3,(a0*p03+a1*p13+a2*p23+a3*p33)*u3,valid&(node<NI))


@triton.jit
def _reduce(J,CHUNKS,COUNT,K:tl.constexpr,B:tl.constexpr):
    chunk=tl.program_id(0);k=tl.program_id(1)
    node=tl.load(CHUNKS+chunk*3);start=tl.load(CHUNKS+chunk*3+1);n=tl.load(CHUNKS+chunk*3+2)
    s=tl.arange(0,B);j=tl.arange(0,16)
    v=tl.load(J+((start+s[:,None])*K+k)*16+j[None,:],s[:,None]<n,other=0.).to(tl.float64)
    tl.atomic_add(COUNT+(node*K+k)*16+j,tl.sum(v,0))


def upward(plan,first,last):
    floor=1e-300 if plan.dtype==torch.float64 else 1e-30
    _up[(triton.cdiv(last-first,128),plan.categories)](plan.up,plan.logs,plan.trans,plan.children,
        plan.mapping,plan.pattern_nodes,plan.offsets,first,last,plan.categories,plan.ni,plan.m,128,floor)


def downward(plan,width,records):
    tile=max(1,min(128,128//width))
    floor=1e-300 if plan.ddtype==torch.float64 else 1e-30
    _down[(triton.cdiv(len(records),tile),plan.categories)](plan.up,plan.theta,plan.joint,plan.trans,
        plan.offsets,plan.links,records,len(records),plan.categories,plan.ni,width,tile,floor)


def reduce_counts(plan):
    _reduce[(len(plan.chunks),plan.categories)](plan.joint,plan.chunks,plan.counts,plan.categories,128)
