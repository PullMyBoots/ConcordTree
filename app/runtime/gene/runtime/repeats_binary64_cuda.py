"""Repeat-DAG upward messages in CLV * 2**int32 form.

Normalize after EACH multiplied child, so an informative product is never left
as an unscaled product across a high-degree root. Every retained positive-rate
category keeps its own exponent. The invariant zero vector remains exactly zero.
Only root mixtures use FP64 log conversion, not every pattern at every node.
"""
import triton
import triton.language as tl
from repeats_generic_cuda import _partial
@triton.jit
def _normalize64(a,b,c,d):
    # Normal values are represented exactly as mantissa * 2**exponent.
    # The floor affects only the normalization factor, never the likelihood.
    norm=tl.maximum(a+b+c+d,tl.full((),1.e-300,tl.float64))
    shift=(((norm.to(tl.uint64,bitcast=True)>>52)&2047).to(tl.int32)-1022)
    factor=((1023-shift).to(tl.uint64)<<52).to(tl.float64,bitcast=True)
    return a*factor,b*factor,c*factor,d*factor,shift


@triton.jit
def _up(U,EXP,P,C,MAP,NODE,OFF,FIRST,LAST,K:tl.constexpr,NI,M,B:tl.constexpr,WIDE:tl.constexpr=False):
    program=tl.program_id(0)
    if WIDE:program=program.to(tl.int64)
    row=FIRST+program*B+tl.arange(0,B);k=tl.program_id(1);valid=row<LAST
    node=tl.load(NODE+row,valid,other=M)
    a=tl.full((B,),1.,tl.float64);b=a;c=a;d=a
    exponent=tl.full((B,),0,tl.int32)
    for h in tl.static_range(3):
        child=tl.load(C+node*3+h,valid,other=M)
        local=tl.load(MAP+row*3+h,valid,other=0)
        mask=valid&(child<M)
        u0,u1,u2,u3,cp=_partial(U,OFF,child,local,k,mask,K,NI,tl.float64)
        off=(child*K+k)*16
        y0=tl.load(P+off,mask,other=0.)*u0+tl.load(P+off+1,mask,other=0.)*u1+tl.load(P+off+2,mask,other=0.)*u2+tl.load(P+off+3,mask,other=0.)*u3
        y1=tl.load(P+off+4,mask,other=0.)*u0+tl.load(P+off+5,mask,other=0.)*u1+tl.load(P+off+6,mask,other=0.)*u2+tl.load(P+off+7,mask,other=0.)*u3
        y2=tl.load(P+off+8,mask,other=0.)*u0+tl.load(P+off+9,mask,other=0.)*u1+tl.load(P+off+10,mask,other=0.)*u2+tl.load(P+off+11,mask,other=0.)*u3
        y3=tl.load(P+off+12,mask,other=0.)*u0+tl.load(P+off+13,mask,other=0.)*u1+tl.load(P+off+14,mask,other=0.)*u2+tl.load(P+off+15,mask,other=0.)*u3
        a*=tl.where(child<M,y0,1.);b*=tl.where(child<M,y1,1.)
        c*=tl.where(child<M,y2,1.);d*=tl.where(child<M,y3,1.)
        a,b,c,d,shift=_normalize64(a,b,c,d)
        exponent+=shift+tl.load(EXP+cp*K+k,mask&(child<NI),other=0)
    off=(row*K+k)*4
    tl.store(U+off,a,valid);tl.store(U+off+1,b,valid)
    tl.store(U+off+2,c,valid);tl.store(U+off+3,d,valid)
    tl.store(EXP+row*K+k,exponent,valid)

def upward(plan,first,last):
    _up[(triton.cdiv(last-first,128),plan.categories)](plan.up,plan.logs,plan.trans,plan.children,
        plan.mapping,plan.pattern_nodes,plan.offsets,first,last,plan.categories,plan.ni,plan.m,128,
        WIDE=plan.r*max(3,plan.categories*4)>2**31-1,
        enable_fp_fusion=True)
