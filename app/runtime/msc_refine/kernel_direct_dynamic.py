"""Same direct Gaussian arithmetic, with runtime row count to avoid JIT churn."""
import triton
import triton.language as tl

@triton.jit
def _direct_runtime(X,P,Z,C,OUT,N,M:tl.constexpr,T:tl.constexpr,
                    SCALE:tl.constexpr,BQ:tl.constexpr,BZ:tl.constexpr):
    q=tl.program_id(0)*BQ+tl.arange(0,BQ);cls=tl.program_id(1);tile=tl.program_id(2)
    z=tile*BZ+tl.arange(0,BZ);distance=tl.full((BQ,BZ),0,tl.float32)
    for d in tl.static_range(15):
        perm=tl.load(P+cls*15+d)
        a=tl.load(X+q*15+perm,q<N,other=0)
        b=tl.load(Z+z*15+d,z<M,other=0)
        difference=a[:,None]-b[None,:]
        distance=tl.fma(difference,difference,distance)
    coefficient=tl.load(C+z,z<M,other=0)
    value=tl.sum(tl.exp(-distance*SCALE)*coefficient[None,:],1)
    tl.store(OUT+(q*3+cls)*T+tile,value,q<N)

