"""Exact subtree repeat compression for reference fitting; no site removal."""
import ctypes
from pathlib import Path
import numpy as np
from . import fast_likelihood as base
LIB=ctypes.CDLL(str(Path(__file__).with_name('repeat_prune.so')))
LIB.repeat_prune.argtypes=[ctypes.c_int]*3+[base.I]*3+[base.U]+[base.D]*7
LIB.repeat_prune.restype=ctypes.c_double

class RepeatPlan(base.Plan):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        inverse={};combinations={};sizes=[]
        for u in range(self.M-1,-1,-1):
            children=[int(v) for v in self.child[u] if v>=0]
            if not children:
                unique,inv=np.unique(self.obs[u],return_inverse=True);combinations[u]=(unique,None)
            else:
                keys=np.column_stack([inverse[v] for v in children]);unique,inv=np.unique(keys,axis=0,return_inverse=True)
                combinations[u]=(unique,children)
            inverse[u]=np.asarray(inv,dtype=np.int32)
        self.offsets=np.r_[0,np.cumsum([len(combinations[u][0]) for u in range(self.M)])].astype(np.int32)
        self.R=int(self.offsets[-1]);self.rc=np.full((self.R,3),-1,dtype=np.int32);self.robs=np.full(self.R,4,dtype=np.uint8)
        for u in range(self.M):
            unique,children=combinations[u];start=self.offsets[u];end=self.offsets[u+1]
            if children is None:self.robs[start:end]=unique
            else:
                for j,ch in enumerate(children):self.rc[start:end,j]=self.offsets[ch]+unique[:,j]
        self.root_count=np.ascontiguousarray(np.bincount(inverse[0],weights=self.count,minlength=self.offsets[1]),dtype=np.float64)
        self.repeat_ratio=self.R/(self.M*self.S)

    def evaluate(self,P,pi,w):
        dp=np.zeros_like(P);dpi=np.zeros(4);dw=np.zeros(len(w))
        ll=LIB.repeat_prune(self.M,self.R,len(w),self.offsets,self.ce,self.rc,self.robs,self.root_count,
            np.ascontiguousarray(P),np.ascontiguousarray(pi),np.ascontiguousarray(w),dp,dpi,dw)
        if not np.isfinite(ll) or not np.isfinite(dp).all():raise ArithmeticError('nonfinite repeat likelihood')
        return ll,dp,dpi,dw

def activate():
    from . import profiles
    profiles.Plan=RepeatPlan
