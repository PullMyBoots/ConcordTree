"""Native, monotone, bounded three-branch trust-Newton placement optimizer."""
import ctypes,time
from pathlib import Path
import numpy as np
from .fast_likelihood import EdgeCache,D,U
LIB=ctypes.CDLL(str(Path(__file__).with_name('insertion_newton.so')))
LIB.insertion_optimize.argtypes=[ctypes.c_int]*2+[D]*3+[U]+[D]*7+[ctypes.c_int]*2+[D]*2
LIB.insertion_optimize.restype=ctypes.c_double
LIB.insertion_derivatives.argtypes=[ctypes.c_int]*2+[D]*3+[U]+[D]*9
LIB.insertion_derivatives.restype=ctypes.c_double

class NewtonCache(EdgeCache):
    def arguments(self,e,query):
        a,b=e;m=self.model
        return (self.S,self.K,self.messages[a,b],self.messages[b,a],
            np.ascontiguousarray(self.scales[a,b]+self.scales[b,a]),np.ascontiguousarray(query,dtype=np.uint8),
            self.count,m['pi'],m['weights'],m['eigenvalues'],np.ascontiguousarray(m['vectors']),m['rates'])

    def place(self,selected,query,pendant,iterations=40):
        start=time.perf_counter();l=self.tree.lengths[selected]
        initial=np.array([max(l/2,1e-6),max(l/2,1e-6),pendant]);lengths=np.zeros(3);info=np.zeros(4)
        ll=LIB.insertion_optimize(*self.arguments(selected,query),initial,iterations,160,lengths,info)
        if not np.isfinite(ll) or not np.isfinite(lengths).all():raise ArithmeticError('nonfinite Newton placement')
        result=dict(edge=list(selected),log_likelihood=float(ll),insertion_lengths=lengths.tolist(),
            converged=bool(info[0]==0),iterations=int(info[1]),evaluations=int(info[2]),
            optimizer='native monotone trust-Newton',native_status=int(info[0]),projected_gradient=float(info[3]),fallback=False)
        if info[0]!=0:
            # A bounded numerical fallback, selected without truth or score gaps.
            # Retain the best observed likelihood; neither path can cause a loop.
            other=super().place(selected,query,pendant,iterations)
            if other['log_likelihood']>result['log_likelihood']:
                result.update(log_likelihood=other['log_likelihood'],insertion_lengths=other['insertion_lengths'],
                    converged=other['converged'],optimizer='native Newton + bounded L-BFGS-B fallback')
            result['fallback']=True;result['fallback_evaluations']=other['evaluations']
        result['seconds']=time.perf_counter()-start
        return result
