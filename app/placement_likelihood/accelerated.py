"""Equivalent native transition/model pullback, optional explicit activation."""
import ctypes
from pathlib import Path
import numpy as np
from scipy.special import gammainc,gammaincinv
from . import fast_likelihood as base
LIB=ctypes.CDLL(str(Path(__file__).with_name('model_gradient.so')))
LIB.model_gradient.argtypes=[ctypes.c_int]*2+[base.D]*12+[ctypes.c_double,base.D]

def pullback(model,t,p,dp,dpi,dw):
    a=model['alpha'];x=np.log(a);lo=max(np.log(.02),x-1e-4);hi=min(np.log(100),x+1e-4)
    def gamma(alpha):
        z=gammaincinv(alpha,np.linspace(0,1,5));r=4*np.diff(gammainc(alpha+1,z));return r/r.mean()
    dr=np.r_[0.,(gamma(np.exp(hi))-gamma(np.exp(lo)))/(hi-lo)/(1-model['pinv'])]
    out=np.zeros(10)
    args=[p,dp,t,model['pi'],model['q'],model['eigenvalues'],model['vectors'],model['rates'],model['exchangeabilities'],dpi,dw,dr]
    LIB.model_gradient(len(t),len(model['rates']),*[np.ascontiguousarray(a,dtype=np.float64) for a in args],model['pinv'],out)
    return out

def activate():base.model_pullback=pullback
