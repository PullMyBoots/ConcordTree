"""Small stationary DNA decoder/Jacobian, original first-order semantics.

No Torch tape. Gamma conditional means and its h=1e-4 log-alpha tangent call
the unchanged release routine, including one-sided original-bound handling.
The matrix-normalization and +FO/+I Jacobians follow ANALYTIC_SCORE_BRIEF.md.
"""
from dataclasses import dataclass
import numpy as np
from concordtree.dna_models import gamma_category_means


@dataclass
class DecodedModel:
    values: np.ndarray  # pi(4), Q(16), site rates(K), category weights(K)
    jacobian: np.ndarray  # values x original free log/logit coordinates
    exchangeabilities: np.ndarray
    alpha: float
    pinv: float


def decode_numpy(parameters, vector):
    x=np.asarray(vector,dtype=np.float64);d=len(parameters.initial)
    if x.shape!=(d,) or not np.all(np.isfinite(x)):
        raise ValueError("invalid small-model coordinates")
    spec=parameters.spec;parts=parameters.parts
    pi=parameters.pi.copy();dpi=np.zeros((4,d))
    if 'pi' in parts:
        z=np.r_[x[parts['pi']],0.];b=np.exp(z-z.max());b/=b.sum()
        pi=(b+1e-8)/(1+4e-8)
        dpi[:,parts['pi']]=(np.diag(b)[:,:3]-b[:,None]*b[None,:3])/(1+4e-8)
    exchange=parameters.rates.copy();ds=np.zeros((6,d))
    if 'rates' in parts:
        free=np.r_[np.exp(x[parts['rates']]),1.]
        exchange=free[np.array(spec.groups)]
        for i,g in enumerate(spec.groups):
            if g>=0:ds[i,parts['rates'].start+g]=exchange[i]
    a,b=np.triu_indices(4,1);s=np.zeros((4,4));s[a,b]=exchange;s[b,a]=exchange
    deriv=np.zeros((4,4,d));deriv[a,b]=ds;deriv[b,a]=ds
    matrix=s*pi[None,:]
    dm=deriv*pi[None,:,None]+s[:,:,None]*dpi[None,:,:]
    diagonal=np.diag_indices(4)
    matrix[diagonal]=-matrix.sum(1);dm[diagonal]=-dm.sum(1)
    mu=-np.sum(pi*matrix.diagonal())
    dmu=-np.sum(dpi*matrix.diagonal()[:,None]+pi[:,None]*dm[diagonal],axis=0)
    q=matrix/mu;dq=dm/mu-q[:,:,None]*(dmu/mu)

    loga=float(x[parts['alpha']][0]) if 'alpha' in parts else float(np.log(parameters.alpha))
    alpha=float(np.exp(loga));k=spec.gamma_categories
    variable=gamma_category_means(min(100.,max(.02,alpha)),k) if k>1 else np.ones(1)
    dv=np.zeros((k,d))
    if 'alpha' in parts:
        lo=max(np.log(.02),loga-1e-4);hi=min(np.log(100.),loga+1e-4)
        dv[:,parts['alpha'].start]=(gamma_category_means(min(100.,np.exp(hi)),k)-
            gamma_category_means(max(.02,np.exp(lo)),k))/(hi-lo)
    pinv=0.;dp=np.zeros(d)
    if spec.invariant:
        pinv=(float(1/(1+np.exp(-x[parts['pinv']][0]))) if 'pinv' in parts else parameters.pinv)
        if 'pinv' in parts:dp[parts['pinv'].start]=pinv*(1-pinv)
        rates=np.r_[0.,variable/(1-pinv)]
        dr=np.vstack((np.zeros((1,d)),dv/(1-pinv)+variable[:,None]*dp[None,:]/(1-pinv)**2))
        weights=np.r_[pinv,np.full(k,(1-pinv)/k)]
        dw=np.vstack((dp,np.broadcast_to(-dp/k,(k,d))))
    else:
        rates=variable;dr=dv;weights=np.full(k,1/k);dw=np.zeros((k,d))
    return DecodedModel(np.r_[pi,q.reshape(-1),rates,weights],
        np.concatenate((dpi,dq.reshape(16,d),dr,dw)),exchange,alpha,pinv)
