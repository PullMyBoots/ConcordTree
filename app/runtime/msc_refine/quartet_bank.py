"""Branch-integrated JC69+Gamma quartet evidence; no branch optimization.

This experimental model is deliberately restrictive, not a GTR implementation.
Observed loci are additional input. A finite quadrature bank is an approximation,
not a general MSC consistency guarantee. Constant-site conditioning removes a
pure invariant mixture but also discards information. See QUARTET_BANK_BRIEF.md.
"""
from __future__ import annotations
import itertools
import numpy as np
from scipy.special import ndtri,logsumexp
from scipy.stats import qmc
import torch


def canonical_pattern(pattern):
    seen={};result=[]
    for x in pattern:
        if x not in seen:seen[x]=len(seen)
        result.append(seen[x])
    return tuple(result)


def pattern_layout():
    all_patterns=list(itertools.product(range(4),repeat=4))
    representatives=sorted(set(canonical_pattern(p) for p in all_patterns))
    index={p:i for i,p in enumerate(representatives)}
    mapping=np.array([[index[canonical_pattern(tuple(p[i] for i in order))] for p in all_patterns]
                     for order in ((0,1,2,3),(0,2,1,3),(0,3,1,2))])
    degeneracy=np.bincount(mapping[0],minlength=15)
    assert len(representatives)==15 and representatives[0]==(0,0,0,0)
    return np.array(representatives),mapping,degeneracy


def fourier_layout():
    representatives,_,degeneracy=pattern_layout()
    group=np.array([(a,b,c,a^b^c) for a,b,c in itertools.product(range(4),repeat=3)])
    masks=np.column_stack((group!=0,(group[:,0]^group[:,1])!=0)).astype(float)
    # Character of Z2 x Z2: parity of the two-bit inner product.
    bit=(group[:,None,:]&representatives[None,:,:])
    parity=((bit&1)+((bit>>1)&1)).sum(2)%2
    signs=(1-2*parity)*degeneracy[None,:]/256
    return masks,signs


def orbit_probabilities(fidelity,alpha=None):
    """Exact JC / continuous-Gamma Laplace probabilities of15 pattern orbits."""
    fidelity=np.asarray(fidelity,dtype=np.float64)
    masks,signs=fourier_layout();exponent=(-np.log(fidelity))@masks.T
    if alpha is None:fourier=np.exp(-exponent)
    else:
        alpha=np.asarray(alpha,dtype=float).reshape(-1,1)
        fourier=np.exp(-alpha*np.log1p(exponent/alpha))
    return fourier@signs


def prior_bank(log2_points=12,seed=20261024):
    """Frozen proper-prior scrambled Sobol rule, independent of input/truth."""
    if log2_points<3:raise ValueError('need at least eight symmetry-orbit points')
    u=qmc.Sobol(6,scramble=True,seed=seed).random_base2(log2_points-3)
    fidelity=np.sin(np.pi*u[:,:5]/2)**2
    alpha=np.exp(ndtri(u[:,5]))
    probability=orbit_probabilities(fidelity,alpha)
    if probability.min() < -64*np.finfo(float).eps or not np.allclose(probability.sum(1),1,atol=1e-12):
        raise ArithmeticError('Fourier prior-bank probabilities invalid')
    numerical_nonpositive=int((probability<=0).sum())
    # Cancellation below FP64 resolution is recorded, not a data pseudocount.
    probability=np.maximum(probability,np.finfo(float).tiny)
    conditional=probability[:,1:]/probability[:,1:].sum(1,keepdims=True)
    _,_,degeneracy=pattern_layout()
    reference=degeneracy[1:]/252.
    logratio=np.log(conditional)-np.log(reference)[None,:]
    # Exactly average the eight labeled-tree automorphisms of AB|CD. A finite
    # nonsymmetric Monte-Carlo bank would otherwise depend on taxon ordering.
    representatives,_,_=pattern_layout();index={tuple(p):i for i,p in enumerate(representatives)}
    symmetrized=[]
    for swap_pairs,swap_left,swap_right in itertools.product((False,True),repeat=3):
        left=[0,1];right=[2,3]
        if swap_left:left.reverse()
        if swap_right:right.reverse()
        order=right+left if swap_pairs else left+right
        mapping=[index[canonical_pattern(p[order])]-1 for p in representatives[1:]]
        symmetrized.append(logratio[:,mapping])
    logratio=np.concatenate(symmetrized,axis=0)
    return logratio,dict(points=len(logratio),base_sobol_points=len(u),automorphism_orbit=8,seed=seed,log2_points=log2_points,
        nuisance_integration='five iid Beta(.5,.5) JC edge fidelities; log(alpha)~N(0,1)',
        constant_conditioning=True,nonpositive_fp64_bank_entries=numerical_nonpositive,
        finite_bank_consistency_certified=False,model='JC69+continuous-Gamma, conditional variable sites')


def collapse_counts(counts):
    """Q x G x256 -> Q x G x3 x14 integer pattern-orbit counts."""
    _,mapping,_=pattern_layout()
    out=torch.zeros((*counts.shape[:-1],3,15),device=counts.device,dtype=counts.dtype)
    for s in range(3):
        ix=torch.as_tensor(mapping[s],device=counts.device).expand(*counts.shape[:-1],256)
        out[...,s,:].scatter_add_(-1,ix,counts)
    return out[...,1:].contiguous()


@torch.inference_mode()
def evidence(counts,bank,*,batch_rows=4096):
    """Equal-locus sum of three numerically integrated posterior weights."""
    collapsed=collapse_counts(counts)
    c=collapsed.reshape(-1,14).to(bank.dtype)
    values=[]
    for start in range(0,len(c),batch_rows):
        loglike=c[start:start+batch_rows]@bank.T
        values.append(torch.logsumexp(loglike,1)-np.log(len(bank)))
    logevidence=torch.cat(values).reshape(*counts.shape[:-1],3)
    posterior=torch.softmax(logevidence,-1)
    return posterior.sum(1),dict(locus_logevidence=logevidence,locus_posterior=posterior,
        variable_counts=collapsed[...,0,:].sum(-1))


def numpy_evidence(counts,bank):
    """Independent small FP64 multinomial sum and SciPy logsumexp audit."""
    _,mapping,_=pattern_layout();counts=np.asarray(counts);output=[]
    for query in counts:
        locus=[]
        for row in query:
            score=[]
            for s in range(3):
                compressed=np.bincount(mapping[s],weights=row,minlength=15)[1:]
                score.append(logsumexp(np.asarray(bank,dtype=float)@compressed)-np.log(len(bank)))
            locus.append(score)
        output.append(locus)
    ll=np.array(output);posterior=np.exp(ll-logsumexp(ll,axis=-1,keepdims=True))
    return posterior.sum(1),ll
