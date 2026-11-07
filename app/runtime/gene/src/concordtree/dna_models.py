"""Stationary reversible DNA models; A,C,G,T and AC,AG,AT,CG,CT,GT order.

Q has unit stationary mean rate. Gamma uses equal-probability conditional means,
not quantile midpoints. A zero-rate invariant class is mixed BEFORE site logs.
"""
from dataclasses import dataclass
import re
import numpy as np
import torch
from scipy.special import gammainc, gammaincinv


@dataclass(frozen=True)
class ModelSpec:
    name: str
    base: str
    groups: tuple
    frequencies: str
    gamma_categories: int
    invariant: bool


def parse_model(name):
    if not isinstance(name, str):
        raise ValueError("model must be a DNA model string")
    parts = name.upper().split("+")
    base = {"JC69":"JC", "K2P":"K80", "HKY85":"HKY", "TN":"TN93"}.get(parts[0], parts[0])
    codes = {"JC": "000000", "F81":"000000", "K80":"010010",
             "HKY":"010010", "TN93":"010020", "SYM":"012345", "GTR":"012345"}
    if base not in codes:
        raise ValueError("GPU DNA models: JC, F81, K80, HKY, TN93, SYM, GTR")
    # Match IQ-TREE's default: empirical frequencies for unequal-frequency models.
    freq = "empirical" if base in {"F81","HKY","TN93","GTR"} else "equal"
    seen=set(); categories=1; invariant=False
    for token in parts[1:]:
        key = "F" if token in {"F","FO","FQ"} else ("G" if re.fullmatch(r"G\d*",token) else token)
        if key in seen: raise ValueError("duplicate model modifier")
        seen.add(key)
        if token in {"F","FO","FQ"}:
            if base in {"JC","K80","SYM"} and token!="FQ":
                raise ValueError("use F81/HKY/GTR for unequal-frequency models")
            freq={"F":"empirical","FO":"ml","FQ":"equal"}[token]
        elif key=="G":
            categories=int(token[1:] or "4")
            if not 2<=categories<=16: raise ValueError("Gamma category count must be 2..16")
        elif token=="I": invariant=True
        else: raise ValueError("unsupported GPU DNA modifier: "+token)
    # Last exchangeability (GT) anchors the unidentifiable global rate scale.
    code=codes[base]; labels=[x for x in dict.fromkeys(code) if x!=code[-1]]
    groups=tuple(labels.index(x) if x in labels else -1 for x in code)
    canonical=base+({"ml":"+FO","equal":"+FQ"}.get(freq, "+F") if "F" in seen else "")
    canonical+=("+I" if invariant else "")+(f"+G{categories}" if categories>1 else "")
    return ModelSpec(canonical,base,groups,freq,categories,invariant)


def gamma_category_means(alpha, categories):
    """Yang discrete Gamma conditional means, Gamma(shape=a, scale=1/a)."""
    if not np.isfinite(alpha) or not .02<=alpha<=100 or not 2<=categories<=16:
        raise ValueError("alpha must be .02..100 and categories 2..16")
    z=gammaincinv(alpha,np.arange(categories+1,dtype=np.float64)/categories)
    cdf=gammainc(alpha+1,z)
    rates=categories*np.diff(cdf)
    return rates/rates.mean()


class _GammaMeans(torch.autograd.Function):
    @staticmethod
    def forward(ctx, log_alpha, categories):
        x=float(log_alpha.detach().cpu()); a=float(np.exp(x))
        # Optimizer bounds are inclusive; tolerate roundoff in exp(log(bound)).
        a=min(100.,max(.02,a))
        rates=gamma_category_means(a,categories)
        h=1e-4
        lo=max(np.log(.02),x-h); hi=min(np.log(100.),x+h)
        derivative=(gamma_category_means(min(100.,np.exp(hi)),categories)-
                    gamma_category_means(max(.02,np.exp(lo)),categories))/(hi-lo)
        ctx.save_for_backward(log_alpha.new_tensor(derivative))
        return log_alpha.new_tensor(rates)

    @staticmethod
    def backward(ctx, grad):
        (derivative,)=ctx.saved_tensors
        return (grad*derivative).sum(),None


def rate_matrix(pi, exchangeabilities):
    pair=torch.triu_indices(4,4,1,device=pi.device)
    s=pi.new_zeros((4,4)); s[pair[0],pair[1]]=exchangeabilities
    s=s+s.T
    q=s*pi[None,:]
    q=q-torch.diag(q.sum(1))
    return q/(-(pi*q.diag()).sum())


class ModelParameters:
    """Small unconstrained parameter vector plus explicit L-BFGS-B bounds."""
    def __init__(self, spec, states, *, frequencies=None, exchangeabilities=None,
                 alpha=.5, pinv=.1, fixed=False):
        self.spec=spec; self.fixed=fixed
        if frequencies is None:
            flat=np.asarray(states).reshape(-1)
            counts=np.zeros(4,dtype=float)
            # bincount casts to int64: bound that temporary for long alignments.
            for start in range(0,len(flat),1048576):
                counts+=np.bincount(flat[start:start+1048576],minlength=5)[:4]
            if not counts.sum(): raise ValueError("alignment contains no observed nucleotides")
            pi=np.maximum(counts/counts.sum(),1e-8); pi/=pi.sum()
        else:
            pi=np.asarray(frequencies,dtype=float)
            if pi.shape!=(4,) or not np.all(np.isfinite(pi)) or np.any(pi<=0) or not np.isclose(pi.sum(),1):
                raise ValueError("frequencies must be four positive values summing to one")
            pi=pi/pi.sum()
        if spec.frequencies=="equal":
            if frequencies is not None and not np.allclose(pi,.25): raise ValueError("model requires equal frequencies")
            pi=np.full(4,.25)
        rates=np.ones(6) if exchangeabilities is None else np.asarray(exchangeabilities,dtype=float)
        if rates.shape!=(6,) or np.any(rates<=0) or not np.all(np.isfinite(rates)):
            raise ValueError("six finite positive exchangeabilities required")
        rates=rates/rates[-1]
        for group in set(spec.groups):
            values=rates[np.array(spec.groups)==group]
            if not np.allclose(values,values[0]) or (group==-1 and not np.allclose(values,1)):
                raise ValueError("exchangeabilities violate model constraints")
        # exp(log(100)) can round to 100.00000000000004 at a legal optimizer
        # boundary. Accept only a four-ulp round trip outside the same bounds.
        if alpha < .02 and .02-alpha <= 4*np.spacing(.02): alpha=.02
        if alpha > 100 and alpha-100 <= 4*np.spacing(100.): alpha=100.
        if pinv > .95 and pinv-.95 <= 4*np.spacing(.95): pinv=.95
        if not .02<=alpha<=100 or not 0<pinv<=.95: raise ValueError("invalid alpha or initial invariant proportion")
        self.pi=pi; self.rates=rates; self.alpha=alpha; self.pinv=pinv
        self.initial=[]; self.bounds=[]; self.parts={}
        def add(key, values, bounds):
            start=len(self.initial); self.initial.extend(values); self.bounds.extend([bounds]*len(values))
            self.parts[key]=slice(start,len(self.initial))
        if not fixed:
            groups=max(spec.groups)+1
            add("rates",[np.log(rates[spec.groups.index(i)]) for i in range(groups)],(-9.21034,9.21034))
            if spec.frequencies=="ml": add("pi",list(np.log(pi[:3]/pi[3])),(-16.,16.))
            if spec.gamma_categories>1: add("alpha",[np.log(alpha)],(np.log(.02),np.log(100.)))
            if spec.invariant: add("pinv",[np.log(pinv/(1-pinv))],(-16.,np.log(19.)))

    def decode(self, x):
        pi=x.new_tensor(self.pi); rates=x.new_tensor(self.rates)
        if "pi" in self.parts:
            pi=torch.softmax(torch.cat((x[self.parts["pi"]],x.new_zeros(1))),0)
            pi=(pi+1e-8)/(1+4e-8)
        if "rates" in self.parts:
            values=torch.cat((x[self.parts["rates"]].exp(),x.new_ones(1)))
            rates=values[torch.tensor(self.spec.groups,device=x.device)]
        alpha=x[self.parts["alpha"]][0] if "alpha" in self.parts else x.new_tensor(np.log(self.alpha))
        variable=(_GammaMeans.apply(alpha,self.spec.gamma_categories) if self.spec.gamma_categories>1 else x.new_ones(1))
        if self.spec.invariant:
            pinv=torch.sigmoid(x[self.parts["pinv"]][0]) if "pinv" in self.parts else x.new_tensor(self.pinv)
            site_rates=torch.cat((x.new_zeros(1),variable/(1-pinv)))
            weights=torch.cat((pinv[None],((1-pinv)/len(variable)).expand(len(variable))))
        else:
            pinv=x.new_zeros(()); site_rates=variable; weights=torch.ones_like(variable)/len(variable)
        return pi,rates,rate_matrix(pi,rates),site_rates,weights,alpha.exp(),pinv


def transition_matrices(q, lengths, site_rates):
    """Differentiable GPU/CPU exponential; stable at repeated JC eigenvalues."""
    return torch.matrix_exp(q[None,None,:,:]*lengths[:,None,None,None]*site_rates[None,:,None,None])
