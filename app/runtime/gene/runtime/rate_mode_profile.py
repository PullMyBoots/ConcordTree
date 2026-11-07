"""Deterministic scale coordinates induced by positive mixture-rate ratios.

Discrete categories can exchange their role when every branch changes scale.
Ordinary local gradients do not cross these separated basins. Evaluate those
model-induced coordinates under the SAME observed likelihood; an improving
coordinate seeds continued shape/model optimization. No fitted/truth scale prior.
"""
import numpy as np
from numpy_model import decode_numpy


def profile_modes(vector,*,origin,scale,edges,coordinate,parameters,objective):
    raw=origin+scale*vector
    lengths=(raw[:edges]/2)**2 if coordinate=='sqrt' else np.exp(raw[:edges])
    model=decode_numpy(parameters,raw[edges:])
    k=parameters.spec.gamma_categories+int(parameters.spec.invariant)
    rates=model.values[20:20+k];positive=rates[rates>0]
    ratios=sorted(set(float(a/b) for a in positive for b in positive))
    candidates=[]
    for ratio in ratios:
        b=np.clip(lengths*ratio,1e-6,10.)
        proposal=raw.copy()
        proposal[:edges]=2*np.sqrt(b) if coordinate=='sqrt' else np.log(b)
        point=(proposal-origin)/scale
        value=objective(point)
        candidates.append(dict(ratio=ratio,value=value,point=point))
    return candidates
