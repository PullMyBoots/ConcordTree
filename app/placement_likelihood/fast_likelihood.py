"""Native scaled pruning and bounded GTR+FO+I+G4 optimization, no truth access."""
import ctypes,time
from pathlib import Path
import numpy as np
from scipy.special import gammainc,gammaincinv,expit,softmax
from scipy.optimize import minimize
from concordtree.treeio import rooted,edge

LIB=ctypes.CDLL(str(Path(__file__).with_name('native.so')))
I=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS')
U=np.ctypeslib.ndpointer(dtype=np.uint8,flags='C_CONTIGUOUS')
D=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
LIB.prune.argtypes=[ctypes.c_int]*3+[I,I,U,D,D,D,D,D,D,D];LIB.prune.restype=ctypes.c_double
LIB.sankoff.argtypes=[ctypes.c_int]*2+[I,U,D,D,D]
LIB.placement.argtypes=[ctypes.c_int]*2+[D,D,D,U,D,D,D,D,D,D]
LIB.placement.restype=ctypes.c_double

class Plan:
    def __init__(self,tree,names,states,compress=True):
        tree.require_binary();self.tree=tree;self.names=names
        self.root,self.parent,self.order,_=rooted(tree);self.ids={u:i for i,u in enumerate(self.order)}
        self.edges=[edge(u,self.parent[u]) for u in self.order[1:]]
        states=np.asarray(states,dtype=np.uint8)
        if compress:states,counts=np.unique(states,axis=1,return_counts=True)
        else:counts=np.ones(states.shape[1])
        self.states=np.ascontiguousarray(states);self.count=np.ascontiguousarray(counts,dtype=np.float64)
        self.sites=int(counts.sum());self.S=states.shape[1];self.M=len(self.order);self.E=self.M-1
        self.child=np.full((self.M,3),-1,dtype=np.int32);self.ce=self.child.copy()
        self.obs=np.full((self.M,self.S),4,dtype=np.uint8);lookup={n:i for i,n in enumerate(names)}
        for u in self.order:
            j=self.ids[u]
            if u in tree.names:self.obs[j]=states[lookup[tree.names[u]]]
            for z,v in enumerate(sorted(v for v in tree.graph[u] if self.parent.get(v)==u)):
                self.child[j,z]=self.ids[v];self.ce[j,z]=self.ids[v]-1

    def seed(self):
        changes=np.zeros(self.M);pairs=np.zeros(6)
        LIB.sankoff(self.M,self.S,self.child,self.obs,self.count,changes,pairs)
        t=np.clip((changes[1:]+.5)/self.sites,1e-6,10.)
        pi=np.bincount(self.states.ravel(),weights=np.tile(self.count,len(self.names)),minlength=5)[:4]+.5
        pi/=pi.sum();a,b=np.triu_indices(4,1);r=(pairs+.5)/(pi[a]*pi[b]);r/=r[-1]
        x=np.r_[np.log(np.clip(r[:5],1e-4,1e4)),np.log(pi[:3]/pi[3]),np.log(.5),np.log(.1/.9)]
        return t,x

    def evaluate(self,P,pi,w):
        dp=np.zeros_like(P);dpi=np.zeros(4);dw=np.zeros(len(w))
        ll=LIB.prune(self.M,self.S,len(w),self.child,self.ce,self.obs,self.count,
                     np.ascontiguousarray(P),np.ascontiguousarray(pi),np.ascontiguousarray(w),dp,dpi,dw)
        if not np.isfinite(ll) or not np.isfinite(dp).all():raise ArithmeticError('nonfinite native likelihood')
        return ll,dp,dpi,dw

BOUNDS=[(-9.210340371976184,9.210340371976184)]*5+[(-16,16)]*3+[(np.log(.02),np.log(100)),(-16,np.log(19))]

def decode(x):
    x=np.asarray(x);pi=(softmax(np.r_[x[5:8],0.])+1e-8)/(1+4e-8)
    r=np.r_[np.exp(x[:5]),1.];q=np.zeros((4,4));a,b=np.triu_indices(4,1)
    q[a,b]=r;q[b,a]=r;q*=pi[None,:];q-=np.diag(q.sum(1));q/=-(pi*np.diag(q)).sum()
    alpha=float(np.clip(np.exp(x[8]),.02,100.));p=float(expit(x[9]))
    z=gammaincinv(alpha,np.linspace(0,1,5));rates=4*np.diff(gammainc(alpha+1,z));rates/=rates.mean()
    rates=np.r_[0.,rates/(1-p)];weights=np.r_[p,np.full(4,(1-p)/4)]
    sq=np.sqrt(pi);lam,v=np.linalg.eigh(q*sq[:,None]/sq[None,:]);lam[-1]=0.
    return dict(pi=pi,q=q,rates=rates,weights=weights,eigenvalues=lam,vectors=v,sqrtpi=sq,
                exchangeabilities=r,alpha=alpha,pinv=p)

def transitions(model,t,derivative=False):
    lam=model['eigenvalues'];v=model['vectors'];sq=model['sqrtpi'];rates=model['rates']
    exponential=np.exp(np.asarray(t)[:,None,None]*rates[None,:,None]*lam[None,None,:])
    p=np.einsum('ij,ekj,lj->ekil',v,exponential,v,optimize=False)*sq[None,None,None,:]/sq[None,None,:,None]
    # Repeated eigenspaces can introduce ~1e-16 negatives at t=0; the invariant
    # category is mathematically exactly identity, independently of Q.
    p[:,0]=np.eye(4);p=np.maximum(p,0.)
    if not derivative:return np.ascontiguousarray(p)
    d=np.einsum('ij,ekj,lj->ekil',v,exponential*rates[None,:,None]*lam[None,None,:],v,optimize=False)*sq[None,None,None,:]/sq[None,None,:,None]
    d[:,0]=0.
    return np.ascontiguousarray(p),np.ascontiguousarray(d)

def model_pullback(model,t,p,dp,dpi,dw):
    """Frechet derivative of exp(Q*t*r), including repeated-eigenvalue limits."""
    pi=model['pi'];sq=model['sqrtpi'];v=model['vectors'];lam=model['eigenvalues'];rates=model['rates']
    right=v/sq[:,None];left=v.T*sq[None,:]
    h=np.einsum('ai,ekab,jb->ekij',right,dp,left,optimize=True)
    z=np.asarray(t)[:,None]*rates[None,:];ex=np.exp(z[:,:,None]*lam)
    diff=lam[:,None]-lam[None,:]
    f=np.divide(ex[:,:,:,None]-ex[:,:,None,:],diff[None,None,:,:],
                out=np.zeros((*z.shape,4,4)),where=np.abs(diff)[None,None,:,:]>1e-9)
    f=np.where(np.abs(diff)[None,None,:,:]>1e-9,f,
               z[:,:,None,None]*(ex[:,:,:,None]+ex[:,:,None,:])/2)
    c=np.sum(h*f,axis=(0,1));gq=left.T@c@right.T
    gr=np.einsum('ekij,il,eklj,e->k',dp,model['q'],p,np.asarray(t),optimize=True)
    r=model['exchangeabilities'];s=np.zeros((4,4));a,b=np.triu_indices(4,1);s[a,b]=r;s[b,a]=r
    q0=s*pi[None,:];q0-=np.diag(q0.sum(1));mu=-(pi*np.diag(q0)).sum()
    out=[]
    for j in range(8):
        dpi0=np.zeros(4);dq0=np.zeros((4,4))
        if j<5:
            dq0[a[j],b[j]]=r[j]*pi[b[j]];dq0[b[j],a[j]]=r[j]*pi[a[j]]
        else:
            raw=pi*(1+4e-8)-1e-8;k=j-5
            dpi0=raw*((np.arange(4)==k)-raw[k])/(1+4e-8)
            dq0=s*dpi0[None,:]
        dq0-=np.diag(dq0.sum(1));dmu=-(dpi0*np.diag(q0)+pi*np.diag(dq0)).sum()
        dq=dq0/mu-q0*dmu/(mu*mu);out.append(np.sum(gq*dq)+dpi@dpi0)
    alpha=model['alpha'];hstep=1e-4;lo=max(np.log(.02),np.log(alpha)-hstep);hi=min(np.log(100),np.log(alpha)+hstep)
    def gamma(a):
        z=gammaincinv(a,np.linspace(0,1,5));r=4*np.diff(gammainc(a+1,z));return r/r.mean()
    da=(gamma(np.exp(hi))-gamma(np.exp(lo)))/(hi-lo)/(1-model['pinv'])
    out.append(gr[1:]@da)
    pinv=model['pinv'];dr=rates*pinv;dr[0]=0.
    dweight=np.r_[pinv*(1-pinv),np.full(4,-pinv*(1-pinv)/4)]
    out.append(gr@dr+dw@dweight)
    return np.array(out)

def fit(tree,names,states,previous=None,iterations=64,model_iterations=True):
    start=time.perf_counter();plan=Plan(tree,names,states)
    initial,m0=plan.seed()
    if tree.lengths:initial=np.array([tree.lengths.get(e,initial[i]) for i,e in enumerate(plan.edges)])
    if previous is not None:m0=np.array(previous['model_vector'])
    free=model_iterations;nb=plan.E
    x0=np.r_[2*np.sqrt(initial),m0] if free else 2*np.sqrt(initial)
    bounds=[(.002,2*np.sqrt(10.))]*nb+(BOUNDS if free else [])
    evaluations=0;best=[np.inf,None]
    def objective(x):
        nonlocal evaluations
        t=(x[:nb]/2)**2;mx=x[nb:] if free else m0;model=decode(mx)
        p,dt=transitions(model,t,True);ll,dp,dpi,dw=plan.evaluate(p,model['pi'],model['weights'])
        grad=np.einsum('ekij,ekij->e',dp,dt)*x[:nb]/2
        if free:grad=np.r_[grad,model_pullback(model,t,p,dp,dpi,dw)]
        value=-ll/plan.sites;g=-grad/plan.sites;evaluations+=1
        if value<best[0]:best[:]=[value,x.copy()]
        if not np.isfinite(g).all():raise ArithmeticError('nonfinite likelihood gradient')
        return value,g
    res=minimize(objective,x0,jac=True,method='L-BFGS-B',bounds=bounds,
                 options=dict(maxiter=iterations,maxls=20,ftol=1e-9,gtol=1e-5,maxcor=10))
    # Always retain the best actually evaluated finite likelihood, even at cap.
    x=best[1];t=(x[:nb]/2)**2;m=x[nb:] if free else m0
    out=tree.copy();out.lengths=dict(zip(plan.edges,map(float,t)));model=decode(m)
    return out,dict(log_likelihood=-best[0]*plan.sites,model_vector=m.tolist(),
        frequencies_acgt=model['pi'].tolist(),Q_acgt=model['q'].tolist(),
        category_rates=model['rates'].tolist(),category_weights=model['weights'].tolist(),
        gamma_alpha=model['alpha'],invariant_proportion=model['pinv'],
        converged=bool(res.success),message=str(res.message),iterations=int(res.nit),
        evaluations=evaluations,patterns=plan.S,seconds=time.perf_counter()-start)

class EdgeCache:
    """Two directed conditional-likelihood vectors for each local backbone edge.

    Cache is independent of queries and retains every original alignment site.
    Category scaling is shared across the whole tree, never averaged per edge.
    """
    def __init__(self,tree,names,states,metrics,counts=None):
        self.tree=tree;self.model=decode(metrics['model_vector']);self.S=states.shape[1];self.K=len(self.model['rates'])
        self.count=np.ones(self.S) if counts is None else np.ascontiguousarray(counts,dtype=np.float64)
        self.sites=float(self.count.sum());self.messages={};self.scales={}
        edges=sorted(tree.lengths);ps=transitions(self.model,[tree.lengths[e] for e in edges]);pm=dict(zip(edges,ps))
        root,parent,order,_=rooted(tree);lookup={n:i for i,n in enumerate(names)};evidence={}
        for u in order:
            if u in tree.names:
                obs=states[lookup[tree.names[u]]];evidence[u]=np.where(obs[:,None]==4,1.,np.eye(4)[np.minimum(obs,3)])
        def message(u,v):
            val=np.broadcast_to(evidence.get(u,np.ones((self.S,4))),(self.K,self.S,4)).copy();ls=np.zeros((self.K,self.S))
            for w in sorted(tree.graph[u]):
                if w==v:continue
                val*=np.einsum('kij,ksj->ksi',pm[edge(u,w)],self.messages[w,u],optimize=False)
                ls+=self.scales[w,u]
            n=np.maximum(val.max(axis=2),1e-300);val/=n[:,:,None];ls+=np.log(n)
            self.messages[u,v]=np.ascontiguousarray(val);self.scales[u,v]=ls
        for u in reversed(order[1:]):message(u,parent[u])
        for u in order:
            for v in sorted(tree.graph[u]):
                if parent.get(v)==u:message(u,v)

    def place(self,selected,query,pendant,iterations=40):
        a,b=selected;l=self.tree.lengths[selected];left=self.messages[a,b];right=self.messages[b,a]
        scale=np.ascontiguousarray(self.scales[a,b]+self.scales[b,a]);query=np.ascontiguousarray(query,dtype=np.uint8)
        grad=np.zeros(3);calls=0;best=[np.inf,None]
        def objective(x):
            nonlocal calls
            t=(x/2)**2;p,dp=transitions(self.model,t,True)
            ll=LIB.placement(self.S,self.K,left,right,scale,query,self.count,p,dp,
                             self.model['pi'],self.model['weights'],grad)
            calls+=1;v=-ll/self.sites;g=-grad.copy()*x/2/self.sites
            if not np.isfinite(v) or not np.isfinite(g).all():raise ArithmeticError('nonfinite cached placement')
            if v<best[0]:best[:]=[v,x.copy()]
            return v,g
        x0=2*np.sqrt([max(l/2,1e-6),max(l/2,1e-6),pendant])
        res=minimize(objective,x0,jac=True,method='L-BFGS-B',bounds=[(.002,2*np.sqrt(10.))]*3,
                     options=dict(maxiter=iterations,maxls=20,ftol=1e-9,gtol=1e-5))
        return dict(edge=list(selected),log_likelihood=-best[0]*self.sites,
            insertion_lengths=((best[1]/2)**2).tolist(),converged=bool(res.success),
            iterations=int(res.nit),evaluations=calls,message=str(res.message))
