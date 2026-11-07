"""Variable projection for the invariant-site mixture.

Let beta=b/(1-p). Conditional on beta, Q, Gamma(alpha), each site has probability
    p*I_site + (1-p)*V_site.
The log likelihood is concave in p. Profile p by a one-dimensional score root;
optimize beta and other model parameters using the envelope theorem. This removes
one weakly coupled numerical coordinate rather than adding a fitting gate.

Prototype bounds: beta in [1e-12,200] covers the incumbent's upper b bound under
p<=.95. Mean-rate b may consequently fall below the incumbent 1e-6 floor or exceed
10. These changed bounds are explicit and require a matched-bound sensitivity
comparison before attributing all gains to the optimizer.
"""
import math,time
import numpy as np
import torch
from scipy.optimize import brentq,minimize,minimize_scalar
from scipy.special import expit
from concordtree.dna_models import ModelParameters,parse_model,gamma_category_means
from frontier_repeats import RepeatPlan
from analytic_score import prepare_model,transitions,endpoint_score
from numpy_model import decode_numpy
from packed_model import packed_transition
from search_progress import ExactRecurrence, MAX_PARAMETER_CONTINUATIONS

PMIN=float(expit(-16.));PMAX=.95

def category_shape_anchors(alpha,categories,bounds):
    """Shape coordinates that align two positive category-rate ratios.

    Preserve the fastest-category exposure. A different shape can assign an
    old relative rate to a different Gamma category. The finite K-dependent
    alignments provide mode proposals without an alpha grid or restart count.
    """
    current=gamma_category_means(min(100.,max(.02,float(alpha))),categories)
    target=np.log(current[:-1]/current[-1]);anchors={math.log(alpha)}
    lo,hi=bounds
    for i in range(categories-1):
        for j in range(categories-1):
            if i==j:continue
            def residual(x):
                g=gamma_category_means(min(100.,max(.02,math.exp(x))),categories)
                return float(math.log(g[j]/g[-1])-target[i])
            if residual(lo)*residual(hi)<0:
                anchors.add(float(brentq(residual,lo,hi)))
    return sorted(anchors)

def optimum_p(log_ratio,frequencies):
    """Root of the monotone mixture score; works for zero invariant likelihood."""
    # Algebraically (I-V)/(p*I+(1-p)*V). Use expm1 around I=V to
    # avoid subtracting posterior mass from p near either mixture boundary.
    # In the exactly flat case the numerator is exactly zero.
    magnitude=np.abs(log_ratio);relative=np.exp(-magnitude)
    positive=log_ratio>=0
    numerator=np.where(positive,-np.expm1(-magnitude),np.expm1(-magnitude))
    def score(p):
        denominator=np.where(positive,p+(1-p)*relative,(1-p)+p*relative)
        return float(np.dot(frequencies,numerator/denominator))
    if score(PMIN)<=0:return PMIN
    if score(PMAX)>=0:return PMAX
    return float(brentq(score,PMIN,PMAX,xtol=4*np.finfo(float).eps,rtol=4*np.finfo(float).eps))


class ProfilePlan(RepeatPlan):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        bits=torch.tensor([1,2,4,8],device=self.device,dtype=torch.uint8)
        self.invariant_basis=((self.root_invariant_masks[:,None]&bits[None,:])!=0).double()
        self.invariant_mass=torch.empty((),device=self.device,dtype=torch.float64)

    @torch.no_grad()
    def _upward_only(self):
        for first,last in self.up_levels:self.upward(self,first,last)
        n=self.root_patterns
        self.root_joint=self.up[:n].double()*self.pi
        self.root_prob=self.root_joint.sum(-1)
        self.root_logcat=torch.where(self.root_prob>0,self.logs[:n].double()*math.log(2)+self.root_prob.clamp_min(1e-300).log(),-torch.inf)
        self.invariant_joint=self.invariant_basis*self.pi
        self.invariant_prob=self.invariant_joint.sum(-1)
        self.invariant_logprob=torch.where(self.invariant_prob>0,self.invariant_prob.clamp_min(1e-300).log(),-torch.inf)
        self.variable_logprob=torch.logsumexp(self.root_logcat,1)-math.log(self.categories)

    @torch.no_grad()
    def upward_state(self,matrices,pi):
        self.trans[self.edge_index]=matrices.to(self.dtype)
        self.pi.copy_(pi)
        if not self.graph_enabled:
            self._upward_only();return
        if not hasattr(self,'upward_graph'):
            current=torch.cuda.current_stream(self.device)
            stream=torch.cuda.Stream(device=self.device);stream.wait_stream(current)
            with torch.cuda.stream(stream):
                for _ in range(2):self._upward_only()
            current.wait_stream(stream);self.upward_graph=torch.cuda.CUDAGraph()
            from capture_scope import capture_scope
            with capture_scope(self.upward_graph,stream):self._upward_only()
            torch.cuda.synchronize(self.device)
        self.upward_graph.replay()

    @torch.no_grad()
    def _counts_from_up(self):
        self.counts.zero_();n=self.root_patterns
        joint,prob=self.root_joint,self.root_prob
        catll=self.root_logcat+self.weights.log()
        invariant_joint,invariant_prob=self.invariant_joint,self.invariant_prob
        invariant_ll=self.invariant_logprob+self.invariant_mass.log()
        site_ll=torch.logaddexp(torch.logsumexp(catll,1),invariant_ll)
        post=(catll-site_ll[:,None]).exp()*self.root_frequency[:,None]
        theta=post[...,None]*joint/prob.clamp_min(1e-300)[...,None]
        invariant_post=(invariant_ll-site_ll).exp()*self.root_frequency
        invariant_root=(invariant_post[:,None]*invariant_joint/invariant_prob.clamp_min(1e-300)[:,None]).sum(0)
        self.theta[:n]=theta
        self.depth_down.run(self)
        self.reduce_counts(self)
        return (site_ll*self.root_frequency).sum(),self.counts.index_select(0,self.edge_index),theta.sum((0,1))+invariant_root,post.sum(0)

    @torch.no_grad()
    def profile_counts(self,p):
        self.weights.fill_((1-p)/self.categories)
        self.invariant_mass.fill_(p)
        if not self.graph_enabled:return self._counts_from_up()
        if not hasattr(self,'profile_graph'):
            current=torch.cuda.current_stream(self.device)
            stream=torch.cuda.Stream(device=self.device);stream.wait_stream(current)
            with torch.cuda.stream(stream):
                for _ in range(2):self._counts_from_up()
            current.wait_stream(stream);self.profile_graph=torch.cuda.CUDAGraph()
            from capture_scope import capture_scope
            with capture_scope(self.profile_graph,stream):self.profile_outputs=self._counts_from_up()
            torch.cuda.synchronize(self.device)
        self.profile_graph.replay();return self.profile_outputs


class ProfileObjective:
    def __init__(self,tree,names,states,model,previous):
        self.spec=parse_model(model)
        assert self.spec.invariant
        self.params=ModelParameters(self.spec,states,frequencies=previous['frequencies_acgt'],
            exchangeabilities=previous['exchangeabilities_ac_ag_at_cg_ct_gt'],
            alpha=previous['gamma_alpha'] or .5,pinv=max(PMIN,previous['invariant_proportion']))
        self.plan=ProfilePlan(tree,names,states,'cuda:0',self.spec.gamma_categories,
            precision='binary64',graph=True,builder='radix',kernel_schedule='matrix-factorized')
        self.edges=len(self.plan.edges);self.sites=states.shape[1]
        self.pindex=self.params.parts['pinv'].start
        self.other=np.array([j for j in range(len(self.params.initial)) if j!=self.pindex])
        self.p0=float(expit(self.params.initial[self.pindex]))
        self.frequencies=self.plan.root_frequency.cpu().numpy()
        masks=self.plan.root_invariant_masks.cpu().numpy()
        compatible=np.flatnonzero(masks!=0)
        self.compatible=torch.tensor(compatible,device=self.plan.device)
        self.profile_frequencies=np.r_[self.frequencies[compatible],self.frequencies[masks==0].sum()]
        k=self.spec.gamma_categories+1
        positive=list(range(20))+list(range(21,20+k))+list(range(21+k,20+2*k))
        self.positive_primitives=torch.tensor(positive,device=self.plan.device)

        self.fast_index=20+self.spec.gamma_categories
        initial_model=decode_numpy(self.params,np.asarray(self.params.initial))
        self.fast_initial=float(initial_model.values[self.fast_index])
        exposure=np.array([tree.lengths[e]*self.fast_initial for e in self.plan.edges])
        self.max_exposure=200*self.spec.gamma_categories
        self.initial=np.r_[np.clip(exposure,1e-12,self.max_exposure),np.asarray(self.params.initial)[self.other]]
        self.bounds=[(1e-12,self.max_exposure)]*self.edges+[self.params.bounds[j] for j in self.other]
        self.calls=0;self.likelihood_calls=0;self.cache={}

    def model_vector(self,values):
        vector=np.asarray(self.params.initial).copy();vector[self.other]=values[self.edges:]
        return vector

    @torch.no_grad()
    def transition_state(self,values):
        vector=self.model_vector(values);decoded=decode_numpy(self.params,vector)
        self.fast=float(decoded.values[self.fast_index])
        self.fast_derivative=decoded.jacobian[self.fast_index]/self.fast
        raw=2*np.sqrt(values[:self.edges]/self.fast)
        pi,q,r,w,jac,p,tape,raw=packed_transition(self.params,vector,raw,self.plan.device,decoded=decoded)
        tape=(tape[0],tape[1],r[1:],*tape[3:])
        return pi,q,r[1:],w[1:],jac.index_select(0,self.positive_primitives),p[:,1:].contiguous(),tape,raw

    @torch.no_grad()
    def forward(self,values,gradient=True):
        pi,q,r,w,jac,matrices,tape,raw=self.transition_state(values)
        self.plan.upward_state(matrices,pi)
        variable=self.plan.variable_logprob
        invariant_ll=self.plan.invariant_logprob
        ratio=(invariant_ll-variable).index_select(0,self.compatible).cpu().numpy()
        p=optimum_p(np.r_[ratio,-np.inf],self.profile_frequencies)
        self.likelihood_calls+=1
        if not gradient:
            site_ll=torch.logaddexp(math.log(p)+invariant_ll,math.log1p(-p)+variable)
            value=-float((site_ll*self.plan.root_frequency).sum().cpu())/self.sites
            return value,p
        ll,counts,root_counts,category_counts=self.plan.profile_counts(p)
        g=endpoint_score(matrices,tape,counts,root_counts,category_counts,self.plan.weights,jac,self.sites,coordinate='sqrt',raw_branches=raw)
        packed=torch.cat((-ll.reshape(1)/self.sites,g)).cpu().numpy()
        value=float(packed[0]);g=packed[1:]
        # Remove the square-root chain from the shared endpoint-score routine.
        # Optimize exposure itself: its derivative stays nonzero at a short edge.
        branch=g[:self.edges]/np.sqrt(values[:self.edges]*self.fast)
        model_score=g[self.edges:]-np.dot(branch,values[:self.edges])*self.fast_derivative
        result=np.r_[branch,model_score[self.other]]
        self.calls+=1
        if not np.isfinite(value) or not np.all(np.isfinite(result)):raise ArithmeticError('nonfinite profiled objective')
        self.cache=dict(point=np.array(values),loss=value,gradient=result,p=p)
        return value,result

    def __call__(self,values):
        if np.array_equal(values,self.cache.get('point')):return self.cache['loss'],self.cache['gradient']
        return self.forward(values)


def fit_profiled(tree,names,states,model,previous=None,max_iter=32):
    start=time.perf_counter();spec=parse_model(model)
    initialization=dict(method='provided lengths/model',seconds=0.)
    if previous is None:
        params=ModelParameters(spec,states,alpha=.5,pinv=.1)
        absent=not any(value is not None for value in tree.lengths.values())
        exchange=params.rates
        if absent:
            from parsimony_seed import seed_parsimony
            tree,initialization=seed_parsimony(tree,names,states,device='cuda:0',
                collect_model_statistics=spec.base=='GTR')
            if spec.base=='GTR':
                a,b=np.triu_indices(4,1)
                exchange=(np.array(initialization['substitutions_ac_ag_at_cg_ct_gt'])+.5)/(params.pi[a]*params.pi[b])
                exchange=np.clip(exchange/exchange[-1],1e-4,1e4)
                initialization['model_seed']='stationary parsimony replacement-count ratio; initializer only'
        previous=dict(frequencies_acgt=params.pi.tolist(),exchangeabilities_ac_ag_at_cg_ct_gt=exchange.tolist(),
            gamma_alpha=.5,invariant_proportion=.1)
    obj=ProfileObjective(tree,names,states,model,previous)
    origin=obj.initial.copy();affine=np.ones_like(origin)
    # Fixed affine Poisson geometry; unlike a square-root reparameterization,
    # it cannot create an artificial stationary point at a zero-length edge.
    affine[:obj.edges]=np.sqrt(origin[:obj.edges])
    affine[obj.edges:]=1/math.sqrt(max(1.,np.sum(origin[:obj.edges])/obj.fast_initial))
    bounds=[((a-origin[j])/affine[j],(b-origin[j])/affine[j]) for j,(a,b) in enumerate(obj.bounds)]
    history=[];profiles=[];iterations=0;stopped=False
    lower,upper=np.asarray(bounds,dtype=np.float64).T
    def objective(x):
        value,g=obj(origin+affine*x);return value,g*affine
    def callback(x):
        nonlocal stopped
        value,g=objective(x);history.append(-value*obj.sites)
        raw=origin+affine*x;score=g/affine
        score[:obj.edges]*=raw[:obj.edges]
        projected=((x<=lower+1e-10)&(score>0))|((x>=upper-1e-10)&(score<0))
        score[projected]=0
        if np.max(abs(score))<=5e-6:stopped=True;raise StopIteration
    point=np.zeros_like(origin)
    recurrence=ExactRecurrence();continuation_stop=None
    while True:
        if recurrence.repeated(point.tobytes()):
            continuation_stop='repeated_parameter_state';break
        seed_loss,_=objective(point)
        stopped=False
        result=minimize(objective,point,jac=True,method='L-BFGS-B',bounds=bounds,callback=callback,
            options=dict(maxiter=max_iter,maxls=30,maxcor=10,ftol=0.,gtol=0.,maxfun=np.iinfo(np.int32).max))
        iterations+=int(result.nit);raw=origin+affine*result.x
        value,_=obj(raw)
        # A failed/restarted local solve must not discard a better seed.
        if value>seed_loss+64*np.finfo(float).eps*max(1.,abs(seed_loss)):
            raw=origin+affine*point;value,_=obj(raw)
            continuation_stop='local_solver_regressed';stopped=False;break
        m=decode_numpy(obj.params,obj.model_vector(raw));k=obj.spec.gamma_categories+1
        rates=m.values[20:20+k];positive=rates[rates>0]
        trials=[];profile_start=time.perf_counter()
        for ratio in sorted(set(float(a/b) for a in positive for b in positive)):
            candidate=raw.copy();candidate[:obj.edges]=np.clip(raw[:obj.edges]*ratio,1e-12,obj.max_exposure)
            loss,p=obj.forward(candidate,gradient=False);trials.append((loss,ratio,p,candidate,'scale'))
        shape_trials=[];anchors=[]
        if 'alpha' in obj.params.parts:
            ai=obj.edges+list(obj.other).index(obj.params.parts['alpha'].start)
            anchors=category_shape_anchors(m.alpha,obj.spec.gamma_categories,obj.bounds[ai])
            for a in anchors:
                candidate=raw.copy();candidate[ai]=a
                loss,p=obj.forward(candidate,gradient=False)
                trials.append((loss,1.,p,candidate,'shape'))
                shape_trials.append(dict(alpha=math.exp(a),loss=loss,pinv=p))
        best=min(trials,key=lambda t:t[0])
        if best[4]=='shape' and best[0]<value-64*np.finfo(float).eps*max(1,abs(value)):
            # Refine only an improving inter-category proposal, not the local
            # coordinate already handled by the joint gradient optimizer.
            knots=sorted(set([*obj.bounds[ai],*anchors]));j=knots.index(float(best[3][ai]))
            lo=(knots[j-1]+knots[j])/2 if j else knots[j]
            hi=(knots[j+1]+knots[j])/2 if j+1<len(knots) else knots[j]
            def shape_loss(a):
                candidate=raw.copy();candidate[ai]=a
                return obj.forward(candidate,gradient=False)[0]
            if hi>lo:
                shape=minimize_scalar(shape_loss,bounds=(lo,hi),method='bounded')
                candidate=raw.copy();candidate[ai]=float(shape.x)
                loss,p=obj.forward(candidate,gradient=False)
                if loss<best[0]:best=(loss,1.,p,candidate,'shape-refinement')
        improved=bool(best[0]<value-64*np.finfo(float).eps*max(1,abs(value)))
        profiles.append(dict(seconds=time.perf_counter()-profile_start,accepted=improved,ratio=best[1],axis=best[4],shape_trials=shape_trials,trials=[dict(loss=t[0],ratio=t[1],pinv=t[2],axis=t[4]) for t in trials]))
        if not improved:break
        if len(profiles)>=MAX_PARAMETER_CONTINUATIONS:
            continuation_stop='parameter_continuation_guard';stopped=False;break
        point=(best[3]-origin)/affine
    value,g=obj(raw);p=obj.cache['p'];theta=obj.model_vector(raw);theta[obj.pindex]=math.log(p/(1-p))
    decoded=decode_numpy(obj.params,theta);k=obj.spec.gamma_categories+1
    fast=float(decoded.values[obj.fast_index])*(1-p)
    beta=raw[:obj.edges]/fast;out=tree.copy()
    for e,b in zip(obj.plan.edges,(1-p)*beta):out.lengths[e]=float(b)
    result_model=dict(model=obj.spec.name,method='variable projection of invariant mass / rate-mode scale coordinates / L-BFGS-B',
        units='expected substitutions per site',device='cuda:0',precision='FP64 positive-rate upward, analytical invariant root, FP32 endpoint, FP64 sums',
        taxa=len(names),sites=obj.sites,max_iter=max_iter,iterations=iterations,converged=bool(stopped and continuation_stop is None),
        continuation_stop=continuation_stop,maximum_parameter_continuations=MAX_PARAMETER_CONTINUATIONS,
        optimizer_success=bool(result.success),optimizer_status=int(result.status),optimizer_message=str(result.message),
        likelihood_evaluations=obj.calls,likelihood_only_evaluations=obj.likelihood_calls-obj.calls,
        log_likelihood=-value*obj.sites,log_likelihood_history=history,
        frequencies_acgt=decoded.values[:4].tolist(),exchangeabilities_ac_ag_at_cg_ct_gt=decoded.exchangeabilities.tolist(),
        Q_acgt=decoded.values[4:20].reshape(4,4).tolist(),gamma_alpha=decoded.alpha,invariant_proportion=p,
        category_rates=decoded.values[20:20+k].tolist(),category_weights=decoded.values[20+k:20+2*k].tolist(),
        rate_mode_profiles=profiles,resident_setup=obj.plan.setup_metrics,initialization=initialization,seconds=time.perf_counter()-start,
        branch_coordinates='fastest-category exposures gamma=b*g_max(alpha)/(1-p), fixed affine seed metric',exposure_bounds=[1e-12,obj.max_exposure],
        warning='development prototype; branch bounds differ from incumbent, profile mixture is exact conditional on beta and remaining model')
    return out,result_model
