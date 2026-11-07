import time
import numpy as np
DENSE_LIMIT=512

def _health(state,*,uninformative=False,unobserved=0):
 info=dict(state.diagnostics());warnings=[]
 if uninformative:warnings.append('No variable observed site: topology is unidentifiable; a deterministic balanced binary completion was returned.')
 if info['jointly_observed_sites']==0:warnings.append('No alignment column jointly observes two taxa; the input has no comparative sequence evidence.')
 if info['zero_overlap_pair_evaluations']:
  warnings.append('Some queried representative pairs have no common observed site. Finite completion distances are placeholders, not observed evidence.')
 if info['gamma_usable_triplets']==0:warnings.append('Gamma shape has no usable triplet evidence; its bounded numerical fallback is not an informative estimate.')
 if unobserved:warnings.append('Some active rows have zero mean observed overlap; their row estimate uses the sampled distance sum without a ratio correction.')
 info.update(informative=not uninformative,warnings=warnings,row_estimator='observed-intersection ratio Taylor control; complete-input exact reduction',zero_mean_overlap_rows=int(unobserved))
 return info

def _uninformative(names,state,started,allocated,model,common,seed,canonical,settings):
 import _ct_apnj_topology as topology
 from concordtree.treeio import Tree
 n=len(names);neighbors=np.full((2*n-2,3),-1,np.int64);active=np.arange(n,dtype=np.int64);nextnode=n
 while len(active)>3:
  count=min(len(active)//2,len(active)-3);dest=np.arange(nextnode,nextnode+count,dtype=np.int64)
  topology.attach(neighbors,np.ascontiguousarray(np.column_stack([active[:2*count:2],active[1:2*count:2],dest])))
  active=np.concatenate([dest,active[2*count:]]);nextnode+=count
 topology.connect_center(neighbors,active,nextnode)
 phase=time.perf_counter();tree=Tree(topology.export_graph(neighbors,n,canonical),dict(enumerate(names)),{});tree.require_binary()
 export=time.perf_counter()-phase
 return tree,dict(upload_seconds=settings.pop('upload_seconds'),export_seconds=export,canonical=canonical,
  local_cleanup=dict(method='skipped: no variable observed site',passes=[],times={},seconds=0.,maximum_passes=4,component_width=4),
  method='uninformative observed alignment: balanced binary completion',constructor_seconds=time.perf_counter()-started,
  projection_seconds=0.,times=dict(candidate=0.,statistics=0.,selection=0.,merge=0.),history=[],sequence_bytes=allocated,taxa=n,all_sites=settings.pop('length'),full_site_input=True,seed=seed,model=model,common_sites=common,gamma_shape=None,
  terminal=dict(active=3,pairs=0,matrix_bytes=0,query_seconds=0.,solve_seconds=0.,limit=DENSE_LIMIT,solver='deterministic topology completion without distance evidence'),active_work=0,candidate_work=0,health=_health(state,uninformative=True),**settings)

def build(names,packed,length,*,seed=20260934,projections=16,window=8,cap=32,representatives=2,landmarks=32,ann_k=16,ann_rounds=2,model=1,common=False,minimum_divisor=4,real_ann=True,row_control=False,sketch_ann=False,q_candidates=True,canonical=False,cpu_threads=16):
 import _ct_apnj_packed as native
 import _ct_apnj_graph as graph
 from concordtree.treeio import Tree
 started=time.perf_counter();n=len(names);R=representatives;workers=min(16,int(cpu_threads))
 if R not in (1,2,4):raise ValueError('representative width must be 1,2,4')
 if not real_ann:raise ValueError('this AP-NJ backend requires real_ann=True')
 if workers<1:raise ValueError('positive CPU thread count required')
 state=native.State(packed,int(length),workers);upload_seconds=time.perf_counter()-started;allocated=state.memory();evidence=state.evidence()
 if evidence['variable_observed_sites']==0:
  return _uninformative(names,state,started,allocated,model,common,seed,canonical,dict(length=length,upload_seconds=upload_seconds,projections=projections,window=window,cap=cap,representatives=R,landmarks=landmarks,ann_k=ann_k,minimum_divisor=minimum_divisor,real_ann=real_ann,row_control=row_control,q_candidates=q_candidates))
 coherent=bool(common);unobserved_rows=0
 alpha=state.gamma_shape() if model>=2 else 100.;phase=time.perf_counter()
 projected=np.zeros((2*n-2,projections),np.float64)
 if n>DENSE_LIMIT:projected[:n]=state.project(seed,projections)
 projection_seconds=time.perf_counter()-phase
 reps=np.zeros((2*n-2,R),np.int32);mass=np.zeros((2*n-2,R),np.float64)
 reps[:n]=np.arange(n,dtype=np.int32)[:,None];mass[:n,0]=1
 offset=np.zeros(2*n-2,np.float64);minleaf=np.arange(2*n-2,dtype=np.int64)
 active=np.arange(n,dtype=np.int32);neighbors=np.full((2*n-2,3),-1,np.int64);nextnode=n;history=[]
 import _ct_apnj_topology as topology
 times=dict(candidate=0.,statistics=0.,selection=0.,merge=0.);rng=np.random.default_rng(seed)
 while len(active)>DENSE_LIMIT:
  m=len(active);phase=time.perf_counter();z=np.ascontiguousarray(projected[active]);order=np.argsort(z[:,0],kind='stable').astype(np.int64)
  pairs=np.asarray(graph.projection_pairs(z,window,workers),dtype=np.int64)
  times['candidate']+=time.perf_counter()-phase
  rr=np.ascontiguousarray(reps[active]);ww=np.ascontiguousarray(mass[active]);phase=time.perf_counter()
  h=min(m,landmarks);sample=np.random.default_rng(seed+len(history)*1000003).choice(m,h,replace=False)
  owners=np.repeat(np.arange(m,dtype=np.int64),h);others=np.tile(sample,m);valid=owners!=others
  probes=np.ascontiguousarray(np.column_stack([owners[valid],others[valid]]),dtype=np.int64)
  ranks=np.ascontiguousarray(minleaf[active]);distance_queries=0
  observed=state.query(rr,ww,probes,model,common,alpha,moments=row_control,overlap=coherent and row_control);probe_d=observed[:,0] if row_control else observed;distance_queries+=len(probe_d)
  counts=np.bincount(owners[valid],minlength=m);sums=np.bincount(owners[valid],weights=probe_d,minlength=m);base=(m-1)*sums/counts
  if row_control and h<m:
   exact=state.raw_rows(rr,ww,overlap=coherent);pbar=exact[:,0]/(m-1);qbar=exact[:,1]/(m-1)
   # The primary metric normalizes on common observations. Differentiate in
   # those units, then chain back to the exactly summed all-site coordinates.
   dp=pbar;dq=qbar
   if coherent:
    coverage=exact[:,2]/(m-1);inverse=np.zeros(m);np.divide(1.,coverage,out=inverse,where=coverage>0)
    dp=pbar+(pbar-1.)*(inverse-1.);dq=qbar*inverse
    unobserved_rows+=int(np.count_nonzero(coverage==0))
   def derivative(f):
    floor=.5/(length*coverage+1.) if coherent else .5/(length+1.)
    usable=f>floor;f=np.maximum(f,floor)
    if model==1:return np.where(usable,1./f,0.)
    exponent=-np.log(f)/alpha;return np.where(usable&(exponent<30),np.exp(np.minimum(exponent,30))/f,0.)
   if model in (1,2):bp=derivative(1-4*dp/3);bq=np.zeros(m)
   elif model==3:
    bp=derivative(1-2*dp+dq);bq=.5*(derivative(1-2*dq)-bp)
   else:bp=np.ones(m);bq=np.zeros(m)
   if coherent:
    bc=(bp*(1.-pbar)-bq*qbar)*inverse**2;bp=bp*inverse;bq=bq*inverse
   sample_p=(m-1)*np.bincount(owners[valid],weights=observed[:,1],minlength=m)/counts
   sample_q=(m-1)*np.bincount(owners[valid],weights=observed[:,2],minlength=m)/counts
   base-=bp*(sample_p-exact[:,0])+bq*(sample_q-exact[:,1])
   if coherent:
    sample_c=(m-1)*np.bincount(owners[valid],weights=observed[:,3],minlength=m)/counts
    base-=bc*(sample_c-exact[:,2])
  for iteration in range(ann_rounds+1):
   d=state.query(rr,ww,pairs,model,common,alpha);distance_queries+=len(d)
   scores=d-(base[pairs[:,0]]+base[pairs[:,1]])/(m-2) if q_candidates else d
   near=graph.select_directed_pairs(pairs,np.ascontiguousarray(scores),ranks,ann_k,workers)
   new_pairs=np.asarray(graph.expand_directed_pairs(near,order,iteration<ann_rounds))
   if iteration==ann_rounds:
    old_keys=pairs[:,0]*m+pairs[:,1];keys=new_pairs[:,0]*m+new_pairs[:,1];positions=np.searchsorted(old_keys,keys);assert np.array_equal(old_keys[positions],keys);d=d[positions]
   pairs=new_pairs
  times['statistics']+=time.perf_counter()-phase
  offs=np.ascontiguousarray(offset[active]);rows=np.ascontiguousarray(base+(m-2)*offs+offs.sum());ranks=np.ascontiguousarray(minleaf[active])
  phase=time.perf_counter();selected,count,degree,reciprocal=graph.select_bounded_nj_merges(pairs,d,ranks,rows,offs,cap,m-3,workers,order,True,minimum_divisor,q_candidates);times['selection']+=time.perf_counter()-phase
  selected=np.asarray(selected);a=active[selected[:,0]];b=active[selected[:,1]];dest=np.arange(nextnode,nextnode+len(a),dtype=np.int32)
  if len(a)<min(m-3,(m+2)//4 if minimum_divisor==4 else (m+minimum_divisor-1)//minimum_divisor):
   raise RuntimeError('AP-NJ contraction invariant violated')
  phase=time.perf_counter()
  if R==2:
   draws=rng.random((len(dest),2));ca=(draws[:,0,None]>=np.cumsum(mass[a],axis=1)).sum(1).clip(max=R-1);cb=(draws[:,1,None]>=np.cumsum(mass[b],axis=1)).sum(1).clip(max=R-1)
   reps[dest,0]=reps[a,ca];reps[dest,1]=reps[b,cb];mass[dest]=.5
  if R!=2:
   for u,v,w in zip(a,b,dest):
    ids=np.concatenate([reps[u,mass[u]>0],reps[v,mass[v]>0]]);weights=.5*np.concatenate([mass[u,mass[u]>0],mass[v,mass[v]>0]])
    if len(ids)<=R:reps[w,:len(ids)]=ids;mass[w,:len(ids)]=weights
    elif R==1:
     choice=rng.choice(len(ids),p=weights);reps[w,0]=ids[choice];mass[w,0]=1.
    else:
     # Stratified half-mixture sample: a fixed number from each child. It is
     # unbiased for the equally weighted NJ merge, without descendant lists.
     choice=np.concatenate([rng.choice(R,size=R//2,p=mass[u]),rng.choice(R,size=R//2,p=mass[v])])
     reps[w,:R//2]=reps[u,choice[:R//2]];reps[w,R//2:]=reps[v,choice[R//2:]];mass[w]=1./R
  topology.attach(neighbors,np.ascontiguousarray(np.column_stack([a,b,dest]),dtype=np.int64))
  projected[dest]=.5*(projected[a]+projected[b]);offset[dest]=-.5*d[selected[:,2]];minleaf[dest]=np.minimum(minleaf[a],minleaf[b])
  keep=np.ones(m,bool);keep[selected[:,:2].ravel()]=False;active=np.concatenate([active[keep],dest]);active=active[np.argsort(minleaf[active],kind='stable')]
  nextnode+=len(dest);times['merge']+=time.perf_counter()-phase
  history.append(dict(active=m,pool=len(pairs),retained=int(count),merges=len(dest),remaining=len(active),distance_queries=distance_queries,reciprocal=int(reciprocal)))
 phase=time.perf_counter();m=len(active)
 terminal_pairs=np.ascontiguousarray(np.array(np.triu_indices(m,1)).T,dtype=np.int64)
 rr=np.ascontiguousarray(reps[active]);ww=np.ascontiguousarray(mass[active])
 distances=state.query(rr,ww,terminal_pairs,model,common,alpha)
 matrix=np.zeros((m,m),np.float64);matrix[terminal_pairs[:,0],terminal_pairs[:,1]]=distances;matrix[terminal_pairs[:,1],terminal_pairs[:,0]]=distances
 terminal_query_seconds=time.perf_counter()-phase;phase=time.perf_counter()
 from _ct_apnj_terminal import solve
 triples,remaining=solve(matrix,np.ascontiguousarray(active,dtype=np.int64),np.ascontiguousarray(minleaf[active]),nextnode)
 topology.attach(neighbors,np.ascontiguousarray(triples,dtype=np.int64))
 nextnode+=len(triples);active=np.asarray(remaining,dtype=np.int32);terminal_solve_seconds=time.perf_counter()-phase
 terminal_meta=dict(active=m,pairs=len(terminal_pairs),matrix_bytes=matrix.nbytes,query_seconds=terminal_query_seconds,solve_seconds=terminal_solve_seconds,limit=DENSE_LIMIT,
                    solver='standard global-minimum NJ; signed reduced distances retained')
 center=nextnode;topology.connect_center(neighbors,np.ascontiguousarray(active,dtype=np.int64),center)
 from apnj_cleanup import refine
 local_meta=refine(neighbors,n,state,model,common,alpha)
 phase=time.perf_counter();g=topology.export_graph(neighbors,n,canonical)
 tree=Tree(g,dict(enumerate(names)),{});tree.require_binary()
 export_seconds=time.perf_counter()-phase
 return tree,dict(upload_seconds=upload_seconds,export_seconds=export_seconds,canonical=canonical,local_cleanup=local_meta,method='sparse Q-aware AP-NJ with bounded exact NJ terminal and fixed distance cleanup',constructor_seconds=time.perf_counter()-started,projection_seconds=projection_seconds,times=times,history=history,sequence_bytes=allocated,
  taxa=n,all_sites=length,full_site_input=True,seed=seed,projections=projections,window=window,cap=cap,representatives=R,landmarks=landmarks,ann_k=ann_k,model=model,common_sites=common,gamma_shape=alpha,minimum_divisor=minimum_divisor,real_ann=real_ann,row_control=row_control,q_candidates=q_candidates,
  terminal=terminal_meta,active_work=sum(x['active'] for x in history),candidate_work=sum(x['pool'] for x in history),health=dict(_health(state,unobserved=unobserved_rows),coherent_overlap_controls=coherent))
