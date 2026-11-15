"""Configurable distance NNI module (default four rounds); no neural model."""
import time
import numpy as np

def refine(neighbors,n,state,model,common,alpha,*,max_rounds=4,stop_ratio=0.,representatives=4,min_gain=0.,checkpoint=None,backbone=None,names=None):
 import _ct_apnj_local as native
 started=time.perf_counter()
 history=[];times=dict(plan=0.,distance=0.,apply=0.)
 from search_progress import ExactRecurrence,EMERGENCY_SWEEPS
 recurrence=ExactRecurrence();stop='round_budget'
 if max_rounds==0:return dict(method='disabled',passes=[],times=times,seconds=0.,maximum_passes=0,component_width=representatives,stop='disabled',converged=True)
 for iteration in range(max_rounds or EMERGENCY_SWEEPS):
  if max_rounds is None and recurrence.repeated(neighbors.tobytes()):stop='repeated_full_search_state';break
  phase=time.perf_counter();edges,branches,reps,mass=native.plan(neighbors,n);E=len(edges);times['plan']+=time.perf_counter()-phase
  if representatives!=4:
   mass[:,representatives:]=0.;mass/=mass.sum(axis=1,keepdims=True)
  pairs=np.ascontiguousarray((4*np.arange(E,dtype=np.int64)[:,None,None]+np.array([[0,1],[2,3],[0,2],[1,3],[0,3],[1,2]],np.int64)[None,:,:]).reshape(-1,2))
  phase=time.perf_counter();dist=state.query(reps,mass,pairs,model,common,alpha).reshape(E,6);times['distance']+=time.perf_counter()-phase
  scores=np.column_stack([dist[:,0]+dist[:,1],dist[:,2]+dist[:,3],dist[:,4]+dist[:,5]]);best=np.argmin(scores,axis=1).astype(np.int64);margin=np.ascontiguousarray(scores[:,0]-scores[np.arange(E),best])
  if min_gain:best[margin<=min_gain]=0
  if backbone is not None:
   allowed=np.asarray(backbone.allowed_edges(neighbors,dict(enumerate(names)),edges),dtype=bool)
   best[~allowed]=0; margin[~allowed]=0.
  phase=time.perf_counter();selected=native.apply(neighbors,edges,branches,best,margin);times['apply']+=time.perf_counter()-phase
  history.append(dict(pass_=iteration+1,edges=E,component_pairs=len(pairs),moves=len(selected),maximum_proposal_gain=float(margin.max())))
  if checkpoint:checkpoint(neighbors,dict(stage='distance',round=iteration+1,moves=len(selected),maximum_proposal_gain=float(margin.max())))
  if len(selected)/(n-3)<=stop_ratio:stop='requested_move_ratio';break
 else:
  if max_rounds is None:raise RuntimeError('Distance NNI emergency sweep guard exhausted')
 return dict(method='four-point mean corrected distance; central-edge closed-neighborhood exclusion',passes=history,times=times,seconds=time.perf_counter()-started,maximum_passes=max_rounds,component_width=representatives,stop=stop,converged=stop=='requested_move_ratio',stop_ratio=stop_ratio,min_gain=min_gain)
