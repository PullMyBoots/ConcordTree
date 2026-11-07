"""Fixed four-pass, distance-only initialization cleanup; no neural model."""
import time
import numpy as np

def refine(neighbors,n,state,model,common,alpha):
 import _ct_apnj_local as native
 started=time.perf_counter()
 history=[];times=dict(plan=0.,distance=0.,apply=0.)
 for iteration in range(4):
  phase=time.perf_counter();edges,branches,reps,mass=native.plan(neighbors,n);E=len(edges);times['plan']+=time.perf_counter()-phase
  pairs=np.ascontiguousarray((4*np.arange(E,dtype=np.int64)[:,None,None]+np.array([[0,1],[2,3],[0,2],[1,3],[0,3],[1,2]],np.int64)[None,:,:]).reshape(-1,2))
  phase=time.perf_counter();dist=state.query(reps,mass,pairs,model,common,alpha).reshape(E,6);times['distance']+=time.perf_counter()-phase
  scores=np.column_stack([dist[:,0]+dist[:,1],dist[:,2]+dist[:,3],dist[:,4]+dist[:,5]]);best=np.argmin(scores,axis=1).astype(np.int64);margin=np.ascontiguousarray(scores[:,0]-scores[np.arange(E),best])
  phase=time.perf_counter();selected=native.apply(neighbors,edges,branches,best,margin);times['apply']+=time.perf_counter()-phase
  history.append(dict(pass_=iteration+1,edges=E,component_pairs=len(pairs),moves=len(selected),maximum_proposal_gain=float(margin.max())))
  if not len(selected):break
 return dict(method='four-point mean corrected distance over four nearest component representatives; fixed four sweeps; central-edge closed-neighborhood exclusion',passes=history,times=times,seconds=time.perf_counter()-started,maximum_passes=4,component_width=4)
