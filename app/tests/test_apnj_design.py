"""Synthetic evidence, streaming and ratio-coordinate regression checks. No input writes."""
from pathlib import Path
import sys,json,time,hashlib
import argparse
p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);p.add_argument('--runtime',type=Path,default=Path(__file__).resolve().parents[1]);a=p.parse_args()
V=a.runtime.resolve();sys.path.insert(0,str(V))
from runtime_config import configure_runtime
configure_runtime(gpu=a.gpu,cpu_threads=16)
import torch,numpy as np
torch.set_num_threads(16);torch.set_num_interop_threads(1)
from apnj_frontend import initialize
import _ct_apnj_packed as native,_ct_apnj_graph as graph
from concordtree.treeio import newick_python,parse_newick,nrf
def pack(x):
 y=np.full((len(x),x.shape[1]+x.shape[1]%2),4,np.uint8);y[:,:x.shape[1]]=x
 return np.ascontiguousarray(y[:,::2]|(y[:,1::2]<<4))
checks=[]
for n in (4,31,32,33,511,512,513,514,515,1024):
 for task in ('gene','species'):
  x=np.zeros((n,1),np.uint8);names=[str(i) for i in range(n)];data=pack(x);before=data.copy();start=time.perf_counter()
  tree,meta=initialize(names,(data,names,1),task=task,seed=20260934)
  tree.require_binary();assert nrf(tree,parse_newick(newick_python(tree)))==0
  assert meta['history']==[] and not meta['health']['informative'] and meta['health']['warnings']
  np.testing.assert_array_equal(data,before)
  checks.append(dict(kind='constant_fast_completion',taxa=n,task=task,seconds=time.perf_counter()-start,health=meta['health']))
for l in (1,15,16,17,131071,131072,131073,262147):
 rng=np.random.default_rng(17+l);x=rng.integers(0,5,(12,l),dtype=np.uint8);x[:,0]=np.arange(12)%4
 s=native.State(pack(x),l,16);actual=s.evidence();expected_var=sum(np.count_nonzero((x==b).any(0)) for b in range(4)) # independent count below
 count=sum((x==b).any(0).astype(np.int16) for b in range(4));assert actual['variable_observed_sites']==int((count>1).sum())
 assert actual['jointly_observed_sites']==int(((x<4).sum(0)>1).sum());assert actual['observed_bases']==int((x<4).sum())
 rr=np.arange(12,dtype=np.int32).reshape(6,2);ww=np.full((6,2),.5);f=np.zeros((6,l,4))
 for i in range(6):
  for r in rr[i]:
   for b in range(4):f[i,:,b]+=.5*(x[r]==b)
 total=f.sum(0);expected=np.zeros((6,2))
 for i in range(6):
  expected[i,0]=5-(f[i]*(total-f[i])).sum()/l
  pur=f[i,:,0]+f[i,:,2];pyr=f[i,:,1]+f[i,:,3];ar=total[:,0]+total[:,2];ay=total[:,1]+total[:,3]
  expected[i,1]=(pur*(ay-pyr)+pyr*(ar-pur)).sum()/l
 np.testing.assert_allclose(s.raw_rows(rr,ww),expected,rtol=1e-13,atol=1e-13)
 assert s.diagnostics()['raw_control_scratch_bytes']<=64*2**20
 checks.append(dict(kind='evidence_and_streamed_exact_rows',sites=l,health=s.diagnostics()))
x=np.full((4,4),4,np.uint8);x[np.arange(4),np.arange(4)]=np.arange(4)
s=native.State(pack(x),4,16);r=np.arange(4,dtype=np.int32)[:,None];w=np.ones((4,1));s.query(r,w,np.array([[0,1]],np.int64),1,True)
assert s.diagnostics()['zero_overlap_pair_evaluations']==1
checks.append(dict(kind='zero_overlap_diagnostic',health=s.diagnostics()))
for function in ('projection','directed','bounded'):
 try:
  p=np.array([[0,1],[1,2],[2,3]],np.int64);r=np.arange(4,dtype=np.int64)
  if function=='projection':graph.projection_pairs(np.full((4,2),np.nan),1,1)
  elif function=='directed':graph.select_directed_pairs(p,np.full(3,np.nan),r,2,1)
  else:graph.select_bounded_nj_merges(p,np.ones(3),r,np.full(4,np.nan),np.zeros(4),2,1,1,r,True,514,True)
 except ValueError:pass
 else:raise AssertionError('Nonfinite values accepted:'+function)
 checks.append(dict(kind='nonfinite_rejected',entry=function))
# Independently verify the added common-observation control and chain rule.
for width in (1,2,4):
 l=131073 if width==4 else 97;rng=np.random.default_rng(41+width);x=rng.integers(0,6,(12,l),dtype=np.uint8);x[:,0]=np.arange(12)%4
 st=native.State(pack(x),l,16);groups=12//width;rr=np.arange(12,dtype=np.int32).reshape(groups,width);ww=np.full((groups,width),1./width)
 pairs=np.array([(i,j) for i in range(groups) for j in range(groups) if i!=j],np.int64)
 out=st.query(rr,ww,pairs,3,True,1.25,True,True);manual=[]
 for i,j in pairs:
  manual.append(sum(float(((x[a]<4)&(x[b]<4)).sum())/l/width**2 for a in rr[i] for b in rr[j]))
 np.testing.assert_allclose(out[:,3],manual,rtol=1e-13,atol=1e-13)
 exact=st.raw_rows(rr,ww,True);np.testing.assert_allclose(exact[:,2],np.array(manual).reshape(groups,groups-1).sum(1),rtol=1e-13,atol=1e-13)
 np.testing.assert_array_equal(st.raw_rows(rr,ww),exact[:,:2])
 checks.append(dict(kind='common_observation_coordinate',width=width,sites=l))
for model in (0,1,2,3):
 a,b,c=.68,.07,.55;alpha=1.25;p0=1-(1-a)/c;q0=b/c
 def value(a,b,c):
  p=1-(1-a)/c;q=b/c
  if model==0:return p
  if model==1:return -.75*np.log(1-4*p/3)
  def inv(f):return alpha*(f**(-1/alpha)-1)
  if model==2:return .75*inv(1-4*p/3)
  return .5*inv(1-2*p+q)+.25*inv(1-2*q)
 def deriv(f):return 1/f if model==1 else f**(-1/alpha-1)
 if model==0:bp,bq=1.,0.
 elif model in (1,2):bp,bq=deriv(1-4*p0/3),0.
 else:bp=deriv(1-2*p0+q0);bq=.5*(deriv(1-2*q0)-bp)
 analytic=np.array([bp/c,bq/c,(bp*(1-a)-bq*b)/c**2]);theta=np.array([a,b,c]);numerical=[]
 for i in range(3):
  high=theta.copy();low=theta.copy();high[i]+=1e-6;low[i]-=1e-6;numerical.append((value(*high)-value(*low))/2e-6)
 np.testing.assert_allclose(analytic,numerical,rtol=1e-8,atol=1e-8)
 checks.append(dict(kind='ratio_chain_rule',model=model))

# Exercise all sixteen aggregate tiles across the streaming boundary.
m,l=1024,131073;x=np.broadcast_to((np.arange(m)%4).astype(np.uint8)[:,None],(m,l)).copy()
s=native.State(pack(x),l,16);rr=np.arange(m,dtype=np.int32)[:,None];ww=np.ones((m,1))
actual=s.raw_rows(rr,ww,True);np.testing.assert_array_equal(actual,np.broadcast_to([768.,512.,1023.],actual.shape))
assert s.diagnostics()['raw_control_scratch_bytes']==64*2**20
checks.append(dict(kind='full_sixteen_tile_streaming_scratch_cap',sites=l,taxa=m))
print('PASS',len(checks),'checks',flush=True)
