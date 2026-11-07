"""Exact-tie matching must make progress without overriding strict Q preferences."""
from pathlib import Path
import sys,argparse,numpy as np
p=argparse.ArgumentParser();p.add_argument('--runtime',type=Path,default=Path(__file__).resolve().parents[1]);a=p.parse_args();sys.path.insert(0,str(a.runtime.resolve()))
import _ct_apnj_graph as graph
n=1024;r=np.arange(n,dtype=np.int64);pairs=np.column_stack([r[:-1],r[1:]]);zeros=np.zeros(n)
def select(distances,workers):return np.asarray(graph.select_bounded_nj_merges(pairs,distances,r,zeros,zeros,32,n-3,workers,r,True,514,True)[0])
outputs=[]
for workers in (1,3,16):
 selected=select(np.zeros(n-1),workers);assert len(selected)==n//2;assert len(np.unique(selected[:,:2]))==n;outputs.append(selected)
for out in outputs[1:]:np.testing.assert_array_equal(out,outputs[0])
strict=select(np.arange(n-1,dtype=np.float64),16)
np.testing.assert_array_equal(strict,np.array([[0,1,0],[2,3,2]]))
d=np.zeros(n-1);d[0]=-1;selected=select(d,16);np.testing.assert_array_equal(selected[0],np.array([0,1,0]));assert len(selected)==n//2
print('PASS exact-tie maximal path matching, strict-preference preservation, deterministic1/3/16 threads')
