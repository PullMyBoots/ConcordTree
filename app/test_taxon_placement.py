"""Exact placement objective, ideal-evidence recovery and invariant checks."""
from pathlib import Path
import sys,unittest,itertools
sys.path.insert(0,str(Path(__file__).resolve().parent))
from runtime_config import configure_runtime
configure_runtime(cpu_threads=4)
import numpy as np
from concordtree.treeio import Tree,parse_newick,split_map
from backbone import restricted_graph,_signature,Backbone
from taxon_placement import PlacementIndex,place,attach,edge

def prune(tree,keep):
 g,n=restricted_graph(tree.graph,tree.names,set(keep));return Tree(g,n,{})
def same(a,b):
 interner={};return _signature(a.graph,a.names,interner)==_signature(b.graph,b.names,interner)
def perfect(truth,names):
 def predict(queries):
  result=[]
  for row in queries:
   labels=[names[int(i)] for i in row];splits=set(split_map(prune(truth,labels),labels).values())
   result.append([.999998 if c in splits else .000001 for c in [3,5,6]])
  return np.array(result)
 return predict

class PlacementTests(unittest.TestCase):
 def test_all_edges_with_exact_quartets(self):
  names=list('abcdefghijq');tree=parse_newick('((a,b),(c,d));');rng=np.random.default_rng(117)
  for n in range(4,11):
   if n>4:
    edges=[edge(u,v) for u in tree.graph for v in tree.graph[u] if u<v]
    tree=attach(tree,names[n-1],edges[int(rng.integers(len(edges)))])
   for representatives in [1,2,4]:
    index=PlacementIndex(tree,names,representatives)
    for selected in index.score_edges(np.zeros((n-2,3))):
     truth=attach(tree,'q',selected)
     result,info=place(tree,list(tree.names.values())+['q'],['q'],perfect(truth,list(tree.names.values())+['q']),representatives=representatives)
     self.assertTrue(same(result,truth));self.assertEqual(info['apnj_calls'],0)
 def test_reroot_scores_equal_brute_objective(self):
  tree=parse_newick('(((a,b),(c,d)),(e,f));');index=PlacementIndex(tree,list('abcdef'),4)
  potentials=np.random.default_rng(16).normal(size=(4,3))
  for target,actual in index.score_edges(potentials).items():
   total=0.
   for j,u in enumerate(index.internal):
    for k,v in enumerate(index.neighbors[u]):
     seen={u};pending=[v];found=False
     while pending:
      x=pending.pop()
      if x in seen:continue
      seen.add(x)
      if x in target:found=True;break
      pending.extend(index.graph[x]-seen)
     if found:total+=potentials[j,k];break
   self.assertAlmostEqual(actual,total,places=11)
 def test_sequential_missing_clade_and_order(self):
  truth=parse_newick('((((a,b),(c,d)),(e,f)),((g,h),(i,j)));');names=list('abcdefghij')
  for missing in [('a','b'),('a','b','c','d'),('a','d','h')]:
   tree=prune(truth,set(names)-set(missing));constraint=Backbone(__import__('concordtree.treeio',fromlist=['newick_python']).newick_python(tree),names)
   for order in itertools.permutations(missing):
    result,info=place(tree,names,order,perfect(truth,names),representatives=2)
    self.assertTrue(same(result,truth));constraint.assert_tree(result)
    self.assertEqual(len(info['placements']),len(missing))
 def test_bad_scores_and_missing_list(self):
  tree=parse_newick('((a,b),(c,d));');index=PlacementIndex(tree,list('abcd'),2)
  with self.assertRaises(ArithmeticError):index.potentials(np.full((len(index.triples),3),np.nan))
  with self.assertRaises(ArithmeticError):index.potentials(np.full((len(index.triples),3),-.5))
  with self.assertRaises(ValueError):PlacementIndex(tree,list('abcd'),0)
  with self.assertRaises(ValueError):place(tree,list('abcde'),[],None)
  with self.assertRaises(ValueError):place(tree,list('abcde'),['e','e'],None)
 def test_uninformative_scores_are_flagged(self):
  tree=parse_newick('((a,b),(c,d));');result,info=place(tree,list('abcde'),['e'],lambda q:np.full((len(q),3),1./3))
  self.assertEqual(info['placements'][0]['score_gap'],0.)
  self.assertFalse(info['placements'][0]['informative']);self.assertTrue(info['warnings'])
  Backbone('((a,b),(c,d));',list('abcde')).assert_tree(result)
 def test_deep_ladder_iterative(self):
  n=2004;names=[f't{i}' for i in range(n)];g={i:set() for i in range(2*n-2)}
  for j in range(n-3):g[n+j].add(n+j+1);g[n+j+1].add(n+j)
  for leaf,parent in [(0,n),(1,n)]+[(j,n+j-1) for j in range(2,n-2)]+[(n-2,2*n-3),(n-1,2*n-3)]:g[leaf].add(parent);g[parent].add(leaf)
  tree=Tree(g,dict(enumerate(names)),{});idx=PlacementIndex(tree,names,2)
  self.assertEqual(len(idx.score_edges(np.zeros((n-2,3)))),2*n-3)

if __name__=='__main__':unittest.main()
