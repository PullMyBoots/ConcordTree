"""DIPPER handoff and lifecycle regressions; no source data is written."""
from pathlib import Path
import sys
import unittest
import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from runtime_config import configure_runtime
configure_runtime(gpu=0, cpu_threads=16)
from dipper_frontend import build_dipper, canonical_topology
from concordtree.treeio import nrf, newick_python

class Handoff(unittest.TestCase):
    def graph(self):
        # An upstream four-leaf tree, including the original degree-two root 4.
        edges = [(0,5),(2,5),(5,4),(4,6),(1,6),(3,6)]
        graph = {}
        for u,v in edges:
            graph.setdefault(u,set()).add(v);graph.setdefault(v,set()).add(u)
        return graph

    def test_suppress_root_restore_names_and_numbering(self):
        names = ['a:1', 'b,2', "c'3", 'd[4]']
        tree = canonical_topology(self.graph(),names,[3,1,0,2])
        tree.require_binary()
        self.assertEqual(tree.names,dict(enumerate(names)))
        self.assertEqual(len(tree.graph),6)
        from concordtree.treeio import parse_newick
        self.assertEqual(nrf(tree,parse_newick(newick_python(tree))),0)

    def test_reject_duplicate_leaf_mapping(self):
        with self.assertRaisesRegex(RuntimeError,'permutation'):
            canonical_topology(self.graph(),['a','b','c','d'],[0,1,2,2])

    def test_reject_cycle(self):
        graph=self.graph();graph[5].add(6);graph[6].add(5)
        with self.assertRaisesRegex(RuntimeError,'cyclic'):
            canonical_topology(graph,['a','b','c','d'],[0,1,2,3])

    def test_reject_noncontiguous_or_wrong_dtype(self):
        names=['a','b','c','d']
        for array in [np.zeros((4,8),dtype=np.float32),np.zeros((4,16),dtype=np.uint8)[:,::2]]:
            with self.assertRaisesRegex(ValueError,'contiguous uint8'):
                build_dipper(names,(array,names,16),seed=1)

class GPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        if not torch.cuda.is_available():raise unittest.SkipTest('CUDA required')

    def test_unknown_taxon_and_odd_tail(self):
        # High padding nibble 0 must not make a completely unknown taxon valid.
        names=['a','b','c','d'];packed=np.zeros((4,1),dtype=np.uint8)
        packed[2,0]=4
        with self.assertRaisesRegex(ValueError,'unobserved taxon'):
            build_dipper(names,(packed,names,1),seed=20260934)

    def test_repeated_calls_have_no_device_memory_leak(self):
        import torch
        from exact_site_patterns import PackedPatterns
        source=HERE.parents[1]/'web/examples/gene.phy'
        x=PackedPatterns(source);data=(x.packed,x.names,x.original_sites)
        tree,_=build_dipper(x.names,data,seed=20260934)
        packed_before=x.packed.copy()
        free_before,_=torch.cuda.mem_get_info()
        for _ in range(8):
            repeated,_=build_dipper(x.names,data,seed=20260934)
            self.assertEqual(newick_python(tree),newick_python(repeated))
        free_after,_=torch.cuda.mem_get_info()
        np.testing.assert_array_equal(packed_before,x.packed)
        self.assertLess(free_before-free_after,65536)

if __name__=='__main__':unittest.main(verbosity=2)
