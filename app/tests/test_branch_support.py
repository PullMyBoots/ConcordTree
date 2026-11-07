"""Statistical interpretation, branch mapping and numerical contract tests."""
from pathlib import Path
import sys,tempfile,unittest,re
import numpy as np

HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
from runtime_config import PROJECT,configure_runtime
configure_runtime(gpu=0)
from branch_support import gene_records,species_records,species_margin_index,split_identifiers,export_support
from concordtree.treeio import Tree,parse_newick,newick_python,split_map
from engine import write_json

def code(e,state=0,alt=1):return ((e+1)<<4)|(alt<<2)|state

class SupportContracts(unittest.TestCase):
    def test_gene_weights_have_defined_local_meaning(self):
        edges=[(10,11),(11,12),(12,13),(13,14)]
        rows=gene_records(edges,[[1,2,3,4]]*4,[[0,0],[-10,-20],[10,0],[10000,-10000]],
                          [0,0,0,0],2000)
        self.assertAlmostEqual(rows[0]['value'],100/3)
        self.assertGreater(rows[1]['value'],99.99)
        self.assertLess(rows[2]['value'],.01)
        self.assertEqual(rows[3]['value'],0.)
        for r in rows:self.assertAlmostEqual(sum(r['topology_weights']),1.)
        shifted=gene_records([edges[1]],[[1,2,3,4]],[[90,80]],[100],2000)
        self.assertEqual(rows[1]['topology_weights'],shifted[0]['topology_weights'])

    def test_species_fractional_votes_and_missing_evidence(self):
        codes=np.array([code(0,1,2)]*5+[code(-1,0,0)],dtype=np.int32)
        # Current=column1; two current wins, one alternative win, one tie,
        # one uninformative quartet. The inactive row never votes.
        weights=np.array([[0,2,0],[0,3,0],[0,0,2],[0,2,2],[7,7,7],[9,0,0]],dtype=float)
        rows=species_records([(10,11),(11,12)],codes,weights,block_size=2)
        self.assertEqual(rows[0]['sampled_quartet_agreement'],62.5)
        self.assertEqual(rows[0]['mean_worst_margin'],.75)
        self.assertEqual(rows[0]['value'],float(species_margin_index(.75)))
        self.assertEqual(rows[0]['informative_quartets'],4)
        self.assertEqual(rows[0]['active_quartets'],5)
        self.assertEqual(rows[0]['uninformative_quartets'],1)
        self.assertEqual(rows[0]['two_way_tied_quartets'],1)
        self.assertEqual(rows[0]['topology_vote_fractions'],[.625,.375,0.])
        self.assertIsNone(rows[1]['value'])
        self.assertEqual(rows[1]['status'],'no_informative_quartets')
        self.assertEqual(rows,species_records([(10,11),(11,12)],codes,weights,block_size=10))
        self.assertEqual(rows,species_records([(10,11),(11,12)],codes,weights,block_size=2,backend='numpy'))

    def test_margin_resolves_full_agreement_and_preserves_signed_evidence(self):
        codes=np.array([code(0),code(1),code(2)],dtype=np.int32)
        weights=np.array([[.001,0.,0.],[2.,0.,0.],[0.,5.,2.]])
        rows=species_records([(4,5),(5,6),(6,7)],codes,weights)
        self.assertEqual([r['sampled_quartet_agreement'] for r in rows],[100.,100.,0.])
        self.assertEqual([r['mean_worst_margin'] for r in rows],[.001,2.,-5.])
        self.assertLess(rows[0]['value'],rows[1]['value'])
        self.assertLess(rows[2]['value'],50.)
        self.assertEqual(float(species_margin_index(0)),50.)
        values=species_margin_index(np.array([-1e100,-5,-1,0,1,5,1e100]))
        self.assertTrue(np.all(np.diff(values)>0))
        self.assertTrue(np.all((values>=0)&(values<=100)))

    def test_native_margin_against_independent_scalar_reference(self):
        rng=np.random.default_rng(20261105);m=11;size=1003
        edge=rng.integers(0,m,size);state=rng.integers(0,3,size)
        alt=(state+rng.integers(1,3,size))%3
        codes=(((edge+1)<<4)|(alt<<2)|state).astype(np.int32)
        weights=rng.normal(size=(size,3));weights[::13]=0
        expected=np.zeros(m);counts=np.zeros(m)
        for i in range(size):
            w=weights[i];tol=64*np.finfo(float).eps*max(1.,max(abs(w)))
            if max(w)-min(w)<=tol:continue
            alternatives=[w[j] for j in range(3) if j!=state[i]]
            expected[edge[i]]+=w[state[i]]-max(alternatives);counts[edge[i]]+=1
        edges=[(20+i,21+i) for i in range(m)]
        rows=species_records(edges,codes,weights)
        np.testing.assert_array_equal([r['mean_worst_margin'] for r in rows],expected/counts)
        reference=species_records(edges,codes,weights,backend='numpy',block_size=127)
        np.testing.assert_allclose([r['mean_worst_margin'] for r in reference],expected/counts,rtol=1e-14)

    def test_uninformative_and_roundoff_ties(self):
        rows=species_records([(4,5)],np.array([code(0)]),np.array([[0.,0.,0.]]))
        self.assertIsNone(rows[0]['value'])
        rows=species_records([(4,5)],np.array([code(0)]),np.array([[1.,np.nextafter(1.,2.),1.]]))
        self.assertIsNone(rows[0]['value'])

    def test_invalid_evidence(self):
        for codes,w in [(np.array([-1]),np.zeros((1,3))),
                        (np.array([code(2)]),np.zeros((1,3))),
                        (np.array([code(0,0,0)]),np.zeros((1,3))),
                        (np.array([code(0)]),np.array([[np.nan,1.,0.]]))]:
            with self.assertRaises((ValueError,ArithmeticError)):species_records([(4,5)],codes,w)
        with self.assertRaises(ArithmeticError):gene_records([(4,5)],[[0,1,2,3]],[[np.inf,0]],[0],1)

    def test_split_ids_and_annotated_newick_preserve_tree(self):
        tree=parse_newick("(('A one':.1,'B''two':.2):.3,E:.4,(C:.5,D:.6):.7);")
        original=newick_python(tree);ids=split_identifiers(tree)
        self.assertEqual(len(ids),2)
        rerooted=parse_newick("(C:.5,D:.6,(E:.4,('B''two':.2,'A one':.1):.3):.7);")
        self.assertEqual({v[0] for v in ids.values()},{v[0] for v in split_identifiers(rerooted).values()})
        edges=sorted(ids);raw=dict(branches=[dict(edge=list(e),value=v,status='available')
                                           for e,v in zip(edges,[98.,55.])])
        with tempfile.TemporaryDirectory(dir=PROJECT/'output',prefix='support-test-') as folder:
            pred=Path(folder)/'prediction.nwk';pred.write_text(original+'\n')
            report=export_support(tree,'gene',raw,write_json=write_json,plain_prediction=pred,
                                  serialize=lambda a:newick_python(tree,a))
            text=pred.read_text();again=parse_newick(text)
            self.assertEqual(set(split_map(tree).values()),set(split_map(again).values()))
            self.assertEqual(again.names,tree.names)
            self.assertEqual(again.lengths,tree.lengths)
            self.assertEqual((Path(folder)/'prediction.unannotated.nwk').read_text(),original+'\n')
            stripped=re.sub(r'\)([0-9.eE+-]+)?\[&support_id=b\d+\]',')',text)
            self.assertEqual(stripped,original+'\n')
            self.assertFalse(report['calibrated']);self.assertEqual(report['available'],2)
            raw['branches'].pop()
            with self.assertRaises(ValueError):export_support(tree,'gene',raw,write_json=write_json,
                plain_prediction=pred,serialize=lambda a:newick_python(tree,a))

    def test_species_newick_labels_preserve_display_value_precision(self):
        tree=parse_newick('((a,b),c,(d,e));');edges=sorted(split_identifiers(tree))
        values=[float(v) for v in species_margin_index([.1234567,.1234568])]
        self.assertNotEqual(values[0],values[1])
        raw=dict(branches=[dict(edge=list(e),value=v,status='available') for e,v in zip(edges,values)])
        with tempfile.TemporaryDirectory(dir=PROJECT/'output',prefix='margin-export-') as folder:
            pred=Path(folder)/'prediction.nwk';pred.write_text(newick_python(tree)+'\n')
            report=export_support(tree,'species',raw,write_json=write_json,plain_prediction=pred,
                serialize=lambda a:newick_python(tree,a))
            labels=re.findall(r'\)([0-9.eE+-]+)\[&support_id=(b\d+)\]',pred.read_text())
            got={bid:float(value) for value,bid in labels}
            self.assertEqual([got[f'b{i:06d}'] for i in [1,2]],values)
            self.assertEqual(report['method'],'mean_worst_quartet_margin')
            self.assertEqual(report['ranking_field'],'mean_worst_margin')

    def test_large_skew_tree_identifiers_are_iterative(self):
        n=10000;graph={i:set() for i in range(2*n-2)}
        for u in range(n,2*n-3):graph[u].add(u+1);graph[u+1].add(u)
        def attach(t,u):graph[t].add(u);graph[u].add(t)
        attach(0,n);attach(1,n)
        for t in range(2,n-2):attach(t,n+t-1)
        attach(n-2,2*n-3);attach(n-1,2*n-3)
        tree=Tree(graph,{i:str(i) for i in range(n)},{})
        ids=split_identifiers(tree)
        self.assertEqual(len(ids),n-3)
        self.assertEqual(len({v[0] for v in ids.values()}),n-3)

if __name__=='__main__':unittest.main(verbosity=2)
