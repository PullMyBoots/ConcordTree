"""Pathological search regressions, including the <203-taxon zero threshold."""
from pathlib import Path
import math,sys,types,unittest
from unittest.mock import patch
import numpy as np
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
from runtime_config import configure_runtime
configure_runtime(gpu=0,cpu_threads=16)
from search_progress import ExactRecurrence,NumericalPlateau
from array_mlp_nni import refine
from concordtree.treeio import Tree,parse_newick,newick_python,newick,rooted,split_map
from exact_topology_key import topology_key


class Guards(unittest.TestCase):
    def test_exact_cycles_and_distinct_parameter_evidence_epochs(self):
        detector=ExactRecurrence()
        self.assertFalse(any(detector.repeated((b'tree',i)) for i in range(100)))
        for period in (1,2,3,5,17):
            detector=ExactRecurrence()
            found=next(i for i in range(100) if detector.repeated(bytes([i%period])))
            self.assertLess(found,4*period+2)

    def test_numerical_oscillation_terminates_with_best_incumbent(self):
        guard=NumericalPlateau(.01)
        for i in range(8):
            stop=guard.observe(-100+i*.0001,-100+(i+1)*.0001,
                               lambda:i%2,lambda:i)
            self.assertEqual(stop,i==7)
        self.assertEqual(guard.best_payload,7)

    def test_small_improvements_accumulating_real_progress_continue(self):
        guard=NumericalPlateau(.01)
        for i in range(40):
            self.assertFalse(guard.observe(i*.005,(i+1)*.005,lambda:i%2))

    def test_new_topologies_are_not_a_cycle(self):
        guard=NumericalPlateau(.01)
        for i in range(40):self.assertFalse(guard.observe(0.,.0001,lambda:i))

    def test_plateau_retains_best_state_when_last_round_is_worse(self):
        guard=NumericalPlateau(.01)
        gains=[.003,.002,.001,.002,.001,.002,.001,.002]
        for i,gain in enumerate(gains):
            stop=guard.observe(-100.,-100.+gain,lambda:i%2,lambda:i)
        self.assertTrue(stop);self.assertEqual(guard.best_payload,0)
        self.assertEqual(guard.best_value,-99.997)

    def test_nonfinite_objective_rejected(self):
        for value in (float('nan'),float('inf'),-float('inf')):
            with self.assertRaises(ArithmeticError):NumericalPlateau(.01).observe(0.,value,lambda:0)

    def test_threshold_boundary_is_preserved(self):
        for n in (4,24,32,64,128,192,202,203,204,256):
            self.assertEqual(math.floor(.005*(n-3)),0 if n<203 else 1)


class MLPCycles(unittest.TestCase):
    def run_search(self,*,probabilities=None,always_move=True):
        initial=np.array([[4,-1,-1],[4,-1,-1],[5,-1,-1],[5,-1,-1],[0,1,5],[2,3,4]],np.int64)
        graph={u:set(int(x) for x in row if x>=0) for u,row in enumerate(initial)}
        class Backend:
            def apply_nni_moves(self,neighbors,*args):
                if not always_move:return []
                # Swap leaves 1/2 across the central edge, then swap them back.
                a,b=int(neighbors[1,0]),int(neighbors[2,0])
                neighbors[a,np.where(neighbors[a]==1)[0][0]]=2
                neighbors[b,np.where(neighbors[b]==2)[0][0]]=1
                neighbors[1,0],neighbors[2,0]=b,a
                for row in neighbors:row.sort()
                return [0]
        plan=types.SimpleNamespace(
            compile_nni_plan=lambda *args:(np.array([[4,5]]),np.array([[0,1,2,3]]),np.array([0,1]),np.array([[0,1,2,3]]),np.array([[0,1,2,3]])),
            reduce_nni_arithmetic=lambda *args:np.array([[0.,1.,0.]]))
        nni=types.SimpleNamespace(_adjacency_neighbor_array=lambda *args:initial.copy())
        predict=lambda q:np.tile([.1,.8,.1],(len(q),1)) if probabilities is None else probabilities
        with patch('concordtree.assets.load_candidate_graph_backend',return_value=Backend()):
            result=refine(graph,4,predict,plan_backend=plan,nni=nni,stop_moves_at_most=0)
        Tree(graph,dict(enumerate('abcd')),{}).require_binary()
        return result

    def test_zero_threshold_cycle_stops_in_few_passes(self):
        result=self.run_search()
        self.assertEqual(result['stop'],'repeated_full_search_state')
        self.assertFalse(result['converged']);self.assertLess(len(result['passes']),8)

    def test_ordinary_stop_unchanged(self):
        result=self.run_search(always_move=False)
        self.assertEqual(result['stop'],'requested_move_ratio');self.assertTrue(result['converged'])
        self.assertEqual(len(result['passes']),1)

    def test_invalid_predictions_do_not_enter_native_selection(self):
        for values in ([[float('nan'),0,1]],[[float('inf'),0,0]],[[-.1,.5,.6]],[[.1,.1,.1]]):
            with self.subTest(values=values),self.assertRaises(ArithmeticError):
                self.run_search(probabilities=np.array(values))


class GraphSafety(unittest.TestCase):
    def test_cycle_does_not_hang_serialization_or_identity(self):
        graph={0:{4},1:{5},2:{6},3:{6},4:{0,5,6},5:{1,4,6},6:{2,3,4,5}}
        tree=Tree(graph,dict(enumerate('abcd')), {})
        for operation in (topology_key,newick_python,newick,rooted,split_map,lambda t:t.require_binary()):
            with self.subTest(operation=operation),self.assertRaises(ValueError):operation(tree)

    def test_malformed_graphs_rejected_by_reachable_traversals(self):
        for graph,names in (({},{}),({0:{1},1:set()},{0:'a'}),
                            ({0:{1},1:{0},2:set()},{0:'a'}),({0:{9}},{0:'a'})):
            tree=Tree(graph,names,{})
            for operation in (newick,newick_python,rooted):
                with self.subTest(graph=graph,operation=operation),self.assertRaises(ValueError):operation(tree)

    def test_ordinary_identity_and_serialization_unchanged(self):
        tree=parse_newick('((a,b),c,d);');tree.require_binary()
        self.assertEqual(topology_key(tree),topology_key(parse_newick(newick_python(tree))))
        self.assertEqual(newick(tree),newick_python(tree))

    def test_species_serializer_rejects_cycles_and_disconnected_nodes(self):
        from species_treeio import graph_to_newick
        good={0:{4},1:{4},2:{5},3:{5},4:{0,1,5},5:{2,3,4}}
        self.assertEqual(graph_to_newick(good,list('abcd'),4),'(a,b,(c,d));')
        for graph in ({**good,6:set()},
                      {**good,4:{0,1,5,6},5:{2,3,4,6},6:{4,5}},
                      {**good,0:set()}, {**good,4:{0,1,5,99}}):
            with self.subTest(graph=graph),self.assertRaises(ValueError):graph_to_newick(graph,list('abcd'),4)


class ParameterContinuationSafety(unittest.TestCase):
    def test_profile_outer_loop_has_a_separate_explicit_guard(self):
        import profile_invariant_fitter as fitter
        tree=parse_newick('((a,b),c,d);')
        class Objective:
            def __init__(self,*args):
                self.initial=np.ones(5);self.edges=5;self.sites=4
                self.fast_initial=1.;self.fast_index=24;self.max_exposure=800.
                self.bounds=[(1e-12,800.)]*5;self.spec=types.SimpleNamespace(gamma_categories=4,name='GTR+FO+I+G4')
                self.params=types.SimpleNamespace(parts={});self.other=np.array([],dtype=int);self.pindex=0
                self.plan=types.SimpleNamespace(edges=list(tree.lengths),setup_metrics={})
                self.calls=0;self.likelihood_calls=0;self.cache={}
            def __call__(self,raw):
                self.calls+=1;self.likelihood_calls+=1;self.cache={'p':.1}
                return float(np.sum(raw)),np.ones(5)
            def forward(self,raw,gradient=False):
                self.likelihood_calls+=1;return float(np.sum(raw)),.1
            def model_vector(self,raw):return np.zeros(1)
        values=np.zeros(30);values[:4]=.25;values[20:25]=[0,1,2,3,4];values[25:]=.2
        decoded=types.SimpleNamespace(values=values,exchangeabilities=np.ones(6),alpha=.5)
        def stalled_local_solver(objective,point,**kwargs):
            return types.SimpleNamespace(x=point.copy(),nit=0,success=False,status=1,message='injected stalled local solver')
        with patch.object(fitter,'ProfileObjective',Objective),patch.object(fitter,'decode_numpy',return_value=decoded),patch.object(fitter,'minimize',side_effect=stalled_local_solver),patch.object(fitter,'MAX_PARAMETER_CONTINUATIONS',2):
            result,info=fitter.fit_profiled(tree,list('abcd'),np.zeros((4,4),np.uint8),'GTR+FO+I+G4',previous={})
        self.assertEqual(info['continuation_stop'],'parameter_continuation_guard')
        self.assertFalse(info['converged']);self.assertEqual(len(info['rate_mode_profiles']),2)
        self.assertTrue(all(math.isfinite(x) and x>0 for x in result.lengths.values()))


class MSCStates(unittest.TestCase):
    def test_compiled_state_encodes_every_tree_edge(self):
        from msc_refine.core import compile_plan
        for text in ('((a,b),c,d);','(((a,b),c),d,e);','((a,b),(c,d),(e,f));'):
            tree=parse_newick(text);names=list(tree.names.values())
            mapping={u:i for i,u in enumerate(tree.names)}
            mapping.update({u:len(names)+i for i,u in enumerate(u for u in tree.graph if u not in tree.names)})
            graph={mapping[u]:{mapping[v] for v in vs} for u,vs in tree.graph.items()}
            plan=compile_plan(graph,len(names));recovered=set()
            for (u,v),(a,b,c,d) in zip(plan.edges,plan.branches):
                recovered.update(tuple(sorted(edge)) for edge in ((u,v),(u,a),(u,b),(v,c),(v,d)))
            self.assertEqual(recovered,{tuple(sorted((u,v))) for u,vs in graph.items() for v in vs})

    def scorer(self,*,growing=False,nonfinite=False):
        class Scorer:
            packed=np.zeros((24,1),np.uint8)
            base=types.SimpleNamespace(check=None)
            queries=np.empty((0,4),np.int32)
            calls=0
            def score(self,g):
                self.calls+=1
                self.queries=np.zeros((self.calls if growing else 1,4),np.int32)
                self.timing=dict(objective=float('nan') if nonfinite else 2.*(self.calls-1),
                                 new_evidence_objective=1. if self.calls>1 else 0.)
                return types.SimpleNamespace(edges=np.array([[4,5]]),branches=np.array([[0,1,2,3]])),np.array([[0.,1.,0.]])
        return Scorer()

    def test_fixed_evidence_cycle_stops(self):
        from msc_refine.active_quartet import refine_active
        scorer=self.scorer();tree=parse_newick('((a,b),c,d);')
        with patch('msc_refine.active_quartet.select_moves',return_value=[(0,1,1.)]),patch('msc_refine.active_quartet.apply_move'):
            _,result=refine_active(tree.graph,scorer)
        self.assertEqual(result['stop'],'repeated_full_search_state')
        self.assertFalse(result['search_converged']);self.assertEqual(scorer.calls,2)

    def test_growing_evidence_is_not_a_recurrence(self):
        from msc_refine.active_quartet import refine_active
        scorer=self.scorer(growing=True);tree=parse_newick('((a,b),c,d);')
        with patch('msc_refine.active_quartet.select_moves',side_effect=[[(0,1,1.)],[(0,1,1.)],[]]),patch('msc_refine.active_quartet.apply_move'):
            _,result=refine_active(tree.graph,scorer)
        self.assertEqual(result['stop'],'requested_move_ratio');self.assertTrue(result['search_converged'])

    def test_nonfinite_objective_stops_before_moves(self):
        from msc_refine.active_quartet import refine_active
        with self.assertRaises(ArithmeticError):refine_active(parse_newick('((a,b),c,d);').graph,self.scorer(nonfinite=True))


if __name__=='__main__':unittest.main(verbosity=2)
