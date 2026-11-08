"""Configuration boundaries and complete-tree pause protocol, without GPU work."""
from pathlib import Path
import itertools,json,sys,tempfile,threading,time,unittest
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
from pipeline_config import defaults,normalize
from runtime_config import configure_runtime
configure_runtime(gpu=0,cpu_threads=16)
from concordtree.treeio import parse_newick
from checkpoints import Checkpoints

class PipelineContract(unittest.TestCase):
    def test_orders_and_disabled_modules(self):
        for task in ('gene','species'):
            base=defaults(task)
            for order in itertools.permutations(base['stages']):
                for bits in itertools.product((False,True),repeat=3):
                    config=defaults(task);config['stages']=[dict(s,enabled=enabled) for s,enabled in zip(order,bits)]
                    self.assertEqual(normalize(config,task),config)
        gene=defaults('gene');species=defaults('species')
        gene['apnj']['seed']=0
        self.assertNotEqual(gene['apnj'],species['apnj'])

    def test_rejects_incompatible_and_invalid_parameters(self):
        for task in ('gene','species'):
            for value in (float('nan'),float('inf'),-1,True,'2'):
                config=defaults(task);config['stages'][0]['max_rounds']=value
                with self.subTest(task=task,value=value),self.assertRaises(ValueError):normalize(config,task)
            config=defaults(task);config['stages'][0]['id']=config['stages'][1]['id']
            with self.assertRaises(ValueError):normalize(config,task)
            config=defaults(task);config['stages'][-1]['id']='kernel' if task=='gene' else 'likelihood'
            with self.assertRaises(ValueError):normalize(config,task)
            config=defaults(task);config['apnj']['unrecognized']=5
            with self.assertRaises(ValueError):normalize(config,task)

    def test_pause_acknowledges_current_request_and_preserves_complete_tree(self):
        (HERE/'runs').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=HERE/'runs',prefix='checkpoint-test-') as name:
            out=Path(name);request=out/'pause.request';request.write_text('first')
            controller=Checkpoints(out,request);tree=parse_newick('((a,b),(c,d));');failures=[]
            def publish():
                try:controller.publish(tree,dict(stage='apnj'))
                except Exception as error:failures.append(error)
            thread=threading.Thread(target=publish,daemon=True);thread.start()
            def wait_token(token):
                deadline=time.monotonic()+5
                while time.monotonic()<deadline:
                    try:
                        if json.loads((out/'pause.state.json').read_text()).get('request_token')==token:return
                    except (OSError,ValueError):pass
                    time.sleep(.01)
                self.fail('Pause request was not acknowledged')
            try:
                wait_token('first');snapshot=(out/'checkpoint.nwk').read_bytes()
                parse_newick(snapshot.decode()).require_binary()
                # Emulate resume/pause faster than the worker sleep interval.
                request.write_text('second');wait_token('second')
                self.assertTrue(thread.is_alive());self.assertEqual(snapshot,(out/'checkpoint.nwk').read_bytes())
                self.assertEqual(json.loads((out/'checkpoint.json').read_text())['stage'],'apnj')
            finally:
                request.unlink(missing_ok=True);thread.join(5)
            self.assertFalse(thread.is_alive());self.assertFalse(failures)
            self.assertFalse((out/'pause.state.json').exists());self.assertGreater(controller.paused_seconds,0)

if __name__=='__main__':unittest.main(verbosity=2)
