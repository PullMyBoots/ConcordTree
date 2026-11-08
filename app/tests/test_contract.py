"""CPU contract checks; all temporary files stay in project output/."""
from pathlib import Path
import hashlib
import math
import sys
import tempfile
import types
import unittest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from runtime_config import PROJECT, owned_path, protected_targets, configure_runtime
configure_runtime(gpu=0)
from inputs import prepare_input, validate_phylip
from frontend import MLP_ASSETS, TASK_PREDICTOR
from concordtree.dna_models import ModelParameters, parse_model
import numpy as np


class Contracts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=PROJECT/'output', prefix='v101-test-')
        self.out = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_immutable_paths(self):
        for path in [PROJECT, PROJECT.parent/'outside', PROJECT/'data'/'never-created',
                     PROJECT/'.ResearchProject'/'never-created', *[p/'never-created' for p in protected_targets()]]:
            with self.subTest(path=str(path)), self.assertRaises(ValueError):owned_path(path)
        source = self.out/'input.fa';source.write_text('input')
        for path in (source, source.parent, source/'child'):
            with self.assertRaises(ValueError):owned_path(path, inputs=[source])
        self.assertEqual(owned_path(self.out/'job', inputs=[source]), self.out/'job')

    def test_fasta_conversion_and_input_unchanged(self):
        source=self.out/'input.fa'
        source.write_text('>a\nacgt\nn-?\n>b\nACGTNNN\n>c\nACGTNNN\n>d\nACGTNNN\n')
        before=source.read_bytes();job=self.out/'job';job.mkdir()
        phy=prepare_input(source,job)
        validate_phylip(phy)
        self.assertEqual(source.read_bytes(),before)
        self.assertTrue(phy.read_text().startswith('4 7\na ACGTN-?\n'))

    def test_bad_alignment_rejected(self):
        fixtures=['4 4\na ACG\nb ACGT\nc ACGT\nd ACGT\n',
                  '4 4\na ACGTT\nb ACGT\nc ACGT\nd ACGT\n',
                  '4 4\na ACGT\na ACGT\nc ACGT\nd ACGT\n',
                  '4 4\na ACEF\nb ACGT\nc ACGT\nd ACGT\n',
                  '4 4\na ACGT\nb ACGT\nc ACGT\nd ACGT\ne ACGT\n',
                  '>a\nACGT\n>b\nACG\n>c\nACGT\n>d\nACGT\n']
        for index,value in enumerate(fixtures):
            source=self.out/f'input{index}';source.write_text(value)
            job=self.out/f'job{index}';job.mkdir()
            with self.subTest(index=index),self.assertRaises(ValueError):prepare_input(source,job)

    def test_checkpoints_and_task_routing(self):
        self.assertEqual(TASK_PREDICTOR,dict(gene='homogeneous',species='heterogeneous'))
        for name,(filename,digest) in MLP_ASSETS.items():
            self.assertEqual(hashlib.sha256((HERE/'assets/mlp'/filename).read_bytes()).hexdigest(),digest,name)

    def test_worker_count_respects_native_extension_capacity(self):
        import os
        try:
            for requested in (16,64,65,256):
                runtime=configure_runtime(cpu_threads=requested)
                self.assertEqual(runtime['requested_cpu_threads'],requested)
                self.assertEqual(runtime['cpu_threads'],min(requested,64))
                self.assertEqual(os.environ['OMP_NUM_THREADS'],str(min(requested,64)))
        finally:configure_runtime(cpu_threads=16)

    def test_roundtrip_parameter_bounds(self):
        spec=parse_model('GTR+FO+I+G4');states=np.array([[0,1,2,3]],dtype=np.uint8)
        for alpha in (.02,100.,math.exp(math.log(.02)),math.exp(math.log(100.))):
            parameters=ModelParameters(spec,states,alpha=alpha)
            self.assertTrue(.02<=parameters.alpha<=100.)
        for alpha in (.019,100.000001,float('nan'),float('inf')):
            with self.assertRaises(ValueError):ModelParameters(spec,states,alpha=alpha)
        for pinv in (0.,.95000001,float('nan')):
            with self.assertRaises(ValueError):ModelParameters(spec,states,pinv=pinv)
        p=ModelParameters(spec,states,pinv=np.nextafter(.95,1.))
        self.assertEqual(p.pinv,.95)

    def test_api_options_and_fresh_output(self):
        from engine import UnifiedEngine
        engine=UnifiedEngine.__new__(UnifiedEngine)
        engine.predictors=dict(gene=None,species=None)
        source=self.out/'in.phy';source.write_text('4 1\na A\nb C\nc G\nd T\n')
        for options in [dict(task='other'),dict(task='gene',loci=source),
                        dict(task='species',model='JC'),dict(task='species',views=2),
                        dict(task='species',max_rounds=1),dict(task='gene',sample_fraction=0),
                        dict(task='gene',stop_ratio=1),dict(task='gene',max_rounds=0)]:
            with self.subTest(options=options),self.assertRaises(ValueError):
                engine.infer(source,self.out/'not-created',**options)
        self.assertFalse((self.out/'not-created').exists())
        with self.assertRaises(ValueError):engine.infer(source,self.out,task='gene')
        self.assertEqual(source.read_text(),'4 1\na A\nb C\nc G\nd T\n')

    def test_foreign_runtime_rejected(self):
        old=sys.modules.get('algorithm');module=types.ModuleType('algorithm')
        module.__file__=str(HERE.parent/'foreign-release/algorithm.py');sys.modules['algorithm']=module
        try:
            with self.assertRaisesRegex(RuntimeError,'incompatible runtime'):configure_runtime()
        finally:
            if old is None:sys.modules.pop('algorithm',None)
            else:sys.modules['algorithm']=old


if __name__=='__main__':unittest.main(verbosity=2)
