"""UI adapter regression tests. No scientific subprocess or GPU is launched."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from web import server

PHY = b'4 8\na ACGTACGT\nb ACGTACGA\nc ACGTTCGT\nd ACGTACGG\n'
NWK = b'((a:0.1,b:0.1):0.1,c:0.2,d:0.2);\n'


SPECIES = ('24 8\n'+''.join(f't{i}  ACGTACG'+('A' if i%2 else 'T')+'\n' for i in range(24))).encode()


class InputTests(unittest.TestCase):
    def test_phylip(self):
        raw, meta, lengths = server.normalize_alignment(PHY)
        self.assertEqual(meta, dict(taxa=4, sites=8, loci=1, partitioned=False))
        self.assertEqual(lengths, [8]); self.assertEqual(raw, PHY)

    def test_fasta_rejected(self):
        with self.assertRaisesRegex(ValueError, 'FASTA'):
            server.normalize_alignment(b'>a\nACGT\n>b\nACGA\n>c\nACGG\n>d\nACGC\n')

    def test_gene_multiple_loci_rejected(self):
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            server.normalize_alignment(PHY + b'\n' + PHY)

    def test_wrapped_and_interleaved(self):
        wrapped=b'4 8\na\nACGT\nACGT\nb ACGT\nACGA\nc ACGT\nTCGT\nd ACGT\nACGG\n'
        interleaved=b'4 8\na ACGT\nb ACGT\nc ACGT\nd ACGT\n\nACGT\nACGA\nTCGT\nACGG\n'
        repeated=b'4 8 I\na ACGT\nb ACGT\nc ACGT\nd ACGT\n\na ACGT\nb ACGA\nc TCGT\nd ACGG\n'
        for raw in [wrapped, interleaved, repeated]:
            with self.subTest(raw=raw): self.assertEqual(server.normalize_alignment(raw)[0], PHY)

    def test_species_auto_partitions_and_missing_taxa(self):
        # Names repeat between loci, and each locus can have a different taxon set.
        raw, meta, lengths=server.normalize_alignment(SPECIES+b'\n2 3\nt1  A-C\nnew  GTG\n', 'species')
        self.assertEqual(meta, dict(taxa=25,sites=11,loci=2,partitioned=True))
        self.assertEqual(lengths,[8,3])
        rows=dict(line.split() for line in raw.decode().splitlines()[1:])
        self.assertEqual(rows['t1'],'ACGTACGAA-C')
        self.assertEqual(rows['t0'],'ACGTACGT???')
        self.assertEqual(rows['new'],'????????GTG')
        self.assertFalse(server.normalize_alignment(SPECIES,'species')[1]['partitioned'])

    def test_invalid_sequences(self):
        for data in [b'', b'3 8\na ACGTACGT\nb ACGTACGT\nc ACGTACGT', PHY.replace(b'd ',b'a '),
                     PHY.replace(b'ACGTACGG',b'ACG'), PHY.replace(b'ACGTACGG',b'ACGTEFGG'),
                     PHY+b'garbage', PHY.replace(b'ACGTACGG',b'NNNNNNNN')]:
            with self.subTest(data=data):
                with self.assertRaises(ValueError): server.normalize_alignment(data)
        with self.assertRaisesRegex(ValueError,'24'): server.normalize_alignment(PHY,'species')

    def test_expansion_and_locus_integrity(self):
        # Small individual loci can expand substantially when their taxa do not overlap.
        raw=SPECIES+b'2 4\nnew1 ACGT\nnew2 ACGT\n'
        matrix, metadata, lengths=server.normalize_alignment(raw,'species')
        self.assertEqual(metadata, dict(taxa=26,sites=12,loci=2,partitioned=True))
        self.assertEqual(lengths,[8,4])
        self.assertIn(b'new1 ????????ACGT\n',matrix)
        with self.assertRaises(ValueError): server.normalize_alignment(SPECIES+b'2 4\na ACGT\na ACGT\n','species')


class AppTests(unittest.TestCase):
    def test_compute_resources_are_validated_and_frozen_per_job(self):
        import json
        resources=dict(cpu_threads=2,gpu=1)
        response=self.upload(resources=json.dumps(resources))
        self.assertEqual(response.status_code,202,response.text)
        job=response.json();directory=self.store.directory/job['id']
        self.assertEqual(job['resources'],resources)
        with patch.dict(server.os.environ,{'CONCORDTREE_GPU':'0','CONCORDTREE_CPU_THREADS':'8'}):
            command=server.command_for(job,directory)
        self.assertEqual(command[command.index('--gpu')+1],'1')
        self.assertEqual(command[command.index('--cpu-threads')+1],'2')
        self.assertEqual(json.loads((directory/'job.json').read_text())['resources'],resources)
        for invalid in [{'gpu':-1},{'gpu':True},{'gpu':1.2},{'cpu_threads':0},{'cpu_threads':65},{'cpu_threads':False},[],{'extra':1}]:
            self.assertEqual(self.upload(resources=json.dumps(invalid)).status_code,422,invalid)

    def test_structured_events_boundaries_partial_writes_and_legacy_logs(self):
        import json
        job=self.upload().json();directory=self.store.directory/job['id'];output=directory/'output';output.mkdir()
        path=output/'events.jsonl'
        path.write_text(''.join(json.dumps(dict(stage='mlp',pass_=i,moves=4,elapsed_seconds=i))+ '\n' for i in range(450))+'{"partial":')
        data=self.client.get(f"/api/jobs/{job['id']}/log").json()
        self.assertEqual(len(data['events']),300);self.assertTrue(data['truncated'])
        self.assertEqual(data['events'][-1]['elapsed_seconds'],449)
        path.unlink()
        (directory/'execution.log').write_text('[     5.21s] gene_refinement round=2 after_ll=-1200.24 moves=3 view=0\n')
        event=self.client.get(f"/api/jobs/{job['id']}/log").json()['events'][0]
        self.assertEqual(event['round'],2);self.assertEqual(event['after_ll'],-1200.24)

    def setUp(self):
        root = server.ROOT / 'output/web-tests'
        root.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=root)
        self.app = server.create_app(True, Path(self.tmp.name))
        self.store = self.app.state.store
        # Only the scheduling boundary is mocked. These are API tests, not a
        # simulated production inference backend.
        self.submit_patch = patch.object(self.store.worker, 'submit')
        self.mock_submit = self.submit_patch.start()
        self.context = TestClient(self.app)
        self.client = self.context.__enter__()

    def tearDown(self):
        self.context.__exit__(None, None, None)
        self.submit_patch.stop()
        self.tmp.cleanup()

    def upload(self, **values):
        return self.client.post('/api/jobs', files={'file': ('input.phy', PHY)}, data=values)

    def test_host_endpoint_serves_cached_sample_without_cuda_import(self):
        with patch.object(self.app.state.monitor, 'snapshot', return_value={'gpu_status': 'unavailable', 'gpus': []}):
            self.assertEqual(self.client.get('/api/host').json(), {'gpu_status': 'unavailable', 'gpus': []})

    def test_runtime_log_tail_is_bounded_and_cancel_removes_outputs(self):
        job = self.upload().json()
        url = f"/api/jobs/{job['id']}/log"
        self.assertEqual(self.client.get(url).json()['text'], '')
        path = self.store.directory / job['id'] / 'execution.log'
        path.write_text(('运行中 <script>alert(1)</script>\n'*5000)+'final log line\n')
        data = self.client.get(url).json()
        self.assertTrue(data['truncated'])
        self.assertLessEqual(len(data['text'].encode()), 64*1024)
        self.assertNotIn('\ufffd', data['text'])
        self.assertTrue(data['text'].endswith('final log line\n'))
        self.store.cancel(job['id'])
        self.assertEqual(self.client.get(url).json()['text'], '')
        self.assertEqual({p.name for p in path.parent.iterdir()},{'job.json'})
        self.assertEqual(self.client.get('/api/jobs/'+'0'*32+'/log').status_code, 404)

    def test_pipeline_snapshot_and_task_validation(self):
        import json
        from pipeline_config import defaults
        config=defaults('gene');config['stages'][2]['advanced']['model']='JC'
        config['stages'][0]['enabled']=False
        job=self.upload(configuration=json.dumps(config)).json()
        self.assertEqual(job['model'],'JC')
        self.assertEqual(job['configuration'],config)
        directory=self.store.directory/job['id']
        self.assertEqual(json.loads((directory/'configuration.json').read_text()),config)
        self.assertIn('--configuration',server.command_for(job,directory))
        config['stages'][2]['id']='kernel'
        self.assertEqual(self.upload(configuration=json.dumps(config)).status_code,422)
        config=defaults('gene');config['apnj']['projections']=0
        self.assertEqual(self.upload(configuration=json.dumps(config)).status_code,422)

    def test_pause_resume_and_cancel_checkpoint_boundary(self):
        import json
        job=self.upload().json();id=job['id'];directory=self.store.directory/id
        self.store.jobs[id]['status']='running'
        response=self.client.post(f'/api/jobs/{id}/pause').json()
        self.assertEqual(response['status'],'pausing')
        self.assertEqual(self.client.get(f'/api/jobs/{id}/checkpoint').status_code,409)
        out=directory/'output';out.mkdir();(out/'checkpoint.nwk').write_bytes(NWK)
        (out/'checkpoint.json').write_text(json.dumps(dict(available=True,stage='apnj')))
        (out/'pause.state.json').write_text(json.dumps(dict(paused=True,request_token=(directory/'pause.request').read_text())))
        self.assertEqual(self.client.get(f'/api/jobs/{id}').json()['status'],'paused')
        self.assertEqual(self.client.get(f'/api/jobs/{id}/checkpoint').content,NWK)
        self.assertEqual(self.client.post(f'/api/jobs/{id}/resume').json()['status'],'running')
        self.assertFalse((directory/'pause.request').exists())
        # A stale acknowledgement must never acknowledge a later pause request.
        self.assertEqual(self.client.post(f'/api/jobs/{id}/pause').json()['status'],'pausing')
        self.client.post(f'/api/jobs/{id}/cancel')
        self.assertEqual({p.name for p in directory.iterdir()},{'job.json'})
        self.assertEqual(self.client.get(f'/api/jobs/{id}/checkpoint').status_code,409)

    def test_static_boundary(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn("frame-ancestors 'none'", response.headers['content-security-policy'])
        self.assertEqual(self.client.get('/server.py').status_code, 404)
        self.assertEqual(self.client.get('/assets/forest.webp').headers['content-type'], 'image/webp')
        self.assertEqual(self.client.get('/assets/grain.svg').status_code, 200)
        self.assertEqual(self.client.get('/assets/SOURCES.json').status_code, 404)
        self.assertEqual(self.client.get('/assets/forest-source.jpg').status_code, 404)
        self.assertEqual(self.client.get('/runs/a/job.json').status_code, 404)

    def test_submit_and_cancel(self):
        response = self.upload(task='gene', model='GTR+G4')
        self.assertEqual(response.status_code, 202, response.text)
        job = response.json()
        self.mock_submit.assert_called_once()
        self.assertEqual(self.client.get('/api/jobs').json()[0]['id'], job['id'])
        result = self.client.get(f"/api/jobs/{job['id']}/tree")
        self.assertEqual(result.status_code, 409)
        cancelled = self.client.post(f"/api/jobs/{job['id']}/cancel").json()
        self.assertEqual(cancelled['status'], 'cancelled')

    def test_local_report_and_resource_boundary(self):
        response = self.client.get('/report/')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, (server.RELEASE/'technical-report/index.html').read_bytes())
        self.assertIn('text/html', response.headers['content-type'])
        for asset in ['site.css', 'benchmark/species-topology-simple.svg', 'benchmark/gene-topology.svg',
                      'benchmark/gene-runtime.svg', 'benchmark/support-roc.svg', 'concordtree-mark.svg']:
            with self.subTest(asset=asset):
                self.assertEqual(self.client.get('/report/assets/'+asset+'?snapshot=1').status_code, 200)
        self.assertIn('href="/report/"', self.client.get('/').text)
        for path in ['/report/run.py', '/report/%2e%2e/run.py', '/report/assets/%2e%2e/%2e%2e/engine.py']:
            self.assertEqual(self.client.get(path).status_code, 404, path)

    def test_report_pages_and_excluded_datasets(self):
        for name in ['index.html', 'accuracy.html', 'performance.html', 'method.html']:
            response = self.client.get('/report/'+name)
            self.assertEqual(response.status_code, 200, name)
            text = response.text
            for excluded in ['RNASim', 'DIPPER 论文', 'DIPPER-paper', '0.14965', '3:08:36.52']:
                self.assertNotIn(excluded, text, name)
            if name.endswith('.html'):
                for target in ['index.html', 'accuracy.html', 'performance.html', 'method.html']:
                    self.assertIn('href="'+target+'"', text)
        self.assertEqual(self.client.get('/report/assets/method-overview.svg').status_code, 404)
        gene = self.client.get('/report/examples/gene.phy')
        species = self.client.get('/report/examples/species.paml')
        self.assertEqual(server.normalize_alignment(gene.content, 'gene')[1]['loci'], 1)
        self.assertGreater(server.normalize_alignment(species.content, 'species')[1]['loci'], 1)

    def test_server_import_is_opt_in(self):
        response = self.client.post('/api/jobs/from-server', data={'path': str(server.HERE/'examples/gene.phy')})
        self.assertEqual(response.status_code, 403)

    def test_server_import_reads_without_changing_source(self):
        from urllib.parse import quote
        source = Path(self.tmp.name)/'alignment with spaces.phy'
        source.write_bytes(PHY)
        app = server.create_app(True, Path(self.tmp.name)/'debug-jobs', allow_server_files=True)
        with patch.object(app.state.store.worker, 'submit'), TestClient(app) as client:
            self.assertTrue(client.get('/api/config').json()['server_files_enabled'])
            uri = 'vscode-remote://ssh-remote+server'+quote(str(source))
            response = client.post('/api/jobs/from-server', data={'path': uri, 'model':'HKY', 'support':'false'})
            self.assertEqual(response.status_code, 202, response.text)
            job = response.json()
            self.assertEqual((job['model'],job['support'],job['metadata']['taxa']), ('HKY',False,4))
            self.assertEqual((app.state.store.directory/job['id']/'alignment.phy').read_bytes(), PHY)
            self.assertEqual(source.read_bytes(), PHY)
            for value, status in [('/etc/passwd',403),('https://example.org/file.phy',422),('relative.phy',422),
                                  ('file:///tmp/bad%00.phy',422),('file://[bad/path.phy',422),
                                  (str(source.parent),422),(str(source.parent/'missing.phy'),404)]:
                self.assertEqual(client.post('/api/jobs/from-server',data={'path':value}).status_code,status,value)
            link=source.parent/'outside.phy';link.symlink_to('/etc/passwd')
            self.assertEqual(client.post('/api/jobs/from-server',data={'path':str(link)}).status_code,403)

    def test_queue_counts_and_positions_follow_job_lifecycle(self):
        jobs = [self.upload().json() for _ in range(3)]
        queue = self.client.get('/api/queue').json()
        self.assertEqual(set(queue), {'running', 'queued', 'capacity', 'compute_enabled', 'accepting_jobs'})
        self.assertEqual((queue['running'], queue['queued']), (0, 3))
        self.assertTrue(queue['accepting_jobs'])
        self.assertEqual([j['queue_position'] for j in jobs], [1, 2, 3])
        # Simulate a worker transition without launching a scientific subprocess.
        with self.store.lock:
            self.store.jobs[jobs[0]['id']]['status'] = 'running'
        queue = self.client.get('/api/queue').json()
        self.assertEqual((queue['running'], queue['queued']), (1, 2))
        self.assertEqual(self.client.get(f"/api/jobs/{jobs[2]['id']}").json()['queue_position'], 2)
        self.client.post(f"/api/jobs/{jobs[1]['id']}/cancel")
        self.assertEqual(self.client.get(f"/api/jobs/{jobs[2]['id']}").json()['queue_position'], 1)
        with self.store.lock:
            self.store.jobs[jobs[0]['id']]['status'] = 'succeeded'
        queue = self.client.get('/api/queue').json()
        self.assertEqual((queue['running'], queue['queued']), (0, 1))

    def test_full_queue_and_reopening_after_cancel(self):
        jobs = [self.upload().json() for _ in range(server.MAX_ACTIVE_JOBS)]
        self.assertFalse(self.client.get('/api/queue').json()['accepting_jobs'])
        self.assertEqual(self.upload().status_code, 429)
        self.client.post(f"/api/jobs/{jobs[0]['id']}/cancel")
        self.assertTrue(self.client.get('/api/queue').json()['accepting_jobs'])
        self.assertEqual(self.upload().status_code, 202)

    def test_bad_input_and_options(self):
        self.assertEqual(self.upload(task='network').status_code, 422)
        self.assertEqual(self.upload(task='summary').status_code, 422)
        self.assertEqual(self.upload(model='; touch /tmp/no').status_code, 422)
        self.assertEqual(self.client.post('/api/jobs', files={'file': ('bad.phy', b'bad')}).status_code, 422)
        self.assertFalse(self.mock_submit.called)

    def test_cross_origin(self):
        response = self.client.post('/api/jobs', headers={'Origin': 'https://unrelated.example'}, files={'file': ('input.phy', PHY)})
        self.assertEqual(response.status_code, 403)

    def test_upload_config_and_large_content_length(self):
        self.assertIsNone(self.client.get('/api/config').json()['max_upload_bytes'])
        response = self.client.post('/api/jobs', files={'file': ('input.phy', PHY)},
                                    headers={'Content-Length': str(513 * 1024 * 1024)})
        self.assertEqual(response.status_code, 202, response.text)

    def test_spooled_upload_preserves_alignment(self):
        # Exceed the multipart in-memory spool threshold, preserving DNA exactly.
        response = self.client.post('/api/jobs', files={'file': ('input.phy', PHY + b' ' * (2 * 1024 * 1024))})
        self.assertEqual(response.status_code, 202, response.text)
        matrix = self.store.directory / response.json()['id'] / 'alignment.phy'
        self.assertEqual(matrix.read_bytes(), PHY)

    def test_result_and_path_boundary(self):
        job = self.upload().json()
        directory = self.store.directory / job['id']
        result = directory / 'output/tree.nwk'
        result.parent.mkdir(); result.write_bytes(NWK)
        self.store.jobs[job['id']].update(status='succeeded', result='output/tree.nwk')
        self.assertEqual(self.client.get(f"/api/jobs/{job['id']}/tree").json()['newick'], NWK.decode())
        self.assertEqual(self.client.get(f"/api/jobs/{job['id']}/download").content, NWK)
        self.store.jobs[job['id']]['result'] = '../outside.nwk'
        self.assertEqual(self.client.get(f"/api/jobs/{job['id']}/download").status_code, 404)

    def test_real_command_mapping_without_execution(self):
        job=self.upload(support='false',model='HKY+G4').json()
        directory=self.store.directory/job['id'];command=server.command_for(job,directory)
        self.assertIn(str(server.RELEASE/'scripts/run_local.sh'),command)
        self.assertIn('--no-support',command)
        self.assertEqual(command[command.index('--model')+1],'HKY+G4')
        self.assertEqual(command[command.index('--cpu-threads')+1],'16')
        self.assertNotIn('--loci',command)
        for count in [1,2]:
            response=self.client.post('/api/jobs',files={'file':('loci.txt',SPECIES*count)},data={'task':'species'})
            self.assertEqual(response.status_code,202,response.text)
            species=response.json();directory=self.store.directory/species['id']
            command=server.command_for(species,directory)
            self.assertNotIn('--model',command);self.assertNotIn('--no-support',command)
            self.assertEqual('--loci' in command,count>1)
            self.assertEqual(species['metadata']['loci'],count)
            self.assertEqual((directory/'loci.jsonl').is_file(),count>1)
            if count>1:
                self.assertEqual((directory/'loci.jsonl').read_text(),'{"alignment_length": 8}\n'*count)

    def test_task_specific_rejections_and_support_boolean(self):
        self.assertEqual(self.client.post('/api/jobs',files={'file':('a.fa',b'>a\nACGT')}).status_code,422)
        self.assertEqual(self.client.post('/api/jobs',files={'file':('a.phy',PHY*2)}).status_code,422)
        self.assertEqual(self.upload(task='species').status_code,422)
        self.assertEqual(self.upload(support='maybe').status_code,422)
        self.assertFalse(self.mock_submit.called)

    def test_worker_reads_v101_result(self):
        import json
        from unittest.mock import MagicMock
        job=self.upload().json();directory=self.store.directory/job['id']
        output=directory/'output';output.mkdir()
        (output/'prediction.nwk').write_bytes(NWK)
        (output/'RESULT.json').write_text(json.dumps(dict(status='complete',branch_support=dict(enabled=True,label='Local likelihood weight'),file_to_tree_seconds=1.2)))
        process=MagicMock();process.wait.return_value=0
        with patch.object(server.subprocess,'Popen',return_value=process): self.store.run(job['id'])
        completed=self.store.get(job['id'])
        self.assertEqual(completed['status'],'succeeded')
        self.assertEqual(completed['branch_support']['label'],'Local likelihood weight')
        self.assertEqual(self.client.get(f"/api/jobs/{job['id']}/download").content,NWK)

    def test_worker_failure_cannot_offer_result(self):
        from unittest.mock import MagicMock
        job=self.upload().json();process=MagicMock();process.wait.return_value=1
        with patch.object(server.subprocess,'Popen',return_value=process): self.store.run(job['id'])
        self.assertEqual(self.store.get(job['id'])['status'],'failed')
        self.assertEqual(self.client.get(f"/api/jobs/{job['id']}/download").status_code,409)

    def test_worker_preserves_search_notice_with_downloadable_result(self):
        import json
        from unittest.mock import MagicMock
        job=self.upload().json();directory=self.store.directory/job['id']
        output=directory/'output';output.mkdir();(output/'prediction.nwk').write_bytes(NWK)
        quality=dict(search_converged=False,guard_interventions=[dict(stage='posterior',reason='repeated_topology_numerical_plateau')])
        (output/'RESULT.json').write_text(json.dumps(dict(status='complete',quality=quality)))
        process=MagicMock();process.wait.return_value=0
        with patch.object(server.subprocess,'Popen',return_value=process):self.store.run(job['id'])
        completed=self.store.get(job['id'])
        self.assertEqual(completed['quality'],quality)
        self.assertEqual(self.client.get(f"/api/jobs/{job['id']}/download").content,NWK)



class PreviewTests(unittest.TestCase):
    def test_default_preview_cannot_compute(self):
        root = server.ROOT / 'output/web-tests'
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as directory:
            with TestClient(server.create_app(False, Path(directory))) as client:
                self.assertFalse(client.get('/api/config').json()['compute_enabled'])
                queue = client.get('/api/queue').json()
                self.assertEqual((queue['running'], queue['queued']), (0, 0))
                self.assertFalse(queue['compute_enabled'])
                self.assertFalse(queue['accepting_jobs'])
                self.assertEqual(client.post('/api/jobs', files={'file': ('input.phy', PHY)}).status_code, 503)


if __name__ == '__main__':
    unittest.main()
