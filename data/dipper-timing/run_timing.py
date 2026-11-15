from pathlib import Path
import argparse,json,hashlib,os,subprocess,time,traceback,datetime,contextlib
root=Path(__file__).resolve().parent
project=root.parents[1]
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(8<<20),b''):h.update(block)
 return h.hexdigest()
def write(path,data):
 temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');temp.replace(path)
@contextlib.contextmanager
def open_alignment(source):
 if Path(source).suffix=='.zst':
  with subprocess.Popen(['zstd','-dc','--',str(source)],stdout=subprocess.PIPE) as process:
   yield process.stdout
   assert process.wait()==0,'zstd decompression failed'
 else:
  with Path(source).open('rb') as stream:yield stream
def convert(source,target,n,m):
 with open_alignment(source) as inp,target.open('wb') as out:
  header=inp.readline().split();assert [int(x) for x in header[:2]]==[n,m]
  seen=0;names=set()
  for line in inp:
   if not line.strip():continue
   name,sequence=line.split(maxsplit=1);sequence=b''.join(sequence.split())
   assert len(sequence)==m and name not in names
   names.add(name);out.write(b'>'+name+b'\n'+sequence+b'\n');seen+=1
  assert seen==n
 return names
p=argparse.ArgumentParser();p.add_argument('--task',choices=['gene','species'],required=True);p.add_argument('--gpu',type=int,required=True);a=p.parse_args()
report=project/'subprojects-main/concordtree-linux-distribution/app/technical-report/data'
inputs={r['id']:r for r in json.loads((report/'INPUTS.json').read_text())}
old=[r for r in json.loads((report/'BASELINES.json').read_text()) if r['method']=='DIPPER GPU' and r['task']==a.task]
old.sort(key=lambda r:(r['taxa'],r['family'],r['id']))
assert len(old)==(200 if a.task=='gene' else 700)
cpus=list(range(0,16)) if a.gpu==0 else list(range(128,144));os.sched_setaffinity(0,cpus)
freeze=json.loads((root/'HARNESS.json').read_text());assert sha(root/'dipper_timed')==freeze['harness_sha256']
journal=root/(a.task+'-RUNS.json');previous={r['id']:r for r in json.loads(journal.read_text())} if journal.exists() else {};rows=[]
start=time.perf_counter()
for index,prior in enumerate(old,1):
 case=inputs[prior['id']]
 if case['id'] in previous:
  row=previous[case['id']];assert row['source_sha256']==case['source_sha256'];rows.append(row);continue
 target=root/'records'/case['id'];target.mkdir(parents=True,exist_ok=False)
 row={'id':case['id'],'task':case['task'],'family':case['family'],'taxa':case['taxa'],'sites':case['sites'],'source':case['source'],'source_sha256':case['source_sha256'],'original_seconds':prior['seconds'],'historical_accuracy_source':prior['source'],'historical_nrf':prior.get('archived_nrf'),'gpu':a.gpu,'cpu_affinity':cpus,'cpu_threads':16,'status':'failed','started_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 fasta=target/'alignment.fasta'
 try:
  assert sha(case['source'])==case['source_sha256'],case['id']
  before=time.perf_counter();names=convert(case['source'],fasta,case['taxa'],case['sites']);conversion=time.perf_counter()-before
  seed=(20261108+int(hashlib.sha256(case['id'].encode()).hexdigest()[:8],16))%(2**32)
  command=[str(root/'dipper_timed'),'-i','m','-o','t','-m','0' if a.task=='gene' else '1','-d','2','--force-binary','-I',str(fasta),'-O',str(target/'tree.raw.nwk')]
  if a.task=='species':command+=['-K','10']
  env={**os.environ,'CUDA_VISIBLE_DEVICES':str(a.gpu),'CUDA_MODULE_LOADING':'EAGER','OMP_NUM_THREADS':'16','OPENBLAS_NUM_THREADS':'16','MKL_NUM_THREADS':'16','DIPPER_TIMING_WARM':'1','DIPPER_TIMING_JSON':str(target/'native-timing.json'),'LD_PRELOAD':str(root/'seed_control.so'),'DIPPER_SHUFFLE_SEED':str(seed)}
  write(target/'COMMAND.json',{'command':command,'environment':{k:env[k] for k in ['CUDA_VISIBLE_DEVICES','CUDA_MODULE_LOADING','OMP_NUM_THREADS','DIPPER_TIMING_WARM','DIPPER_SHUFFLE_SEED','LD_PRELOAD']},'cpu_affinity':cpus})
  with (target/'stdout.log').open('wb') as stdout,(target/'stderr.log').open('wb') as stderr:
   before=time.perf_counter();run=subprocess.run(command,cwd=target,env=env,stdout=stdout,stderr=stderr);wall=time.perf_counter()-before
  assert run.returncode==0,run.returncode
  timing=json.loads((target/'native-timing.json').read_text());assert timing['warm'] and timing['cuda_sync_code']==0 and timing['native_file_to_tree_seconds']>0
  tree=target/'tree.raw.nwk';assert tree.is_file() and tree.read_text().strip().endswith(';')
  row.update(status='complete',seconds=conversion+timing['native_file_to_tree_seconds'],conversion_seconds=conversion,native_file_to_tree_seconds=timing['native_file_to_tree_seconds'],excluded_initialization_seconds=timing['initialization_seconds'],process_wall_seconds=wall,tree=str(tree),tree_sha256=sha(tree),seed=seed,fasta_sha256=sha(fasta),input_leaf_count=len(names))
 except Exception as exc:
  row.update(error=repr(exc),traceback=traceback.format_exc());print(row['traceback'],flush=True)
 finally:
  if fasta.exists():fasta.unlink()
 write(target/'MEASUREMENT.json',row);rows.append(row);write(journal,rows)
 print(json.dumps({'index':index,'total':len(old),'id':case['id'],'status':row['status'],'seconds':row.get('seconds')}),flush=True)
 if row['status']!='complete':raise RuntimeError('Stopped at failed measurement; earlier results preserved')
write(root/(a.task+'-TERMINAL.json'),{'planned':len(old),'complete':len(rows),'wall_seconds':time.perf_counter()-start,'gpu':a.gpu,'cpu_affinity':cpus})
