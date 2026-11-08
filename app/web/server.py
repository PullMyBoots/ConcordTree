"""Local desktop browser UI for the downloaded ConcordTree application.

Scientific inference runs in this release's local environment as a subprocess.
The web environment never imports CUDA or any historical scientific runtime.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import signal
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import uuid
from urllib.parse import unquote, urlsplit

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from starlette.middleware.trustedhost import TrustedHostMiddleware

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
RELEASE = HERE.parent
sys.path.insert(0,str(RELEASE))
from pipeline_config import normalize as normalize_pipeline, schema as pipeline_schema
ROOT = next((p for p in RELEASE.parents if (p / '.ResearchProject').is_dir()), RELEASE)
MAX_ACTIVE_JOBS = 16
ACTIVE = {'queued', 'running', 'pausing', 'paused'}
MODELS = ('GTR+FO+I+G4', 'GTR+I+G4', 'GTR+G4', 'HKY+G4',
          'GTR', 'SYM', 'TN93', 'HKY', 'K80', 'F81', 'JC')
RELEASE_VERSION = '1.0.2'
SCIENTIFIC_PYTHON = RELEASE / '.conda/env/bin/python'
DEFAULT_RUNS = RELEASE / 'runs/web'

if __package__:
    from .alignments import normalize_alignment
    from .host_metrics import HostMonitor
    from .compute_settings import normalize as normalize_compute, schema as compute_schema
else:
    from alignments import normalize_alignment
    from host_metrics import HostMonitor
    from compute_settings import normalize as normalize_compute, schema as compute_schema


def now():
    return datetime.now(timezone.utc).isoformat()


def read_server_alignment(resource, roots):
    if len(resource) > 8192 or '\x00' in resource:
        raise HTTPException(422, 'Invalid server file path.')
    try: uri = urlsplit(resource)
    except ValueError: raise HTTPException(422, 'Invalid server file URI.')
    if uri.scheme:
        if uri.scheme not in {'file', 'vscode-remote'} or (uri.scheme == 'file' and uri.netloc not in {'', 'localhost'}):
            raise HTTPException(422, 'Use a file from this server workspace.')
        value = unquote(uri.path)
    else:
        value = resource
    if '\x00' in value: raise HTTPException(422, 'Invalid server file path.')
    requested = Path(value)
    if not requested.is_absolute(): raise HTTPException(422, 'An absolute server file path is required.')
    try: path = requested.resolve(strict=True)
    except (OSError, RuntimeError): raise HTTPException(404, 'Server file not found.')
    base = next((root for root in roots if path.is_relative_to(root)), None)
    if base is None or any(part.startswith('.') for part in path.relative_to(base).parts):
        raise HTTPException(403, 'Choose an alignment within this project or its linked input datasets.')
    if path.suffix.lower() not in {'.phy', '.phylip', '.paml', '.txt'}:
        raise HTTPException(422, 'Choose a PHYLIP/PAML alignment file.')
    try:
        with os.fdopen(os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW), 'rb') as source:
            info = os.fstat(source.fileno())
            if not stat.S_ISREG(info.st_mode): raise HTTPException(422, 'Choose a regular alignment file.')
            raw = source.read()
    except OSError: raise HTTPException(422, 'The server file could not be read.')
    return raw, path.name


def command_for(job, directory):
    if job['task'] not in {'gene', 'species'}:
        raise ValueError('Invalid task options.')
    resources = normalize_compute(job.get('resources'))
    command = ['/usr/bin/bash',
               str(RELEASE / 'scripts/run_local.sh'), '--task', job['task'],
               '--input', str(directory / 'alignment.phy'), '--out', str(directory / 'output'),
               '--gpu', str(resources['gpu']), '--cpu-threads', str(resources['cpu_threads']), '--pause-request', str(directory/'pause.request')]
    if job['task'] == 'gene':
        if job['model'] not in MODELS: raise ValueError('Invalid evolutionary model.')
        command += ['--model', job['model']]
    elif job['metadata']['partitioned']:
        command += ['--loci', str(directory / 'loci.jsonl')]
    if job.get('configuration') is not None:
        command += ['--configuration',str(directory/'configuration.json')]
    if not job['support']: command.append('--no-support')
    return command


class JobStore:
    def __init__(self, directory: Path, enabled=False):
        # Reuse the release's path policy without importing the scientific runtime.
        import importlib.util
        spec = importlib.util.spec_from_file_location('web_path_policy', RELEASE / 'runtime_config.py')
        policy = importlib.util.module_from_spec(spec); spec.loader.exec_module(policy)
        self.directory = policy.owned_path(directory)
        self.input_roots = (ROOT.resolve(), *[p for p in policy.protected_targets() if p != ROOT / '.ResearchProject'])

        self.directory.mkdir(parents=True, exist_ok=True)
        self.enabled = enabled
        self.jobs, self.processes = {}, {}
        self.lock = threading.RLock()
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix='concordtree-web')
        for path in self.directory.glob('*/job.json'):
            try:
                job = json.loads(path.read_text())
                if not re.fullmatch('[0-9a-f]{32}', path.parent.name) or job['id'] != path.parent.name: continue
                if job['status'] in ACTIVE:
                    job.update(status='failed', error='The service restarted. Please resubmit.', finished_at=now())
                    path.write_text(json.dumps(job, ensure_ascii=False, indent=2))
                self.jobs[job['id']] = job
            except (ValueError, KeyError):
                continue

    def persist(self, job):
        directory = self.directory / job['id']
        tmp = directory / 'job.json.tmp'
        tmp.write_text(json.dumps(job, ensure_ascii=False, indent=2))
        tmp.replace(directory / 'job.json')

    def get(self, identifier):
        with self.lock:
            if not re.fullmatch('[0-9a-f]{32}', identifier) or identifier not in self.jobs:
                raise HTTPException(404, 'Task not found.')
            source_job=self.jobs[identifier]
            pause_state=self.directory/identifier/'output/pause.state.json'
            if source_job['status']=='pausing':
                try:
                    token=(self.directory/identifier/'pause.request').read_text()
                    if json.loads(pause_state.read_text()).get('request_token')==token:
                        source_job['status']='paused';self.persist(source_job)
                except (OSError,ValueError):pass
            job = dict(source_job)
            if job['status'] in {'running','pausing','paused'}:
                try:job['checkpoint']=json.loads((self.directory/identifier/'output/checkpoint.json').read_text())
                except (OSError,ValueError):pass
            progress = self.directory / identifier / 'output/progress.json'
            if job['status'] in {'running','pausing','paused'} and progress.is_file():
                try: job['progress'] = json.loads(progress.read_text())
                except (ValueError, OSError): pass
            if job['status'] == 'queued':
                waiting = [j['id'] for j in self.jobs.values() if j['status'] == 'queued']
                job['queue_position'] = waiting.index(identifier) + 1
            return job

    def queue_status(self):
        with self.lock:
            running = sum(j['status'] in {'running','pausing','paused'} for j in self.jobs.values())
            queued = sum(j['status'] == 'queued' for j in self.jobs.values())
            return dict(running=running, queued=queued, capacity=MAX_ACTIVE_JOBS,
                        compute_enabled=self.enabled,
                        accepting_jobs=self.enabled and running + queued < MAX_ACTIVE_JOBS)

    def add(self, raw, filename, task, model, support, configuration=None, resources=None):
        if not self.enabled: raise HTTPException(503, 'Inference is paused.')
        if task not in {'gene', 'species'} or (task == 'gene' and model not in MODELS):
            raise HTTPException(422, 'Invalid task options.')
        try:
            resources=normalize_compute(json.loads(resources) if isinstance(resources,str) else resources)
            explicit=configuration is not None
            configuration=normalize_pipeline(json.loads(configuration) if isinstance(configuration,str) else configuration,task)
            if not explicit and task=='gene':next(s for s in configuration['stages'] if s['id']=='likelihood')['advanced']['model']=model
            if task=='gene':model=next(s for s in configuration['stages'] if s['id']=='likelihood')['advanced']['model']
            data, metadata, lengths = normalize_alignment(raw, task)
        except ValueError as error:
            raise HTTPException(422, str(error))
        with self.lock:
            if sum(j['status'] in ACTIVE for j in self.jobs.values()) >= MAX_ACTIVE_JOBS:
                raise HTTPException(429, 'The queue is full. Please try again later.')
            identifier = uuid.uuid4().hex
            directory = self.directory / identifier
            directory.mkdir()
            (directory / 'alignment.phy').write_bytes(data)
            (directory/'configuration.json').write_text(json.dumps(configuration,indent=2)+'\n')
            if metadata['partitioned']:
                (directory / 'loci.jsonl').write_text(''.join(json.dumps(dict(alignment_length=n))+'\n' for n in lengths))
            job = dict(id=identifier, task=task, filename=Path(filename or 'input.phy').name[:180],
                       model=model if task == 'gene' else None, support=support, metadata=metadata,configuration=configuration,resources=resources,
                       version=RELEASE_VERSION, status='queued', created_at=now(), error=None)
            self.jobs[identifier] = job
            self.persist(job)
            self.worker.submit(self.run, identifier)
            return self.get(identifier)

    def pause(self, identifier):
        with self.lock:
            job=self.jobs[self.get(identifier)['id']]
            if job['status']=='running':
                (self.directory/identifier/'pause.request').write_text(uuid.uuid4().hex)
                job['status']='pausing';self.persist(job)
            return self.get(identifier)

    def resume(self, identifier):
        with self.lock:
            job=self.jobs[self.get(identifier)['id']]
            if job['status'] in {'paused','pausing'}:
                (self.directory/identifier/'pause.request').unlink(missing_ok=True)
                job['status']='running';self.persist(job)
            return self.get(identifier)

    def clean_cancelled(self, identifier):
        directory=self.directory/identifier
        for path in directory.iterdir():
            if path.name in {'job.json','job.json.tmp'}:continue
            if path.is_dir():shutil.rmtree(path)
            else:path.unlink(missing_ok=True)

    def cancel(self, identifier):
        with self.lock:
            job = self.jobs[self.get(identifier)['id']]
            if job['status'] in ACTIVE:
                job.update(status='cancelled', finished_at=now())
                process = self.processes.get(identifier)
                if process:
                    try: os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError: pass
                if not process:self.clean_cancelled(identifier)
                job.pop('checkpoint',None);job.pop('result',None)
                self.persist(job)
            return dict(job)

    def run(self, identifier):
        directory = self.directory / identifier
        with self.lock:
            job = self.jobs[identifier]
            if job['status'] != 'queued': return
            job.update(status='running', started_at=now())
            self.persist(job)
        try:
            with (directory / 'execution.log').open('w') as log:
                with self.lock:
                    if job['status'] == 'cancelled': return
                    environment = os.environ.copy()
                    environment.pop('PYTHONPATH', None)
                    environment['PYTHONDONTWRITEBYTECODE'] = '1'
                    environment['PYTHONUNBUFFERED'] = '1'
                    environment['CONCORDTREE_LIVE_LOG'] = '1'
                    process = subprocess.Popen(command_for(job, directory), cwd=ROOT, stdout=log,
                                               stderr=subprocess.STDOUT, start_new_session=True, env=environment)
                    self.processes[identifier] = process
                code = process.wait()
                if code != 0:
                    detail = None
                    try: detail = json.loads((directory / 'output/RESULT.json').read_text()).get('error')
                    except (ValueError, OSError): pass
                    raise RuntimeError(detail or 'Inference failed. Check your PHYLIP alignment or try again later.')
            output = directory / 'output'
            result = output / 'prediction.nwk'
            if not result.is_file(): raise RuntimeError('No tree file was produced.')
            details = json.loads((output / 'RESULT.json').read_text())
            if details.get('status') != 'complete': raise RuntimeError('Inference did not complete.')
            with self.lock:
                if job['status'] != 'cancelled':
                    job.update(status='succeeded', result=str(result.relative_to(directory)), finished_at=now(),
                               quality=details.get('quality', {}),
                               branch_support=details.get('branch_support', {'enabled': False}),
                               elapsed_seconds=details.get('file_to_tree_seconds'),
                               input_mode=details.get('details', {}).get('posterior', {}).get('input_mode'))
        except Exception as error:
            with self.lock:
                if job['status'] != 'cancelled': job.update(status='failed', error=str(error), finished_at=now())
        finally:
            with self.lock:
                self.processes.pop(identifier, None)
                if job['status']=='cancelled':self.clean_cancelled(identifier)
                self.persist(job)

    def result(self, identifier):
        job = self.get(identifier)
        if job['status'] != 'succeeded': raise HTTPException(409, 'The result is not ready.')
        directory = self.directory / identifier
        path = (directory / job['result']).resolve()
        if not path.is_relative_to(directory.resolve()) or not path.is_file():
            raise HTTPException(404, 'Result file not found.')
        return path

    def close(self):
        for identifier in list(self.jobs): self.cancel(identifier)
        with self.lock:
            for process in self.processes.values():
                try: os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError: pass
        self.worker.shutdown(wait=True, cancel_futures=True)


class LocalBoundary:
    def __init__(self, app): self.app = app

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http': return await self.app(scope, receive, send)
        headers = dict(scope['headers'])
        if scope['method'] == 'POST':
            origin = headers.get(b'origin', b'').decode()
            host = headers.get(b'host', b'').decode()
            if origin and origin not in {f'http://{host}', f'https://{host}'} or headers.get(b'sec-fetch-site') == b'cross-site':
                return await JSONResponse({'detail': 'Only same-origin requests are accepted.'}, status_code=403)(scope, receive, send)
        async def secure_send(message):
            if message['type'] == 'http.response.start':
                message['headers'] += [(b'x-content-type-options', b'nosniff'), (b'cache-control', b'no-store'),
                    (b'content-security-policy', b"default-src 'self';script-src 'self';style-src 'self';img-src 'self' blob:;connect-src 'self';object-src 'none';base-uri 'none';frame-ancestors 'none'"),
                    (b'referrer-policy', b'no-referrer')]
            await send(message)

        await self.app(scope, receive, secure_send)


def create_app(enabled=False, directory=None, *, allow_server_files=False):
    temporary = RELEASE / '.runtime/cache/web-upload'
    temporary.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(temporary)
    store = JobStore(directory or DEFAULT_RUNS, enabled)
    monitor = HostMonitor()

    @asynccontextmanager
    async def lifespan(app):
        monitor.start()
        try: yield
        finally:
            monitor.close()
            store.close()

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    app.state.store = store
    app.state.monitor = monitor
    app.add_middleware(LocalBoundary)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=['127.0.0.1', 'localhost', '[::1]', 'testserver'])

    @app.get('/api/config')
    def config(): return dict(compute_enabled=enabled, max_upload_bytes=None, version=RELEASE_VERSION, gene_models=MODELS,
                              server_files_enabled=allow_server_files and enabled,pipeline_settings=pipeline_schema(),compute_settings=compute_schema())

    @app.get('/api/queue')
    def queue(): return store.queue_status()

    @app.get('/api/host')
    def host(): return monitor.snapshot()

    @app.get('/api/jobs')
    def jobs():
        with store.lock: return sorted([dict(j) for j in store.jobs.values()], key=lambda j: j['created_at'], reverse=True)

    @app.post('/api/jobs', status_code=202)
    async def submit(file: UploadFile = File(...), task: str = Form('gene'), model: str = Form('GTR+FO+I+G4'),
                     support: bool = Form(True), configuration: str = Form(None), resources: str = Form(None)):
        try:
            if not enabled: raise HTTPException(503, 'Inference is not available yet.')
            raw = await file.read()
            return await run_in_threadpool(store.add, raw, file.filename, task, model, support,configuration,resources)
        finally:
            await file.close()

    @app.post('/api/jobs/from-server', status_code=202)
    def submit_server_file(path: str = Form(...), task: str = Form('gene'),
                           model: str = Form('GTR+FO+I+G4'), support: bool = Form(True), configuration: str = Form(None), resources: str = Form(None)):
        if not enabled or not allow_server_files: raise HTTPException(403, 'Server file import is disabled. Start with --allow-server-files for local debugging.')
        raw, filename = read_server_alignment(path, store.input_roots)
        return store.add(raw, filename, task, model, support,configuration,resources)

    @app.get('/api/jobs/{identifier}')
    def job(identifier: str): return store.get(identifier)

    @app.get('/api/jobs/{identifier}/log')
    def log(identifier: str):
        store.get(identifier)
        path = store.directory / identifier / 'execution.log'
        try:
            with path.open('rb') as source:
                size = os.fstat(source.fileno()).st_size
                offset = max(0, size - 64 * 1024)
                source.seek(offset)
                data = source.read(64 * 1024)
            if offset:
                # Discard a partial first line (and any split UTF-8 character).
                first = data.find(b'\n')
                data = data[first+1:] if first >= 0 else data
            text = data.decode('utf-8', errors='replace')
        except FileNotFoundError:
            size, offset, text = 0, 0, ''
        events=[]
        journal=store.directory / identifier / 'output/events.jsonl'
        try:
            with journal.open('rb') as stream:
                journal_size=os.fstat(stream.fileno()).st_size
                stream.seek(max(0,journal_size-256*1024))
                lines=stream.read(256*1024).splitlines()
            for line in lines:
                try:events.append(json.loads(line))
                except (ValueError,UnicodeDecodeError):pass
        except FileNotFoundError:
            # Older tasks predate the JSON event journal.
            for line in text.splitlines():
                match=re.match(r'\[\s*([\d.]+)s\]\s+(\w+)\s*(.*)',line)
                if not match:continue
                event=dict(stage=match[2],elapsed_seconds=float(match[1]))
                for key,value in re.findall(r'(\w+)=([^\s]+)',match[3]):
                    try:event[key]=json.loads(value)
                    except ValueError:event[key]=value
                events.append(event)
        return dict(text=text, bytes=size, truncated=offset > 0 or len(events)>300, events=events[-300:])

    @app.post('/api/jobs/{identifier}/pause')
    def pause(identifier: str): return store.pause(identifier)

    @app.post('/api/jobs/{identifier}/resume')
    def resume(identifier: str): return store.resume(identifier)

    @app.get('/api/jobs/{identifier}/checkpoint')
    def checkpoint(identifier: str):
        job=store.get(identifier)
        if job['status']!='paused' or not job.get('checkpoint',{}).get('available'):
            raise HTTPException(409,'Pause the task at a complete tree before downloading.')
        path=store.directory/identifier/'output/checkpoint.nwk'
        if not path.is_file():raise HTTPException(404,'No complete intermediate tree is available.')
        return FileResponse(path,media_type='text/plain',filename='intermediate-tree.nwk')

    @app.post('/api/jobs/{identifier}/cancel')
    def cancel(identifier: str): return store.cancel(identifier)

    @app.get('/api/jobs/{identifier}/tree')
    def tree(identifier: str): return dict(newick=store.result(identifier).read_text(), branch_support=store.get(identifier).get('branch_support', {'enabled': False}))

    @app.get('/api/jobs/{identifier}/download')
    def download(identifier: str): return FileResponse(store.result(identifier), media_type='text/plain', filename='tree.nwk')

    @app.get('/')
    def index(): return FileResponse(HERE / 'index.html')

    @app.get('/examples/{name}')
    def example(name: str):
        if name not in {'gene.phy', 'species.paml'}: raise HTTPException(404)
        return FileResponse(HERE / 'examples' / name, media_type='text/plain', filename=name)

    @app.get('/assets/{asset}')
    def background_asset(asset: str):
        allowed = {f'{scene}{size}.webp' for scene in ('forest', 'reef', 'fern', 'hummingbird', 'chameleon', 'sundew', 'mushroom', 'tyrannosaurus', 'ammonite', 'seed-fern', 'cells', 'virus', 'diatom', 'fungal-culture', 'tiger') for size in ('', '-mobile')} | {'grain.svg'}
        if asset not in allowed: raise HTTPException(404)
        return FileResponse(HERE / 'assets' / asset)

    app.mount('/report', StaticFiles(directory=RELEASE / 'technical-report', html=True), name='technical-report')

    @app.get('/{asset}')
    def asset(asset: str):
        if asset not in {'app.js', 'dashboard.js', 'settings.js', 'settings.css', 'geometry.js', 'seasons.js', 'seasons.css', 'styles.css', 'tokens.css', 'mark.svg', 'backgrounds.json', 'credits.html'}: raise HTTPException(404)
        return FileResponse(HERE / asset)

    return app


if __name__ == '__main__':
    import uvicorn
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--host', choices=['127.0.0.1', '0.0.0.0'], default='127.0.0.1')
    parser.add_argument('--preview', action='store_true', help='serve the UI without accepting inference jobs')
    parser.add_argument('--enable-compute', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--runs', type=Path, default=DEFAULT_RUNS)
    parser.add_argument('--allow-server-files', action='store_true', help='temporarily accept workspace paths from VS Code drag-and-drop')
    args = parser.parse_args()
    uvicorn.run(create_app(not args.preview, args.runs, allow_server_files=args.allow_server_files), host=args.host, port=args.port, log_level='warning')
