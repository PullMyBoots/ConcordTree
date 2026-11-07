"""Exercise real HTTP uploads, GPU inference, results and input errors."""
from pathlib import Path
import argparse
import hashlib
import json
import time
import urllib.error
import urllib.request
import uuid

p = argparse.ArgumentParser()
p.add_argument('--url', required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
root = Path(__file__).resolve().parents[1]
a.output.mkdir(parents=True, exist_ok=True)

def request(path, body=None, headers=None):
    return urllib.request.urlopen(urllib.request.Request(a.url + path, data=body, headers=headers or {}), timeout=30)

def upload(filename, task, raw):
    boundary = 'ct-' + uuid.uuid4().hex
    body = b''
    for key, value in [('task', task), ('support', 'true')]:
        body += f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode()
    body += f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{filename}"\r\nContent-Type: text/plain\r\n\r\n'.encode() + raw
    body += f'\r\n--{boundary}--\r\n'.encode()
    with request('/api/jobs', body, {'Content-Type': 'multipart/form-data; boundary=' + boundary}) as response:
        assert response.status == 202
        return json.load(response)

report = dict(url=a.url, config=json.load(request('/api/config')), tasks=[])
try:
    upload('invalid.phy', 'gene', b'invalid input\n')
    raise AssertionError('Invalid input accepted')
except urllib.error.HTTPError as exc:
    assert exc.code == 422, exc.code
    report['invalid_input_http_status'] = exc.code
for task, filename in [('gene', 'gene.phy'), ('species', 'species.paml')]:
    start = time.perf_counter()
    job = upload(filename, task, (root / 'app/web/examples' / filename).read_bytes())
    identifier = job['id']
    deadline = time.monotonic() + 600
    while job['status'] not in ('succeeded', 'failed', 'cancelled') and time.monotonic() < deadline:
        time.sleep(.5)
        job = json.load(request('/api/jobs/' + identifier))
    assert job['status'] == 'succeeded', job
    downloaded = request('/api/jobs/' + identifier + '/download').read()
    tree = json.load(request('/api/jobs/' + identifier + '/tree'))
    assert downloaded.decode() == tree['newick']
    assert downloaded.strip().endswith(b';')
    (a.output / (task + '.nwk')).write_bytes(downloaded)
    report['tasks'].append(dict(task=task, job=job, web_wall_seconds=time.perf_counter()-start,
        tree_sha256=hashlib.sha256(downloaded).hexdigest()))
    (a.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
report['status'] = 'ok'
(a.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
