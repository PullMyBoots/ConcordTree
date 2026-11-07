from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
context = ROOT / 'container/context'
context.mkdir(parents=True, exist_ok=True)
for source, destination in [(ROOT / 'app/.conda/env', context / 'science'),
                             (ROOT / 'app/.web-python', context / 'web-python')]:
    if not destination.exists():
        subprocess.run(['cp', '-al', str(source), str(destination)], check=True)
code = context / 'app'
code.mkdir(exist_ok=True)
subprocess.run(['rsync', '-a', '--delete', '--exclude=.conda', '--exclude=.web-python',
    '--exclude=.runtime', '--exclude=runs', '--exclude=__pycache__',
    str(ROOT / 'app') + '/', str(code) + '/'], check=True)
(context / 'Dockerfile').write_text('''FROM nvidia/cuda:12.1.1-base-ubuntu22.04@sha256:457a4076c56025f51217bff647ca631c7880ad3dbf546b03728ba98297ebbc22
RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev && rm -rf /var/lib/apt/lists/*
COPY science /opt/concordtree/.conda/env
COPY web-python /opt/concordtree/.web-python
COPY app /opt/concordtree
ENV CONCORDTREE_BIND=0.0.0.0 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
ENV LD_LIBRARY_PATH=/opt/concordtree/.web-python/lib:/opt/concordtree/.conda/env/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
WORKDIR /opt/concordtree
EXPOSE 8765
ENTRYPOINT ["/opt/concordtree/concordtree"]
CMD ["--no-browser"]
''')
print(context)
