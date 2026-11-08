"""Cached Linux host telemetry; never imports the scientific/CUDA runtime."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import io
import os
from pathlib import Path
import shutil
import select
import time
import subprocess
import threading


def cpu_ticks():
    # The first eight columns exclude guest time, already included in user/nice.
    values = [int(v) for v in Path('/proc/stat').read_text().splitlines()[0].split()[1:9]]
    return sum(values), values[3] + values[4]


def memory_usage():
    fields = {key: int(value.split()[0]) * 1024 for key, value in
              (line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())}
    total = fields['MemTotal']
    available = min(total, fields.get('MemAvailable', fields['MemFree']))
    return dict(total_bytes=total, available_bytes=available, used_bytes=total-available,
                utilization_percent=round(100 * (total-available) / total, 1))


def numeric(value):
    try:
        number = float(value.strip())
        return number if 0 <= number < float('inf') else None
    except ValueError:
        return None


def gpu_usage():
    fields = 'index,uuid,name,utilization.gpu,memory.used,memory.total'
    command = shutil.which('nvidia-smi') or '/usr/lib/wsl/lib/nvidia-smi'
    try:
        result = subprocess.run([command, '--query-gpu='+fields, '--format=csv,noheader,nounits'],
                                capture_output=True, text=True, timeout=5, check=True)
        full = True
    except subprocess.CalledProcessError:
        # Some WSL/driver versions expose identity but not every telemetry counter.
        result = subprocess.run([command, '--query-gpu=index,uuid,name', '--format=csv,noheader,nounits'],
                                capture_output=True, text=True, timeout=5, check=True)
        full = False
    return parse_gpus(result.stdout, full)

def parse_gpus(output, full=True):
    selected = os.environ.get('CONCORDTREE_GPU', '0')
    devices = []
    for row in csv.reader(io.StringIO(output), skipinitialspace=True):
        if not full and len(row) == 3: row += ['N/A'] * 3
        if len(row) != 6:
            continue
        index, uuid, name, utilization, used, total = [v.strip() for v in row]
        used, total = numeric(used), numeric(total)
        devices.append(dict(index=index, uuid=uuid, name=name,
            selected=selected in {index, uuid}, utilization_percent=numeric(utilization),
            memory_used_bytes=None if used is None else round(used * 1024**2),
            memory_total_bytes=None if total is None else round(total * 1024**2)))
    # Containers may expose one selected physical GPU under a different index.
    if len(devices) == 1:
        devices[0]['selected'] = True
    return devices


class HostMonitor:
    def __init__(self):
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.thread = None
        self.gpu_thread = None
        self.gpu_process = None
        self.previous_cpu = None
        self.value = dict(sampled_at=None, cpu=dict(utilization_percent=None, logical_cores=os.cpu_count()),
                          memory=None, gpus=[], gpu_status='sampling', gpu_sampled_at=None)

    def start(self):
        try: self.previous_cpu = cpu_ticks()
        except (OSError, ValueError, IndexError): pass
        self.thread = threading.Thread(target=self.run, name='host-load', daemon=True)
        self.thread.start()
        self.gpu_thread = threading.Thread(target=self.watch_gpu, name="gpu-watch", daemon=True)
        self.gpu_thread.start()

    def sample(self, include_gpu=True):
        value = dict(sampled_at=datetime.now(timezone.utc).isoformat(),
                     cpu=dict(utilization_percent=None, logical_cores=os.cpu_count()), memory=None,
                     gpus=[], gpu_status='unavailable')
        try:
            current = cpu_ticks()
            if self.previous_cpu is not None:
                total = current[0] - self.previous_cpu[0]
                idle = current[1] - self.previous_cpu[1]
                if total > 0 and 0 <= idle <= total:
                    value['cpu']['utilization_percent'] = round(100 * (total-idle) / total, 1)
            self.previous_cpu = current
        except (OSError, ValueError, IndexError): pass
        try: value['memory'] = memory_usage()
        except (OSError, ValueError, KeyError, ZeroDivisionError): pass
        if include_gpu:
            value['gpu_sampled_at'] = None
            try:
                value['gpus'] = gpu_usage()
                if value['gpus']:
                    value['gpu_status'] = 'available'
                    value['gpu_sampled_at'] = value['sampled_at']
            except (OSError, subprocess.SubprocessError, ValueError): pass
        with self.lock:
            if not include_gpu:
                for key in ('gpus','gpu_status','gpu_sampled_at'):value[key]=self.value.get(key)
            elif not value['gpus'] and self.value['gpus']:
                value.update(gpus=self.value['gpus'],gpu_status='stale',gpu_sampled_at=self.value.get('gpu_sampled_at'))
            self.value = value

    def gpu_interrupted(self):
        with self.lock:
            self.value['gpu_status']='stale' if self.value['gpus'] else 'unavailable'

    def watch_gpu(self):
        # One driver process streams samples, just like nvidia-smi -l 2.
        # CPU/RAM sampling and HTTP handlers never wait for the driver.
        full=True
        while not self.stop.is_set():
            process=None
            try:
                command=shutil.which('nvidia-smi') or '/usr/lib/wsl/lib/nvidia-smi'
                fields='index,uuid,name'+(',utilization.gpu,memory.used,memory.total' if full else '')
                with self.lock:
                    if self.stop.is_set():return
                    process=subprocess.Popen([command,'--query-gpu='+fields,'--format=csv,noheader,nounits','-l','2'],
                                             stdout=subprocess.PIPE,stderr=subprocess.PIPE)
                    self.gpu_process=process
                pending=b'';last=time.monotonic()
                while not self.stop.is_set():
                    if select.select([process.stdout],[],[],.25)[0]:
                        chunk=os.read(process.stdout.fileno(),65536)
                        if not chunk:break
                        pending+=chunk
                        lines=pending.split(b'\n');pending=lines.pop()
                        devices=parse_gpus(b'\n'.join(lines).decode('utf-8',errors='replace'),full)
                        if devices:
                            stamp=datetime.now(timezone.utc).isoformat();last=time.monotonic()
                            with self.lock:
                                merged={d['index']:d for d in self.value['gpus']}
                                merged.update({d['index']:{**d,'sampled_at':stamp} for d in devices})
                                self.value.update(gpus=sorted(merged.values(),key=lambda d:int(d['index'])),
                                                  gpu_status='available',gpu_sampled_at=stamp)
                    if time.monotonic()-last>6:self.gpu_interrupted()
                    if time.monotonic()-last>15:break
                if process.poll() is not None:
                    error=process.stderr.read().decode('utf-8',errors='replace').lower()
                    if 'not a valid field' in error or 'not supported' in error:full=False
            except (OSError,ValueError,subprocess.SubprocessError):pass
            finally:
                if process is not None:
                    self.stop_gpu_process(process)
                    process.stdout.close();process.stderr.close()
                with self.lock:self.gpu_process=None
            if not self.stop.is_set():
                self.gpu_interrupted();self.stop.wait(2)

    @staticmethod
    def stop_gpu_process(process):
        if process.poll() is None:
            process.terminate()
            try:process.wait(timeout=1)
            except subprocess.TimeoutExpired:process.kill();process.wait(timeout=1)

    def run(self):
        # Allow a nonzero interval for the first CPU percentage.
        if self.stop.wait(.25): return
        while not self.stop.is_set():
            self.sample(include_gpu=False)
            self.stop.wait(2)

    def snapshot(self):
        with self.lock: return dict(self.value)

    def close(self):
        self.stop.set()
        with self.lock:process=self.gpu_process
        if process is not None:self.stop_gpu_process(process)
        if self.thread:self.thread.join(timeout=3)
        if self.gpu_thread:self.gpu_thread.join(timeout=3)
