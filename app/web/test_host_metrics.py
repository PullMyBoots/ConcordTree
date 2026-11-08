"""Telemetry must degrade cleanly and preserve host units/device selection."""
import subprocess
import sys
import time
import unittest
from unittest.mock import patch

from web import host_metrics


class HostMetricsTests(unittest.TestCase):
    def test_watch_reconnects_after_driver_exit(self):
        original = subprocess.Popen
        launches=[]
        def launch(*args, **kwargs):
            launches.append(args)
            usage=10 if len(launches)==1 else 80
            script=f"import time; print('0, GPU-a, NVIDIA Test, {usage}, 1024, 8192',flush=True); time.sleep({.1 if usage==10 else 20})"
            return original([sys.executable,'-u','-c',script], **kwargs)
        monitor=host_metrics.HostMonitor()
        with patch.object(host_metrics.subprocess,'Popen',side_effect=launch):
            monitor.start()
            try:
                deadline=time.monotonic()+3
                while time.monotonic()<deadline and monitor.snapshot()['gpu_status']!='stale':time.sleep(.02)
                self.assertEqual(monitor.snapshot()['gpus'][0]['name'],'NVIDIA Test')
                self.assertEqual(monitor.snapshot()['gpu_status'],'stale')
                deadline=time.monotonic()+4
                while time.monotonic()<deadline:
                    if monitor.snapshot()['gpus'][0]['utilization_percent']==80:break
                    time.sleep(.02)
                self.assertEqual(monitor.snapshot()['gpu_status'],'available')
                self.assertEqual(monitor.snapshot()['gpus'][0]['utilization_percent'],80)
                self.assertEqual(len(launches),2)
            finally:monitor.close()

    def test_watch_uses_one_streaming_process_for_multiple_samples(self):
        original = subprocess.Popen
        script = "import time; print('0, GPU-a, NVIDIA Test, 10, 1024, 8192',flush=True); time.sleep(.1); print('0, GPU-a, NVIDIA Test, 80, 2048, 8192',flush=True); time.sleep(20)"
        def launch(*args, **kwargs): return original([sys.executable,'-u','-c',script], **kwargs)
        monitor = host_metrics.HostMonitor()
        with patch.object(host_metrics.subprocess,'Popen',side_effect=launch) as start:
            monitor.start()
            try:
                deadline=time.monotonic()+3
                while time.monotonic()<deadline:
                    devices=monitor.snapshot()['gpus']
                    if devices and devices[0]['utilization_percent']==80:break
                    time.sleep(.02)
                self.assertEqual(monitor.snapshot()['gpus'][0]['utilization_percent'],80)
                self.assertEqual(start.call_count,1)
                self.assertEqual(start.call_args.args[0][-2:],['-l','2'])
                process=monitor.gpu_process
            finally: monitor.close()
        self.assertIsNotNone(process.poll())

    def test_transient_failure_preserves_gpu_identity_and_recovers(self):
        monitor = host_metrics.HostMonitor()
        device = dict(index='1', name='NVIDIA Test', selected=True, utilization_percent=25)
        with patch.object(host_metrics, 'gpu_usage', return_value=[device]): monitor.sample()
        previous = monitor.snapshot()['gpu_sampled_at']
        with patch.object(host_metrics, 'gpu_usage', side_effect=subprocess.TimeoutExpired('nvidia-smi', 5)):
            monitor.sample()
        self.assertEqual(monitor.snapshot()['gpus'], [device])
        self.assertEqual(monitor.snapshot()['gpu_status'], 'stale')
        self.assertEqual(monitor.snapshot()['gpu_sampled_at'], previous)
        with patch.object(host_metrics, 'gpu_usage', return_value=[{**device, 'utilization_percent': 80}]): monitor.sample()
        self.assertEqual(monitor.snapshot()['gpu_status'], 'available')
        self.assertEqual(monitor.snapshot()['gpus'][0]['utilization_percent'], 80)

    def test_identity_fallback_for_unsupported_telemetry_query(self):
        with patch.object(host_metrics.subprocess, 'run', side_effect=[
            subprocess.CalledProcessError(1, 'nvidia-smi'),
            subprocess.CompletedProcess([], 0, '0, GPU-a, NVIDIA WSL\n')]):
            device = host_metrics.gpu_usage()[0]
        self.assertEqual(device['name'], 'NVIDIA WSL')
        self.assertIsNone(device['utilization_percent'])

    def test_cpu_is_interval_usage_not_load_average(self):
        monitor = host_metrics.HostMonitor()
        monitor.previous_cpu = (1000, 800)
        with patch.object(host_metrics, 'cpu_ticks', return_value=(1200, 850)), \
             patch.object(host_metrics, 'memory_usage', return_value=None), \
             patch.object(host_metrics, 'gpu_usage', side_effect=FileNotFoundError):
            monitor.sample()
        data = monitor.snapshot()
        self.assertEqual(data['cpu']['utilization_percent'], 75)
        self.assertEqual(data['gpu_status'], 'unavailable')
        self.assertEqual(data['gpus'], [])

    def test_unavailable_counters_are_not_zero(self):
        monitor = host_metrics.HostMonitor()
        with patch.object(host_metrics, 'cpu_ticks', side_effect=OSError), \
             patch.object(host_metrics, 'memory_usage', side_effect=OSError), \
             patch.object(host_metrics, 'gpu_usage', side_effect=subprocess.TimeoutExpired('nvidia-smi', 2)):
            monitor.sample()
        self.assertIsNone(monitor.snapshot()['cpu']['utilization_percent'])
        self.assertIsNone(monitor.snapshot()['memory'])
        self.assertEqual(monitor.snapshot()['gpu_status'], 'unavailable')

    def test_memory_uses_available_including_reclaimable_cache(self):
        with patch.object(host_metrics.Path, 'read_text', return_value='MemTotal: 1000 kB\nMemAvailable: 600 kB\nMemFree: 100 kB\n'):
            data = host_metrics.memory_usage()
        self.assertEqual(data['used_bytes'], 400*1024)
        self.assertEqual(data['utilization_percent'], 40)

    def test_multiple_gpu_selection_and_unsupported_counter(self):
        output = '0, GPU-a, NVIDIA A, 60, 1024, 8192\n1, GPU-b, NVIDIA B, [N/A], 2048, 16384\n'
        with patch.object(host_metrics.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, output)), \
             patch.dict(host_metrics.os.environ, {'CONCORDTREE_GPU': '1'}):
            devices = host_metrics.gpu_usage()
        self.assertEqual([d['selected'] for d in devices], [False, True])
        self.assertEqual(devices[0]['memory_used_bytes'], 1024**3)
        self.assertIsNone(devices[1]['utilization_percent'])

    def test_container_single_gpu_is_selected(self):
        with patch.object(host_metrics.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, '0, GPU-b, NVIDIA B, 0, 0, 16384\n')), \
             patch.dict(host_metrics.os.environ, {'CONCORDTREE_GPU': '2'}):
            self.assertTrue(host_metrics.gpu_usage()[0]['selected'])

    def test_wsl_driver_query_when_driver_directory_is_not_on_path(self):
        with patch.object(host_metrics.shutil, 'which', return_value=None), \
             patch.object(host_metrics.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, '0, GPU-a, NVIDIA A, 1, 0, 8192\n')) as query:
            host_metrics.gpu_usage()
        self.assertEqual(query.call_args.args[0][0], '/usr/lib/wsl/lib/nvidia-smi')


if __name__ == '__main__': unittest.main()
