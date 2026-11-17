"""macOS process/system resource samples; all byte units are explicit."""
import json
import os
import re
import subprocess
import threading
import time


def parse_swap(text):
    match = re.search(r'used\s*=\s*([\d.]+)([KMG]?)', text)
    if not match:
        raise ValueError('Cannot parse swap usage')
    return int(float(match[1]) * {'': 1, 'K': 1024, 'M': 1024**2, 'G': 1024**3}[match[2]])


def parse_vm_stat(text):
    size = re.search(r'page size of (\d+) bytes', text)
    if not size:
        raise ValueError('Cannot parse VM page size')
    pages = int(size[1])
    result = {'page_size_bytes': pages}
    for name, key in [('Pages free', 'free_bytes'), ('Pages inactive', 'inactive_bytes'),
                      ('Pages speculative', 'speculative_bytes'), ('Pages wired down', 'wired_bytes'),
                      ('Pages occupied by compressor', 'compressor_bytes')]:
        match = re.search(re.escape(name) + r':\s*(\d+)', text)
        if match:
            result[key] = int(match[1]) * pages
    return result


def resource_violation(sample, initial_swap):
    if sample['rss_bytes'] > 10 * 1024**3:
        return 'Process RSS exceeded 10 GiB'
    if sample['swap_used_bytes'] - initial_swap > 2 * 1024**3:
        return 'System swap grew by more than 2 GiB'
    return None


def sample_resources():
    def command(args):
        return subprocess.check_output(args, text=True, timeout=5)
    rss = int(command(['/bin/ps', '-o', 'rss=', '-p', str(os.getpid())]).strip()) * 1024
    swap = parse_swap(command(['/usr/sbin/sysctl', '-n', 'vm.swapusage']))
    return dict(rss_bytes=rss, swap_used_bytes=swap, **parse_vm_stat(command(['/usr/bin/vm_stat'])))


class ResourceMonitor:
    def __init__(self, path):
        self.path = path
        self.samples = []
        self.error = None
        self.stop_event = threading.Event()
        self.started = time.perf_counter()

    def start(self):
        self._sample()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _sample(self):
        try:
            sample = dict(seconds=time.perf_counter() - self.started, **sample_resources())
            self.samples.append(sample)
            with self.path.open('a') as handle:
                handle.write(json.dumps(sample) + '\n')
            violation = resource_violation(sample, self.samples[0]['swap_used_bytes'])
            if violation:
                self.error = violation
        except Exception as exc:
            self.error = 'Resource sampling failed: ' + repr(exc)

    def _loop(self):
        while not self.stop_event.wait(2):
            self._sample()

    def check(self):
        if self.error:
            raise RuntimeError(self.error)

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=20)
        self._sample()

    def summary(self):
        if not self.samples:
            return {'error': self.error, 'n_samples': 0}
        first, last = self.samples[0], self.samples[-1]
        return dict(n_samples=len(self.samples), error=self.error,
                    peak_sampled_rss_bytes=max(s['rss_bytes'] for s in self.samples),
                    initial_swap_used_bytes=first['swap_used_bytes'],
                    final_swap_used_bytes=last['swap_used_bytes'],
                    peak_swap_growth_bytes=max(s['swap_used_bytes'] for s in self.samples) - first['swap_used_bytes'],
                    initial_vm=first, final_vm=last)
