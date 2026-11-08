"""Per-job execution resources, independent of scientific pipeline parameters."""
import os


def thread_limit():
    try: available = len(os.sched_getaffinity(0))
    except (AttributeError, OSError): available = os.cpu_count() or 1
    return max(1, min(available, 64))


def defaults():
    try: threads = int(os.environ.get('CONCORDTREE_CPU_THREADS', '16'))
    except ValueError: threads = 16
    try: gpu = max(0, int(os.environ.get('CONCORDTREE_GPU', '0')))
    except ValueError: gpu = 0
    return dict(cpu_threads=max(1, min(threads, thread_limit())), gpu=gpu)


def normalize(value):
    if value is None: return defaults()
    if not isinstance(value, dict) or set(value) - {'cpu_threads', 'gpu'}:
        raise ValueError('Invalid compute settings.')
    result = {**defaults(), **value}
    for name, maximum in [('cpu_threads', thread_limit()), ('gpu', 65535)]:
        v = result[name]
        if isinstance(v, bool) or not isinstance(v, int) or not (1 if name == 'cpu_threads' else 0) <= v <= maximum:
            raise ValueError(f'{name} must be an integer between {1 if name == "cpu_threads" else 0} and {maximum}.')
    return result


def schema():
    return dict(defaults=defaults(), max_cpu_threads=thread_limit())
