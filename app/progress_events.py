"""Small durable UI events; never serializes per-edge arrays or search state."""
import json
import math


def compact(event):
    value = {k: v for k, v in event.items()
             if isinstance(v, (str, int, float, bool)) and
             (not isinstance(v, float) or math.isfinite(v))}
    if isinstance(event.get('detail'), dict): value['detail'] = compact(event['detail'])
    selected = event.get('selected')
    if selected:
        value['mean_probability_gain'] = sum(s['margin'] for s in selected) / len(selected)
    return value


def append(path, event):
    with path.open('a') as stream:
        stream.write(json.dumps(compact(event), allow_nan=False, separators=(',', ':')) + '\n')
