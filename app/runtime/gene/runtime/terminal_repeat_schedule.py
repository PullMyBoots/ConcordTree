"""Exact dependency scheduling: terminal leaf endpoints can run last.

Only internal posterior states feed later descendants. Leaf endpoint factors
are consumed by the final fixed-order count reduction, so independent leaf
records can be grouped by their existing CSR width after all internal states.
No parent-link order, record fields, dtype, reduction or numerical knob changes.
"""
import numpy as np

def terminal_repeat_schedule(buckets,internal_nodes):
    ordered=[];leaves={};leaf_rows=0;last_depth=0
    for depth,power,records in buckets:
        last_depth=max(last_depth,int(depth))
        mask=records[:,0]<internal_nodes
        if np.all(mask):ordered.append((depth,power,records));continue
        if np.any(mask):ordered.append((depth,power,records[mask]))
        terminal=records if not np.any(mask) else records[~mask]
        leaves.setdefault(int(power),[]).append(terminal)
        leaf_rows+=len(terminal)
    for power,arrays in sorted(leaves.items()):
        records=arrays[0] if len(arrays)==1 else np.concatenate(arrays,axis=0)
        ordered.append((last_depth+1,power,records))
    return ordered,dict(terminal_leaf_schedule=True,terminal_leaf_records=leaf_rows,
        reverse_buckets_before=len(buckets),reverse_buckets_after=len(ordered))
