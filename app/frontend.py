"""Shared task-specific MLP-NNI frontend for the unified AP-NJ pipeline.

The unified engine initializes with apnj_frontend, then calls this MLP.
This file does not invoke SplitBank. Load the original
learned-NNI implementation by file to avoid mixing two concordtree packages.
The small network below matches the frozen model's names and forward exactly.
"""
from __future__ import annotations

import hashlib
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class QuartetMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, 3)

    def forward(self, value):
        return self.fc3(F.relu(self.fc2(F.relu(self.fc1(value)))))


MLP_ASSETS = {
    'homogeneous': ('homogeneous.pth', '8554764e33885992cc31e4d044a82a4ae8e876f119197e72b07ae3a1dcbc533b'),
    'heterogeneous': ('heterogeneous.pth', 'a74e89e7b6091938f0b413a5c1446864bc18a8db7176f5c77a204018bc8e2edb'),
}
TASK_PREDICTOR = {'gene': 'homogeneous', 'species': 'heterogeneous'}


class MLPWarmstart:
    def __init__(self, predictor):
        if predictor not in MLP_ASSETS:
            raise ValueError('unknown MLP predictor')
        self.predictor = predictor
        self.root = Path(__file__).resolve().parent
        self.loaded = False

    def load(self):
        if self.loaded:
            return 0.
        start = time.perf_counter()
        from concordtree.assets import load_backends, _load_extension
        self.sequence, self.pattern = load_backends()
        from concordtree.assets import PACKAGE_ROOT
        package = PACKAGE_ROOT
        filename, expected = MLP_ASSETS[self.predictor]
        checkpoint = self.root / 'assets/mlp' / filename
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if digest != expected:
            raise RuntimeError('MLP asset hash mismatch: ' + self.predictor)
        self.model = QuartetMLP().to('cuda:0').eval()
        self.model.load_state_dict(torch.load(checkpoint, map_location='cuda:0', weights_only=True))
        from concordtree._core import learned_nni
        self.nni = learned_nni
        path = package / 'assets/backends/learned_nni_plan_backend.cpython-310-x86_64-linux-gnu.so'
        self.plan = _load_extension('learned_nni_plan_backend', path)
        self.checkpoint_sha256 = digest
        self.loaded = True
        torch.cuda.synchronize()
        return time.perf_counter() - start

    def validate_features(self):
        self.load()
        rng = np.random.default_rng(20261105)
        # Odd length and missing characters also exercise packed tail handling.
        states = rng.integers(0, 5, size=(12, 131), dtype=np.uint8)
        quartets = np.array([rng.choice(12, 4, replace=False) for _ in range(16)])
        padded = np.full((12, 132), 4, dtype=np.uint8)
        padded[:, :131] = states
        packed = padded[:, ::2] | (padded[:, 1::2] << 4)
        actual = self.pattern.compute_pattern_frequencies_cuda_packed(
            torch.tensor(packed, device='cuda:0'),
            torch.tensor(quartets, device='cuda:0', dtype=torch.long), 131).cpu().numpy()
        expected = []
        for q in quartets:
            values = states[q].astype(np.int64)
            valid = np.all(values < 4, axis=0)
            ids = np.array([64, 16, 4, 1]) @ values[:, valid]
            expected.append(np.bincount(ids, minlength=256) / valid.sum())
        error = float(np.abs(actual - np.array(expected)).max())
        assert error < 1e-7, error
        return dict(status='passed', rows=16, maximum_absolute_error=error,
                    checkpoint_sha256=self.checkpoint_sha256)

    @torch.no_grad()
    def refine(self, tree, names, msa, *, stop_ratio=.005, progress=None, packed_data=None, max_rounds=None, representatives=4, min_margin=.10, batch_size=65536, checkpoint=None):
        start = time.perf_counter()
        load_seconds = self.load()
        if packed_data is None:
            packed, raw_names, length = self.sequence.load_phy_to_packed_tensor(str(msa), True)
        else:
            # Reuse the same native parser's packed original observations.
            # Apply the original raw-constant filter without parsing MSA again.
            raw_packed, raw_names, original_sites = packed_data
            packed, length = self.sequence.filter_conserved_packed(
                raw_packed, int(original_sites), torch.get_num_threads())
        if length < 1:
            tree.require_binary()
            return tree,dict(passes=[],moves=0,seconds=time.perf_counter()-start,
                stop='no_variable_sites',converged=False,variable_sites=0,
                checkpoint_sha256=self.checkpoint_sha256,
                warnings=['No variable sites: MLP warmstart has no topology information'])
        if len(raw_names)!=len(names) or set(raw_names)!=set(names):
            raise ValueError('MLP taxon names do not match the alignment')
        names = list(raw_names)
        name_to_row = {name: i for i, name in enumerate(names)}
        relabel = {node: name_to_row[name] for node, name in tree.names.items()}
        for i, node in enumerate(sorted(set(tree.graph) - set(tree.names))):
            relabel[node] = len(names) + i
        adjacency = {relabel[u]: {relabel[v] for v in adjacent}
                     for u, adjacent in tree.graph.items()}
        sequences = torch.from_numpy(packed).to('cuda:0')
        torch.cuda.synchronize()
        prep_seconds = time.perf_counter() - start

        def predict(quartets):
            indices = torch.as_tensor(quartets, device='cuda:0', dtype=torch.long)
            features = self.pattern.compute_pattern_frequencies_cuda_packed(
                sequences, indices, int(length)).contiguous()
            probabilities=torch.softmax(self.model(features), dim=-1)
            # An all-zero feature row means no jointly observed quartet site.
            # A network bias is not sequence evidence and must not drive NNI.
            probabilities=torch.where(features.any(dim=-1,keepdim=True),probabilities,
                                      torch.full_like(probabilities,1./3.))
            return probabilities.cpu().numpy()

        search_start = time.perf_counter()
        from array_mlp_nni import refine
        def checkpoint_neighbors(neighbors,record):
            from concordtree.treeio import Tree
            import _ct_apnj_topology as topology
            if checkpoint:checkpoint(Tree(topology.export_graph(neighbors,len(names),True),dict(enumerate(names)),{}),dict(stage='mlp',round=record['pass']))
        result = refine(
            adjacency, len(names), predict, passes=max_rounds or 10000, representatives=representatives,
            min_margin=min_margin,batch_size=batch_size, stop_moves_at_most=math.floor(stop_ratio * (len(names)-3)),
            plan_backend=self.plan, nni=self.nni, progress=progress,checkpoint=checkpoint_neighbors if checkpoint else None)
        torch.cuda.synchronize()
        search_seconds = time.perf_counter() - search_start
        if result['stop']=='emergency_pass_guard' and max_rounds is not None:
            result.update(stop='round_budget',converged=False,warnings=['MLP stopped at requested maximum rounds'])
        if result['stop']=='emergency_pass_guard':
            raise RuntimeError('MLP emergency guard exhausted before requested stop ratio')
        from concordtree.treeio import Tree, edge
        refined = Tree(adjacency, {i: name for i, name in enumerate(names)},
                       {edge(u,v): None for u in adjacency for v in adjacency[u] if u<v})
        refined.require_binary()
        result.update(seconds=time.perf_counter()-start, load_seconds=load_seconds,
                      preparation_seconds=prep_seconds, search_seconds=search_seconds,
                      variable_sites=int(length), checkpoint_sha256=self.checkpoint_sha256,
                      source='v0.5.9 optimized MLP-NNI; task-specific checkpoint; no SplitBank',
                      stop_ratio=stop_ratio, model=self.predictor, representatives=representatives, min_margin=min_margin, max_rounds=max_rounds, batch_size=batch_size)
        return refined, result
