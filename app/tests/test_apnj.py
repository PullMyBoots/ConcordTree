"""Numerical, lifecycle and public-boundary checks for the relocatable closure.

Run only when this worker has exclusive use of the selected GPU. All synthetic
observations are created in memory; source datasets are never modified.
"""
from pathlib import Path
import sys, math
import numpy as np
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import argparse
import gc
import unittest

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, required=True)
args = parser.parse_args()
from runtime_config import configure_runtime
configure_runtime(gpu=args.gpu, cpu_threads=16)
from apnj_frontend import initialize, MAX_SITES
import _ct_apnj_packed as packed_native
from concordtree.treeio import newick_python, parse_newick, nrf
import torch


def pack(states):
    padded = np.full((len(states), states.shape[1]+states.shape[1] % 2), 4, np.uint8)
    padded[:, :states.shape[1]] = states
    return padded[:, ::2] | (padded[:, 1::2] << 4)


def hash32(value):
    value &= 0xffffffff
    value ^= value >> 16
    value = value * 0x7feb352d & 0xffffffff
    value ^= value >> 15
    value = value * 0x846ca68b & 0xffffffff
    return value ^ (value >> 16)


class Boundaries(unittest.TestCase):
    def setUp(self):
        self.names = ['a', 'b', 'c', 'd']
        self.packed = np.zeros((4, 8), np.uint8)

    def call(self, **kw):
        params = dict(task='gene', seed=20260934, cpu_threads=16)
        params.update(kw)
        return initialize(self.names, (self.packed, self.names, 16), **params)

    def test_invalid_public_parameters(self):
        for task in ['', 'other', None]:
            with self.subTest(task=task), self.assertRaisesRegex(ValueError, 'task'):
                self.call(task=task)
        for seed in [-1, 2**32, True, 1.25]:
            with self.subTest(seed=seed), self.assertRaisesRegex(ValueError, 'seed'):
                self.call(seed=seed)
        for threads in [0, -1, True, 1.25]:
            with self.subTest(threads=threads), self.assertRaisesRegex(ValueError, 'thread'):
                self.call(cpu_threads=threads)
        for sites in [0, -1, MAX_SITES+1, True, 1.25]:
            with self.subTest(sites=sites), self.assertRaisesRegex(ValueError, 'site count'):
                initialize(self.names, (self.packed, self.names, sites), task='gene', seed=1)

    def test_invalid_layouts_and_names(self):
        for array in [self.packed.astype(np.float32), np.zeros((4, 16), np.uint8)[:, ::2],
                      np.zeros((4, 7), np.uint8), np.zeros(32, np.uint8)]:
            with self.subTest(shape=array.shape), self.assertRaisesRegex(ValueError, 'contiguous uint8'):
                initialize(self.names, (array, self.names, 16), task='gene', seed=1)
        with self.assertRaisesRegex(ValueError, 'order mismatch'):
            initialize(self.names, (self.packed, self.names[::-1], 16), task='gene', seed=1)
        for names in [['a', 'a', 'c', 'd'], ['a', '', 'c', 'd'], ['a', 1, 'c', 'd']]:
            with self.assertRaises(ValueError):
                initialize(names, (self.packed, names, 16), task='gene', seed=1)
        with self.assertRaisesRegex(ValueError, 'taxa'):
            initialize(self.names[:3], (self.packed[:3], self.names[:3], 16), task='gene', seed=1)

    def test_entirely_missing_even_and_odd_tail(self):
        for sites in [1, 2, 15, 16, 17]:
            x = np.zeros((4, sites), np.uint8)
            x[2] = 4
            data = pack(x)
            if sites % 2:
                # Deliberately make the unused final nibble look like A.
                data[:, -1] &= 15
            with self.subTest(sites=sites), self.assertRaisesRegex(ValueError, 'observed A/C/G/T'):
                initialize(self.names, (data, self.names, sites), task='gene', seed=1)


class Numerics(unittest.TestCase):
    def test_projection_queries_rows_and_gamma(self):
        rng = np.random.default_rng(202611062012)
        checks = []
        for sites in [1, 2, 15, 16, 17, 63, 64, 65, 129, 2051]:
            x = rng.integers(0, 8, (12, sites), dtype=np.uint8)
            x[:, 0] = np.arange(12, dtype=np.uint8) % 4
            data = pack(x)
            before = data.copy()
            new = packed_native.State(data, sites, 16)
            seed, projections = 20260934, 4
            actual = new.project(seed, projections)
            expected = np.zeros_like(actual)
            for row in range(len(x)):
                for p in range(projections):
                    salt = hash32(seed+p*0x9e3779b9)
                    expected[row, p] = sum((1 if hash32(s*4+int(base)+salt)&1 else -1)
                                           for s, base in enumerate(x[row]) if base < 4)
            np.testing.assert_array_equal(actual, expected)
            self.assertTrue(.01 <= new.gamma_shape() <= 100.)
            for width in [1, 2, 4]:
                groups = 12//width
                reps = np.arange(groups*width, dtype=np.int32).reshape(groups, width)
                mass = np.full((groups, width), 1./width)
                pairs = np.array([(i, j) for i in range(groups) for j in range(groups)], np.int64)
                f = np.zeros((groups, sites, 4))
                for i in range(groups):
                    for r, weight in zip(reps[i], mass[i]):
                        for base in range(4):
                            f[i, :, base] += weight*(x[r] == base)
                total = f.sum(axis=0)
                raw = np.zeros((groups, 2))
                for i in range(groups):
                    raw[i, 0] = groups-1-(f[i]*(total-f[i])).sum()/sites
                    pur, pyr = f[i, :, 0]+f[i, :, 2], f[i, :, 1]+f[i, :, 3]
                    ar, ay = total[:, 0]+total[:, 2], total[:, 1]+total[:, 3]
                    raw[i, 1] = (pur*(ay-pyr)+pyr*(ar-pur)).sum()/sites
                np.testing.assert_allclose(new.raw_rows(reps, mass), raw, atol=2e-13, rtol=2e-13)
                for common in [False, True]:
                    for model in [0, 1, 2, 3]:
                        answer = new.query(reps, mass, pairs, model, common, 1.25, True)
                        self.assertTrue(np.isfinite(answer).all())
                        manual = []
                        for i, j in pairs:
                            value = 0.
                            if i != j:
                                for a, wa in zip(reps[i], mass[i]):
                                    for b, wb in zip(reps[j], mass[j]):
                                        both = (x[a] < 4)&(x[b] < 4)
                                        usable_mask = both if common else ((x[a] < 4)|(x[b] < 4))
                                        usable = int(usable_mask.sum()) if a != b else 0
                                        same = (x[a] == x[b])&both
                                        p = 0. if a == b else (1.-same.sum()/usable if usable else .75)
                                        q = 0. if a == b else (int((both&((x[a] % 2) != (x[b] % 2))).sum())/usable if usable else .5)
                                        floor = .5/(usable+1.)
                                        def inverse(f):
                                            return 1.25*math.expm1(min(30., -math.log(max(f, floor))/1.25))
                                        if model == 0: corrected = p
                                        elif model == 1: corrected = -.75*math.log(max(1.-4.*p/3., floor))
                                        elif model == 2: corrected = .75*inverse(1.-4.*p/3.)
                                        else: corrected = .5*inverse(1.-2.*p+q)+.25*inverse(1.-2.*q)
                                        value += wa*wb*corrected
                            manual.append(value)
                        np.testing.assert_allclose(answer[:, 0], manual, atol=3e-13, rtol=3e-13)
            np.testing.assert_array_equal(before, data)
            checks.append(dict(sites=sites, widths=[1, 2, 4], models=[0, 1, 2, 3], masks=['union', 'intersection']))
            del new


    def test_representative_validation(self):
        x = np.zeros((4, 8), np.uint8)
        state = packed_native.State(x, 16, 1)
        reps = np.arange(4, dtype=np.int32)[:, None]
        pairs = np.array([[0, 1]], np.int64)
        for weights in [np.full((4, 1), .5), np.full((4, 1), np.nan), np.full((4, 1), -1.)]:
            with self.assertRaises(ValueError):
                state.raw_rows(reps, weights)
            with self.assertRaises(ValueError):
                state.query(reps, weights, pairs, 1, True)


class Lifecycle(unittest.TestCase):
    def test_thread_counts_labels_ties_and_input_immutability(self):
        rng = np.random.default_rng(202611062013)
        names = [f"taxon {i}:a,'[]" for i in range(513)]
        x = rng.integers(0, 5, (513, 65), dtype=np.uint8)
        x[:, 0] = 0
        data = pack(x)
        before = data.copy()
        outputs = []
        for workers in [1, 3, 16]:
            tree, meta = initialize(names, (data, names, 65), task='gene', seed=20260934, cpu_threads=workers)
            outputs.append(newick_python(tree))
            self.assertEqual(meta['cpu_threads'], workers)
            self.assertEqual(nrf(tree, parse_newick(outputs[-1])), 0.)
        self.assertEqual(len(set(outputs)), 1)
        renamed = [f'renamed_{i}' for i in range(len(names))]
        other, _ = initialize(renamed, (data, renamed, 65), task='gene', seed=20260934)
        self.assertEqual(tree.graph, other.graph)
        np.testing.assert_array_equal(before, data)
        # Exactly tied observations must still produce a connected binary tree.
        data[:] = 0
        tied, _ = initialize(names, (data, names, 65), task='species', seed=20260934)
        tied.require_binary()
        self.assertEqual(len(tied.graph), 2*len(names)-2)

    def test_native_memory_released_after_repeated_calls(self):
        torch.cuda.init()
        names = [str(i) for i in range(32)]
        rng = np.random.default_rng(202611062014)
        x = rng.integers(0, 5, (32, 129), dtype=np.uint8)
        x[:, 0] = 0
        data = pack(x)
        first, _ = initialize(names, (data, names, 129), task='gene', seed=20260934)
        text = newick_python(first)
        torch.cuda.synchronize()
        gc.collect()
        before, _ = torch.cuda.mem_get_info()
        for _ in range(12):
            tree, _ = initialize(names, (data, names, 129), task='gene', seed=20260934)
            self.assertEqual(newick_python(tree), text)
        torch.cuda.synchronize()
        gc.collect()
        after, _ = torch.cuda.mem_get_info()
        self.assertLessEqual(before-after, 65536)


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
