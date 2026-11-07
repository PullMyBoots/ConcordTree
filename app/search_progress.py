"""Bounded search diagnostics; ordinary scientific stopping is unchanged.

Exact recurrence compares complete deterministic states, not hashes or topology
alone. Numerical plateau detection is separate from convergence: it requires a
window of repeated topologies AND negligible total verified objective change.
"""
from collections import Counter, deque
import math

EMERGENCY_SWEEPS = 10000
MAX_PARAMETER_CONTINUATIONS = 64


class ExactRecurrence:
    """Brent recurrence detection with one immutable full-state checkpoint."""
    def __init__(self):
        self.checkpoint = None
        self.power = 1
        self.distance = 0

    def repeated(self, state):
        if self.checkpoint is None:
            self.checkpoint = state
            return False
        self.distance += 1
        if state == self.checkpoint:
            return True
        if self.distance == self.power:
            self.checkpoint = state
            self.power *= 2
            self.distance = 0
        return False


class NumericalPlateau:
    """Only repeated topology wandering within the existing LL check tolerance.

Eight consecutive low-gain rounds are required; every topology in the window
must occur at least twice. A new topology, or meaningful cumulative progress,
prevents this diagnostic. Payloads retain only the best verified incumbent.
"""
    def __init__(self, tolerance, window=8):
        self.tolerance = float(tolerance)
        self.rows = deque(maxlen=window)
        self.best_value = -math.inf
        self.best_payload = None

    def observe(self, before, after, state, payload=lambda: None):
        if not math.isfinite(before) or not math.isfinite(after):
            raise ArithmeticError('nonfinite search objective')
        if after-before > self.tolerance:
            self.rows.clear()
            self.best_value = -math.inf
            self.best_payload = None
            return False
        self.rows.append((state(), before, after))
        if after > self.best_value:
            self.best_value = after
            self.best_payload = payload()
        if len(self.rows) < self.rows.maxlen:
            return False
        counts = Counter(row[0] for row in self.rows)
        span = max(row[2] for row in self.rows)-min(row[1] for row in self.rows)
        return min(counts.values()) >= 2 and span <= self.tolerance
