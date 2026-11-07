"""Experimental all-taxon/all-site gene-tree ML-NNI; no reference topology input.

Site-rate categories are shared across the entire tree. Incoming messages retain
their category/site scales until AFTER category mixing. No quartet rank test,
neural scorer, taxon subsampling or site subsampling is used here.
"""
from __future__ import annotations
import math
import time
import numpy as np
import torch
from concordtree.treeio import edge, rooted
from concordtree.dna_models import transition_matrices
from concordtree.reversible_lengths import PruningPlan


def tensors(metrics, device):
    return tuple(torch.tensor(metrics[k], dtype=torch.float64, device=device)
                 for k in ('Q_acgt', 'frequencies_acgt', 'category_rates', 'category_weights'))


@torch.no_grad()
def independent_ll(tree, names, states, metrics, device='cuda:0'):
    """Independent release pruning, matrix_exp transitions, FP64 throughout."""
    q, pi, rates, weights = tensors(metrics, device)
    plan = PruningPlan(tree, names, states, device, len(rates))
    lengths = torch.tensor([tree.lengths[e] for e in plan.edges], device=device, dtype=torch.float64)
    p = transition_matrices(q, lengths, rates)
    ll = plan.expectation(p, pi, weights, need_counts=False,
                          backend='triton' if torch.device(device).type == 'cuda' else 'torch')[0]
    result = float(ll)
    if not math.isfinite(result):
        raise RuntimeError('nonfinite independent likelihood')
    return result


class FullContextNNI:
    """Exact conditional messages; bounded central-length optimization per NNI.

    Memory O(N*K*L*4) for directed messages, time O(N*K*L) per scan for fixed
    length-search budget. This prototype retains all messages, not a large-N
    streaming implementation. Central length optimization is not a global ML
    search; the best of every evaluated length (including incumbent) is retained.
    """
    @torch.no_grad()
    def __init__(self, tree, names, states, metrics, device='cuda:0'):
        tree.require_binary()
        states = np.asarray(states)
        if set(names) != set(tree.names.values()) or len(names) != len(set(names)):
            raise ValueError('alignment/tree taxa mismatch')
        if states.shape[0] != len(names) or states.ndim != 2 or states.min() < 0 or states.max() > 4:
            raise ValueError('invalid DNA alignment')
        self.tree, self.device = tree, torch.device(device)
        self.q, self.pi, self.rates, self.weights = tensors(metrics, device)
        self.sites = states.shape[1]
        self.root, parent, order, _ = rooted(tree)
        edges = [edge(u, parent[u]) for u in order if u != self.root]
        lengths = torch.tensor([tree.lengths[e] for e in edges], dtype=torch.float64, device=device)
        p = transition_matrices(self.q, lengths, self.rates)
        self.transitions = dict(zip(edges, p))
        lookup = {n: i for i, n in enumerate(names)}
        seq = torch.as_tensor(states.astype(np.int64), device=device)
        self.obs = {}
        for u, name in tree.names.items():
            x = seq[lookup[name]]
            val = torch.nn.functional.one_hot(x.clamp_max(3), 4).double()
            self.obs[u] = torch.where((x < 4)[:, None], val, torch.ones_like(val))[None].expand(len(self.rates), -1, -1)
        self.messages = {}
        self.scales = {}
        for u in reversed(order):
            if u != self.root:
                self._send(u, parent[u])
        for u in order:
            for v in sorted(tree.graph[u]):
                if parent.get(v) == u:
                    self._send(u, v)
        product = torch.ones_like(next(iter(self.messages.values())))
        scale = torch.zeros_like(next(iter(self.scales.values())))
        for v in tree.graph[self.root]:
            product *= self.messages[v, self.root]
            scale += self.scales[v, self.root]
        category = (product * self.pi).sum(-1).log() + scale + self.weights.log()[:, None]
        self.log_likelihood = float(torch.logsumexp(category, 0).sum())
        # Reversible symmetric eigensystem for repeated scalar central lengths.
        self.sqrtpi = self.pi.sqrt()
        sym = self.sqrtpi[:, None] * self.q / self.sqrtpi[None, :]
        self.eigenvalues, self.vectors = torch.linalg.eigh((sym + sym.T) / 2)
        self.eigenvalues[-1] = 0.  # stationary eigenvalue, eliminate roundoff drift
        self.exponents = self.rates[:, None] * self.eigenvalues[None, :]

    def _send(self, u, v):
        val = self.obs.get(u)
        if val is None:
            val = torch.ones_like(next(iter(self.obs.values())))
        scale = torch.zeros(val.shape[:-1], dtype=val.dtype, device=val.device)
        for w in sorted(self.tree.graph[u] - {v}):
            val = val * self.messages[w, u]
            scale = scale + self.scales[w, u]
        # Normalize both before and after transition. Zero invariant-category
        # likelihood stays zero, never becomes a spurious positive category.
        norm = val.amax(-1).clamp_min(1e-300)
        val = val / norm[..., None]
        scale = scale + norm.log()
        val = torch.einsum('kij,ksj->ksi', self.transitions[edge(u, v)], val)
        norm = val.amax(-1).clamp_min(1e-300)
        self.messages[u, v] = val / norm[..., None]
        self.scales[u, v] = scale + norm.log()

    def internal_edges(self):
        return [edge(u, v) for u in sorted(self.tree.graph) if u not in self.tree.names
                for v in sorted(self.tree.graph[u]) if v > u and v not in self.tree.names]

    @torch.no_grad()
    def coefficients(self, edges):
        """Return [edge*3,category,site,eigenstate] with category scaling included."""
        coefficients, shifts, branches = [], [], []
        for u, v in edges:
            a, b = sorted(self.tree.graph[u] - {v})
            c, d = sorted(self.tree.graph[v] - {u})
            branches.append((a, b, c, d))
            incoming = [self.messages[x, y] for x, y in ((a, u), (b, u), (c, v), (d, v))]
            scale = sum(self.scales[x, y] for x, y in ((a, u), (b, u), (c, v), (d, v)))
            logweight = scale + self.weights.log()[:, None]
            shift = logweight.amax(0)
            scaledweight = (logweight - shift[None]).exp()
            for i, j, k, l in ((0, 1, 2, 3), (0, 2, 1, 3), (0, 3, 1, 2)):
                left, right = incoming[i] * incoming[j], incoming[k] * incoming[l]
                x = (left * self.sqrtpi) @ self.vectors
                y = (right * self.sqrtpi) @ self.vectors
                coeff = x * y
                # Invariant class is an exact identity transition, not an
                # eigenvector cancellation across mutually exclusive states.
                zero = self.rates == 0
                coeff[zero] = 0
                coeff[zero, :, -1] = (left[zero] * right[zero] * self.pi).sum(-1)
                coefficients.append(coeff * scaledweight[..., None])
                shifts.append(shift.sum())
        return torch.stack(coefficients), torch.stack(shifts), branches

    def _values(self, coeff, shift, loglength):
        exponential = (loglength.exp()[:, None, None] * self.exponents[None]).exp()
        p = torch.einsum('nksd,nkd->ns', coeff, exponential)
        return p.clamp_min(1e-300).log().sum(-1) + shift

    @torch.no_grad()
    def scan(self, batch_edges=8, golden_steps=24, grid_points=15):
        if batch_edges < 1 or golden_steps < 0 or grid_points < 3:
            raise ValueError('invalid search budget')
        proposals = []
        max_baseline_error = 0.
        edges = self.internal_edges()
        grid = torch.linspace(math.log(1e-6), math.log(10.), grid_points, dtype=torch.float64, device=self.device)
        for start in range(0, len(edges), batch_edges):
            batch = edges[start:start + batch_edges]
            coeff, shift, branches = self.coefficients(batch)
            current = torch.tensor([self.tree.lengths[e] for e in batch], device=self.device, dtype=torch.float64).repeat_interleave(3).log()
            best_x = current.clone()
            best = self._values(coeff, shift, current)
            max_baseline_error = max(max_baseline_error, float((best[::3] - self.log_likelihood).abs().max()))
            values = torch.stack([self._values(coeff, shift, torch.full_like(current, x)) for x in grid], -1)
            idx = values.argmax(-1)
            gx = grid[idx]
            gv = values.gather(1, idx[:, None])[:, 0]
            improve = gv > best
            best_x, best = torch.where(improve, gx, best_x), torch.maximum(gv, best)
            lo, hi = grid[(idx - 1).clamp_min(0)], grid[(idx + 1).clamp_max(grid_points - 1)]
            ratio = (math.sqrt(5.) - 1) / 2
            x1, x2 = hi - ratio * (hi - lo), lo + ratio * (hi - lo)
            f1, f2 = self._values(coeff, shift, x1), self._values(coeff, shift, x2)
            for _ in range(golden_steps):
                for x, f in ((x1, f1), (x2, f2)):
                    improve = f > best
                    best_x, best = torch.where(improve, x, best_x), torch.maximum(f, best)
                right = f1 < f2
                lo, hi = torch.where(right, x1, lo), torch.where(right, hi, x2)
                x1, x2 = hi - ratio * (hi - lo), lo + ratio * (hi - lo)
                f1, f2 = self._values(coeff, shift, x1), self._values(coeff, shift, x2)
            for x, f in ((x1, f1), (x2, f2)):
                improve = f > best
                best_x, best = torch.where(improve, x, best_x), torch.maximum(f, best)
            scores, lengths = best.reshape(-1, 3).cpu().numpy(), best_x.exp().reshape(-1, 3).cpu().numpy()
            for e, nodes, ll, t in zip(batch, branches, scores, lengths):
                alt = int(np.argmax(ll))
                if alt and ll[alt] > max(ll[0], self.log_likelihood) + 1e-4:
                    proposals.append(dict(edge=e, branches=nodes, alternative=alt,
                        length=float(t[alt]), log_likelihood=float(ll[alt]),
                        gain=float(ll[alt] - self.log_likelihood), local_gain=float(ll[alt] - ll[0])))
        if max_baseline_error > max(1e-5, abs(self.log_likelihood) * 1e-10):
            raise RuntimeError(f'central likelihood disagrees with tree: {max_baseline_error}')
        return sorted(proposals, key=lambda x: (-x['gain'], x['edge'])), max_baseline_error


def apply_proposals(tree, proposals):
    out = tree.copy()
    for p in proposals:
        u, v = p['edge']
        a, b, c, d = p['branches']
        x = c if p['alternative'] == 1 else d
        lb, lx = out.lengths.pop(edge(u, b)), out.lengths.pop(edge(v, x))
        out.graph[u].remove(b); out.graph[b].remove(u)
        out.graph[v].remove(x); out.graph[x].remove(v)
        out.graph[u].add(x); out.graph[x].add(u)
        out.graph[v].add(b); out.graph[b].add(v)
        out.lengths[edge(u, x)], out.lengths[edge(v, b)] = lx, lb
        out.lengths[edge(u, v)] = p['length']
    return out


def independent_subset(tree, proposals):
    chosen, blocked = [], set()
    for p in proposals:
        u, v = p['edge']
        if u in blocked or v in blocked:
            continue
        chosen.append(p)
        blocked.update({u, v} | tree.graph[u] | tree.graph[v])
    return chosen


def fit_model(tree, names, states, previous=None, max_iter=256, device='cuda:0'):
    # Import only during scientific execution through the retained wrapper.
    from fitter import fit_reversible_reference_optimizer
    warm = {} if previous is None else dict(
        frequencies=previous['frequencies_acgt'],
        exchangeabilities=previous['exchangeabilities_ac_ag_at_cg_ct_gt'],
        alpha=previous['gamma_alpha'] or .5,
        pinv=max(previous['invariant_proportion'], 1e-7))
    return fit_reversible_reference_optimizer(tree, names, states, model='GTR+I+G4',
        device=device, max_iter=max_iter, precision='mixed', length_coordinate='sqrt',
        seed_method='parsimony' if previous is None else None,
        plan_backend='repeat-matrix', transition_backend='spectral-cuda',
        coordinate_metric='incumbent', stopping_policy='physical',
        model_backend='device', score_backend='analytic-cuda', optimizer_backend='lbfgsb',
        coupled_linear_solver='woodbury', **warm)


def refine(tree, names, states, *, passes=3, max_iter=256, device='cuda:0', callback=None, time_limit=180):
    if passes < 0 or max_iter < 1:
        raise ValueError('invalid refinement budget')
    started = time.perf_counter()
    tree, model = fit_model(tree, names, states, max_iter=max_iter, device=device)
    ll = independent_ll(tree, names, states, model, device)
    history = [dict(pass_index=0, log_likelihood=ll, seconds=time.perf_counter()-started,
                    fit=model, moves=0)]
    if callback:
        callback(tree, history[-1])
    for step in range(1, passes + 1):
        if time.perf_counter() - started > time_limit:
            raise TimeoutError('per-record refinement time cap')
        scanner = FullContextNNI(tree, names, states, model, device)
        if abs(scanner.log_likelihood - ll) > max(1e-5, abs(ll)*1e-10):
            raise RuntimeError('directed messages disagree with independent pruning')
        proposals, error = scanner.scan()
        chosen = independent_subset(tree, proposals)
        del scanner
        trials = []
        candidate, verified = tree, ll
        # Simultaneous moves need not be additive under global rate mixtures.
        # Only whole-tree verified increases are accepted. Shrink to best single.
        while chosen:
            proposed = apply_proposals(tree, chosen)
            value = independent_ll(proposed, names, states, model, device)
            trials.append(dict(moves=len(chosen), log_likelihood=value))
            if value > ll + 1e-5:
                candidate, verified = proposed, value
                break
            chosen = chosen[:len(chosen)//2]
        old_ll = ll
        refit = None
        refit_accepted = False
        if chosen:
            tree, ll = candidate, verified
            fitted, refit = fit_model(tree, names, states, model, max_iter=max_iter, device=device)
            fitted_ll = independent_ll(fitted, names, states, refit, device)
            refit_accepted = fitted_ll >= ll
            if refit_accepted:
                tree, model, ll = fitted, refit, fitted_ll
        item = dict(pass_index=step, moves=len(chosen), candidates=len(proposals),
            log_likelihood=ll, previous_log_likelihood=old_ll,
            seconds=time.perf_counter()-started, fixed_model_verified_ll=verified,
            baseline_max_error=error, batch_trials=trials,
            fit=refit, refit_accepted=refit_accepted,
            proposals=chosen)
        history.append(item)
        if ll < old_ll:
            raise AssertionError('accepted likelihood decreased')
        if callback:
            callback(tree, item)
    return tree, model, history
