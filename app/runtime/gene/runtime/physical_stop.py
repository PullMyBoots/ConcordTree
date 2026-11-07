"""Coordinate-invariant first-order fixed-topology ML stopping (per site).

The numerical target is frozen by PHYSICAL_STOP_BRIEF.md, not tuned per input.
This is a box-stationarity test, not a local/global optimum certificate.
"""
import numpy as np

INDEPENDENT_TARGET = 1e-5
INTERNAL_TARGET = 5e-6


def projected_log_score(vector, gradient, origin, scale, edges, coordinate, log_bounds):
    """Undo affine/sqrt conditioning and project in ORIGINAL log/model box.

    Input gradient is d(loss/site)/d(vector). s=2sqrt(t) implies
    d(loss)/d(log t) = d(loss)/ds * sqrt(t). Model log/logit coordinates
    are unchanged. Inactive directions must not be suppressed by a scale's
    transformed distance to a bound.
    """
    raw = origin + scale * vector
    score = np.asarray(gradient, dtype=np.float64).copy() / scale
    physical = raw.copy()
    if coordinate == "sqrt":
        score[:edges] *= raw[:edges] / 2
        physical[:edges] = 2 * np.log(raw[:edges] / 2)
    elif coordinate != "log":
        raise ValueError("coordinate must be log or sqrt")
    box = np.asarray(log_bounds)
    active = ((physical <= box[:, 0] + 1e-10) & (score > 0)) | (
        (physical >= box[:, 1] - 1e-10) & (score < 0))
    score[active] = 0.
    if not np.all(np.isfinite(score)) or not np.all(np.isfinite(physical)):
        raise RuntimeError("nonfinite physical score or point")
    return score


def physical_convergence(result, score_inf, callback_stopped, initial_stopped):
    """Status99 is custom success ONLY when our callback actually requested it.

    Never relabel an iteration cap or line-search failure as success, even if
    the last numerical point happens to have a small score.
    """
    acceptable = (bool(result.success) and result.status not in {1, 2, 99}) or (
        callback_stopped and result.status == 99)
    converged = bool(acceptable and score_inf <= INTERNAL_TARGET)
    if converged:
        reason = ("physical-initial" if initial_stopped else
                  "physical-callback" if callback_stopped else "physical-final")
    elif result.status == 1:
        reason = "iteration-or-evaluation-limit"
    elif result.status == 2:
        reason = "line-search-failure"
    elif result.status == 99:
        reason = "unverified-callback-stop"
    else:
        reason = "physical-target-not-met"
    return converged, reason
