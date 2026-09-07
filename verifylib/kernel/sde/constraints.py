"""§21's structural checks, and §14's mandatory confidence intervals.

The confidence interval is not an ornament. Every comparison -- to a reference, to
another plan, to a threshold -- must be made against one, because without it you
cannot distinguish "biased" from "noisy", and with no reference that distinction
is load-bearing. A point estimate landing inside a threshold band proves nothing
if its own error bar is wider than the band, which is why every check here returns
**pass / fail / inconclusive** rather than a boolean.

``inconclusive`` gets its own score (6). Reporting it as a pass, or as a solver
bug, are both wrong: the fix is more paths, and the feedback must say exactly that.
"""

from __future__ import annotations

import numpy as np


def moment_stats(X):
    """Point estimates and standard errors for the mean and the variance.

    The general ``se_v``, never the Gaussian ``v*sqrt(2/(M-1))``: GBM, CIR and
    Exp-OU are heavy-tailed and the Gaussian form badly understates the error there.
    """
    X = np.asarray(X, dtype=float).ravel()
    M = X.size
    m, v = float(np.mean(X)), float(np.var(X, ddof=1))
    se_m = float(np.std(X, ddof=1) / np.sqrt(M))
    mu4 = float(np.mean((X - m) ** 4))
    se_v = float(np.sqrt(max(mu4 - v ** 2, 0.0) / M))
    return {"mean": m, "variance": v, "se_mean": se_m, "se_variance": se_v, "num_paths": M}


def resolution(stats, v_ref, var_tol, ci_mult=2.0):
    """§14. Is the estimate sharp enough to decide at all?"""
    width = ci_mult * stats["se_variance"] / (abs(float(v_ref)) + 1e-14)
    return {"resolved": bool(width < 0.5 * float(var_tol)), "ci_width_rel": float(width),
            "bar": 0.5 * float(var_tol), "ci_mult": float(ci_mult)}


def compare_moment(empirical, se, reference, tol, *, near_zero=None, ci_mult=2.0):
    """``(outcome, detail)`` -- ``True`` / ``False`` / ``"unresolved"``."""
    reference = float(reference)
    rel = abs(float(empirical) - reference) / (abs(reference) + 1e-14)
    detail = {"empirical": float(empirical), "reference": reference, "rel_error": rel,
              "tol": float(tol), "se": float(se)}
    if near_zero is not None and abs(reference) < float(near_zero):
        # An |exact| below the near-zero threshold makes a relative error
        # meaningless; the spec says so, so compare absolutely against the CI.
        ok = abs(float(empirical) - reference) < max(ci_mult * float(se), float(near_zero))
        return bool(ok), {**detail, "near_zero": float(near_zero), "mode": "absolute"}
    if ci_mult * float(se) / (abs(reference) + 1e-14) > float(tol):
        return "unresolved", {**detail, "reason": "the confidence interval is wider than "
                                                  "the tolerance band"}
    return bool(rel < float(tol)), detail
