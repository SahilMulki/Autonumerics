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
    """Point estimates and standard errors for moments one through four.

    The general ``se_v``, never the Gaussian ``v*sqrt(2/(M-1))``: GBM, CIR and
    Exp-OU are heavy-tailed and the Gaussian form badly understates the error there.
    The same argument decides the third and fourth moments, so their standard errors
    come from the estimator's **influence function** rather than from the normal-case
    ``sqrt(6/M)`` and ``sqrt(24/M)`` -- which those reduce to exactly on a Gaussian
    sample, and which are wrong by a large factor on a lognormal one.

    Moments 3 and 4 are **reported, never graded** (plan §11 D11). Censused across
    all seven live specs, ``analytic_moments`` carries only ``mean_*`` and
    ``variance_*`` spellings and the §15-17 surrogate ladder returns two moments, so
    there is no third-moment reference even in principle until a ``terminal_law`` is
    declared. The one formulator who thought about it wrote the target as prose, in
    ``sde_ornstein_uhlenbeck``'s note: *"sample skewness ~ 0 and excess kurtosis ~ 0
    within MC error is a free extra check the closed-form moments do not cover"*.
    """
    X = np.asarray(X, dtype=float).ravel()
    M = X.size
    m, v = float(np.mean(X)), float(np.var(X, ddof=1))
    se_m = float(np.std(X, ddof=1) / np.sqrt(M))
    centred = X - m
    mu4 = float(np.mean(centred ** 4))
    se_v = float(np.sqrt(max(mu4 - v ** 2, 0.0) / M))
    out = {"mean": m, "variance": v, "se_mean": se_m, "se_variance": se_v,
           "num_paths": M}
    out.update(_standardized(centred, M))
    return out


def _standardized(centred, M):
    """The third and fourth standardized central moments, with their own bars.

    ``g_r = mu_r / sigma^r`` has influence function
    ``z^r - g_r - r*g_{r-1}*z - (r/2)*g_r*(z^2 - 1)`` at ``z = (x - mu)/sigma``, and
    ``se(g_r) = sd(IF)/sqrt(M)``. On a Gaussian sample that is ``sqrt(6/M)`` for the
    skewness and ``sqrt(24/M)`` for the excess kurtosis -- the textbook values, which
    is the check that the algebra is right rather than a reason to use them.
    """
    sigma = float(np.sqrt(np.mean(centred ** 2)))
    if M < 4 or sigma <= 0.0:
        return {}
    z = centred / sigma
    g3 = float(np.mean(z ** 3))
    g4 = float(np.mean(z ** 4))
    if_3 = z ** 3 - g3 - 3.0 * z - 1.5 * g3 * (z ** 2 - 1.0)
    if_4 = z ** 4 - g4 - 4.0 * g3 * z - 2.0 * g4 * (z ** 2 - 1.0)
    return {"skewness": g3, "excess_kurtosis": g4 - 3.0,
            "se_skewness": float(np.std(if_3, ddof=1) / np.sqrt(M)),
            "se_excess_kurtosis": float(np.std(if_4, ddof=1) / np.sqrt(M))}


def cross_moment_stats(Xi, Xj):
    """``E[XY]``, the covariance and the correlation of one component pair, each
    with its own standard error.

    Finding 3: ``verification.cross_moment`` is declared with a ``tol`` and read by
    nothing, and ``sde_gbm_2d_high_corr``'s own note says in as many words that it is
    the only detector it has -- *"This is the ONLY place rho appears in any terminal
    statistic. A solver that drops the correlation entirely (independent increments)
    still passes both per-component moment checks"*.

    The correlation's bar comes from its influence function,
    ``u*v - 0.5*r*(u^2 + v^2)`` at the standardized ``u``, ``v``, for the reason
    ``moment_stats`` gives: the Gaussian ``(1 - r^2)/sqrt(M)`` understates the error
    on the lognormal marginals this check exists for.
    """
    Xi = np.asarray(Xi, dtype=float).ravel()
    Xj = np.asarray(Xj, dtype=float).ravel()
    M = min(Xi.size, Xj.size)
    Xi, Xj = Xi[:M], Xj[:M]
    mi, mj = float(np.mean(Xi)), float(np.mean(Xj))
    products = Xi * Xj
    centred = (Xi - mi) * (Xj - mj)
    cov = float(np.sum(centred) / max(M - 1, 1))
    si, sj = float(np.std(Xi, ddof=1)), float(np.std(Xj, ddof=1))
    out = {"cross_moment": float(np.mean(products)),
           "se_cross_moment": float(np.std(products, ddof=1) / np.sqrt(M)),
           "covariance": cov,
           "se_covariance": float(np.std(centred, ddof=1) / np.sqrt(M)),
           "num_paths": M}
    if si > 0.0 and sj > 0.0:
        r = cov / (si * sj)
        u, v = (Xi - mi) / si, (Xj - mj) / sj
        influence = u * v - 0.5 * r * (u ** 2 + v ** 2)
        out["correlation"] = float(r)
        out["se_correlation"] = float(np.std(influence, ddof=1) / np.sqrt(M))
    return out


def resolution(stats, v_ref, var_tol, ci_mult=2.0):
    """§14. Is the estimate sharp enough to decide at all?"""
    width = ci_mult * stats["se_variance"] / (abs(float(v_ref)) + 1e-14)
    return {"resolved": bool(width < 0.5 * float(var_tol)), "ci_width_rel": float(width),
            "bar": 0.5 * float(var_tol), "ci_mult": float(ci_mult)}


def compare_moment(empirical, se, reference, tol, *, near_zero=None, ci_mult=2.0,
                   abs_ci_mult=None):
    """``(outcome, detail)`` -- ``True`` / ``False`` / ``"unresolved"``."""
    reference = float(reference)
    rel = abs(float(empirical) - reference) / (abs(reference) + 1e-14)
    detail = {"empirical": float(empirical), "reference": reference, "rel_error": rel,
              "tol": float(tol), "se": float(se)}
    if near_zero is not None and abs(reference) < float(near_zero):
        # An |exact| below the near-zero threshold makes a relative error
        # meaningless; the spec says so, so compare absolutely against the CI.
        #
        # **And do not report the relative one.** It was left in `detail`, and
        # `_compare` folds every float `rel_error` into `worst_rel_error`, which is
        # the plan's `estimated_rel_error` and the ranker's accuracy input. Against
        # an exact mean of 0 the denominator is the 1e-14 guard: on N01
        # (2026-11-15) U's mean of -0.044 -- a pass, correctly judged absolutely --
        # became an estimated error of 4.4e12 on all four plans. The harness has
        # always left near-zero means out of its worst error (`verify.py`), and so
        # does this now; the absolute difference is reported in its place.
        detail.pop("rel_error")
        #
        # **At the family-wise multiplier.** This branch has no tolerance band: its
        # band is `max(k * se, near_zero)`, and `near_zero` (0.01) is far below the
        # standard error at any realistic path count, so it is a bare k-sigma test
        # that fails a correct solver ~4.6% of the time at k = 2 -- per zero-mean
        # component, and, since the graded call now shares the kernel's increments
        # across a pool, for every plan in the pool at once. Measured on N01
        # (2026-11-16): U's exact mean is 0, both plans returned 0.043 at 2.3 sigma,
        # both failed `moments_ok`, both were capped at 8 and the pool tied. The
        # relative branch below has a tolerance to clear and keeps `ci_mult`; only
        # this pure significance test is held at the family-wise level, which is
        # what `dynkin.family_wise_multiplier` already does for the same reason.
        diff = abs(float(empirical) - reference)
        k = float(abs_ci_mult if abs_ci_mult is not None else ci_mult)
        ok = diff < max(k * float(se), float(near_zero))
        return bool(ok), {**detail, "abs_error": diff, "near_zero": float(near_zero),
                          "mode": "absolute", "ci_mult_used": k}
    if ci_mult * float(se) / (abs(reference) + 1e-14) > float(tol):
        return "unresolved", {**detail, "reason": "the confidence interval is wider than "
                                                  "the tolerance band"}
    return bool(rel < float(tol)), detail
