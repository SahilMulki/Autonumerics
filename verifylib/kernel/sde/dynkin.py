"""Dynkin's identity: the reference-free residual, and D1's SDE counterpart.

``verification_manual.md`` §20. For any smooth ``phi``, with generator
``L phi = f phi' + 0.5 g^2 phi''``::

    E[phi(X_T)] = phi(x0) + E[ integral_0^T (L phi)(X_s) ds ]

available for **every** SDE, with no closed form, no surrogate and no assumptions.

The trapezoid path integral carries its own ``O(dt)`` bias, so the residual does
not go to zero at fixed ``dt`` -- the meaningful test is that it *shrinks at the
expected rate*. That sentence is where the PDE-side D1 slope test came from; this
is the original.

``L phi`` is formed in the sandbox child by central-differencing the test
function, because only expressions cross the process boundary. The differencing
error is ~1e-8 relative and, unlike the trapezoid bias, is independent of ``dt`` --
so it survives the Richardson extrapolation as a fixed offset. It sits five orders
below the Monte Carlo standard error at the path counts these specs use, which is
what makes it acceptable rather than merely small.
"""

from __future__ import annotations

import numpy as np

DEFAULT_TEST_FUNCTIONS = ("X", "X**2", "np.tanh(X)")


def family_wise_multiplier(ci_mult, n_functions):
    """Widen the per-test bar so ``K`` simultaneous tests keep the stated level.

    §20 tests several test functions and the check passes only if **all** of them
    are inside the interval. At ``ci_mult = 2`` that is a ~95% test per function,
    so requiring three of them to pass is a ~86% test overall -- one correct solver
    in seven fails, and the failure moves a Path-B score from 9 to 7. Bonferroni is
    the cheapest correction that fixes it: hold the family-wise level at what
    ``ci_mult`` states.

    Measured on a correct Euler-Maruyama OU solver at 40 000 paths across six
    seeds, the extrapolated residual lands between 0.2 and 3.2 standard errors of
    zero. The dominant term is the *sample mean of the martingale part*, which is a
    property of the seed rather than of the scheme, so the check's resolution is
    set by the path count and a residual false-failure rate survives this
    correction. It is reported honestly (``z`` is in the metrics) rather than tuned
    away, and ``ci_mult`` is a spec-controlled knob.
    """
    from math import erf, sqrt
    if n_functions <= 1:
        return float(ci_mult)
    # Per-test two-sided level implied by ci_mult, split across the family.
    alpha = 1.0 - erf(float(ci_mult) / sqrt(2.0))
    target = alpha / n_functions
    # Invert the normal tail by bisection: no scipy needed, and 60 steps is exact
    # to double precision over this range.
    lo, hi = float(ci_mult), 10.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if 1.0 - erf(mid / sqrt(2.0)) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def test_functions(spec):
    declared = (spec.get("verification") or {}).get("dynkin_test_functions")
    if isinstance(declared, list) and declared:
        return [e for e in declared if isinstance(e, str)]
    return list(DEFAULT_TEST_FUNCTIONS)


def observables_payload(spec, phis):
    """``{name: {...}}`` for :func:`verifylib.kernel.sandbox._make_observables`."""
    params = dict(spec.get("parameters") or {})
    drift, diffusion = spec.get("drift_expression"), spec.get("diffusion_expression")
    if not isinstance(drift, str) or not isinstance(diffusion, str):
        return None
    return {f"L{i}": {"kind": "generator", "expr": phi, "drift": drift,
                      "diffusion": diffusion, "params": params}
            for i, phi in enumerate(phis)}


def _phi_values(phis, X, params):
    from ...operator import evaluate
    X = np.asarray(X, dtype=float).reshape(len(X), -1)[:, 0]
    return [np.asarray(evaluate(phi, {**params, "X": X}), dtype=float) for phi in phis]


def per_path_residual(spec, phis, run, x0):
    """``phi(X_T) - phi(x0) - integral(L phi)`` **per path**, one array per test function.

    Per path rather than per mean, because the extrapolation and its confidence
    interval are both formed from these. ``r_extrap = 2*r_fine - r_coarse`` has a
    larger variance than ``r_fine`` and, under CRN, a strongly *correlated* one --
    so taking ``se(r_fine)`` as the error bar of the extrapolated value is wrong in
    both directions at once. Extrapolating path by path and taking the standard
    error of the result is exact and needs no covariance model.
    """
    params = dict(spec.get("parameters") or {})
    result = run["result"]
    integrals = result.get("path_integrals") or {}
    if not integrals:
        return None, {"reason": "the solver returned no `path_integrals`, so the Dynkin "
                                "identity has no right-hand side"}
    values = _phi_values(phis, result["terminal_paths"], params)
    phi0 = _phi_values(phis, np.asarray([float(x0)]), params)
    out = []
    for i, phi_T in enumerate(values):
        key = f"L{i}"
        if key not in integrals:
            return None, {"reason": f"the solver returned no path integral for {key}"}
        out.append(phi_T - float(phi0[i][0])
                   - np.asarray(integrals[key], dtype=float).ravel())
    return out, {}


def residual(spec, phis, run, x0):
    """The mean residual per test function, plus each one's standard error."""
    per_path, detail = per_path_residual(spec, phis, run, x0)
    if per_path is None:
        return None, detail
    return (np.asarray([float(np.mean(r)) for r in per_path]),
            {"se": [float(np.std(r, ddof=1) / np.sqrt(r.size)) for r in per_path]})


#: Band the measured residual order has to fall in to be believed. Below 0.5 the
#: residual is barely shrinking and Richardson is not the right tool; above 4 the
#: estimate is almost certainly noise in the denominator rather than a real rate.
ORDER_BAND = (0.5, 4.0)


def residual_order(means, diffs, ci_mult):
    """``(order, detail)`` for one test function's residual sequence, coarse to fine.

    The standard Richardson order estimate off three levels:
    ``r(dt) = r_inf + C dt^p`` gives ``(r0 - r1) / (r1 - r2) = 2^p``, so
    ``p = log2`` of that ratio. Estimated from the *differences* rather than from
    ``r0 / r1`` deliberately: the difference form tolerates a constant offset, and
    §20's central-differencing of ``L phi`` contributes exactly such an offset.

    Why three levels and not two: with two, fitting ``p`` and then extrapolating
    with it is circular. ``p = log2(r0/r1)`` gives ``2^p = r0/r1``, and
    ``r1 + (r1 - r0)/(2^p - 1)`` is then identically **0** for any input -- the
    "test" would pass everything. Measuring the rate and testing the limit need
    separate information, so the ladder carries a third level.

    ``None`` when the estimate cannot be believed: the finer difference is not
    separated from its own Monte-Carlo noise, the ratio is non-positive (a
    non-monotone sequence, so there is no rate to read), or ``p`` lands outside
    ``ORDER_BAND``. The caller then falls back to the ``O(dt)`` assumption, which is
    what this function replaced -- a fallback, not a silent default: it is recorded.
    """
    d0, d1 = diffs
    m0, m1 = float(np.mean(d0)), float(np.mean(d1))
    se1 = float(np.std(d1, ddof=1) / np.sqrt(d1.size)) if d1.size > 1 else 0.0
    detail = {"d_coarse": m0, "d_fine": m1, "se_d_fine": se1}
    if not np.isfinite(m0) or not np.isfinite(m1):
        return None, {**detail, "reason": "a non-finite residual difference"}
    if abs(m1) <= float(ci_mult) * se1:
        return None, {**detail, "reason": "the finer residual difference is inside its own "
                                         "Monte-Carlo noise, so no rate can be read from it"}
    ratio = m0 / m1
    if not np.isfinite(ratio) or ratio <= 0.0:
        return None, {**detail, "ratio": ratio,
                      "reason": "the residual differences do not share a sign, so the "
                                "sequence is not converging monotonically"}
    order = float(np.log2(ratio))
    if not ORDER_BAND[0] <= order <= ORDER_BAND[1]:
        return None, {**detail, "ratio": float(ratio), "order": order,
                      "reason": f"the measured order {order:.2f} is outside the believable "
                                f"band {ORDER_BAND}"}
    return order, {**detail, "ratio": float(ratio), "order": order}


def check(spec, phis, runs, x0, ci_mult=2.0):
    """``(outcome, detail)``. Richardson-extrapolate the residual to ``dt -> 0`` **at
    the order the residual is measured to converge at**, and require the
    extrapolated value inside the MC CI of zero.

    ``runs`` is the dt ladder, coarse to fine (at least two levels; three is what
    the driver sends and what a measured order needs).

    **Both runs must be driven by the same Brownian path** (§18). Measured on a
    correct Euler-Maruyama OU solver: with independent draws per level the residual
    at dt = 0.002 and dt = 0.001 extrapolated to -6.5e-3 against a 5.0e-3 bar and
    the check failed a correct scheme; with CRN, the same extrapolation lands at
    -3.9e-3 and passes. The difference is not bias, it is that ``r_fine -
    r_coarse`` between independent estimates is mostly noise, and Richardson
    amplifies it.

    **Why the order is measured rather than assumed.** This check used to hard-code
    ``r_extrap = 2 r_fine - r_coarse``, i.e. ``p = 1``, on the grounds that the
    trapezoid path integral carries an ``O(dt)`` bias. That is the bias of the
    *quadrature*; the residual as a whole converges at the weak order of the
    **scheme**, and for a weak-order-2 scheme the order-1 formula overshoots through
    zero and reports a large residual of the *opposite sign*. Measured on
    `sde_double_well_langevin` (2026-09-26), where it capped two correct plans at 3
    -- both with mean and variance errors inside the graded tolerances and a clean
    strong-order study, and with the residual *shrinking by 4x per halving*, which
    is the manual's own criterion ("it shrinks at the expected rate") passing while
    the implementation of it failed:

    ==================================  ==================  ==============  ====
    plan                                assumed p = 1        measured p      z
    ==================================  ==================  ==============  ====
    ``1-tamed-heun-drift-trapezoid``    +0.1220 at 25.6 s.e.  2.21  0.0076   1.7
    ``3-strang-splitting-exact-flows``  +0.0928 at 19.7 s.e.  1.96  0.0079   1.8
    ``2-drift-implicit-trapezoid``      -0.0024 at  0.5 s.e.  1.23  0.0034   1.3
    ==================================  ==================  ==============  ====

    The measured order comes out at ~2 for the two weak-order-2 schemes and all
    three plans are then clean, which is the right answer: the harness passes all
    three. The winner used to be the only one that passed, and only because its
    residual happened to already sit at the noise floor.
    """
    levels, detail_bad = [], None
    for run in runs:
        got, why = per_path_residual(spec, phis, run, x0)
        if got is None:
            detail_bad = detail_bad or why
        else:
            levels.append(got)
    if detail_bad is not None:
        return "unavailable", detail_bad
    if len(levels) < 2:
        return "unavailable", {"reason": "fewer than two usable dt levels"}
    shapes = {tuple(a.shape for a in lv) for lv in levels}
    if len(shapes) != 1:
        return "unavailable", {"reason": "the dt levels returned different path counts, "
                                         "so they cannot share a Brownian path"}

    n_phi = len(levels[0])
    bar = family_wise_multiplier(ci_mult, n_phi)
    extrapolated, orders, order_detail, sources = [], [], [], []
    for i in range(n_phi):
        seq = [lv[i] for lv in levels]                     # coarse -> fine, per path
        order = None
        if len(seq) >= 3:
            order, od = residual_order([np.mean(r) for r in seq],
                                       (seq[-3] - seq[-2], seq[-2] - seq[-1]), ci_mult)
            order_detail.append(od)
        else:
            order_detail.append({"reason": "only two dt levels; no order can be measured"})
        orders.append(order)
        sources.append("measured" if order is not None else "assumed_1")
        p = 1.0 if order is None else order
        # r_extrap = r_fine + (r_fine - r_mid) / (2^p - 1); at p = 1 this is exactly
        # the historical 2*r_fine - r_coarse, so a two-level ladder is unchanged.
        extrapolated.append(seq[-1] + (seq[-1] - seq[-2]) / (2.0 ** p - 1.0))

    means = np.asarray([float(np.mean(r)) for r in extrapolated])
    ses = np.asarray([float(np.std(r, ddof=1) / np.sqrt(r.size)) for r in extrapolated])
    inside = np.abs(means) < bar * ses
    return bool(np.all(inside)), {
        "residuals": [[float(np.mean(lv[i])) for lv in levels] for i in range(n_phi)],
        "residual_coarse": [float(np.mean(levels[-2][i])) for i in range(n_phi)],
        "residual_fine": [float(np.mean(levels[-1][i])) for i in range(n_phi)],
        "extrapolated": means.tolist(), "se": ses.tolist(),
        "z": (means / np.maximum(ses, 1e-300)).tolist(),
        "extrapolation_order": orders,
        "extrapolation_order_source": sources,
        "order_detail": order_detail,
        "levels": len(levels),
        "ci_mult": float(ci_mult), "family_wise_multiplier": float(bar),
        "test_functions": list(phis),
        "per_function_ok": [bool(a) for a in inside]}
