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


def check(spec, phis, coarse_run, fine_run, x0, ci_mult=2.0):
    """``(outcome, detail)``. The residual is ``O(dt)``, so Richardson-extrapolate
    it to ``dt -> 0`` and require the extrapolated value inside the MC CI of zero.

    **Both runs must be driven by the same Brownian path** (§18). Measured on a
    correct Euler-Maruyama OU solver: with independent draws per level the residual
    at dt = 0.002 and dt = 0.001 extrapolated to -6.5e-3 against a 5.0e-3 bar and
    the check failed a correct scheme; with CRN, the same extrapolation lands at
    -3.9e-3 and passes. The difference is not bias, it is that ``r_fine -
    r_coarse`` between independent estimates is mostly noise, and Richardson
    amplifies it.
    """
    coarse, detail_c = per_path_residual(spec, phis, coarse_run, x0)
    fine, detail_f = per_path_residual(spec, phis, fine_run, x0)
    if coarse is None or fine is None:
        return "unavailable", detail_c if coarse is None else detail_f
    if any(a.shape != b.shape for a, b in zip(coarse, fine, strict=True)):
        return "unavailable", {"reason": "the two dt levels returned different path "
                                         "counts, so they cannot share a Brownian path"}

    extrapolated = [2.0 * b - a for a, b in zip(coarse, fine, strict=True)]
    means = np.asarray([float(np.mean(r)) for r in extrapolated])
    ses = np.asarray([float(np.std(r, ddof=1) / np.sqrt(r.size)) for r in extrapolated])
    bar = family_wise_multiplier(ci_mult, len(extrapolated))
    inside = np.abs(means) < bar * ses
    return bool(np.all(inside)), {
        "residual_coarse": [float(np.mean(r)) for r in coarse],
        "residual_fine": [float(np.mean(r)) for r in fine],
        "extrapolated": means.tolist(), "se": ses.tolist(),
        "z": (means / np.maximum(ses, 1e-300)).tolist(),
        "ci_mult": float(ci_mult), "family_wise_multiplier": float(bar),
        "test_functions": list(phis),
        "per_function_ok": [bool(a) for a in inside]}
