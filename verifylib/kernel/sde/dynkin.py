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
    """``{name: {...}}`` for :func:`verifylib.kernel.sandbox._make_observables`, or
    ``None`` with a reason.

    Returns ``(payload, reason)``. The guard tests that the drift and diffusion
    **evaluate**, not that they are strings, which is [REV 8]'s correction and worth
    making even though the multi-dimensional generator is deferred.
    ``isinstance(drift, str)`` admits *prose*: ``sde_multichannel_stiff_m13``'s drift
    is ``"X @ F.T  (row-major paths, X shape (num_paths, 2), F = [[F11, F12], [F21,
    F22]])"``, a perfectly good ``str``, so a payload was built, three sandbox runs
    were spent, and each one crashed inside the child with *"operands could not be
    broadcast together with shapes (500,) (500,2)"*. The net outcome was already
    ``"unavailable"`` -- via a burned run per level and a reason that blamed the
    solver for the formulator's prose.
    """
    from ...operator import evaluate

    params = dict(spec.get("parameters") or {})
    declared = (spec.get("verification") or {}).get("generator")
    if isinstance(declared, dict):
        payload, why = _vector_payload(declared, phis, params, spec)
        if payload is not None or why is not None:
            return payload, why
    drift, diffusion = spec.get("drift_expression"), spec.get("diffusion_expression")
    for name, expr in (("drift_expression", drift), ("diffusion_expression", diffusion)):
        if not isinstance(expr, str):
            return None, (f"{name} is {type(expr).__name__}, not an expression the "
                          f"scalar generator can be built from (the multi-dimensional "
                          f"generator needs verification.generator.*)")
        probe = np.asarray([float(params.get("X_0", 1.0))] * 3, dtype=float)
        try:
            np.asarray(evaluate(expr, {**params, "X": probe}), dtype=float)
        except Exception as exc:  # noqa: BLE001 -- prose here is normal and expected
            return None, (f"{name} does not evaluate as an expression in X "
                          f"({type(exc).__name__}: {exc}); it is guidance prose, which "
                          f"schema.GUIDANCE_PATHS allows, so the generator cannot be "
                          f"built from it")
    return {f"L{i}": {"kind": "generator", "expr": phi, "drift": drift,
                      "diffusion": diffusion, "params": params}
            for i, phi in enumerate(phis)}, None


def _vector_payload(declared, phis, params, spec):
    """``({name: {...}}, None)`` or ``(None, reason)`` from
    ``verification.generator``.

    The multi-dimensional generator, 2b: ``L phi = sum_i f_i d_i phi + 0.5 sum_ij
    (G G^T)_ij d_ij phi``. ``drift`` is one expression per state component and
    ``diffusion_matrix`` is ``d`` rows of ``m`` expressions, each written in terms of
    the ``(num_paths, d)`` state array ``X``. The child builds the Hessian by central
    differences, which is ``1 + 2d + 2d(d-1)`` test-function evaluations -- 9 at
    ``d = 2``, against 3 for the scalar form. ``m`` enters only through ``G G^T``,
    which is ``d x d`` whatever ``m`` is, so 13 noise channels cost the same as 2.
    """
    drift = declared.get("drift")
    matrix = declared.get("diffusion_matrix")
    if not isinstance(drift, list) or not drift:
        return None, "verification.generator.drift is not a list of per-component expressions"
    if not (isinstance(matrix, list) and matrix
            and all(isinstance(row, list) and row for row in matrix)):
        return None, ("verification.generator.diffusion_matrix is not a list of rows of "
                      "expressions")
    if len(matrix) != len(drift):
        return None, (f"verification.generator declares {len(drift)} drift components "
                      f"and {len(matrix)} diffusion rows")
    widths = {len(row) for row in matrix}
    if len(widths) != 1:
        return None, f"verification.generator.diffusion_matrix rows have widths {widths}"
    return ({f"L{i}": {"kind": "generator_nd", "expr": phi, "drift": list(drift),
                       "diffusion_matrix": [list(row) for row in matrix],
                       "params": params, "state_dimension": len(drift)}
             for i, phi in enumerate(phis)}, None)


def _phi_values(phis, X, params):
    """``[phi(X) as (M,), ...]``, with the state in the shape the spec declared.

    [REV 11]: this opened with ``reshape(len(X), -1)[:, 0]``, the fourth site of
    Finding 11's component-0 reduction and the worst of them.
    ``sde_multichannel_stiff_m13`` declares ``"X[:, 0]**2 + X[:, 1]**2"`` as a test
    function, so a 1-D ``X`` does not evaluate half of it -- it raises, and the
    residual the caller then forms is not the identity the spec asked for.

    A test function must still return one value per path: the Dynkin identity pairs
    it with a single ``L phi`` path integral from the child. A vector-valued one is
    reported rather than silently reduced, which is what the deferred
    multi-dimensional generator (plan §3 2b) has to supply.
    """
    from ...operator import evaluate

    state = np.asarray(X, dtype=float).reshape(len(X), -1)
    namespace = state[:, 0] if state.shape[1] == 1 else state
    out = []
    for phi in phis:
        value = np.asarray(evaluate(phi, {**params, "X": namespace}), dtype=float)
        if value.ndim == 0:
            value = np.full(len(state), float(value))
        out.append(value.reshape(len(state), -1)[:, 0] if value.ndim > 1 else value)
    return out


#: Gauss-Hermite nodes for ``E[phi(X_0)]`` under a declared normal initial law. 40
#: nodes integrates a smooth ``phi`` against a Gaussian to double precision, and the
#: test functions in use are polynomials and ``tanh``.
_GH_NODES = 40


def initial_moments(spec, phis):
    """``(E[phi(X_0)] per test function, detail)``, or ``(None, reason detail)``.

    Finding 5. ``x0`` was read as ``parameters.get("X_0", 0.0)`` and the identity was
    formed with ``phi(x0)``, which assumes a **deterministic** initial state.
    ``sde_quintic_random_ic`` is ``dX = -X**5 dt`` with ``X(0) = xi ~ Normal(0,
    sigma_bar**2)`` and no ``X_0`` key at all, so ``x0`` silently became ``0.0`` and
    for ``phi = X**2`` the identity used ``phi(0) = 0`` where it needs
    ``E[phi(X_0)] = sigma_bar**2 = 0.1111``. The residual then carried a fixed
    -0.111 offset at every level, survived the Richardson extrapolation, and landed
    ``rubric_sde``'s 3 row -- *"the wrong equation integrated correctly"* -- on a
    correct solver.

    Three cases, and the third is the one that matters:

    * a numeric ``X_0`` (or ``x0``): deterministic, ``phi(x0)``, exactly as before;
    * a declared ``initial_condition`` law: ``E[phi(X_0)]`` by quadrature;
    * **neither**: ``None``, so the caller reports ``"unavailable"`` with the random
      initial condition as its stated reason. Never a failure -- the kernel cannot
      form the identity, which is a property of what the spec declares, not of the
      solver.
    """
    params = dict(spec.get("parameters") or {})
    declared = spec.get("initial_condition")
    law = (declared.get("law") if isinstance(declared, dict) else None)
    if isinstance(law, str) and law != "deterministic":
        return _law_moments(spec, phis, declared, params)
    value = _initial_state(spec, declared, params)
    if value is None:
        return None, {"reason": "no deterministic initial value is declared and no "
                                "initial_condition law is given, so E[phi(X_0)] cannot "
                                "be formed. The Dynkin identity needs the initial "
                                "state's law, not a default of zero -- declare "
                                "initial_condition: {law, ...} or parameters.X_0",
                      "random_initial_condition": True}
    state = np.asarray(value, dtype=float).reshape(1, -1)
    if state.shape[1] == 1:
        state = state.reshape(1)
    return ([float(v[0]) for v in _phi_values(phis, state, params)],
            {"initial_condition": "deterministic",
             "x0": [float(v) for v in np.ravel(value)]})


def _initial_state(spec, declared, params):
    """The deterministic initial state, as a scalar or one value per component.

    The ``X_0`` / ``Y_0`` / ``Z_0`` spelling, which is how every live vector spec
    writes it -- the same shape ``driver._claimed`` reads for the moments. A declared
    ``initial_condition.value`` overrides it, and may itself be a list.
    """
    if isinstance(declared, dict) and declared.get("value") is not None:
        value = declared["value"]
        if isinstance(value, list) and value:
            return [float(v) for v in value]
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        return None
    dimension = spec.get("state_dimension")
    if isinstance(dimension, int) and not isinstance(dimension, bool) and dimension > 1:
        per = [params.get(f"{c}_0") for c in ("X", "Y", "Z")[:dimension]]
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in per):
            return [float(v) for v in per]
        return None
    value = params.get("X_0", params.get("x0"))
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _law_moments(spec, phis, declared, params):
    """``E[phi(X_0)]`` under a declared initial law, by quadrature.

    One law for now, ``normal``, because that is the only one any spec in the set
    needs and because guessing at the rest would be the inference this repo refuses.
    An undeclared law is reported, not assumed.
    """
    law = declared.get("law")
    if law != "normal":
        return None, {"reason": f"initial_condition.law {law!r} is not one the kernel "
                                f"can integrate against (known: 'normal', "
                                f"'deterministic')"}
    try:
        mean = float(evaluate_number(declared.get("mean", 0.0), params))
        variance = float(evaluate_number(declared.get("variance"), params))
    except Exception as exc:  # noqa: BLE001
        return None, {"reason": f"initial_condition's mean/variance did not evaluate: "
                                f"{type(exc).__name__}: {exc}"}
    if not variance > 0.0:
        return None, {"reason": f"initial_condition.variance evaluated to {variance}"}
    nodes, weights = np.polynomial.hermite_e.hermegauss(_GH_NODES)
    sample = mean + np.sqrt(variance) * nodes
    weights = weights / float(np.sum(weights))
    values = _phi_values(phis, sample, params)
    return ([float(np.dot(weights, v)) for v in values],
            {"initial_condition": "normal", "mean": mean, "variance": variance,
             "quadrature_nodes": _GH_NODES})


def evaluate_number(value, params):
    """A declared scalar that may be a literal or an expression in the parameters."""
    from ...operator import evaluate

    if isinstance(value, str):
        return float(evaluate(value, params))
    return float(value)


def per_path_residual(spec, phis, run, phi0):
    """``phi(X_T) - E[phi(X_0)] - integral(L phi)`` **per path**, one array per test
    function.

    ``phi0`` is a list of ``E[phi(X_0)]`` values from :func:`initial_moments`, one
    per test function, rather than an ``x0`` this function evaluates itself: on a
    random initial condition there is no single ``x0`` to evaluate at (Finding 5).

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
    out = []
    for i, phi_T in enumerate(values):
        key = f"L{i}"
        if key not in integrals:
            return None, {"reason": f"the solver returned no path integral for {key}"}
        out.append(phi_T - float(phi0[i])
                   - np.asarray(integrals[key], dtype=float).ravel())
    return out, {}


def residual(spec, phis, run, phi0):
    """The mean residual per test function, plus each one's standard error."""
    per_path, detail = per_path_residual(spec, phis, run, phi0)
    if per_path is None:
        return None, detail
    return (np.asarray([float(np.mean(r)) for r in per_path]),
            {"se": [float(np.std(r, ddof=1) / np.sqrt(r.size)) for r in per_path]})


#: Band the measured residual order has to fall in to be believed. Below 0.5 the
#: residual is barely shrinking and Richardson is not the right tool; above 4 the
#: estimate is almost certainly noise in the denominator rather than a real rate.
#:
#: The two edges fail for opposite reasons and are reported separately: an order
#: *above* the band is noise, so the ``O(dt)`` fallback is the safer guess, while an
#: order *below* it is a real rate the extrapolation cannot use -- the residual is
#: converging and the ladder cannot reach the limit. The second case carries
#: ``unresolved`` and ``check`` abstains on it rather than reporting the leftover
#: bias as a generator violation. See ``check``'s third measured case.
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

    ``None`` when the estimate cannot be believed, in four distinct cases, each with
    its own recorded reason: the finer difference is not separated from its own
    Monte-Carlo noise; the ratio is non-positive (a non-monotone sequence, so there
    is no rate to read); ``p`` is above ``ORDER_BAND`` (noise in the denominator);
    or ``p`` is below it. The caller falls back to the ``O(dt)`` assumption in all
    four -- a fallback, not a silent default: it is recorded.

    The last case is the only one that also sets ``unresolved``, and only when the
    residual is **shrinking** across the levels. That combination -- real,
    same-signed differences, a shrinking residual, and a rate too low for three
    levels to locate a limit -- is a statement about the ladder rather than about
    the solver, and ``check`` abstains on it. The distinction matters because the
    other three cases are what a defective solver produces: a solver that never
    integrates the SDE has a ``dt``-independent residual, so its differences are
    noise and the *first* guard catches it. See ``check``'s third measured case.
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
    if order < ORDER_BAND[0]:
        # The differences are real -- outside their own noise, and same-signed --
        # and they are not shrinking fast enough for three levels to locate a
        # limit. Two sub-cases, and only one of them is an abstention: the residual
        # itself has to be *shrinking*. A residual that grows under refinement is a
        # diverging scheme, which is evidence, so it keeps the `O(dt)` fallback and
        # the hard verdict that comes with it.
        shrinking = all(abs(means[i + 1]) < abs(means[i])
                        for i in range(len(means) - 1))
        if not shrinking:
            return None, {**detail, "ratio": float(ratio), "order": order,
                          "reason": f"the measured order {order:.2f} is below the "
                                    f"believable band {ORDER_BAND} and the residual is "
                                    f"not shrinking under refinement, which is a "
                                    f"diverging sequence rather than a slow one"}
        # Distinct from every other `None` here, which is why it is flagged: those
        # say "no rate can be read" and fall back to a verdict, this one says the
        # limit is not reachable from this ladder at all.
        return None, {**detail, "ratio": float(ratio), "order": order,
                      "unresolved": True, "residual_means": [float(v) for v in means],
                      "reason": f"the residual is shrinking under refinement -- "
                                f"{abs(means[0]):.4g} to {abs(means[-1]):.4g} over "
                                f"{len(means)} levels -- but the rate the ladder reads "
                                f"from it, {order:.2f}, is below the believable band "
                                f"{ORDER_BAND}, so this sequence is not in an "
                                f"asymptotic regime and no extrapolation on it reaches "
                                f"the limit: the leftover is the scheme's own bias, not "
                                f"evidence about the generator"}
    if order > ORDER_BAND[1]:
        return None, {**detail, "ratio": float(ratio), "order": order,
                      "reason": f"the measured order {order:.2f} is above the believable "
                                f"band {ORDER_BAND}, which is noise in the denominator "
                                f"rather than a real rate"}
    return order, {**detail, "ratio": float(ratio), "order": order}


def check(spec, phis, runs, phi0, ci_mult=2.0):
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

    **And why a *low* measured order abstains rather than fails.** The same
    hard-coded ``p = 1`` has a mirror-image failure: where the residual converges
    *slower* than ``O(dt)`` the formula under-corrects, and the leftover -- which is
    the scheme's own bias, not a generator violation -- is reported as one. Measured
    on `sde_cir_feller_violated/2-milstein` (2026-10-15), a full-truncation Milstein
    on a CIR whose Feller condition is violated (``2 kappa theta = 1.0`` against
    ``sigma^2 = 4.0``, so ``sqrt(X)`` is non-Lipschitz at an origin the process
    reaches):

    ===============================  ===========  ===========  ===========  ======
    quantity                         dt = 0.005   dt = 0.0025  dt = 0.00125  exact
    ===============================  ===========  ===========  ===========  ======
    ``E[tanh(X_T)]``                 0.296        0.291        0.290        0.280
    Dynkin residual, same function   0.0289       0.0246       0.0205       0
    ===============================  ===========  ===========  ===========  ======

    The residual is shrinking, monotonically and well outside its own noise -- the
    differences are 11 s.e. from zero -- but the rate the ladder reads off them is
    **0.07**, and on ``X`` it is **-0.05**: the differences are flat, so the
    sequence is not in an asymptotic regime at all and three levels spanning a 4x
    ``dt`` range cannot locate the limit. The limit does exist. Against the exact
    law the weak error of ``E[tanh(X_T)]`` falls from 0.0161 to 0.0061 over a 64x
    refinement, i.e. order ~0.2, which is the known rate for a truncated scheme on
    a Feller-violated CIR. Extrapolating at ``p = 1`` left +0.0164 against a 0.0032
    s.e. -- **z = 5.1** -- and capped a solver that converges to the exact
    noncentral-chi-squared law at 3.

    A *growing* residual is the opposite case and keeps its hard verdict: the
    abstention requires the sequence to be shrinking, so a diverging scheme is
    still evidence rather than an abstention.

    Extrapolating at the *measured* 0.07 is not the fix: ``2^p - 1 = 0.046``
    multiplies the correction, and its noise, by ~22, so as ``p -> 0`` everything
    passes and ``ORDER_BAND``'s lower edge exists precisely to stop that. The
    outcome is ``"unresolved"`` instead -- the ladder cannot reach the limit here,
    which is a statement about the ladder.

    This cannot reopen Finding 1's hole, because the three failure modes land in
    **different branches** of ``residual_order``:

    =============================  =========================  ===================
    solver                         residual vs dt             branch
    =============================  =========================  ===================
    fabricated sampler             dt-independent             inside its own noise
    `ginzburg_landau_s6` 1 and 2   order 1.20 / 1.36          measured, fails
    this CIR plan                  order 0.07, shrinking      **unresolved**
    =============================  =========================  ===================

    A solver that does not integrate the SDE has a residual that does not move with
    ``dt``, so the difference between levels is noise and the first guard in
    ``residual_order`` catches it before any rate is read.
    """
    if phi0 is None or (isinstance(phi0, dict) and phi0.get("reason")):
        # Finding 5: the identity cannot be formed, which is a property of what the
        # spec declares. `"unavailable"` with the reason, never `False`.
        return "unavailable", (phi0 if isinstance(phi0, dict)
                               else {"reason": "no initial-state moments"})
    if isinstance(phi0, (int, float)):
        # The old positional contract: a bare deterministic x0.
        phi0 = [float(v[0]) for v in _phi_values(phis, np.asarray([float(phi0)]),
                                                 dict(spec.get("parameters") or {}))]
    levels, detail_bad = [], None
    for run in runs:
        got, why = per_path_residual(spec, phis, run, phi0)
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
    # A failure the extrapolation could not resolve is not a verdict. A *resolved*
    # failure on any one function outranks it: an unreachable limit on `tanh(X)`
    # says nothing about a real violation on `X**2`.
    failing = [i for i in range(n_phi) if not inside[i]]
    slow = [i for i in failing if order_detail[i].get("unresolved")]
    outcome = (True if not failing
               else "unresolved" if len(slow) == len(failing) else False)
    unresolved_detail = ({"unresolved_functions": [list(phis)[i] for i in slow],
                          "reason": order_detail[slow[0]]["reason"]}
                         if outcome == "unresolved" else {})
    return outcome, {**unresolved_detail,
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
