"""The analytic-reference check: is the claimed closed form actually a solution?

This is the mechanism behind the guardrails plan's central rule (§4b):

    An analytic_solution enters Tier A when it is VALIDATED against the problem's
    own declared structure. Where validation cannot run, a verbatim quote from
    problem.md is accepted in its place and the gap is recorded. Otherwise the
    field is null.

Three tests, all on the *formula*, none on the solver:

  1. term-balanced operator residual  -- wrong PDE / coefficient / sign
  2. formula at t = 0 vs initial_condition -- right PDE, wrong branch or mode
  3. formula on the boundary vs boundary_conditions -- wrong BC selection

Test 1 is the only one that reaches outside the spec; 2 and 3 are consistency
checks. All three are needed: on pde_advection_1d, widening the Gaussian gives a
*different but still exact* solution, so the residual stays small and only test 2
catches it.

Evaluating a closed form is nearly free, which is what makes this better calibrated
than a residual on solver output: put the formula on a fine grid, drive the check's
own discretization error to 1e-10, and use a tight tolerance. A solver cannot do
that, which is why plan-no-closed-form's D1 must gate rather than certify.
"""

from __future__ import annotations

import numpy as np

from .findings import Finding, error, warning
from .operator import (
    STENCIL_HALO,
    build_namespace,
    coord_aliases,
    evaluate,
    normalize_helpers,
    split_terms,
)

#: median term-balanced residual below which a formula is accepted.
#: Measured headroom: correct-and-resolved formulas land at <= 4.8e-09, every
#: wrong formula at >= 2.5e-01. Seven orders; this sits in the gap.
TOL = 1e-6

#: probe resolutions, coarse first. A formula whose median residual is above TOL
#: at the coarse grid but falls at the fine one was under-resolved, not wrong.
PROBE_N = (257, 513, 1025)

#: Test 2 thresholds. Below EXACT the two transcriptions agree; above WRONG they
#: describe different functions. Measured basis: a doubled mode or a widened
#: Gaussian differs by O(1), while pde_advection_1d's genuinely non-periodic
#: initial_condition differs from its periodically-wrapped solution by 1.2e-04 --
#: real, benign, and two orders below the problem's own 1e-2 accuracy target. That
#: middle band is reported, not failed.
IC_EXACT_TOL = 1e-6
IC_WRONG_TOL = 1e-3

#: outcomes, in the order of §4g. The blind-evaluator ranker maps `validated` and
#: `validated_off_singularity` to provenance `analytic`, and `quoted` /
#: `unavailable` / `unvalidated_at_feature` to `analytic_unvalidated` (§14 C4).
#:
#: ``unvalidated_at_feature`` (findings F5 (2)): the median residual passed but
#: the set above tolerance is either more than :data:`OFF_SINGULARITY_MAX_FRAC` of
#: the domain or the problem declares a ``layer`` / ``shock`` in
#: ``verification.structural_facts``. A closed form accepted everywhere except
#: where the problem is hard is A- evidence, not A. Measured on
#: ``pde_burgers_viscous_1d``: median 7e-9, max 2.2 over 14.9% of the domain at the
#: layer the problem exists to test.
OUTCOMES = ("validated", "validated_off_singularity", "unvalidated_at_feature", "quoted",
            "unavailable", "failed")

#: Above this fraction of the domain **at the finest probe**, "off a localized
#: singularity" is no longer localized. Measured 2026-09-20 on the two staged
#: cases: the fraction above tol shrinks with the probe on both -- Black-Scholes'
#: kink 23% -> 12% -> 8.5% at N = 257/513/1025, Burgers' layer 14.9% -> 7.3% ->
#: 5.4% -- so it is the same stencil-footprint effect on both and a fixed-level
#: fraction cannot tell an exact formula at a kink from one at a layer (the
#: Burgers formula agrees with the harness's own to 2.2e-14, [rev2]). The
#: fraction rule therefore fires only on a gross, non-shrinking set; the declared
#: feature rule (``FEATURE_KEYS``) is what carries "waived exactly where the
#: problem is hard".
OFF_SINGULARITY_MAX_FRAC = 0.10

#: ``structural_facts`` keys that name a feature the residual check must not be
#: waived at.
FEATURE_KEYS = ("layer", "shock")

#: Constructs that make a closed form a *numerical* closed form (findings F5 (1)).
#: A quadrature or a series written as an expression is a legitimate route to a
#: reference -- the harness's own ``_heston_call`` and Mittag-Leffler series are
#: exactly that -- and it is flagged, never demoted: the formulator's Cole-Hopf
#: quadrature on 4001 nodes agrees with the harness's own to 2.2e-14. What is
#: recorded is that "exact" rests on a discretization, and how many nodes it has.
NUMERICAL_CONSTRUCTORS = ("linspace", "arange", "geomspace", "logspace")
NUMERICAL_REDUCERS = ("sum", "mean", "trapz", "trapezoid", "cumsum", "dot", "prod")


def numerical_closed_form(expr):
    """``(flag, detail)``: does this expression discretize something?

    Flags a ``lambda``, or a constructed axis (``np.linspace`` and friends)
    reduced over (``np.sum`` / ``mean`` / ``trapezoid`` ...). ``detail["nodes"]``
    lists the literal node counts of the constructed axes.
    """
    import ast
    if not isinstance(expr, str):
        return False, {}
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return False, {}
    lambdas, constructors, reducers, nodes = 0, [], [], []
    for node in ast.walk(tree):
        if isinstance(node, ast.Lambda):
            lambdas += 1
        elif isinstance(node, ast.Call):
            name = (node.func.attr if isinstance(node.func, ast.Attribute)
                    else node.func.id if isinstance(node.func, ast.Name) else None)
            if name in NUMERICAL_CONSTRUCTORS:
                constructors.append(name)
                for arg in node.args[2:3] if name != "arange" else ():
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, (int, float)):
                        nodes.append(int(arg.value))
            elif name in NUMERICAL_REDUCERS:
                reducers.append(name)
    flag = bool(lambdas or (constructors and reducers))
    if not flag:
        return False, {}
    return True, {"lambdas": lambdas, "constructors": constructors, "reducers": reducers,
                  "nodes": nodes,
                  "label": (f"{constructors[0]}({max(nodes)})" if constructors and nodes
                            else "lambda" if lambdas else constructors[0] if constructors
                            else "reduction")}


# --- what operator to check against (§4d) ------------------------------------

def resolve_operator(spec):
    """Return ``(operator, provenance)`` -- the residual form to check against.

    ``operator`` is one of::

        {"kind": "scalar", "terms": {name: expr}, "source": expr or None}
        {"kind": "system", "equations": {field: {"terms": ..., "source": ...}},
         "combine": "rms"}

    or ``None``, with ``provenance`` then carrying the reason. Four sources, in
    descending order of what they can express:

    1. ``verification.operator`` -- the field this plan defines (§4d). Carries a
       source term and a per-equation system form, which is why it exists.
    2. ``verification.residual_operator`` -- already source-aware: 11 specs carry
       forms like ``"-lap_u - f"``, and ``f`` binds to the top-level
       ``source_term``, which is a real evaluable expression in every spec that
       has one.
    3. ``mms_probe.operator_check`` -- the homogeneous operator, which 19 of 22
       PDE specs carry. When the spec also declares ``source_term``, it is
       subtracted, because a claimed solution of an inhomogeneous problem does not
       satisfy the homogeneous operator.
    4. nothing.

    Routes 2 and 3 are the legacy path. They work better than the plan's rev-3
    estimate: ``source_term`` turned out to be a real field rather than prose
    inside ``governing_equation``, so Poisson, Helmholtz, Monge-Ampere and
    Cahn-Hilliard are all checkable without any spec change.
    """
    verification = spec.get("verification") or {}

    declared = verification.get("operator")
    if isinstance(declared, dict):
        if "equations" in declared:
            return {"kind": "system", "equations": declared["equations"],
                    "combine": declared.get("combine", "rms")}, "verification.operator"
        return {"kind": "scalar", "terms": declared.get("terms") or {},
                "source": declared.get("source")}, "verification.operator"
    if declared is None and "operator" in verification:
        return None, verification.get("operator_note") or "operator declared null"

    for key, subtract_source in (("residual_operator", False),
                                 (("mms_probe", "operator_check"), True)):
        raw = (verification.get(key[0], {}) or {}).get(key[1]) if isinstance(key, tuple) \
            else verification.get(key)
        if not isinstance(raw, str) or not raw.strip():
            continue
        terms = {f"t{i}": src if sign > 0 else f"-({src})"
                 for i, (sign, src) in enumerate(split_terms(normalize_helpers(raw)))}
        source = spec.get("source_term") if subtract_source else None
        name = key[0] + "." + key[1] if isinstance(key, tuple) else key
        return {"kind": "scalar", "terms": terms, "source": source}, f"verification.{name}"

    return None, "no operator declaration and no legacy operator string"


# --- evaluating the claimed solution on a grid -------------------------------

def _grid(spec, N):
    sv = spec["spatial_variables"]
    bounds = spec["domain"]["bounds"]
    axes = [np.linspace(*bounds[v], N) for v in sv]
    hs = [float(a[1] - a[0]) for a in axes]
    coords = coord_aliases(dict(zip(sv, np.meshgrid(*axes, indexing="ij"), strict=True)))
    return sv, coords, hs, axes


def claimed_fields(spec):
    """``{field_name: expression}`` for the claimed solution, or ``{}``."""
    sol = spec.get("analytic_solution")
    if not isinstance(sol, dict):
        return {}
    if isinstance(sol.get("fields"), dict):
        return dict(sol["fields"])
    if isinstance(sol.get("expression"), str):
        return {"u": sol["expression"]}
    return {}


def _t_probe(spec):
    """The time to evaluate at. ``t0`` when declared -- a similarity solution is
    singular at t = 0 and states its initial data at t0 instead."""
    params = spec.get("parameters") or {}
    return float(params.get("t_final", params.get("T", 0.5)))


def _sample(spec, exprs, coords, sv, t, ht):
    """Evaluate every claimed field at ``t``, plus its first two time derivatives."""
    params = dict(spec.get("parameters") or {})
    ones = np.ones_like(coords[sv[0]])

    def at(expr, tt):
        return np.asarray(evaluate(expr, {**params, **coords, "t": tt}), dtype=float) * ones

    fields = {name: at(e, t) for name, e in exprs.items()}
    derivs = {}
    if spec.get("time_dependent", True):
        for name, e in exprs.items():
            f = [at(e, t + k * ht) for k in (-2, -1, 1, 2)]
            u = fields[name]
            derivs[name] = (
                (-f[3] + 8 * f[2] - 8 * f[1] + f[0]) / (12 * ht),
                (-f[3] + 16 * f[2] - 30 * u + 16 * f[1] - f[0]) / (12 * ht * ht),
            )
    return fields, derivs


def _interior(shape, nd):
    mask = np.zeros(shape, dtype=bool)
    mask[tuple([slice(STENCIL_HALO, -STENCIL_HALO)] * nd)] = True
    return mask


def _scaled_residual(terms, source, ns, sv, hs, fields, stencil=None):
    """``|sum(terms) - source| / max_k |term_k|`` -- term-balanced, never absolute.

    Helmholtz at large k and 100:1 anisotropy both have individually huge terms
    that cancel; an absolute tolerance rejects correct formulas on both.
    """
    ones = np.ones_like(next(iter(fields.values())))
    vals = [np.asarray(evaluate(src, ns), dtype=float) * ones for src in terms.values()]
    total = sum(vals)
    if source is not None:
        src_val = np.asarray(evaluate(source, ns), dtype=float) * ones
        total = total - src_val
        vals = [*vals, src_val]

    if len(vals) >= 2:
        den = np.maximum.reduce([np.abs(v) for v in vals])
    else:
        # A single-term operator makes residual/max|term| identically 1, so Laplace
        # fails on a correct harmonic solution. Decompose one level down.
        from .operator import default_stencil
        st = stencil or default_stencil()
        u = next(iter(fields.values()))
        den = np.maximum.reduce([np.abs(st.d2(u, i, hs[i])) for i in range(len(sv))])
    return np.abs(total) / np.maximum(den, 1e-300)


def residual_field(spec, operator, N=257, ht=1e-3, t=None, exprs=None, stencil=None):
    """Pointwise term-balanced residual of the claimed solution. ``(scaled, mask)``.

    ``operator`` is either a resolved dict from :func:`resolve_operator` or a bare
    operator string, which is accepted so the legacy call form keeps working.
    """
    if isinstance(operator, str):
        operator = resolve_operator({"verification": {"residual_operator": operator}})[0]
    sv, coords, hs, _ = _grid(spec, N)
    exprs = exprs or claimed_fields(spec)
    if not exprs:
        raise ValueError("no analytic_solution to check")
    t = _t_probe(spec) if t is None else t

    fields, derivs = _sample(spec, exprs, coords, sv, t, ht)
    ns = build_namespace(fields, coords, sv, hs, time_derivs=derivs, stencil=stencil)
    ns.update(spec.get("parameters") or {})
    ns["t"] = t  # a source term is a function of space *and* time
    if spec.get("source_term") is not None:
        ns["f"] = np.asarray(evaluate(spec["source_term"], ns), dtype=float)

    if operator["kind"] == "scalar":
        scaled = _scaled_residual(operator["terms"], operator.get("source"),
                                  ns, sv, hs, fields, stencil)
    else:
        per_eq = [
            _scaled_residual(eq.get("terms") or {}, eq.get("source"), ns, sv, hs, fields,
                             stencil)
            for eq in operator["equations"].values()
        ]
        stack = np.stack(per_eq)
        scaled = (np.sqrt(np.mean(stack ** 2, axis=0)) if operator.get("combine", "rms") == "rms"
                  else np.max(stack, axis=0))
    return scaled, _interior(scaled.shape, len(sv))


def _stats(scaled, mask):
    s = scaled[mask]
    return {"median": float(np.median(s)), "q95": float(np.quantile(s, 0.95)),
            "max": float(np.max(s)), "frac_above_tol": float(np.mean(s > TOL))}


def check_operator_residual(spec, operator, tol=TOL, exprs=None, stencil=None):
    """Test 1. Classify a claimed solution. Returns ``(outcome, detail)``.

    The gate is the **median**, not the max. An earlier design gated on the max
    plus a "singular set < 2% of the domain" threshold; measured, that bar is
    fragile (real singular cases land at 0.8/1.2/1.6%) and cannot classify
    pde_black_scholes_call at all, whose kink at the strike keeps ~4% of the
    domain above tol at N=1025 while the formula is exactly right.

    Spatial concentration is retained as a *diagnostic* -- it records where the
    singularity is and how large -- but it no longer decides pass/fail.
    """
    trace = []
    for N in PROBE_N:
        scaled, mask = residual_field(spec, operator, N=N, exprs=exprs, stencil=stencil)
        st = _stats(scaled, mask)
        st["N"] = N
        trace.append(st)
        if st["median"] < tol:
            outcome = "validated" if st["max"] < tol else "validated_off_singularity"
            if outcome == "validated_off_singularity":
                # Keep probing so the fraction above tol is reported at the finest
                # level, where a localized singularity has shrunk to its footprint
                # and a region the formula is wrong in has not.
                for finer in PROBE_N[PROBE_N.index(N) + 1:]:
                    scaled, mask = residual_field(spec, operator, N=finer, exprs=exprs,
                                                  stencil=stencil)
                    fine_st = _stats(scaled, mask)
                    fine_st["N"] = finer
                    trace.append(fine_st)
                return outcome, {"trace": trace, **st, "frac_above_tol_finest": trace[-1][
                    "frac_above_tol"], "finest_N": trace[-1]["N"]}
            return outcome, {"trace": trace, **st}
    # The median never fell below tol: a wrong formula, unless it is still
    # converging -- which is under-resolution of the *probe*, not of the formula.
    falling = trace[-1]["median"] < 0.1 * trace[0]["median"]
    return ("unavailable" if falling else "failed"), {"trace": trace, **trace[-1]}


# --- test 2: the formula at the initial time ---------------------------------

def _initial_times(spec):
    """The times at which an ``initial_condition`` can actually be stated.

    Two, both named by the spec rather than searched for. ``t0`` for a similarity
    solution, which is singular at t = 0 and states its data at t0 instead; and
    ``T_maturity`` for a backward parabolic problem, where the "initial" condition
    is a *terminal* payoff -- ``pde_black_scholes_call`` marches forward in
    tau = T_maturity - t from the payoff at tau = 0.
    """
    params = spec.get("parameters") or {}
    times = [("t0", float(params.get("t0", 0.0)))]
    for key in ("T_maturity", "T_expiry"):
        if key in params:
            # Approached, not evaluated at: V is 0/0 exactly at maturity.
            times.append((key, float(params[key]) * (1 - 1e-12) - 1e-12))
    return times


def check_initial_condition(spec, N=513):
    """Test 2. Right PDE, wrong solution branch or mode.

    The residual alone cannot see initial data: on ``pde_advection_1d``, widening
    the Gaussian gives a *different but still exact* solution of
    ``u_t + a*u_x = 0``, so the residual correctly stays small. Only this test
    catches it, which is the concrete argument for keeping all three.
    """
    ic = spec.get("initial_condition")
    exprs = claimed_fields(spec)
    if not spec.get("time_dependent", True) or not exprs:
        return "unavailable", {"reason": "steady state or no claimed solution"}
    if not isinstance(ic, str) or not ic.strip():
        return "unavailable", {"reason": "no initial_condition expression"}
    if len(exprs) > 1:
        # A system's initial_condition is prose by design ("u = sin(x)cos(y); v = ...").
        return "unavailable", {"reason": "multi-field initial_condition is not a single expression"}

    sv, coords, _, _ = _grid(spec, N)
    params = dict(spec.get("parameters") or {})
    ones = np.ones_like(coords[sv[0]])
    expr = next(iter(exprs.values()))

    best, per_time = None, {}
    for label, t0 in _initial_times(spec):
        try:
            claimed = np.asarray(evaluate(expr, {**params, **coords, "t": t0}),
                                 dtype=float) * ones
            declared = np.asarray(evaluate(ic, {**params, **coords, "t": t0}),
                                  dtype=float) * ones
        except (NameError, SyntaxError, TypeError, ValueError, KeyError) as exc:
            per_time[label] = f"unevaluable: {exc}"
            continue
        scale = max(float(np.max(np.abs(declared))), 1e-300)
        err = float(np.max(np.abs(claimed - declared))) / scale
        per_time[label] = err
        if best is None or err < best[1]:
            best = (label, err, t0)

    if best is None:
        return "unavailable", {"reason": "initial_condition is not evaluable",
                               "times": per_time}
    label, err, t0 = best
    detail = {"max_rel_error": err, "matched_at": label, "t0": t0, "times": per_time}
    if err < IC_EXACT_TOL:
        return "validated", detail
    if err > IC_WRONG_TOL:
        return "failed", detail
    return "inconsistent", detail


# --- test 3: the formula on the boundary -------------------------------------

def _dirichlet_faces(spec):
    """``(axis_name, coordinate, declared_value_expr)`` for each conforming face.

    Keys look like ``"x=0"``. Non-conforming keys -- ``"note"``, ``"outflow"``,
    ``"x>=s(t)"``, a moving boundary ``"x=s(t)"`` -- are skipped rather than
    guessed at.
    """
    bc = spec.get("boundary_conditions") or {}
    if not str(bc.get("type", "")).startswith("dirichlet"):
        return []
    out = []
    for key, value in (bc.get("values") or {}).items():
        if "=" not in key or not isinstance(value, (str, int, float)):
            continue
        axis, _, coord = key.partition("=")
        axis, coord = axis.strip(), coord.strip()
        if axis not in (spec.get("spatial_variables") or []):
            continue
        try:
            position = float(coord)
        except ValueError:
            continue  # a moving or symbolic boundary
        out.append((axis, position, value))
    return out


def check_boundary_conditions(spec, rtol=1e-8, N=257):
    """Right PDE and IC, wrong BC selection."""
    exprs = claimed_fields(spec)
    faces = _dirichlet_faces(spec)
    if not exprs or len(exprs) > 1:
        return "unavailable", {"reason": "no single-field claimed solution"}
    if not faces:
        return "unavailable", {"reason": "no conforming Dirichlet faces to check"}

    sv, coords, _, axes1d = _grid(spec, N)
    params = dict(spec.get("parameters") or {})
    t = _t_probe(spec)
    ones = np.ones_like(coords[sv[0]])
    expr = next(iter(exprs.values()))
    u = np.asarray(evaluate(expr, {**params, **coords, "t": t}), dtype=float) * ones

    worst, per_face = 0.0, {}
    for axis, position, declared in faces:
        i = sv.index(axis)
        idx = int(np.argmin(np.abs(axes1d[i] - position)))
        face = np.take(u, idx, axis=i)
        face_coords = {k: np.take(v, idx, axis=i) for k, v in coords.items()}
        try:
            want = np.asarray(evaluate(str(declared), {**params, **face_coords, "t": t}),
                              dtype=float) * np.ones_like(face)
        except (NameError, SyntaxError, TypeError, ValueError) as exc:
            per_face[f"{axis}={position}"] = f"unevaluable: {exc}"
            continue
        scale = max(float(np.max(np.abs(want))), float(np.max(np.abs(u))), 1e-300)
        err = float(np.max(np.abs(face - want))) / scale
        per_face[f"{axis}={position}"] = err
        worst = max(worst, err)

    checked = [v for v in per_face.values() if isinstance(v, float)]
    if not checked:
        return "unavailable", {"reason": "no face was evaluable", "faces": per_face}
    return ("validated" if worst < rtol else "failed"), {"max_rel_error": worst,
                                                         "faces": per_face}


# --- SDE: the claimed moments against the declared moment ODE ----------------

def _moment_observables(spec):
    """``(name, from_expr, claimed, at_T_only)`` pairs, matched by *declared*
    quantity, never guessed.

    The one trap here: ``cross_from`` is a **covariance** (``"mxy - m1x*m1y"``)
    while ``cross_moment.expression`` is the raw ``E[XY]``. Pairing those two by
    name reports a 1.0 mismatch on a perfectly consistent spec, so the covariance
    is matched against ``cross_moment.covariance_value_at_T`` -- both say
    covariance, and the spec commits to that number.
    """
    am = spec.get("analytic_moments") or {}
    verification = spec.get("verification") or {}
    mo = verification.get("moment_ode") or {}
    out = []

    def add(name, from_expr, claimed, at_T=False):
        if isinstance(from_expr, str) and isinstance(claimed, (str, int, float)):
            out.append((name, from_expr, claimed, at_T))

    for kind in ("mean", "variance"):
        from_expr, claimed = mo.get(f"{kind}_from"), am.get(f"{kind}_expression")
        if isinstance(from_expr, list) and isinstance(claimed, list) \
                and len(from_expr) == len(claimed):
            for i, (f, c) in enumerate(zip(from_expr, claimed, strict=True)):
                add(f"{kind}[{i}]", f, c)
        else:
            add(kind, from_expr, claimed)
        for comp in ("X", "Y", "Z"):
            add(f"{kind}_{comp}", mo.get(f"{kind}_{comp}_from"),
                am.get(f"{kind}_{comp}") or am.get(f"{kind}_{comp}_expression"))
    add("covariance", mo.get("cross_from"),
        (verification.get("cross_moment") or {}).get("covariance_value_at_T"), at_T=True)
    return out


def check_moment_ode_consistency(spec, *, rtol=1e-6, n_points=12):
    """Integrate the declared moment ODE and compare against the claimed moments.

    Returns ``(outcome, detail)``. The moment ODE is a self-contained IVP -- state,
    rhs, initial -- so integrating it forward needs nothing from the claimed
    moments and works at any state dimension. That is what covers the two vector
    specs (5 coupled moments including cross terms) that differentiating the
    claimed moments could not: ``multichannel_stiff_m13``'s ``p12`` appears in no
    claimed moment at all, so there is nothing to differentiate, but it integrates
    like any other component.

    Measured: all five Tier-A SDE specs agree to <= 2.1e-10, and every corrupted
    moment separates at >= 2.5e-01.
    """
    from scipy.integrate import solve_ivp

    am = spec.get("analytic_moments") or {}
    mo = (spec.get("verification") or {}).get("moment_ode") or {}
    if not am.get("has_analytic_solution"):
        return "unavailable", {"reason": "no analytic moments claimed"}
    if not mo.get("closes_exactly"):
        return "unavailable", {"reason": "moment ODE does not close exactly; "
                                         "a closure approximation is not a reference"}
    state, rhs = mo.get("state") or [], mo.get("rhs") or []
    if not state or len(state) != len(rhs) or len(mo.get("initial") or []) != len(state):
        return "unavailable", {"reason": f"moment ODE shape is incomplete (state {state})"}

    observables = _moment_observables(spec)
    if not observables:
        return "unavailable", {"reason": "no claimed moment pairs with a declared "
                                         "mean_from / variance_from"}

    params = dict(spec.get("parameters") or {})
    T = float(spec["time_interval"]["T"])
    y0 = [float(evaluate(e, params)) for e in mo["initial"]]

    def rhs_fn(t, y):
        ns = {**params, "t": t, **dict(zip(state, y, strict=True))}
        return [float(evaluate(r, ns)) for r in rhs]

    # Radau: the multichannel spec is genuinely stiff. rtol here is 6 orders tighter
    # than the comparison tolerance, so the integrator is never what decides.
    sol = solve_ivp(rhs_fn, (0.0, T), y0, rtol=1e-12, atol=1e-14,
                    dense_output=True, method="Radau")
    if not sol.success:
        return "unavailable", {"reason": f"moment ODE integration failed: {sol.message}"}

    worst, per = 0.0, {}
    for name, from_expr, claimed, at_T in observables:
        times = [T] if at_T else np.linspace(0.05 * T, T, n_points)
        w = 0.0
        for t in times:
            ns = {**params, "t": t, **dict(zip(state, sol.sol(t), strict=True))}
            integrated = float(evaluate(from_expr, ns))
            want = float(evaluate(claimed, {**params, "t": t})) if isinstance(claimed, str) \
                else float(claimed)
            w = max(w, abs(integrated - want) / max(abs(integrated), abs(want), 1e-8))
        per[name] = w
        worst = max(worst, w)

    return ("validated" if worst < rtol else "failed"), {"max_rel_mismatch": worst,
                                                         "per_observable": per}


# --- the §4b rule, applied ---------------------------------------------------

def _quote_route(spec):
    sol = spec.get("analytic_solution")
    text = (sol or {}).get("analytic_solution_source_text") if isinstance(sol, dict) else None
    return text or spec.get("analytic_solution_source_text")


def check_reference(spec):
    """Apply §4b end to end. Returns ``(outcome, detail)`` with ``outcome`` in
    :data:`OUTCOMES`.

    The precedence that matters: ``failed`` overrides a quote **unconditionally**.
    A verbatim quotation establishes blame, not correctness, so a quoted formula
    that fails validation is still ``null``. That is the case neither quote-or-null
    nor graded provenance caught, and the one that matters in production, where
    there is no ``benchmark/verify.py`` behind the user.
    """
    if spec.get("chaotic"):
        # §14 C9: no closed form can apply, so `unavailable` would misreport the
        # metrics. Short-circuit instead.
        return "unavailable", {"reason": "chaotic problem; no closed form applies",
                               "short_circuit": True}

    is_sde = "state_dimension" in spec or "sde_family" in spec
    if is_sde:
        outcome, detail = check_moment_ode_consistency(spec)
        detail["tests"] = {"moment_ode": outcome}
    else:
        if not claimed_fields(spec):
            return "unavailable", {"reason": "analytic_solution is null"}
        operator, provenance = resolve_operator(spec)
        detail = {"operator_provenance": provenance}
        if operator is None:
            outcome, sub = "unavailable", {"reason": provenance}
            detail.update(sub)
            detail["tests"] = {"residual": "unavailable"}
        else:
            try:
                outcome, sub = check_operator_residual(spec, operator)
            except NameError as exc:
                # A composite helper the legacy operator_check leans on (`adv_u`,
                # `grad_p`, `lorentz_u`) that only makes sense per equation. This is
                # the systems ceiling of §4d, and the fix is a verification.operator
                # declaration, not a bigger helper namespace: `adv_u` means something
                # different in every system that uses it.
                outcome = "unavailable"
                sub = {"reason": f"operator references {exc}, which is not a generic "
                                 f"helper. Declare verification.operator.equations "
                                 f"and write the term out"}
            detail.update(sub)
            if outcome == "validated_off_singularity":
                outcome, why = _demote_off_singularity(spec, sub)
                if why:
                    detail["demoted_from"] = "validated_off_singularity"
                    detail["reason"] = why
            ic_outcome, ic_detail = check_initial_condition(spec)
            bc_outcome, bc_detail = check_boundary_conditions(spec)
            detail["tests"] = {"residual": outcome, "initial_condition": ic_outcome,
                               "boundary_conditions": bc_outcome}
            detail["initial_condition"] = ic_detail
            detail["boundary_conditions"] = bc_detail
            if outcome != "failed" and "failed" in (ic_outcome, bc_outcome):
                outcome = "failed"
            if spec.get("source_term") is not None:
                detail["used_declared_source"] = True

    if not is_sde:
        flagged, numerical = _numerical_fields(spec)
        if flagged:
            detail["analytic_numerical"] = numerical
    if outcome == "unavailable" and _quote_route(spec):
        return "quoted", {**detail, "reason": "validation could not run; accepted on the "
                                              "verbatim quote from problem.md"}
    return outcome, detail


def _numerical_fields(spec):
    """The F5 (1) flag over every claimed field: ``(any, {field: detail})``."""
    out = {}
    for name, expr in (claimed_fields(spec) or {}).items():
        flag, sub = numerical_closed_form(expr)
        if flag:
            out[name] = sub
    return bool(out), out


def _demote_off_singularity(spec, stats):
    """``(outcome, reason)``. ``validated_off_singularity`` survives only when the
    set above tolerance is small *and* the problem declares no layer or shock."""
    frac = float(stats.get("frac_above_tol_finest", stats.get("frac_above_tol")) or 0.0)
    facts = ((spec.get("verification") or {}).get("structural_facts") or {})
    declared = [k for k in (facts if isinstance(facts, dict) else [])
                if any(key in str(k).lower() for key in FEATURE_KEYS)]
    if frac > OFF_SINGULARITY_MAX_FRAC:
        return "unvalidated_at_feature", (
            f"the residual is above tolerance over {frac:.1%} of the domain at the finest "
            f"probe, more than {OFF_SINGULARITY_MAX_FRAC:.0%}: that is not a localized "
            f"singularity but the "
            f"region the problem is hard in, and a closed form accepted everywhere "
            f"except there is A- evidence (analytic_unvalidated), not A")
    if declared:
        return "unvalidated_at_feature", (
            f"the residual is above tolerance over {frac:.1%} of the domain at the finest "
            f"probe and the spec declares {declared} in verification.structural_facts: "
            f"the check was waived "
            f"exactly at the feature the problem exists to test, which is A- evidence "
            f"(analytic_unvalidated), not A")
    return "validated_off_singularity", None


def reference_findings(spec, name="") -> list[Finding]:
    """The reference check as findings, for the gate and the hooks."""
    outcome, detail = check_reference(spec)
    path = f"{name}analytic_solution" if name else "analytic_solution"
    if outcome == "failed":
        where = detail.get("tests") or {}
        return [error("reference", path,
                      f"the claimed closed form does not satisfy the spec's own declared "
                      f"structure ({where}); median term-balanced residual "
                      f"{detail.get('median', detail.get('max_rel_mismatch', float('nan'))):.3e}. "
                      f"Per §4b this field must be null -- a quote does not rescue it")]
    if outcome in ("unavailable", "quoted"):
        return [warning("reference", path,
                        f"reference check {outcome}: {detail.get('reason', '')}. "
                        f"Tier A here rests on provenance, not validation; "
                        f"the gap is recorded")]
    sub = (detail.get("tests") or {})
    extra = []
    if "inconsistent" in sub.values():
        which = [k for k, v in sub.items() if v == "inconsistent"]
        extra.append(warning(
            "reference", path,
            f"{which} disagree with the claimed solution by more than round-off but "
            f"less than a wrong branch would. Two transcriptions of the same function "
            f"that do not quite agree -- recorded, not fatal",
        ))
    if outcome == "validated_off_singularity":
        extra.append(warning(
            "reference", path,
            f"validated off a localized singularity: median {detail['median']:.2e} "
            f"passes, max {detail['max']:.2e} does not, over "
            f"{detail['frac_above_tol']:.1%} of the domain at N={detail.get('N')}"
            + (f" ({detail['frac_above_tol_finest']:.1%} at N={detail['finest_N']})"
               if "frac_above_tol_finest" in detail else "")))
    if outcome == "unvalidated_at_feature":
        extra.append(warning(
            "reference", path,
            f"reference check unvalidated_at_feature: {detail.get('reason', '')}. The "
            f"closed form is priced as analytic_unvalidated (9), not analytic"))
    numerical = detail.get("analytic_numerical")
    if numerical:
        labels = ", ".join(f"{k}: {v.get('label')}" for k, v in numerical.items())
        extra.append(warning(
            "reference", path,
            f"the closed form is a numerical closed form ({labels}): it discretizes a "
            f"quadrature or series inside the expression. Recorded, not capped -- the "
            f"residual check is the ceiling -- so a reader can see what 'exact' rests on"))
    return extra
