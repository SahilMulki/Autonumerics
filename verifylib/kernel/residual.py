"""D1: the operator residual, as a **slope test** rather than a magnitude test.

Rev 1 of the plan made D1 a hard gate and never said what failure was. No fixed
threshold can exist, and the plan's own table says so: residual magnitude depends
on operator conditioning, grid and stencil order. The only calibrated number in
the repo, ``reference.TOL = 1e-6``, was measured for *a closed form on a
1025-point grid whose own discretization error is 1e-10*. A solver's residual is
dominated by its own truncation error, ``O(h^p) + O(dt^q)``, and does not approach
zero at fixed resolution -- so a constant threshold either fails every correct
coarse-grid solver or passes everything.

The SDE half of this manual already solved exactly this. §20, for Dynkin: *"the
trapezoid path integral carries its own O(dt) bias, so the residual does not go to
zero at fixed dt -- the meaningful test is that it shrinks at the expected rate."*
Transplanted here::

    r(h)  = median over the interior of  |sum(terms) - source| / max_k |term_k|
    p_res = log2(r(h) / r(h/2))          # both grids are already on the ladder

Two things keep this from punishing correct work. The residual is gated on the
**median**, never the max -- the max-plus-"singular set < 2%" design was measured
to be fragile and could not classify ``pde_black_scholes_call`` at all, and solver
fields have shocks and layers too. And the whole measurement is abandoned as
``unresolved`` when two kernel stencils disagree, because a 6th-order or spectral
solver differenced with lower-order stencils measures the *kernel's* error, not
the solver's.
"""

from __future__ import annotations

import numpy as np

from ..operator import Stencil, evaluate
from ..reference import TOL as FORMULA_TOL
from ..reference import _scaled_residual, claimed_fields, residual_field, resolve_operator

#: ``p_res`` must reach ``p_design - SLOPE_SLACK`` to count as a circularity break.
#: One order of slack, mirroring manual §4's ``p_sane = p < theoretical_order + 1``
#: -- the same band, applied on the other side. A ratio of medians of a noisy
#: pointwise quantity does not deserve a tighter bar than the order estimate the
#: whole ladder exists to produce.
SLOPE_SLACK = 1.0

#: Below this the residual is not falling at all: ``r(h/2) > 0.84 * r(h)``.
STALL_P = 0.25

#: ``r`` at or below this is round-off, not accuracy, and the slope carries no
#: information. Calibrated downward from ``reference.TOL``: that check accepts a
#: *formula* at median 1e-6 and measures correct-and-resolved formulas at
#: <= 4.8e-9, and a solver field cannot beat the formula's own floor. Four orders
#: below the formula acceptance bar is therefore round-off. It is a floor
#: *detector*, never a pass threshold -- the pass is the slope.
RESIDUAL_ROUNDOFF = 1e-10

#: The §5b pair disagreeing by more than this makes the check, not the solver, the
#: limit. Deliberately loose: two stencils of different order legitimately differ
#: by an O(1) factor on a resolved field, and only a gross disagreement means the
#: residual is measuring the kernel.
INSENSITIVITY_FACTOR = 2.0

#: Outcomes. ``slow`` is not in the plan's four-row table and is named here rather
#: than folded into a neighbour: the table defines ``clean`` as reaching the design
#: slope and ``stalled`` as "r does not fall", which leaves the band where r falls
#: but under-performs unnamed. Routing it to ``stalled`` would gate a solver at 3
#: for converging slowly; routing it to ``clean`` would let it certify. It reports,
#: and it neither gates nor counts as a circularity break.
OUTCOMES = ("clean", "slow", "stalled", "unresolved", "unavailable")


# --- the time term: Fornberg weights over the returned snapshot times ---------

def fornberg_weights(z, nodes, m):
    """Finite-difference weights for derivatives ``0..m`` at ``z`` from ``nodes``.

    Arbitrary spacing, which is the point: a solver's final step is whatever
    ``t_final / Nt`` left over, and an adaptive integrator's is not uniform at all.
    Fornberg (1988), transcribed.
    """
    x = np.asarray(nodes, dtype=float)
    n = len(x) - 1
    c = np.zeros((n + 1, m + 1))
    c1, c4 = 1.0, x[0] - z
    c[0, 0] = 1.0
    for i in range(1, n + 1):
        mn = min(i, m)
        c2, c5, c4 = 1.0, c4, x[i] - z
        for j in range(i):
            c3 = x[i] - x[j]
            c2 *= c3
            if j == i - 1:
                for k in range(mn, 0, -1):
                    c[i, k] = c1 * (k * c[i - 1, k - 1] - c5 * c[i - 1, k]) / c2
                c[i, 0] = -c1 * c5 * c[i - 1, 0] / c2
            for k in range(mn, 0, -1):
                c[j, k] = (c4 * c[j, k] - k * c[j, k - 1]) / c3
            c[j, 0] = c4 * c[j, 0] / c3
        c1 = c2
    return c


def time_derivatives(snapshots, field_names, t_eval=None):
    """``{name: (u_t, u_tt)}`` from the solver's returned ``snapshots``.

    ``snapshots`` is ``[{"t": float, "fields": {name: array}}, ...]`` ending at
    ``t_final`` -- the **return**-contract extension in plan §5c. Rev 1 asked
    ``override`` to "support returning u(T-dt)", but ``override`` is an input dict
    with no return channel, and both routes that exist today fail:

    * a second solve at ``t_final = T - dt`` picks a different internal step
      sequence, so each run approximates its *own* exact solution to O(dt^p)
      independently and the difference quotient carries O(dt^(p-1)) -- O(1) for a
      first-order scheme, and it costs a full extra solve;
    * the solver's own last step is what the scheme computes, so the time term is
      consistent by construction and D1 degenerates to a spatial-operator test.

    Returns ``(derivs, order)``. ``order`` is ``K - 1``; ``K = 2`` runs but records
    its own truncation floor, and fewer than 2 is nothing.
    """
    if not snapshots or len(snapshots) < 2:
        return None, 0
    times = np.asarray([float(s["t"]) for s in snapshots], dtype=float)
    if not np.all(np.diff(times) > 0):
        order = np.argsort(times)
        times = times[order]
        snapshots = [snapshots[i] for i in order]
    z = float(times[-1]) if t_eval is None else float(t_eval)
    m = 2 if len(times) >= 3 else 1
    weights = fornberg_weights(z, times, m)

    derivs = {}
    for name in field_names:
        stack = [np.asarray(s["fields"][name], dtype=float) for s in snapshots
                 if name in (s.get("fields") or {})]
        if len(stack) != len(times):
            continue
        first = sum(weights[i, 1] * stack[i] for i in range(len(times)))
        second = (sum(weights[i, 2] * stack[i] for i in range(len(times)))
                  if m >= 2 else None)
        derivs[name] = (first, second)
    return (derivs or None), len(times) - 1


def snapshot_fields(snapshots):
    """The final state, taken from the snapshots themselves, so the residual and
    its time derivative are evaluated on one consistent set of arrays."""
    if not snapshots:
        return {}
    last = max(snapshots, key=lambda s: float(s["t"]))
    return {k: np.asarray(v, dtype=float) for k, v in (last.get("fields") or {}).items()}


# --- the residual on solver output -------------------------------------------

def _interior(shape, halo, domain_mask=None):
    mask = np.zeros(shape, dtype=bool)
    mask[tuple([slice(halo, -halo)] * len(shape))] = True
    if domain_mask is not None:
        # np.roll-based stencils difference straight across a hole. Intersecting
        # with the spec's own domain mask is what keeps `poisson_lshape` and the
        # masked benchmark domains from being scored on differences taken through
        # the excluded region -- and where the mask cannot be built, D1 reports
        # `unavailable` rather than quietly differencing across it.
        mask &= np.asarray(domain_mask, dtype=bool)
    return mask


def residual_on_fields(operator, fields, coords, spatial_vars, hs, params, *,
                       time_derivs=None, stencil=None, t=None, source_term=None,
                       domain_mask=None):
    """``(median, stats, mask)`` for the term-balanced residual of solver output.

    Shares ``reference._scaled_residual`` rather than re-deriving it, so the gate
    and the check it gates for can never disagree about what the residual is
    (guardrails §14 C1).
    """
    from ..operator import build_namespace
    stencil = stencil or Stencil(8)
    ns = build_namespace(fields, coords, spatial_vars, hs, time_derivs=time_derivs,
                         stencil=stencil)
    ns.update(params or {})
    if t is not None:
        ns["t"] = float(t)
    if source_term is not None:
        ns["f"] = np.asarray(evaluate(source_term, ns), dtype=float)

    if operator["kind"] == "scalar":
        scaled = _scaled_residual(operator["terms"], operator.get("source"),
                                  ns, spatial_vars, hs, fields, stencil)
    else:
        stack = np.stack([
            _scaled_residual(eq.get("terms") or {}, eq.get("source"), ns, spatial_vars,
                             hs, fields, stencil)
            for eq in operator["equations"].values()])
        scaled = (np.sqrt(np.mean(stack ** 2, axis=0))
                  if operator.get("combine", "rms") == "rms" else np.max(stack, axis=0))

    mask = _interior(scaled.shape, stencil.halo or 4, domain_mask)
    if not mask.any():
        return None, {"reason": "the interior mask is empty after the domain mask"}, mask
    sample = scaled[mask]
    stats = {"median": float(np.median(sample)), "q95": float(np.quantile(sample, 0.95)),
             "max": float(np.max(sample)), "n": int(sample.size),
             "stencil": str(stencil.order)}
    return stats["median"], stats, mask


# --- the slope test -----------------------------------------------------------

def slope_test(levels, p_design, *, slack=SLOPE_SLACK):
    """Classify the residual's rate of decrease. ``levels`` is coarse-to-fine
    ``[(h_label, median), ...]`` with at least two entries."""
    if len(levels) < 2:
        return "unavailable", {"reason": "the residual is available at one grid only"}
    (_, r_coarse), (_, r_fine) = levels[-2], levels[-1]
    detail = {"r_coarse": r_coarse, "r_fine": r_fine, "p_design": float(p_design)}

    if r_fine <= RESIDUAL_ROUNDOFF:
        # Manual §4's not_at_roundoff branch, on the residual instead of the
        # ladder difference. Not hypothetical: the KS spec's own note documents a
        # spectral scheme collapsing d21 to round-off on the 64/127/253 ladder.
        return "clean", {**detail, "p_res": float("inf"), "reason": "round-off floor",
                         "roundoff_floor": RESIDUAL_ROUNDOFF}
    if r_coarse <= 0.0 or r_fine <= 0.0:
        return "unavailable", {**detail, "reason": "a residual of exactly zero"}

    p_res = float(np.log2(r_coarse / r_fine))
    detail["p_res"] = p_res
    if p_res >= float(p_design) - slack:
        return "clean", detail
    if p_res <= STALL_P:
        return "stalled", detail
    return "slow", detail


def insensitive(median_a, median_b, factor=INSENSITIVITY_FACTOR):
    """§5b. True when two stencil families agree closely enough that the residual
    is measuring the solver rather than the check.

    The same escalation idiom ``reference.PROBE_N = (257, 513, 1025)`` already uses
    to separate "under-resolved" from "wrong", moved from grid to stencil.
    """
    lo, hi = sorted((abs(median_a), abs(median_b)))
    if hi <= RESIDUAL_ROUNDOFF:
        return True
    return hi <= factor * max(lo, RESIDUAL_ROUNDOFF)


def stencil_pair(scheme_family, spatial_order, periodic):
    """The two families §5b differences with, and how the choice was made.

    Rule 1 of §3b says difference at higher order than the solver used; rule 2 says
    use a different *family*. Where the plan frontmatter declares neither -- every
    plan written before this landed -- the kernel takes 8th-order FD and records
    ``stencil_choice: default``, and the insensitivity guard catches the case where
    that was not enough.
    """
    order = int(spatial_order) if isinstance(spatial_order, (int, float)) else None
    family = (scheme_family or "").strip().lower() or None
    choice = "declared" if (order is not None or family) else "default"

    if family == "spectral" and periodic and all(periodic):
        # A spectral solver cannot be out-resolved by any finite difference, so the
        # honest pair is FD8 against a spectral derivative: agreement then says the
        # residual is real, and disagreement says the check is the limit.
        return (Stencil(8), Stencil("spectral", periodic)), choice, "fd8+spectral"
    if order is not None and order >= 8:
        return ((Stencil("spectral", periodic), Stencil(8)) if periodic and all(periodic)
                else (Stencil(8), Stencil(6))), choice, "at-or-above-kernel-order"
    if order is not None and order >= 6:
        return (Stencil(8), Stencil(6)), choice, "fd8+fd6"
    return (Stencil(8), Stencil(6)), choice, "fd8+fd6"


# --- §5d: validating the operator itself --------------------------------------

VALIDATION_ROUTES = ("reference", "mms", "degenerate", "none")


def validate_operator(spec, operator, *, reference_outcome=None, tol=FORMULA_TOL):
    """Is ``verification.operator`` itself right? ``(route, ok, detail)``.

    D1's residual and the solver both descend from this one declaration, so if it
    is wrong they are wrong *consistently* and D1 is not a circularity break at
    all. Three routes, descending in strength -- and none of them catches an
    operator misread the same way ``problem.md`` was. That is the requirements
    ledger's job, and it is the only check that reaches outside the spec.
    """
    if operator is None:
        return "none", False, {"reason": "no operator declaration"}

    if reference_outcome in ("validated", "validated_off_singularity"):
        # Path A. A wrong operator makes a correct closed form fail, so the
        # reference check having passed is itself the validation.
        return "reference", True, {"reference_outcome": reference_outcome}

    verification = spec.get("verification") or {}
    probe = verification.get("mms_probe") or {}
    exact, source = probe.get("exact"), probe.get("source")
    if isinstance(exact, str) and isinstance(source, str):
        ok, detail = _residual_of_formula(spec, {**operator, "source": source},
                                          {"u": exact}, tol)
        if ok is not None:
            return "mms", ok, {"route": "mms_probe", **detail}
    if isinstance(exact, dict) and isinstance(source, dict):
        ok, detail = _residual_of_formula(spec, _system_with_source(operator, source),
                                          dict(exact), tol)
        if ok is not None:
            return "mms", ok, {"route": "mms_probe(system)", **detail}

    degenerate = verification.get("degenerate_limit") or {}
    if isinstance(degenerate.get("exact"), (str, dict)):
        merged = dict(spec)
        merged["parameters"] = {**(spec.get("parameters") or {}),
                                **(degenerate.get("params") or {})}
        exprs = ({"u": degenerate["exact"]} if isinstance(degenerate["exact"], str)
                 else dict(degenerate["exact"]))
        ok, detail = _residual_of_formula(merged, operator, exprs, tol)
        if ok is not None:
            return "degenerate", ok, {"route": "degenerate_limit",
                                      "params": degenerate.get("params"), **detail}

    return "none", False, {"reason": "no reference validation, no MMS probe with a "
                                     "source, and no degenerate limit with an exact "
                                     "solution -- the operator is unaudited"}


def _system_with_source(operator, sources):
    if operator["kind"] != "system":
        return operator
    equations = {name: {**eq, "source": sources.get(name, eq.get("source"))}
                 for name, eq in operator["equations"].items()}
    return {**operator, "equations": equations}


def _residual_of_formula(spec, operator, exprs, tol):
    """Apply the operator to a *formula* on a fine grid. Nearly free, and
    ``reference.residual_field`` already takes the ``exprs=`` override that makes
    it a call rather than an implementation."""
    try:
        scaled, mask = residual_field(spec, operator, N=257, exprs=exprs)
    except Exception as exc:  # noqa: BLE001 -- an unevaluable probe is not a failure
        return None, {"reason": f"{type(exc).__name__}: {exc}"}
    sample = scaled[mask]
    median = float(np.median(sample))
    return median < tol, {"median": median, "tol": tol, "max": float(np.max(sample))}


# --- §5f: the three trivial-attractor guards ----------------------------------

#: ``rms(u_T)`` below this multiple of ``rms(u_0)`` is a collapsed field.
TRIVIAL_TOL = 1e-3

#: The solver's own t = 0 state must reproduce the declared initial condition.
IC_CONSISTENT_TOL = 1e-10


def trivial_guards(u_final, u_initial, *, triv_tol=TRIVIAL_TOL):
    """A residual rewards any field sitting on a stable steady state of the
    operator, whether or not it is the state the initial condition evolves to.

    Gray-Scott is the sharpest case: ``u = 1, v = 0`` is an exact homogeneous
    solution, so a solver that over-diffuses and destroys the pattern satisfies
    every term to near machine precision, having eliminated the only phenomenon the
    problem is about. The degenerate case is worse -- if ``u -> 0`` the numerator
    and the denominator both vanish and the epsilon leaves you dividing zero by
    1e-300, a perfect score for solving nothing.
    """
    from .ladder import rms
    if u_initial is None:
        return "not_reported", {"reason": "no initial field to compare against"}
    scale_final, scale_initial = rms(u_final), rms(u_initial)
    ratio = scale_final / (scale_initial + 1e-300)
    return bool(ratio > triv_tol), {"rms_final": scale_final, "rms_initial": scale_initial,
                                    "ratio": float(ratio), "tol": triv_tol}


def ic_consistency(u_at_zero, u_declared, *, tol=IC_CONSISTENT_TOL):
    """Did the run start where it should? ``not_reported`` when either side is
    missing -- an unavailable check never counts against a plan."""
    from .ladder import rel_err
    if u_at_zero is None or u_declared is None:
        return "not_reported", {"reason": "the solver's t = 0 state or the declared "
                                          "initial condition is unavailable"}
    err = rel_err(u_at_zero, u_declared)
    return bool(err < tol), {"rel_error": err, "tol": tol}


def operator_for(spec):
    """The resolved operator, or ``(None, reason)``."""
    return resolve_operator(spec)


def has_claimed_solution(spec):
    return bool(claimed_fields(spec))
