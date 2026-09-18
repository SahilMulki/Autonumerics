"""Declared scalar quantities of interest, checked by self-convergence.

Some problems are not scored on a field alone. An eigenvalue problem's answer is a
number *and* a mode; a flow problem's may be a drag coefficient. The formulator
declares those in ``evaluation_thresholds.functionals``::

    "functionals": {"lambda_1": {"rel_tol": 5e-4, "gate": true, "note": "..."}}

**The kernel has no reference value for them, and must not pretend otherwise.** On a
problem with no closed form there is no ``lambda_1`` to compare against -- that is the
whole premise. What the kernel can do is exactly what it already does for the field on
Path B: carry the quantity up the ladder and ask whether it has *converged* to the
declared tolerance. Richardson on three levels gives an error estimate; two levels give
the raw difference, which is the same quantity without the extrapolation and is
therefore conservative.

So a functional check answers "is this number resolved to the accuracy the problem
demands", never "is this number right". Those are different claims and the metrics keep
them apart: the outcome vocabulary here is the ladder's, and a functional check never
contributes to a provenance tier.

The outcomes, in the closed skip vocabulary of :mod:`.metrics`:

``clean``          the estimated relative error is inside ``rel_tol``
``stalled``        it is not -- the quantity has not converged to the declared bar
``not_reported``   declared, but the solver returned no value for it
``unavailable``    nothing declared, or fewer than two ladder levels carried a value

``not_reported`` is deliberately distinct from a failure, and for a **gated**
functional it caps the score at 9 rather than passing -- plan §6's rule that a declared
gate the kernel cannot evaluate is worse than no gate, because it reads as evidence.
"""

from __future__ import annotations

import math

#: Below this multiple of the value itself, successive levels differ by round-off
#: rather than by discretization error, and the quantity is converged whatever the
#: declared tolerance says. Mirrors ``richardson.ROUNDOFF_FLOOR``.
ROUNDOFF_FLOOR = 1e-13

#: Clamp on the order inferred from three levels. An apparent order far above the
#: scheme's cannot be real, and using it would *understate* the error -- the direction
#: that turns a fail into a pass (plan §7a got this wrong in rev 1).
MAX_INFERRED_ORDER = 4.0


def declared(thresholds):
    """The ``{name: config}`` map, tolerant of a malformed declaration."""
    decl = (thresholds or {}).get("functionals")
    if not isinstance(decl, dict):
        return {}
    return {str(k): v for k, v in decl.items() if isinstance(v, dict)}


def _series(runs, name):
    """The value of ``name`` at each ladder level that reported it, coarse to fine."""
    out = []
    for run in runs:
        got = (run.get("result") or {}).get("functionals")
        if isinstance(got, dict) and name in got:
            try:
                value = float(got[name])
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                out.append(value)
    return out


def estimate_error(values):
    """Relative error of the finest value, from the ladder alone.

    Three or more levels give Richardson: with a refinement ratio of 2 the observed
    order is ``log2(|d10| / |d21|)`` and the error estimate is ``|d21| / (2**p - 1)``.
    Two levels give ``|d10|`` itself -- no order is measurable, so the difference
    stands in for the error, which overstates it for any convergent scheme and is the
    safe direction.

    Returns ``(rel_err, detail)``, or ``(None, detail)`` when nothing is measurable.
    """
    detail = {"levels": len(values), "value": values[-1] if values else None}
    if len(values) < 2:
        return None, detail
    scale = abs(values[-1])
    if scale < 1e-300:
        scale = max((abs(v) for v in values), default=0.0) or 1.0
    d21 = abs(values[-1] - values[-2])
    detail["d21"] = d21
    if d21 <= ROUNDOFF_FLOOR * scale:
        detail["at_roundoff"] = True
        return 0.0, detail
    if len(values) >= 3:
        d10 = abs(values[-2] - values[-3])
        detail["d10"] = d10
        if d10 > d21 > 0.0:
            order = min(math.log2(d10 / d21), MAX_INFERRED_ORDER)
            detail["observed_order"] = order
            denom = 2.0 ** order - 1.0
            if denom > 0.0:
                return (d21 / denom) / scale, detail
    detail["observed_order"] = None
    return d21 / scale, detail


def run_declared(thresholds, runs):
    """Evaluate every declared functional across the ladder.

    ``runs`` is the kernel's ladder: a list of ``{"result": ..., "N": ...}`` coarse to
    fine, so the finest level is last.
    """
    decl = declared(thresholds)
    out = {"outcomes": {}, "errors": {}, "values": {}, "detail": {},
           "gate_failed": [], "not_reported_gated": [], "not_reported": [],
           "functionals_ok": "unavailable"}
    if not decl:
        return out

    any_ran, any_failed = False, False
    for name, cfg in sorted(decl.items()):
        gate = bool(cfg.get("gate", False))
        try:
            tol = float(cfg.get("rel_tol"))
        except (TypeError, ValueError):
            tol = None
        values = _series(runs, name)
        if not values:
            out["outcomes"][name] = "not_reported"
            out["not_reported"].append(name)
            if gate:
                out["not_reported_gated"].append(name)
            continue
        out["values"][name] = values[-1]
        rel_err, detail = estimate_error(values)
        detail["gate"] = gate
        detail["rel_tol"] = tol
        out["detail"][name] = detail
        if rel_err is None or tol is None:
            # One level, or no tolerance to test against: the value is recorded and
            # nothing is claimed about it.
            out["outcomes"][name] = "unavailable"
            continue
        out["errors"][name] = rel_err
        any_ran = True
        if rel_err < tol:
            out["outcomes"][name] = "clean"
        else:
            out["outcomes"][name] = "stalled"
            if gate:
                out["gate_failed"].append(name)
                any_failed = True

    if any_failed:
        out["functionals_ok"] = False
    elif any_ran:
        out["functionals_ok"] = True
    return out
