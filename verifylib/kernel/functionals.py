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

``unresolved``     the ladder cannot measure a discretization bias here -- F18's
                   exactly-sampled transition, which has none at any ``dt``

``not_reported`` is deliberately distinct from a failure, and for a **gated**
functional it caps the score at 9 rather than passing -- plan §6's rule that a declared
gate the kernel cannot evaluate is worse than no gate, because it reads as evidence.
``unresolved`` is the abstention and costs the evidence rather than charging a penalty,
exactly as ``crn.abstention`` does one layer over.
"""

from __future__ import annotations

import math
import operator
import re

#: Below this multiple of the value itself, successive levels differ by round-off
#: rather than by discretization error, and the quantity is converged whatever the
#: declared tolerance says. Mirrors ``richardson.ROUNDOFF_FLOOR``.
ROUNDOFF_FLOOR = 1e-13

#: Clamp on the order inferred from three levels. An apparent order far above the
#: scheme's cannot be real, and using it would *understate* the error -- the direction
#: that turns a fail into a pass (plan §7a got this wrong in rev 1).
MAX_INFERRED_ORDER = 4.0

#: The other end of the same clamp, and the one that was missing. The Richardson
#: denominator ``2**p - 1`` goes to zero with ``p``, so an *under*-performing inferred
#: order inflates the error estimate without bound -- and a ladder whose differences
#: have stopped shrinking because the quantity has **converged** reports exactly that
#: small ``p``. Measured on ``sde_sabr_cev_call_ii_lowvol`` plan 3:
#: ``d21 = 6.906e-05`` (7.3x tighter than its pool sibling's ``5.062e-04``) with
#: ``d10 = 7.333e-05`` gives ``p = 0.0866``, a denominator of 0.0619, and a 16.2x
#: inflation that ranked the tightest ladder in the pool last.
#:
#: So the denominator is trusted only above a floor, and below it the estimate falls
#: back to ``d21 / scale`` -- the two-level bound this function's own docstring names
#: as "the safe direction". Flat differences at a small absolute level mean converged,
#: not diverging, and the fallback says the weaker of the two true things rather than
#: a false one.
#:
#: **The floor is a fraction of the plan's declared design order, not a constant.**
#: Half, because an observed rate within a factor of two of the design rate is the
#: band the ladder is credited for everywhere else in the kernel, and because the one
#: number that works for every scheme does not exist: half of a Milstein scheme's rate
#: is 0.5 and half of an Euler-Maruyama strong rate is 0.25, and a constant 0.5 -- which
#: reproduces the measurement above -- would reject a correctly converging strong-order-
#: 0.5 ladder as readily as it rejects a stalled one.
ORDER_FLOOR_FRACTION = 0.5

#: The floor where no design order is declared, which is most plans today. Fixed, and
#: deliberately on the conservative side: rejecting the denominator *overstates* the
#: error, so a floor that is too high costs resolving power and a floor that is too low
#: manufactures accuracy out of a converged ladder. The detail block records which of
#: the two floors was applied, because an inference that silently stands in for a
#: declaration is the defect this kernel's declared-never-inferred rule exists to stop.
DEFAULT_ORDER_FLOOR = 0.5


#: Why an exactly-sampled transition's ladder differences are not a bias (F18).
#: Stated here as :data:`..sde.crn.EXACT_SAMPLING_REASON` is stated there, and worded
#: for the ladder rather than for the order study, because the two measure different
#: things off the same cause.
EXACT_SAMPLING_ABSTENTION = (
    "the plan declares **every** state component's transition to be sampled exactly, "
    "so the whole state draws from its exact conditional law instead of integrating "
    "it: there is no discretization bias to measure at any dt, and the level-to-level "
    "differences are the auxiliary randomness each step drew. An abstention is not a "
    "pass -- it forfeits the ladder's bias evidence, leaving the solver's reported "
    "sampling term to stand alone. A plan that draws only *some* components exactly is "
    "not abstained on: the graded functional depends on the whole state, so a single "
    "discretised leg biases it however exactly the others are drawn")


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


def estimate_error(values, *, design_order=None):
    """Relative error of the finest value, from the ladder alone.

    Three or more levels give Richardson: with a refinement ratio of 2 the observed
    order is ``log2(|d10| / |d21|)`` and the error estimate is ``|d21| / (2**p - 1)``.
    Two levels give ``|d10|`` itself -- no order is measurable, so the difference
    stands in for the error, which overstates it for any convergent scheme and is the
    safe direction.

    ``design_order`` is the scheme's declared rate, which sets the floor the inferred
    order must clear before that denominator is trusted -- see
    :data:`ORDER_FLOOR_FRACTION` for the measurement that put it there. Below the
    floor the two-level bound is used instead, and ``detail["order_rejected"]`` says
    so.

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
            floor, source = order_floor(design_order)
            detail["order_floor"] = floor
            detail["order_floor_source"] = source
            denom = 2.0 ** order - 1.0
            if order >= floor and denom > 0.0:
                return (d21 / denom) / scale, detail
            # Converged, stalled, or outside its asymptotic range -- the ladder
            # cannot tell which, so the extrapolation is not applied and the raw
            # difference stands in for the error. `detail` keeps the order it
            # measured: the number is a real diagnostic, it is just not a divisor.
            detail["order_rejected"] = True
            return d21 / scale, detail
    detail["observed_order"] = None
    return d21 / scale, detail


def order_floor(design_order):
    """``(floor, source)``: the inferred order a Richardson denominator must clear.

    Capped at :data:`MAX_INFERRED_ORDER`, because that is also the ceiling the order is
    *reported* at: a floor above the clamp can never be cleared, so a plan declaring
    ``spatial_order: 10`` would have the extrapolation silently disabled on every
    functional and get the conservative two-level bound forever. The cap turns that
    into "trusted whenever the ladder reaches the clamp", which is the strongest thing
    the estimator can say about a scheme that converges faster than it can measure.
    """
    try:
        design = float(design_order)
    except (TypeError, ValueError):
        design = None
    if design is not None and math.isfinite(design) and design > 0.0:
        return min(ORDER_FLOOR_FRACTION * design, MAX_INFERRED_ORDER), "design_order"
    return DEFAULT_ORDER_FLOOR, "kernel_default"


#: The comparisons an ``expected_cross_check`` may state between two functionals.
_COMPARATORS = {"<": operator.lt, "<=": operator.le,
                ">": operator.gt, ">=": operator.ge}


def cross_check(statement, values):
    """Check a declared ordering between functionals against the values returned.

    ``verification.expected_cross_check`` is prose with a checkable claim inside it::

        "price is monotone decreasing in strike and S0-ordered:
         price_S75 < price_S100 < price_S125"

    The claim is an *ordering*, and an ordering is a far stronger statement about a
    multi-functional solver than any one functional's tolerance. Three prices from one
    set of paths can each sit inside their own gate while being out of order, which is
    the signature of a mis-wired strike or a payoff evaluated at the wrong spot --
    a defect the per-functional errors cannot see, because it is a relationship
    between them.

    So only the relations are read, never the prose: this scans for runs of identifiers
    and comparison symbols separated by whitespace alone, and keeps the triples whose
    operator is a comparison and whose two names are both functionals the solver
    returned. Everything else falls out -- a sentence's words form runs too, and they
    fail the comparison test. A statement with no surviving triple yields
    ``"unavailable"``: the field was declared and no claim could be read out of it,
    which is a measurement and not a failure.

    Returns ``(outcome, detail)`` in the vocabulary of :mod:`.metrics`.
    """
    detail = {"statement": statement, "relations": [], "violations": []}
    if not isinstance(statement, str) or not isinstance(values, dict) or not values:
        return "unavailable", detail
    # Identifiers rather than the returned names, so a chain still *parses* when one
    # of its names is a functional the solver did not return -- "a < b < c" with no `c`
    # is still a claim about a and b. Each relation is filtered on the values below,
    # which is where a missing name drops out, and it drops out as a skip: an
    # unreturned functional is already `not_reported` above and charging it twice
    # punishes one gap twice.
    token = re.compile(r"([A-Za-z_][A-Za-z_0-9]*)|(<=|>=|<|>)")
    # A chain is tokens separated by whitespace only. "a < b" is a claim; "a < 3, b"
    # is not, and neither is a name that merely appears in the same sentence -- which
    # is what most of an `expected_cross_check` is, since it is written for a reader.
    runs, current, end = [], [], None
    for match in token.finditer(statement):
        if end is not None and statement[end:match.start()].strip():
            runs.append(current)
            current = []
        current.append(match.group(0))
        end = match.end()
    runs.append(current)

    for run in runs:
        # name op name [op name ...]: names on the even positions. A run of prose
        # tokens reaches here too and falls out at the comparator test below.
        if len(run) < 3:
            continue
        for i in range(0, len(run) - 2, 2):
            left, op, right = run[i], run[i + 1], run[i + 2]
            if left not in values or right not in values or op not in _COMPARATORS:
                continue
            try:
                a, b = float(values[left]), float(values[right])
            except (TypeError, ValueError):
                continue
            held = _COMPARATORS[op](a, b)
            detail["relations"].append(
                {"left": left, "op": op, "right": right,
                 "left_value": a, "right_value": b, "held": bool(held)})
            if not held:
                detail["violations"].append(f"{left} ({a:.6g}) {op} {right} ({b:.6g})")
    if not detail["relations"]:
        return "unavailable", detail
    return not detail["violations"], detail


def run_declared(thresholds, runs, *, design_order=None, exact_sampling=(),
                 state_dimension=None):
    """Evaluate every declared functional across the ladder.

    ``runs`` is the kernel's ladder: a list of ``{"result": ..., "N": ...}`` coarse to
    fine, so the finest level is last.

    ``design_order`` is the plan's declared rate, which sets the Richardson floor --
    see :func:`estimate_error`.

    ``exact_sampling`` is the plan's declared ``exact_sampling_components`` (F18) and
    ``state_dimension`` the spec's. **The bias term abstains only when every state
    component is exactly sampled**, because only then is there no discretization
    anywhere for the ladder's differences to be measuring. ``crn.abstention`` has
    excused that case since F18 and this function never consulted the field, so the
    ladder went on charging a bias to the schemes least able to carry one: measured on
    ``sde_hhw_call_zerocorr`` plan 1, which declares all three of three components
    exactly sampled, a bias of **0.0130** against a tolerance of 0.003 -- while the
    independent harness grades the same solver at 7.8% of its gate, so the charge is
    wrong by roughly fifty times, in the direction that condemns the plan.

    **A partially exact plan is not abstained on, and that is a correction made by
    measurement.** The first cut here abstained whenever *any* component was exactly
    sampled -- the conservative reading, and the one ``crn.abstention`` takes, since an
    order study that cannot see one leg cannot see the order. It is wrong for a
    *functional*: the graded quantity depends on the whole state, so a discretised leg
    biases it however exactly the other legs are drawn. Measured on
    ``sde_sabr_cev_call_i`` plan 1, which draws the forward exactly (component 0 of 2)
    and steps the volatility: its ladder shows ``d10/d21`` ratios near 4 at observed
    orders 1.90 to 2.29 -- a clean, real second-order convergence worth 5.7e-04 to
    1.5e-03 -- and abstaining on it dropped the plan's charged error by a factor of two,
    promoted it from third to first in its pool, and cost the selection a plan whose
    true error was 2.3x better. Abstention destroys evidence, so the bar for it is that
    there is provably no bias to measure, not that part of the state has none.

    Where ``state_dimension`` is unknown the declaration cannot be checked for
    completeness and nothing is abstained, which is the same safe direction.
    """
    decl = declared(thresholds)
    out = {"outcomes": {}, "errors": {}, "values": {}, "detail": {},
           "gate_failed": [], "not_reported_gated": [], "not_reported": [],
           "functionals_ok": "unavailable"}
    if not decl:
        return out
    exact_sampling = tuple(exact_sampling or ())
    try:
        dimension = int(state_dimension)
    except (TypeError, ValueError):
        dimension = None
    fully_exact = (dimension is not None and dimension > 0
                   and set(exact_sampling) >= set(range(dimension)))

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
        rel_err, detail = estimate_error(values, design_order=design_order)
        detail["gate"] = gate
        detail["rel_tol"] = tol
        if exact_sampling:
            detail["exact_sampling_components"] = list(exact_sampling)
            detail["state_dimension"] = dimension
        if fully_exact:
            # The estimate is kept in `detail` -- it is what the ladder saw and a
            # reader has to be able to disagree -- and withheld from `errors`, which
            # is what `_graded_functionals` charges and what the gate below reads.
            detail["bias_abstained"] = list(exact_sampling)
            detail["bias_abstained_reason"] = EXACT_SAMPLING_ABSTENTION
            detail["ladder_rel_err"] = rel_err
            out["detail"][name] = detail
            out["outcomes"][name] = "unresolved"
            continue
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
