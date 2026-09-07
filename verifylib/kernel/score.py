"""The rubrics, as code: manual §12 and §23, under plan §4c's certification table.

**Why the kernel scores at all.** The rubrics are decision tables with no
judgement in them. Manual §12 reads ``verified and converged and order_ok and
invariants_ok and constraints_ok and temporal_ok -> 10``; §23 is its SDE twin; §4c
is the same shape. Every input is a boolean the kernel already computed. Handing
those booleans to a language model and asking it to apply the table is a pure
loss: it adds no information and adds a drift surface. The measured shape of that
drift is ``workspace/pde_kuramoto_sivashinsky``, where ``observed_order: 0.003``
and ``observed_order: 10.9`` both arrived at 10.

**The agent keeps one authority, and it is asymmetric: it may cap the score
downward, never raise it.** A cap needs a machine-readable reason, and it is the
escape hatch for what no check measures -- a ``SOLUTION.md`` describing a
different scheme than ``solver.py`` implements, an answer smuggled in as a lookup
table, a plan that "converges" by returning its input. It is the shape
``benchmark/verify.py``'s ``ledger_audit`` already uses (``score_cap: 3``).
Asymmetry is what preserves the anti-drift property: an agent that cannot inflate
cannot launder a 4 into a 10, and one that can deflate can still stop a wrong 10
from shipping.

**Where §4c and manual §12 disagree, §4c wins**, because §4c is the manual
tightened and the manual is being reframed as this module's specification. The one
substantive difference: §12 gives 9 for a clean self-convergence run with no
Tier-B evidence, and §4c caps that at **8** (``self_convergence``), reserving 9
for a run with exactly one circularity break (``manufactured_partial``) and 10 for
two. Existing Path-B plans that scored 9 on self-convergence alone therefore score
8 here. That is the intended tightening, not a regression: "a 10 without a closed
form requires two independent circularity breaks", and 9 should mean one.
"""

from __future__ import annotations

from .metrics import failed, is_skip, passed

#: Most to least, per project_manual.md. The conductor ranks on this order.
PROVENANCE_ORDER = ("analytic", "analytic_unvalidated", "surrogate", "manufactured",
                    "manufactured_partial", "self_convergence", "none")


def provenance_rank(tag):
    try:
        return PROVENANCE_ORDER.index(tag)
    except ValueError:
        return len(PROVENANCE_ORDER)


#: The evidence tier reached, highest first.
TIERS = ("A", "A_quoted", "A_unavailable", "A_prime", "B", "C", "none")


def tier_of(evidence):  # noqa: D401
    """The highest tier the run actually reached.

    ``A_quoted`` and ``A_unavailable`` are both provenance ``analytic_unvalidated``
    but they are **not** the same evidence, which is the hole guardrails §14 C4
    opened and did not price. A verbatim quote is evidence external to the pipeline
    -- someone wrote that formula in ``problem.md``, and ``schema.check_source_text``
    confirms it contains arithmetic. "The check could not run" is no evidence at
    all, and should not buy the same score. The split reads ``reference_outcome``
    from the metrics rather than the provenance tag, so C4's total order is
    untouched.
    """
    outcome = evidence.get("reference_outcome")
    if evidence.get("has_reference"):
        if outcome in ("validated", "validated_off_singularity"):
            return "A"
        if outcome == "quoted":
            return "A_quoted"
        if outcome == "unavailable":
            return "A_unavailable"
    if passed(evidence.get("surrogate_ok")):
        return "A_prime"
    if passed(evidence.get("tier_b")):
        return "B"
    if evidence.get("converged_measurable"):
        return "C"
    return "none"


def d1_is_break(evidence):
    """D1 counts as a circularity break only with a validated operator.

    B and D1 break the circle in different directions -- MMS checks the
    *discretization* against a known solution of a *modified* problem, D1 checks
    the *produced field* against the *stated* operator on the real problem, where
    the shock, layer or stiffness the manufactured problem lacks actually lives.
    But they are independent only given a correct operator: a misread equation
    fools both. Rev 1 asserted the independence without the precondition.
    """
    return (evidence.get("d1_outcome") == "clean"
            and bool(evidence.get("operator_validated")))


def gates_clean(evidence):
    """D2 clean, and D1 not stalled with a validated operator."""
    if evidence.get("gate_violation"):
        return False
    return not (evidence.get("d1_outcome") == "stalled"
                and evidence.get("operator_validated"))


def ceiling(evidence):
    """Plan §4c. Returns ``(max_score, provenance, reason)``.

    Every row is a conjunction of booleans the kernel already has, which is exactly
    why it belongs in code rather than in a markdown table an agent reinterprets
    once per cycle.
    """
    if evidence.get("crashed"):
        # A crash gathered no evidence, so it carries no tier and no tag. Manual
        # §25: the absence of ground truth is a property of the problem, but a
        # crash is a property of the code, and the two must not share a label.
        return 1, "none", "the solver crashed"
    if not gates_clean(evidence):
        why = ("a gate:true structural violation" if evidence.get("gate_violation")
               else "D1 stalled with a validated operator: the residual does not fall "
                    "under refinement and the operator it is measured against is known "
                    "to be right")
        return 3, _fallback_tag(evidence), why
    if not evidence.get("any_test_ran"):
        return 2, "none", "no test in the manual could be run"

    tier = tier_of(evidence)
    c_clean = passed(evidence.get("converged")) and passed(evidence.get("order_ok"))
    if tier == "A":
        return 10, "analytic", "Tier A: the closed form was validated against the "\
                               "spec's own operator, IC and BC"
    if tier == "A_quoted":
        return 10, "analytic_unvalidated", "Tier A-: accepted on a verbatim quote from "\
                                           "problem.md"
    if tier == "A_unavailable":
        return 9, "analytic_unvalidated", "Tier A-: the reference check could not run, "\
                                          "which is no evidence rather than weak evidence"
    if tier == "A_prime":
        return 10, "surrogate", "Tier A': a deterministic surrogate passed its own trust guard"
    if tier == "B":
        if d1_is_break(evidence) and c_clean:
            return 10, "manufactured", "two independent circularity breaks: Tier B and a "\
                                       "clean D1 against a validated operator"
        if c_clean:
            return 9, "manufactured_partial", "one circularity break (Tier B); D1 is "\
                                              f"{evidence.get('d1_outcome')}"
        return 8, "manufactured_partial", "Tier B evidence, but self-convergence is not clean"
    if d1_is_break(evidence) and c_clean:
        return 9, "manufactured_partial", "one circularity break (D1 against a validated "\
                                          "operator); no Tier-B evidence"
    if tier == "C":
        return 8, "self_convergence", "self-convergence and the gates only: no circularity "\
                                      "break, so this cannot reach 9"
    return 2, "none", "no test in the manual could be run"


def _fallback_tag(evidence):
    """A gate violation does not change what evidence was gathered (§4c: "tag
    unchanged"), so the tag still names the tier that was reached."""
    tier = tier_of(evidence)
    return {"A": "analytic", "A_quoted": "analytic_unvalidated",
            "A_unavailable": "analytic_unvalidated", "A_prime": "surrogate",
            "B": "manufactured", "C": "self_convergence"}.get(tier, "none")


# --- the rubrics --------------------------------------------------------------

def rubric_pde_analytic(evidence):
    """``project_manual.md``'s analytic PDE rubric, for Path A. ``(score, reason)``.

    Kept separate from manual §12, which is explicitly the *no-analytic-solution*
    rubric: §12 requires Tier-B evidence for a 10, and on Path A the closed form is
    strictly better evidence than a manufactured one, so requiring MMS there would
    fail every correct Path-A plan. Running one rubric for both paths was the shape
    of the bug, not a simplification.
    """
    if evidence.get("crashed"):
        return 1, "the solver crashed"
    if evidence.get("gate_violation"):
        return 3, "a hard structural gate is violated, however accurate the field is"
    err = evidence.get("e_fine")
    tol = float(evidence.get("rel_err_tol") or 0.01)
    if err is None:
        return 2, "no error could be measured against the closed form"
    err = float(err)
    order_ok = passed(evidence.get("order_ok"))
    if err < tol and order_ok and passed(evidence.get("invariants_ok")):
        return 10, f"error {err:.3e} < {tol:g} against the closed form, order clears the "\
                   f"floor, invariants and structural constraints hold"
    if err < tol and not order_ok:
        return 7, f"error {err:.3e} clears the tolerance but the observed order does not "\
                  f"clear its floor: accurate on one grid, not converging"
    if err < tol:
        return 5, f"error {err:.3e} clears the tolerance and the order clears its floor, "\
                  f"but a declared invariant drifts materially"
    if err < 5 * tol:
        return 8, f"error {err:.3e} is within 5x the tolerance"
    if err < 20 * tol:
        return 6, f"error {err:.3e} is within 20x the tolerance"
    if err < 50 * tol:
        return 4, f"error {err:.3e} is within 50x the tolerance"
    return 2, f"the code ran but the error is {err:.3e}, at least 50x the tolerance"


def rubric_sde_analytic(evidence):
    """``project_manual.md``'s analytic SDE rubric, plus §14's mandatory CI logic."""
    if evidence.get("crashed"):
        return 1, "the solver crashed or returned non-finite output"
    if evidence.get("gate_violation"):
        return 4, "a gate constraint is violated"
    if failed(evidence.get("resolved")):
        return 6, "MC-inconclusive: the error bars are too wide to decide. Raise "\
                  "num_paths -- this is not a solver bug"
    if passed(evidence.get("moments_ok")) and passed(evidence.get("resolved")):
        return 10, "every checked moment is within tolerance of the exact moments, and "\
                   "the estimates are MC-resolved"
    if passed(evidence.get("variance_ok")):
        return 8, "the variance passes; the mean is slightly above threshold"
    return 4, "the variance check fails, though the code ran cleanly"


def rubric_pde(evidence):
    """Manual §12, verbatim as a decision table. ``(score, reason)``."""
    if evidence.get("crashed"):
        return 1, "the solver crashed"
    v = evidence.get
    verified, converged = passed(v("tier_b")), passed(v("converged"))
    order_ok, invariants_ok = passed(v("order_ok")), passed(v("invariants_ok"))
    constraints_ok, temporal_ok = passed(v("constraints_ok")), passed(v("temporal_ok"))
    # A check skipped for cost never counts against a plan (manual §24, §25). Only
    # temporal isolation is ever skipped for cost, and only when the plan is
    # already failing for a reason that has nothing to do with dt.
    if is_skip(v("temporal_ok")):
        temporal_ok = True

    if v("gate_violation"):
        return 3, "a hard-gate structural constraint or gate invariant is violated"
    if verified and converged and order_ok and invariants_ok and constraints_ok and temporal_ok:
        return 10, "Tier-B evidence, converged, order clears the floor, invariants and "\
                   "constraints hold, temporal error subdominant"
    if converged and order_ok and invariants_ok and constraints_ok:
        return 9, "converged and clean, with no Tier-B evidence"
    if failed(v("asymptotic")):
        return 4, "the asymptotic guards fail: the order estimate is not stable across "\
                  "the ladder"
    if order_ok and invariants_ok and not converged:
        return 7, "converging, but not yet accurate enough"
    if converged and (not temporal_ok or failed(v("non_gate_invariants_ok"))):
        return 5, ("the temporal error dominates" if not temporal_ok
                   else "a non-gate invariant drifts materially")
    if converged and invariants_ok and constraints_ok and not order_ok:
        # §12's table has no row for "accurate at this grid, but converging slower
        # than the scheme claims", and a plan in that state used to fall through to
        # the default 4 with a message about instability that was simply untrue --
        # measured on the over-diffusion and fake-second-order canaries, which
        # converge cleanly at order 1 against a floor of 1.8. The analytic rubric in
        # project_manual.md scores exactly this case 7 ("accurate on one grid, not
        # converging"), so the no-closed-form table gets the same row.
        return 7, "accurate at the finest grid, but the observed order does not clear "\
                  "its floor: it is not converging at the rate the scheme claims"
    if failed(v("shrinking")):
        return 2, "the differences do not shrink at all under refinement"
    return 4, "the run neither converged nor produced a usable order estimate"


def rubric_sde(evidence):
    """Manual §23, verbatim as a decision table. ``(score, reason)``."""
    if evidence.get("crashed"):
        return 1, "the solver crashed or returned non-finite output"
    v = evidence.get
    surrogate, moments_ok = passed(v("surrogate_ok")), passed(v("moments_ok"))
    resolved, constraints_ok = passed(v("resolved")), passed(v("constraints_ok"))
    orders_ok, dynkin_ok = passed(v("orders_ok")), passed(v("dynkin_ok"))

    if v("gate_violation"):
        return 4, "a gate constraint is violated (negative paths, support breach)"
    if surrogate and moments_ok and resolved and constraints_ok:
        return 10, "a surrogate reference passed its trust guard and the moments match it"
    if orders_ok and dynkin_ok and constraints_ok and passed(v("richardson_stable")):
        return 9, "no surrogate: the orders, Dynkin and the constraints all hold and the "\
                  "Richardson moments are stable across the ladder"
    if failed(v("resolved")):
        return 6, "MC-inconclusive: the error bars are too wide to decide. Raise "\
                  "num_paths -- this is not a solver bug"
    if orders_ok and constraints_ok:
        return 7, "the Richardson-extrapolated moments are unstable, or Dynkin failed"
    return 3, "the estimator is erratic in dt: orders far from expectation, or the "\
              "differences are not shrinking"


# --- composition --------------------------------------------------------------

def certify(evidence):
    """The one function that assigns a score. Returns a dict, never a bare number.

    ``score = min(rubric, ceiling, every cap)``. The reasons are carried out so the
    review can say *which* rule bound the score rather than asserting the number.
    """
    kind, path = evidence.get("kind", "pde"), evidence.get("path", "B")
    rubric = {("pde", "A"): rubric_pde_analytic, ("pde", "B"): rubric_pde,
              ("sde", "A"): rubric_sde_analytic, ("sde", "B"): rubric_sde}[(kind, path)]
    rubric_score, rubric_reason = rubric(evidence)
    cap, provenance, ceiling_reason = ceiling(evidence)

    caps = [{"source": "rubric", "score": rubric_score, "reason": rubric_reason},
            {"source": "certification", "score": cap, "reason": ceiling_reason}]

    not_reported = list(evidence.get("not_reported_invariants") or [])
    if not_reported:
        # Manual §8's rule for a missing invariant_trace, generalised by plan §6:
        # a declared gate the kernel cannot evaluate is not a pass and not a
        # failure, and a plan carrying one cannot be certified.
        caps.append({"source": "not_reported", "score": 9,
                     "reason": f"declared invariant(s) {not_reported} could not be "
                               f"evaluated and are reported as not_reported, never as a "
                               f"pass"})
    if evidence.get("interpolated"):
        caps.append({"source": "interpolated", "score": 9,
                     "reason": "the ladder did not nest, so the order estimate rests on "
                               "interpolation and must not certify a 10 (manual §3)"})

    agent_cap = evidence.get("agent_cap")
    if isinstance(agent_cap, dict) and agent_cap.get("score") is not None:
        proposed = int(agent_cap["score"])
        floor_so_far = min(c["score"] for c in caps)
        if proposed > floor_so_far:
            # Recorded, and refused. The asymmetry is the whole anti-drift property.
            caps.append({"source": "agent_cap_rejected", "score": floor_so_far,
                         "reason": f"the agent proposed {proposed}, above the computed "
                                   f"{floor_so_far}. A cap may only move the score "
                                   f"downward"})
        else:
            caps.append({"source": "agent_cap", "score": proposed,
                         "reason": agent_cap.get("reason") or "capped by the evaluator "
                                                              "with no reason given"})

    score = min(c["score"] for c in caps)
    binding = [c for c in caps if c["score"] == score]
    return {"score": int(score), "provenance": provenance, "tier": tier_of(evidence),
            "caps": caps, "bound_by": binding[0]["source"],
            "reason": binding[0]["reason"], "d1_is_break": d1_is_break(evidence),
            "gates_clean": gates_clean(evidence)}
