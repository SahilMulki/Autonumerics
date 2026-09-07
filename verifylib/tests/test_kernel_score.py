"""The scoring tables. These are the whole of §2c: the kernel emits the score.

Every case here is a row of ``verification_manual.md`` §12/§23 or of the plan's §4c
certification table, and the point of testing them exhaustively is that they are
the one place in this repo where changing a boolean silently changes what "10"
means.
"""
import pytest

from verifylib.kernel.score import PROVENANCE_ORDER, certify, provenance_rank

CLEAN_B = dict(kind="pde", path="B", any_test_ran=True, converged=True, order_ok=True,
               invariants_ok=True, non_gate_invariants_ok=True, constraints_ok=True,
               temporal_ok=True, converged_measurable=True, asymptotic=True,
               shrinking=True)
CLEAN_A = dict(CLEAN_B, path="A", has_reference=True, reference_outcome="validated",
               e_fine=2e-3, rel_err_tol=1e-2)


def score(base, **kw):
    return certify({**base, **kw})


# --- §4c: the certification table --------------------------------------------

def test_two_circularity_breaks_reach_ten():
    got = score(CLEAN_B, tier_b=True, d1_outcome="clean", operator_validated=True)
    assert (got["score"], got["provenance"]) == (10, "manufactured")
    assert got["d1_is_break"] is True


def test_one_break_caps_at_nine_whichever_it_is():
    """Tier B alone and a clean D1 alone are the same amount of evidence."""
    tier_b_only = score(CLEAN_B, tier_b=True, d1_outcome="unavailable")
    d1_only = score(CLEAN_B, d1_outcome="clean", operator_validated=True,
                    tier_b="unavailable")
    assert tier_b_only["score"] == d1_only["score"] == 9
    assert tier_b_only["provenance"] == d1_only["provenance"] == "manufactured_partial"


def test_self_convergence_alone_caps_at_eight():
    """§4c tightens manual §12, which gives 9 here. 9 is reserved for one
    circularity break, 10 for two -- so no break tops out at 8."""
    got = score(CLEAN_B, tier_b="unavailable", d1_outcome="unavailable")
    assert (got["score"], got["provenance"]) == (8, "self_convergence")
    assert got["bound_by"] == "certification"


def test_d1_is_not_a_break_without_a_validated_operator():
    """A misread equation makes the solver and the residual wrong the same way, so
    D1 is independent of Tier B only given a correct operator. Rev 1 of the plan
    asserted the independence without this precondition."""
    got = score(CLEAN_B, tier_b=True, d1_outcome="clean", operator_validated=False)
    assert (got["score"], got["provenance"]) == (9, "manufactured_partial")
    assert got["d1_is_break"] is False


def test_a_stalled_d1_gates_only_when_the_operator_was_validated():
    gated = score(CLEAN_B, tier_b=True, d1_outcome="stalled", operator_validated=True)
    assert gated["score"] == 3 and gated["gates_clean"] is False
    ungated = score(CLEAN_B, tier_b=True, d1_outcome="stalled", operator_validated=False)
    assert ungated["score"] == 9, "with an unaudited operator, both are suspects"


def test_a_gate_violation_is_three_and_keeps_its_tag():
    got = score(CLEAN_B, tier_b=True, d1_outcome="clean", operator_validated=True,
                gate_violation=True)
    assert got["score"] == 3
    assert got["provenance"] == "manufactured", "§4c: tag unchanged"


def test_the_a_minus_split_prices_a_quote_above_no_evidence():
    """Guardrails C4 opened this hole and did not price it: a verbatim quote is
    evidence external to the pipeline, and "the check could not run" is none."""
    quoted = score(CLEAN_A, reference_outcome="quoted")
    unavailable = score(CLEAN_A, reference_outcome="unavailable")
    assert quoted["score"] == 10 and unavailable["score"] == 9
    assert quoted["provenance"] == unavailable["provenance"] == "analytic_unvalidated"


def test_nothing_runnable_is_two_and_a_crash_is_one():
    nothing = score(CLEAN_B, any_test_ran=False, converged_measurable=False)
    assert (nothing["score"], nothing["provenance"]) == (2, "none")
    crash = score(CLEAN_B, crashed=True)
    assert (crash["score"], crash["provenance"]) == (1, "none")


# --- the two rubrics ----------------------------------------------------------

def test_the_analytic_path_does_not_require_tier_b():
    """§12 is the *no-analytic-solution* rubric and requires Tier B for a 10.
    Applying it on Path A, where the closed form is strictly better evidence, would
    fail every correct Path-A plan."""
    got = score(CLEAN_A)
    assert (got["score"], got["provenance"]) == (10, "analytic")
    assert got["bound_by"] == "rubric"


@pytest.mark.parametrize("err,expected", [(2e-3, 10), (3e-2, 8), (1.5e-1, 6),
                                          (4e-1, 4), (9e-1, 2)])
def test_the_analytic_rubric_bands(err, expected):
    assert score(CLEAN_A, e_fine=err)["score"] == expected


def test_accurate_but_under_converging_is_seven_not_four():
    """§12's table has no row for this and such a run used to fall through to 4
    with a message about instability that was simply untrue -- measured on the
    over-diffusion canary, which converges cleanly at order 1."""
    got = score(CLEAN_B, tier_b="unavailable", d1_outcome="unavailable", order_ok=False)
    assert got["score"] == 7
    assert "not converging at the rate" in got["reason"]


def test_the_sde_inconclusive_outcome_is_its_own_score():
    got = certify(dict(kind="sde", path="A", any_test_ran=True, has_reference=True,
                       reference_outcome="validated", resolved=False,
                       moments_ok="unavailable", constraints_ok=True,
                       converged_measurable=True))
    assert got["score"] == 6
    assert "not a solver bug" in got["reason"]


# --- caps ---------------------------------------------------------------------

def test_an_agent_may_cap_downward():
    got = score(CLEAN_B, tier_b=True, d1_outcome="clean", operator_validated=True,
                agent_cap={"score": 4, "reason": "SOLUTION.md describes a different scheme"})
    assert got["score"] == 4 and got["bound_by"] == "agent_cap"
    assert "different scheme" in got["reason"]


def test_an_agent_may_not_cap_upward_and_the_attempt_is_recorded():
    """The asymmetry is the whole anti-drift property: an agent that cannot inflate
    cannot launder a 4 into a 10."""
    got = score(CLEAN_B, tier_b="unavailable", d1_outcome="unavailable",
                agent_cap={"score": 10, "reason": "looks fine to me"})
    assert got["score"] == 8
    rejected = [c for c in got["caps"] if c["source"] == "agent_cap_rejected"]
    assert rejected and "only move the score downward" in rejected[0]["reason"]


def test_a_gated_invariant_the_kernel_cannot_evaluate_caps_at_nine():
    got = score(CLEAN_B, tier_b=True, d1_outcome="clean", operator_validated=True,
                not_reported_invariants=["energy_bounded"])
    assert got["score"] == 9 and got["bound_by"] == "not_reported"


def test_an_interpolated_ladder_cannot_certify_a_ten():
    got = score(CLEAN_B, tier_b=True, d1_outcome="clean", operator_validated=True,
                interpolated=True)
    assert got["score"] == 9 and got["bound_by"] == "interpolated"


# --- the provenance total order ----------------------------------------------

def test_the_provenance_order_is_total_and_matches_the_manual():
    assert PROVENANCE_ORDER == ("analytic", "analytic_unvalidated", "surrogate",
                                "manufactured", "manufactured_partial",
                                "self_convergence", "none")
    ranks = [provenance_rank(p) for p in PROVENANCE_ORDER]
    assert ranks == sorted(ranks) == list(range(len(PROVENANCE_ORDER)))
    assert provenance_rank("invented") > provenance_rank("none")
