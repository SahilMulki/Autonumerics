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


# --- edges of the table: measured before the tests were written ---------------

@pytest.mark.xfail(strict=True, reason=(
    "plan §2c: 'A cap requires a machine-readable reason.' certify() accepts "
    "{'score': 3} and substitutes 'capped by the evaluator with no reason given'. "
    "Measured through the driver: the honest canary drops to 3 on a reason-less cap"))
def test_an_agent_cap_without_a_reason_is_refused():
    """A downward cap is the agent's only authority and its justification is what
    makes it auditable. A cap with no reason should be recorded as rejected, not
    applied with a placeholder."""
    got = score(CLEAN_B, tier_b=True, d1_outcome="clean", operator_validated=True,
                agent_cap={"score": 3})
    assert got["score"] == 10
    assert any(c["source"] == "agent_cap_rejected" for c in got["caps"])


@pytest.mark.xfail(strict=True, reason=(
    "driver._reference_outcome returns 'unavailable (ValueError)' when check_reference "
    "raises, and tier_of matches only the exact string 'unavailable' -- so a Path-A "
    "plan whose reference check threw falls through to Tier C (8, self_convergence) "
    "instead of A- unavailable (9, analytic_unvalidated)"))
def test_a_reference_check_that_raises_is_priced_as_a_minus_unavailable():
    """§4c: 'the check could not run' is A- at 9. An exception inside the check is
    exactly that case, and must not be priced *below* it."""
    got = score(CLEAN_A, reference_outcome="unavailable (ValueError)")
    assert (got["score"], got["provenance"]) == (9, "analytic_unvalidated"), got


def test_a_reference_check_reporting_unavailable_exactly_is_a_minus():
    """The control: the exact string is priced as designed."""
    got = score(CLEAN_A, reference_outcome="unavailable")
    assert (got["score"], got["provenance"]) == (9, "analytic_unvalidated")


def test_a_crash_outranks_a_gate_violation():
    """A crash gathered no evidence, so it carries no tier and no tag: 1, 'none',
    whatever else the evidence dict says."""
    got = score(CLEAN_B, crashed=True, gate_violation=True, tier_b=True,
                d1_outcome="clean", operator_validated=True)
    assert (got["score"], got["provenance"]) == (1, "none")
    assert got["bound_by"] in ("rubric", "certification")


def test_order_check_off_still_lets_a_failing_asymptotic_guard_score_four():
    """Plan §12a (2): ``order_check: false`` makes ``order_ok`` vacuously true but
    the asymptotic guards still run, and a failing one still returns 4 -- a
    spec-level opt-out must not become a tier-level one."""
    got = score(CLEAN_B, order_ok=True, asymptotic=False, converged=False,
                tier_b="unavailable", d1_outcome="unavailable")
    assert got["score"] == 4
    assert "asymptotic" in got["reason"]


SKIPS = ("not_run", "unavailable", "unresolved", "not_reported", "waived_chaotic",
         "faulty_probe")


@pytest.mark.parametrize("skip", SKIPS)
def test_every_skip_marker_is_neither_pass_nor_fail_at_certify(skip):
    """Whichever check reports it, a skip scores exactly as ``None`` does: it can
    never lift a score and never sink one. The one exception the manual states is
    ``temporal_ok``, which is forgiven when skipped for cost (§24)."""
    for field in ("tier_b", "d1_outcome", "orders_ok", "dynkin_ok", "surrogate_ok"):
        with_skip = score(CLEAN_B, **{field: skip})
        with_none = score(CLEAN_B, **{field: None})
        assert with_skip["score"] == with_none["score"], (field, skip)
    assert score(CLEAN_B, tier_b=True, temporal_ok=skip)["score"] == \
        score(CLEAN_B, tier_b=True, temporal_ok=True)["score"]
    assert score(CLEAN_B, tier_b=True, temporal_ok=False)["score"] < 10


def test_the_drivers_supply_every_evidence_key_certify_reads():
    """Static: a driver that forgets a key silently hands ``certify`` a ``None``.
    Collect the keys each scoring function reads and check the driver that feeds
    it names each one -- the shared functions (tier, ceiling, caps) by both
    drivers, the PDE rubrics by the PDE driver, the SDE rubrics by the SDE one."""
    import inspect
    import re

    from verifylib.kernel import driver
    from verifylib.kernel import score as score_mod
    from verifylib.kernel.sde import driver as sde_driver

    def keys_of(fn):
        src = inspect.getsource(fn)
        found = set(re.findall(r"""evidence\.get\(["']([a-z_]+)["']""", src))
        found |= set(re.findall(r"""\bv\(["']([a-z_]+)["']\)""", src))
        return found - {"kind", "path"}

    shared = set()
    for fn in (score_mod.tier_of, score_mod.d1_is_break, score_mod.gates_clean,
               score_mod.ceiling, score_mod.certify):
        shared |= keys_of(fn)
    pde_only = keys_of(score_mod.rubric_pde) | keys_of(score_mod.rubric_pde_analytic)
    sde_only = keys_of(score_mod.rubric_sde) | keys_of(score_mod.rubric_sde_analytic)
    assert shared and pde_only and sde_only, "no evidence keys found -- did score.py change?"

    # Two caps are PDE-only by construction: an SDE has no spatial ladder to
    # interpolate and declares no scalar functionals. Their absence means "no cap".
    pde_only_caps = {"interpolated", "not_reported_functionals"}

    pde = inspect.getsource(driver._evidence)
    sde = inspect.getsource(sde_driver.evaluate_sde)
    for key in sorted(shared | pde_only):
        assert f'"{key}"' in pde, f"PDE evidence never sets {key!r}"
    for key in sorted((shared | sde_only) - pde_only_caps):
        assert f'"{key}"' in sde, f"SDE evidence never sets {key!r}"


def test_an_imported_order_carries_the_larger_safety_factor():
    """§7a precondition 3, in the direction rev 1 got wrong: an order imported
    from the probe understates the error if it is too high, so the GCI uses
    Roache's ``Fs = 3.0`` rather than 1.25, and the estimate is *larger* than the
    measured-order estimate would be for the same ``d10``."""
    import numpy as np

    from verifylib.kernel import richardson as rich

    x = np.linspace(0, 1, 33)
    u0 = np.sin(np.pi * x)
    u1 = np.sin(np.pi * np.linspace(0, 1, 65)) * (1 + 1e-3)
    probe = rich.from_probe([u0, u1], [[x], [np.linspace(0, 1, 65)]], 2.0, 2.0)
    assert probe["fs"] == rich.FS_IMPORTED == 3.0
    _, gci_measured = rich.estimate(probe["d10"], probe["d10"], u1, 2.0, fs=rich.FS_MEASURED)
    assert probe["gci"] > gci_measured
    assert probe["gci"] == pytest.approx(gci_measured * 3.0 / 1.25)
