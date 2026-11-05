"""Declared scalar quantities of interest: transported, measured, and gated.

The gap these close was found by running the pipeline, not by reading the code.
``benchmark/verify.py`` grew a ``functionals`` return key so an eigenvalue problem
could be scored on its eigenvalue as well as its eigenfunction, and the kernel's
sandbox -- which documents itself as speaking *the whole* solver contract -- silently
dropped it. The formulator noticed and wrote it into its own spec as a limitation:
"The kernel sandbox does not transport `functionals`, so this gate cannot be evaluated
by cli.py." A declared gate nobody can evaluate is exactly what plan §6 calls worse
than no gate, because it reads as evidence.
"""
import pytest

from verifylib.kernel import functionals, sandbox
from verifylib.kernel.score import certify

SOLVER = """
import numpy as np

def solve_pde(N, override=None):
    x = np.linspace(0.0, 1.0, N)
    u = np.sin(np.pi * x)
    # A converging eigenvalue: exact value 2.0, error ~ 1/N^2.
    return {"numerical_solution": u, "grid": {"x": x}, "t_final": 0.0,
            "functionals": {"lambda_1": 2.0 - 1.0 / (N * N)}}
"""

MESSY = """
import numpy as np

def solve_pde(N, override=None):
    x = np.linspace(0.0, 1.0, N)
    return {"numerical_solution": np.sin(np.pi * x), "grid": {"x": x}, "t_final": 0.0,
            "functionals": {"good": 1.5, "an_array": np.ones(3), "a_string": "nope",
                            "not_finite": float("nan")}}
"""


def _write(tmp_path, text):
    (tmp_path / "solver.py").write_text(text)
    return str(tmp_path)


def _runs(*values, key="lambda_1"):
    """A ladder whose levels reported ``values``, coarse to fine."""
    return [{"N": 8 * 2 ** i, "result": {"functionals": {key: v}}}
            for i, v in enumerate(values)]


# --- transport ---------------------------------------------------------------


def test_functionals_cross_the_process_boundary(tmp_path):
    got = sandbox.run(_write(tmp_path, SOLVER), "pde", {"N": 9})
    assert got["status"] == "ok"
    assert got["result"]["functionals"] == pytest.approx({"lambda_1": 2.0 - 1 / 81})


def test_a_non_scalar_functional_is_dropped_rather_than_crashing_the_solve(tmp_path):
    """A malformed entry must reach the checks as ``not_reported``, which is a
    measurement, rather than as ``crashed``, which would blame the solver for the
    wrong thing."""
    got = sandbox.run(_write(tmp_path, MESSY), "pde", {"N": 9})
    assert got["status"] == "ok"
    assert got["result"]["functionals"] == {"good": 1.5}


def test_a_solver_reporting_no_functionals_still_round_trips(tmp_path):
    plain = "import numpy as np\n\ndef solve_pde(N, override=None):\n" \
            "    x = np.linspace(0.0, 1.0, N)\n" \
            "    return {'numerical_solution': x, 'grid': {'x': x}, 't_final': 0.0}\n"
    got = sandbox.run(_write(tmp_path, plain), "pde", {"N": 9})
    assert got["status"] == "ok"
    assert "functionals" not in got["result"]


# --- the error estimate ------------------------------------------------------


def test_three_levels_give_richardson_and_two_give_the_raw_difference():
    """Second-order convergence toward 2.0: 1.9, 1.975, 1.99375."""
    rel3, detail3 = functionals.estimate_error([1.9, 1.975, 1.99375])
    assert detail3["observed_order"] == pytest.approx(2.0, abs=1e-9)
    # e_est = |d21| / (2^2 - 1) = 0.01875/3
    assert rel3 == pytest.approx((0.01875 / 3.0) / 1.99375, rel=1e-9)

    rel2, detail2 = functionals.estimate_error([1.975, 1.99375])
    assert detail2["observed_order"] is None
    assert rel2 == pytest.approx(0.01875 / 1.99375, rel=1e-9)
    assert rel2 > rel3, "two levels must not report a smaller error than three"


def test_an_apparent_order_above_the_clamp_cannot_understate_the_error():
    """A too-high order *understates* the error, which is the direction that turns a
    fail into a pass -- so it is clamped (plan §7a's correction)."""
    rel, detail = functionals.estimate_error([1.0, 1.5, 1.5 + 1e-12])
    assert detail["observed_order"] <= functionals.MAX_INFERRED_ORDER


def test_successive_levels_at_round_off_are_converged():
    rel, detail = functionals.estimate_error([2.0, 2.0, 2.0])
    assert detail["at_roundoff"] is True
    assert rel == 0.0


def test_one_level_measures_nothing():
    rel, _ = functionals.estimate_error([2.0])
    assert rel is None


# --- the declared check ------------------------------------------------------


def test_a_converged_functional_is_clean():
    thresholds = {"functionals": {"lambda_1": {"rel_tol": 1e-2, "gate": True}}}
    got = functionals.run_declared(thresholds, _runs(1.9, 1.975, 1.99375))
    assert got["outcomes"]["lambda_1"] == "clean"
    assert got["functionals_ok"] is True
    assert got["gate_failed"] == []


def test_a_functional_short_of_its_tolerance_is_stalled_and_gates():
    thresholds = {"functionals": {"lambda_1": {"rel_tol": 1e-6, "gate": True}}}
    got = functionals.run_declared(thresholds, _runs(1.9, 1.975, 1.99375))
    assert got["outcomes"]["lambda_1"] == "stalled"
    assert got["gate_failed"] == ["lambda_1"]
    assert got["functionals_ok"] is False


def test_an_ungated_functional_is_measured_but_never_sinks_the_run():
    thresholds = {"functionals": {"lambda_1": {"rel_tol": 1e-9, "gate": False}}}
    got = functionals.run_declared(thresholds, _runs(1.9, 1.975, 1.99375))
    assert got["outcomes"]["lambda_1"] == "stalled"
    assert got["gate_failed"] == []
    assert got["functionals_ok"] is True


def test_a_declared_functional_the_solver_never_reported_is_not_a_pass():
    thresholds = {"functionals": {"lambda_1": {"rel_tol": 1e-3, "gate": True}}}
    runs = [{"N": 8, "result": {}}, {"N": 16, "result": {}}]
    got = functionals.run_declared(thresholds, runs)
    assert got["outcomes"]["lambda_1"] == "not_reported"
    assert got["not_reported_gated"] == ["lambda_1"]
    # Not a failure either: nothing was measured, so nothing is claimed.
    assert got["functionals_ok"] == "unavailable"


def test_declaring_nothing_is_a_no_op():
    got = functionals.run_declared({}, _runs(1.9, 1.975))
    assert got["functionals_ok"] == "unavailable"
    assert got["outcomes"] == {}


def test_a_malformed_declaration_is_ignored_rather_than_crashing():
    assert functionals.declared({"functionals": ["lambda_1"]}) == {}
    assert functionals.declared({"functionals": {"a": "not-a-dict"}}) == {}


# --- the score consequence ---------------------------------------------------


def _evidence(**over):
    base = {"kind": "pde", "path": "B", "crashed": False, "has_reference": False,
            "reference_outcome": None, "tier_b": True, "surrogate_ok": "unavailable",
            "d1_outcome": "clean", "operator_validated": True, "converged": True,
            "converged_measurable": True, "order_ok": True, "invariants_ok": True,
            "non_gate_invariants_ok": True, "constraints_ok": True, "temporal_ok": True,
            "asymptotic": True, "shrinking": True, "gate_violation": False,
            "not_reported_invariants": [], "interpolated": False, "any_test_ran": True,
            "e_fine": 1e-4, "rel_err_tol": 1e-2, "chaotic": False, "agent_cap": None}
    base.update(over)
    return base


def test_an_unevaluable_gated_functional_caps_the_score_at_nine():
    clean = certify(_evidence())
    capped = certify(_evidence(not_reported_functionals=["lambda_1"]))
    assert clean["score"] > capped["score"]
    assert capped["score"] == 9
    assert any(c["source"] == "not_reported_functional" for c in capped["caps"])


def test_the_cap_is_absent_when_nothing_is_declared():
    got = certify(_evidence(not_reported_functionals=[]))
    assert not any(c["source"] == "not_reported_functional" for c in got["caps"])


# --- the Richardson denominator's lower clamp (11-sde-accuracy-fixes.md §2) ----


#: ``sde_sabr_cev_call_ii_lowvol`` plan 3's own ladder, which is what put the clamp
#: there: three prices around 0.05 whose differences have stopped shrinking, read as
#: an order of 0.0866 and a denominator of 0.0619.
_FLAT_LADDER = (0.05 - 7.333e-05 - 6.906e-05, 0.05 - 6.906e-05, 0.05)


def test_a_near_zero_inferred_order_does_not_divide_the_estimate():
    """The measurement in :data:`functionals.ORDER_FLOOR_FRACTION`. Plan 3 holds the
    tightest ladder in its pool -- ``d21`` is 7.3x smaller than its sibling's -- and
    the denominator turned that into the pool's *worst* error estimate."""
    rel, detail = functionals.estimate_error(list(_FLAT_LADDER), design_order=0.5)
    assert detail["observed_order"] == pytest.approx(0.0866, abs=5e-3)
    assert detail["order_rejected"] is True
    # The two-level bound, not the extrapolation: d21 / scale.
    assert rel == pytest.approx(6.906e-05 / 0.05, rel=1e-9)
    # ...and the extrapolation would have been 16x worse, which is the whole point.
    inflated = (6.906e-05 / (2.0 ** detail["observed_order"] - 1.0)) / 0.05
    assert inflated / rel > 10.0


def test_a_healthy_ladder_still_gets_its_extrapolation():
    """The clamp must not cost the resolving power it exists to protect. Halving
    differences are a clean first order, so the denominator is 1.0 and the estimate
    is the difference -- but via Richardson, not via the fallback."""
    rel, detail = functionals.estimate_error([1.6, 1.8, 1.9], design_order=1.0)
    assert detail["observed_order"] == pytest.approx(1.0)
    assert "order_rejected" not in detail
    assert rel == pytest.approx(0.1 / 1.9)


def test_the_floor_follows_the_declared_design_order():
    """Half the design rate, so the same ladder is judged differently by a scheme that
    claims first order and one that claims a strong order of 0.5. An inferred 0.415 is
    a stalled first-order scheme and a converging half-order one, and a single constant
    cannot say both."""
    d21, d10 = 0.1, 0.1 * 2 ** 0.415              # p = log2(d10/d21) = 0.415
    ladder = [1.0, 1.0 + d10, 1.0 + d10 + d21]
    _, half = functionals.estimate_error(ladder, design_order=0.5)
    _, first = functionals.estimate_error(ladder, design_order=1.0)
    assert half["order_floor"] == pytest.approx(0.25)
    assert first["order_floor"] == pytest.approx(0.5)
    assert "order_rejected" not in half
    assert first["order_rejected"] is True


def test_an_undeclared_design_order_falls_back_and_says_so():
    """Declared-never-inferred: the kernel's own constant is used and is *named* as
    the kernel's, so a reader can tell it from a rate the plan stated."""
    _, detail = functionals.estimate_error(list(_FLAT_LADDER))
    assert detail["order_floor"] == functionals.DEFAULT_ORDER_FLOOR
    assert detail["order_floor_source"] == "kernel_default"
    _, declared_detail = functionals.estimate_error(list(_FLAT_LADDER), design_order=2)
    assert declared_detail["order_floor_source"] == "design_order"


def test_the_design_order_reaches_the_ladder_from_run_declared():
    """The clamp is worthless if the call site does not pass the rate. It did not --
    ``run_declared`` had no parameter for it."""
    thresholds = {"functionals": {"lambda_1": {"rel_tol": 1e-2, "gate": True}}}
    got = functionals.run_declared(thresholds, _runs(*_FLAT_LADDER), design_order=0.5)
    assert got["detail"]["lambda_1"]["order_rejected"] is True
    assert got["detail"]["lambda_1"]["order_floor"] == pytest.approx(0.25)


# --- F18's abstention, on the ladder this time (11-sde-accuracy-fixes.md §2) ---


def test_a_fully_exactly_sampled_plan_abstains_on_the_bias():
    """``crn.abstention`` has excused this since F18 and the ladder did not, so the
    scheme family least able to carry a discretization bias was charged one anyway.
    An abstention is not a pass: the outcome is a skip marker, the error is withheld
    from what `_graded_functionals` charges, and nothing is claimed either way.

    `sde_hhw_call_zerocorr` plan 1's shape -- three of three components drawn exactly,
    charged a bias of 0.0130 against a tolerance of 0.003 while the independent harness
    graded the same solver at 7.8% of its gate."""
    from verifylib.kernel import metrics

    thresholds = {"functionals": {"lambda_1": {"rel_tol": 1e-6, "gate": True}}}
    # A ladder that would be `stalled` on its own, so the abstention is doing the work.
    charged = functionals.run_declared(thresholds, _runs(1.9, 1.975, 1.99375))
    assert charged["outcomes"]["lambda_1"] == "stalled"
    assert charged["functionals_ok"] is False

    got = functionals.run_declared(thresholds, _runs(1.9, 1.975, 1.99375),
                                   exact_sampling=(0, 1), state_dimension=2)
    assert got["outcomes"]["lambda_1"] == "unresolved"
    assert metrics.is_skip(got["outcomes"]["lambda_1"])
    assert "lambda_1" not in got["errors"]
    assert got["gate_failed"] == []
    # Not a pass either -- the evidence is forfeited, as `crn` forfeits it.
    assert got["functionals_ok"] == "unavailable"


def test_a_partially_exact_plan_is_still_charged_its_bias():
    """The correction that the acceptance test forced, and the measurement behind it.

    The first cut abstained whenever *any* component was declared exact -- the
    conservative reading, and the one `crn.abstention` takes, because an order study
    that cannot see one leg cannot see the order. It is wrong for a *functional*: the
    graded quantity depends on the whole state, so one discretised leg biases it
    however exactly the others are drawn.

    Measured on `sde_sabr_cev_call_i` plan 1 -- forward drawn exactly (component 0 of
    2), volatility stepped. Its ladder reads observed orders 1.90 to 2.29 with
    `d10/d21` near 4: a clean real second-order convergence worth 5.7e-04 to 1.5e-03.
    Abstaining on it halved the plan's charged error, moved it from third to first in
    its pool, and cost the selection a plan whose true error was 2.3x better.
    """
    thresholds = {"functionals": {"lambda_1": {"rel_tol": 1e-6, "gate": True}}}
    got = functionals.run_declared(thresholds, _runs(1.9, 1.975, 1.99375),
                                   exact_sampling=(0,), state_dimension=2)
    assert got["outcomes"]["lambda_1"] == "stalled"
    assert "lambda_1" in got["errors"]
    assert "bias_abstained" not in got["detail"]["lambda_1"]
    # The declaration is still recorded, so a reader sees why it did not fire.
    assert got["detail"]["lambda_1"]["exact_sampling_components"] == [0]
    assert got["detail"]["lambda_1"]["state_dimension"] == 2


def test_an_unknown_state_dimension_does_not_abstain():
    """Completeness cannot be checked without the dimension, and abstention destroys
    evidence, so the unknown case takes the same safe direction."""
    thresholds = {"functionals": {"lambda_1": {"rel_tol": 1e-6, "gate": True}}}
    got = functionals.run_declared(thresholds, _runs(1.9, 1.975, 1.99375),
                                   exact_sampling=(0, 1))
    assert got["outcomes"]["lambda_1"] == "stalled"


def test_the_abstained_estimate_is_recorded_rather_than_discarded():
    """A reader has to be able to disagree with the abstention, which means seeing the
    number it withheld and the declaration it rested on."""
    thresholds = {"functionals": {"lambda_1": {"rel_tol": 1e-6, "gate": True}}}
    got = functionals.run_declared(thresholds, _runs(1.9, 1.975, 1.99375),
                                   exact_sampling=(0,), state_dimension=1)
    detail = got["detail"]["lambda_1"]
    assert detail["bias_abstained"] == [0]
    assert detail["ladder_rel_err"] > 0.0
    assert "every" in detail["bias_abstained_reason"]


def test_declaring_no_exact_components_changes_nothing():
    thresholds = {"functionals": {"lambda_1": {"rel_tol": 1e-2, "gate": True}}}
    base = functionals.run_declared(thresholds, _runs(1.9, 1.975, 1.99375))
    for empty in ((), None, []):
        got = functionals.run_declared(thresholds, _runs(1.9, 1.975, 1.99375),
                                       exact_sampling=empty, state_dimension=2)
        assert got["outcomes"] == base["outcomes"]
        assert got["errors"] == base["errors"]


def test_both_spellings_of_the_declaration_parse_to_the_same_components():
    """The field is written ``0, 1`` in some plans and ``[0, 1]`` in others, and the
    abstention would be silently empty -- reintroducing the whole defect -- if one of
    the two parsed to nothing."""
    from verifylib.kernel.plan_meta import parse_components

    assert parse_components("0, 1") == (0, 1)
    assert parse_components("[0, 1]") == (0, 1)
    assert parse_components("0 1") == (0, 1)
    # One unparseable entry voids the declaration rather than half-honouring it.
    assert parse_components("0, two") == ()


# --- the declared ordering between functionals (11-sde-accuracy-fixes.md §4) ---


_ORDERED = ("price is monotone decreasing in strike and S0-ordered: "
            "price_S75 < price_S100 < price_S125")


def test_a_declared_ordering_is_checked_against_the_values():
    """`verification.expected_cross_check` was declared in a live spec and read by
    nothing. An ordering is a stronger claim about a multi-functional solver than any
    one tolerance: three prices from one set of paths can each sit inside their own
    gate while being out of order, which is a mis-wired strike rather than an accuracy
    problem -- and no per-functional error can see it, because it is a relationship
    between them."""
    ok, detail = functionals.cross_check(
        _ORDERED, {"price_S75": 4.0, "price_S100": 12.0, "price_S125": 30.0})
    assert ok is True
    assert [r["held"] for r in detail["relations"]] == [True, True]
    assert detail["violations"] == []


def test_a_violated_ordering_names_which_relation_broke():
    ok, detail = functionals.cross_check(
        _ORDERED, {"price_S75": 30.0, "price_S100": 12.0, "price_S125": 4.0})
    assert ok is False
    assert len(detail["violations"]) == 2
    assert "price_S75 (30)" in detail["violations"][0]


def test_only_the_relations_are_read_and_never_the_prose():
    """The claim is the chain; everything around it is a sentence. A statement whose
    prose happens to mention two functionals without comparing them states nothing
    checkable, and must come back `unavailable` rather than True."""
    outcome, detail = functionals.cross_check(
        "price_S75 and price_S100 are both priced from the same variance paths",
        {"price_S75": 4.0, "price_S100": 12.0})
    assert outcome == "unavailable"
    assert detail["relations"] == []


def test_a_chain_broken_by_anything_but_whitespace_is_not_one_claim():
    """`a < 3, b < 4` is two statements about constants, not an ordering of a against
    b. Tokens have to be adjacent."""
    outcome, _ = functionals.cross_check(
        "price_S75 < 3, price_S100 < price_S125",
        {"price_S75": 1.0, "price_S100": 9.0, "price_S125": 4.0})
    # The second half *is* a claim, and it is violated; the first half is not read.
    assert outcome is False


def test_the_other_comparators_are_honoured():
    values = {"a": 2.0, "b": 2.0}
    assert functionals.cross_check("a <= b", values)[0] is True
    assert functionals.cross_check("a >= b", values)[0] is True
    assert functionals.cross_check("a < b", values)[0] is False
    assert functionals.cross_check("a > b", values)[0] is False


def test_no_declaration_and_no_values_are_both_unavailable():
    for statement, values in ((None, {"a": 1.0}), (_ORDERED, {}), (_ORDERED, None),
                              ("", {"a": 1.0})):
        assert functionals.cross_check(statement, values)[0] == "unavailable"


def test_a_relation_naming_a_functional_the_solver_did_not_return_is_skipped():
    """Not a failure: an unreturned functional is already `not_reported` elsewhere, and
    charging it here would punish one gap twice."""
    outcome, detail = functionals.cross_check(
        _ORDERED, {"price_S75": 4.0, "price_S100": 12.0})
    assert outcome is True
    assert len(detail["relations"]) == 1


def test_the_floor_never_exceeds_the_order_the_ladder_can_report():
    """`MAX_INFERRED_ORDER` clamps the order *upward* at 4, so a floor above 4 can never
    be cleared and would disable the extrapolation outright on a high-order scheme --
    `pde_kdv_zabusky_kruskal`'s plan 1 declares `spatial_order: 8`. Capped, the rule
    reads "trusted whenever the ladder reaches the clamp", which is the strongest claim
    available about a scheme converging faster than the estimator can measure."""
    assert functionals.order_floor(8)[0] == pytest.approx(functionals.MAX_INFERRED_ORDER)
    assert functionals.order_floor(20)[0] == pytest.approx(functionals.MAX_INFERRED_ORDER)
    # A ladder at the clamp still gets its denominator.
    _, detail = functionals.estimate_error([0.0, 16.0, 17.0], design_order=8)
    assert detail["observed_order"] == pytest.approx(functionals.MAX_INFERRED_ORDER)
    assert "order_rejected" not in detail
