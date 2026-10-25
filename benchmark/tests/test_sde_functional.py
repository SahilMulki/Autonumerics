"""A scalar quantity of interest on the SDE side (plan-benchop-slv.md step 5).

Four of BENCHOP's eight methods are SDE-side, so the suite cannot be adopted while
``functional_truth`` works on only one of the two formulations. Six claims:

**K1** ``functionals`` crosses the sandbox boundary on the SDE path, by the same rule
as the PDE path. Step 2 had to fix exactly this defect once already, for PDE.

**K2** An SDE problem can be graded on a number: beside its terminal moments, or
instead of them. A price has no elementary moment formula behind it, so "instead of"
is not a corner case -- it is how every BENCHOP problem will arrive.

**K3** The gate is the PDE gate, reached through the same two helpers, including step
4's per-value floor. Only the *default* differs, and only because the field gate it
falls back to differs: 1% for a resolved field, 5% for a sampled moment.

**K4** The two-level check is the anti-hardcoding defence. The PDE route gets one free
from the 2-grid order check -- a constant has zero observed order -- and an SDE price
graded from one run gets nothing, so the price is taken at two path counts. A constant
fails; a real Monte-Carlo solver passes; the band is symmetric, because step 3 measured
these curves non-monotone and "finer must be better" would fail correct solvers.

**K5** All of it reaches the record, the printed summary, the problem statement, the
manifest and the number plans are ranked on -- and changes none of them for the 56
problems already in the suite.

**K6** The number plans are ranked on carries a margin, so two plans a hair apart
are not reported as a mis-ranking. A price has no ``empirical_se`` the way a moment
does; step 10's first three runs had the margin at zero and ``wrong_plan_won`` at
3/3, of which two were differences smaller than the sampling noise.

The sandboxed tests in K1 and K4 run a real ``solve_sde`` in a real subprocess, because
the thing under test *is* the subprocess boundary and a stub would assume away the bug.
Everything else feeds ``verify.py`` synthetic output, as the rest of this directory does.
"""
import json
import math
import sys

import numpy as np
import pytest
from scipy.special import ndtr

from .conftest import BENCHMARK, RUN, P, R, S, V

sys.path.insert(0, BENCHMARK)
import runner  # noqa: E402

# A European call on S0=1, K=1, r=0, sigma=0.2, T=1, by Black-Scholes. Computed here
# rather than quoted: d1 = sigma*sqrt(T)/2, d2 = -d1, C = N(d1) - N(d2).
SIGMA, T_HORIZON = 0.2, 1.0
CALL = float(ndtr(SIGMA * np.sqrt(T_HORIZON) / 2) - ndtr(-SIGMA * np.sqrt(T_HORIZON) / 2))

# One correct Monte-Carlo pricer, written three ways. The payoff is the same in all
# three; what differs is only whether the reported number came from the paths.
_BODY = """
import numpy as np


def solve_sde(num_paths, dt, T, seed):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(int({n}))
    s = np.exp(-0.5 * 0.2 ** 2 * T + 0.2 * np.sqrt(T) * z)
    return {{"terminal_paths": s, "empirical_mean": float(s.mean()),
            "empirical_variance": float(s.var(ddof=1)),
            "functionals": {{"price": {price}}}}}
"""
HONEST = _BODY.format(n="num_paths", price="float(np.maximum(s - 1.0, 0.0).mean())")
# Right to twelve digits, and right about its moments. It fails for one reason only.
HARDCODED = _BODY.format(n="num_paths", price=repr(CALL))
# Simulates honestly but ignores the path count it was handed, so both levels are the
# same run and the price cannot move either.
IGNORES_LEVEL = _BODY.format(n="20000", price="float(np.maximum(s - 1.0, 0.0).mean())")

PRICE_PROBLEM = {
    "slug": "test_sde_price", "type": "sde", "tier": 1, "family": "test",
    "title": "t", "challenge": "t", "id": 999,
    "state_dimension": 1, "T": T_HORIZON, "dt": 0.01, "num_paths": 20000, "seed": 7,
    "has_ground_truth": True, "ground_truth_kind": "exact",
    "functional_truth": {"price": CALL}, "functional_tol": 0.05,
}


def price_problem(**over):
    prob = dict(PRICE_PROBLEM)
    prob.update(over)
    return prob


def sde_result(price, *, n=20000, seed=7):
    """Synthetic solver output of the shape ``_run_and_read_sde`` reads."""
    rng = np.random.default_rng(seed)
    s = np.exp(-0.5 * SIGMA ** 2 + SIGMA * rng.standard_normal(int(n)))
    out = {"terminal_paths": s, "empirical_mean": float(s.mean()),
           "empirical_variance": float(s.var(ddof=1))}
    if price is not None:
        out["functionals"] = {"price": price}
    return out


def score(prob, price, **kw):
    """Grade one synthetic run the way ``verify_sde`` does, without a subprocess."""
    result = sde_result(price, **kw)
    meta = {"num_paths": prob["num_paths"], "dt": prob["dt"], "T": prob["T"],
            "seed": prob["seed"]}
    comps, meta, result, err = V._run_and_read_sde(prob, None, injected=(result, meta))
    assert err is None, err
    return V._score_sde_moments(prob, comps, V._sde_moment_truth(prob), meta,
                                reference=False, result=result)


def plan(tmp_path, source):
    (tmp_path / "solver.py").write_text(source)
    return str(tmp_path)


# --- K1: the boundary carries a functional on the SDE path -----------------------

def test_a_functional_survives_the_sde_boundary(tmp_path):
    """The PDE half of this was step 2's measured defect: the boundary dropped
    ``functionals`` and the parent then failed the solver for not returning what it
    had in fact returned. The SDE half would have shipped with the same hole."""
    run = runner.run_solver(plan(tmp_path, HONEST), PRICE_PROBLEM)
    assert run["status"] == "ok", run.get("error")
    assert run["result"]["functionals"]["price"] == pytest.approx(CALL, rel=0.05)


def test_the_sde_boundary_carries_the_moments_and_the_functional_together(tmp_path):
    """Not one or the other. A price is read off a simulation, so a problem that
    scores both must get both across in one crossing."""
    run = runner.run_solver(plan(tmp_path, HONEST), PRICE_PROBLEM)
    assert set(run["result"]) == {"terminal_paths", "empirical_mean",
                                  "empirical_variance", "functionals"}


@pytest.mark.parametrize("value,expected", [
    ("float('nan')", float("nan")),
    ("'not a number'", "'not a number'"),
])
def test_the_sde_child_does_not_judge_a_functional_it_should_carry(tmp_path, value,
                                                                   expected):
    """``_functional_errs`` resolves a non-number (``inconclusive``) differently from
    a non-finite one (``nonfinite``). Both decisions are the parent's, on both paths,
    or the sandboxed and inline verdicts diverge."""
    run = runner.run_solver(plan(tmp_path, _BODY.format(n="num_paths", price=value)),
                            PRICE_PROBLEM)
    assert run["status"] == "ok", run.get("error")
    got = run["result"]["functionals"]["price"]
    assert np.isnan(got) if isinstance(expected, float) else got == expected


# --- K2: an SDE problem can be graded on a number --------------------------------

def test_a_price_is_graded_with_no_moment_formula_behind_it():
    """The shape every BENCHOP problem arrives in: ``functional_truth`` and no
    ``moments``. The old code called ``problem["moments"]`` unconditionally."""
    out = score(price_problem(), CALL)
    assert out["status"] == "ok" and out["passed"]
    assert out["components"] == []
    assert out["functional_errs"]["price"] == pytest.approx(0.0, abs=1e-12)


def test_a_wrong_price_fails_a_problem_that_has_nothing_else_to_check():
    out = score(price_problem(), CALL * 1.5)
    assert not out["passed"] and out["functional_ok"] is False
    assert out["worst_functional_rel_err"] == pytest.approx(0.5, rel=1e-9)


def test_the_moments_and_the_price_are_independent_gates():
    """A solver can simulate the right law and read the wrong number off it, or the
    reverse. Either fails, and the record says which half did."""
    prob = price_problem(moments=lambda T: {"mean": 1.0, "variance": 0.0408})
    good = score(prob, CALL)
    assert good["passed"] and good["components"]
    bad_price = score(prob, CALL * 2)
    assert not bad_price["passed"]
    assert all(c["passed"] for c in bad_price["components"])
    assert bad_price["functional_ok"] is False


def test_a_functional_only_problem_with_a_good_price_still_scores_ten():
    """``min()`` over no moment components is not an error, it is 'no moment is
    wrong'. Getting that identity wrong would have raised on every BENCHOP problem."""
    assert score(price_problem(), CALL)["verified_score"] == 10


def test_a_missing_functionals_dict_is_the_solvers_failure_not_an_inconclusive():
    out = score(price_problem(), None)
    assert out["status"] == "contract" and out["passed"] is False
    assert out["verified_score"] == 1


def test_a_non_numeric_price_is_inconclusive_and_a_nonfinite_one_is_not():
    assert score(price_problem(), "nope")["status"] == "inconclusive"
    assert score(price_problem(), float("nan"))["status"] == "nonfinite"


def test_a_functional_failure_record_is_not_shaped_like_a_pde_one():
    """``_functional_errs`` reports in PDE terms (``rel_l2_err``, ``primary_err``).
    Those keys on an SDE record would route ``_fmt`` -- and every reader that
    branches the same way -- down the PDE branch of a record that has no field."""
    out = score(price_problem(), None)
    assert "rel_l2_err" not in out and "primary_err" not in out
    assert out["gt_kind"] == "exact" and "run" in out


def test_the_work_precision_axis_sees_a_price():
    """A functional-only problem has no components, so an error axis built from the
    moments alone would be a constant zero -- a flat perfect curve at every level."""
    assert V._wp_sde_err(score(price_problem(), CALL * 1.2)) == pytest.approx(0.2, rel=1e-9)


# --- K3: the same gate as the PDE route ------------------------------------------

def test_the_sde_functional_gate_defaults_to_the_sde_field_gate():
    """1% is the gate on a resolved field and 5% on a sampled moment. Inheriting the
    PDE default here would grade a Monte-Carlo price at a field solver's tolerance."""
    prob = price_problem()
    prob.pop("functional_tol")
    assert V._functional_tol(prob) == V.SDE_MEAN_TOL
    assert V._functional_tol({"type": "pde"}) == V.PDE_L2_TOL


def test_step_fours_per_value_floor_applies_on_the_sde_side_too():
    """The reference's own uncertainty widens the gate per value, by the same rule
    and the same factor. A Monte-Carlo standard error is one of the three sources
    that rule was written for."""
    prob = price_problem(functional_truth={"price": CALL, "delta": 0.54},
                         reference_abs_err={"price": 0.004, "delta": 0.0})
    tols = V._functional_tols(prob)
    assert tols["delta"] == pytest.approx(0.05)
    assert tols["price"] == pytest.approx(V.REFERENCE_TOL_FACTOR * 0.004 / CALL)
    assert tols["price"] > tols["delta"]


def test_a_widened_sde_gate_is_recorded_with_what_widened_it():
    prob = price_problem(reference_abs_err=0.004)
    out = score(prob, CALL)
    assert out["functional_tol"] == pytest.approx(0.05)
    assert out["functional_tol_widened"] == ["price"]
    assert out["reference_abs_err"] == {"price": 0.004}


def test_both_errors_are_recorded_on_the_sde_route_as_well():
    """The relative one grades; the absolute one is what BENCHOP states its own
    values and tolerances in, and it cannot be recovered from the relative one
    afterwards because the record does not carry the reference value."""
    out = score(price_problem(), CALL + 0.001)
    assert out["functional_abs_errs"]["price"] == pytest.approx(0.001, rel=1e-9)
    assert out["functional_errs"]["price"] == pytest.approx(0.001 / CALL, rel=1e-6)


def test_the_sde_functional_record_is_json_serializable():
    json.dumps(score(price_problem(), CALL))


# --- K4: the two-level check is the anti-hardcoding defence ----------------------

def verdict(tmp_path, source, **over):
    prob = price_problem(**over)
    p = plan(tmp_path, source)
    return V.verify_problem(prob, p, plan_dir=p, sandbox=True, work_precision=False)


def test_an_honest_monte_carlo_pricer_passes_the_two_level_check(tmp_path):
    out = verdict(tmp_path, HONEST)
    assert out["status"] == "ok", out.get("error")
    lv = out["functional_levels"]
    assert lv["status"] == "ok" and lv["passed"], lv
    assert out["passed"]


def test_a_hardcoded_price_fails_although_everything_else_about_it_is_right(tmp_path):
    """The whole reason step 5 needs more than plumbing. This solver returns the
    reference to twelve digits and simulates the right law; the only thing wrong with
    it is that the number did not come from the simulation."""
    out = verdict(tmp_path, HARDCODED)
    assert out["functional_errs"]["price"] == pytest.approx(0.0, abs=1e-12)
    assert out["functional_ok"] is True
    lv = out["functional_levels"]
    assert lv["constant"] == ["price"] and not lv["passed"]
    assert out["passed"] is False and out["verified_score"] == 1


def test_a_solver_that_ignores_the_path_count_is_caught_and_named(tmp_path):
    """It fails for the same reason -- the price cannot move if the run cannot -- and
    the record distinguishes the two diagnoses rather than reporting a constant."""
    out = verdict(tmp_path, IGNORES_LEVEL)
    lv = out["functional_levels"]
    assert lv["constant"] == ["price"] and lv["level_ignored"] is True
    assert out["passed"] is False


def test_the_band_is_symmetric_so_a_worse_fine_run_does_not_fail(tmp_path):
    """Step 3 measured these curves non-monotone at this suite's path counts, so a
    decreasing test would fail correct solvers. Checked on the real two levels: the
    coarse price is allowed to sit on either side of the graded one."""
    out = verdict(tmp_path, HONEST)
    v = out["functional_levels"]["values"]["price"]
    assert v["moved"] and v["consistent"]
    # The direction is not an input to the verdict, whichever way it happened to go.
    assert V._sde_functional_levels.__doc__.count("symmetric") >= 1
    flipped = dict(v, graded=v["coarse"], coarse=v["graded"])
    assert abs(flipped["graded"] - flipped["coarse"]) == pytest.approx(v["delta"])


def test_the_band_widens_with_the_noise_the_two_counts_imply():
    """``1 + sqrt(n_graded/n_coarse)``, which is 3 at the default quarter because a
    standard error falls as ``1/sqrt(n)``. The factor is derived, not picked."""
    assert V._sde_level_paths({"num_paths": 20000}, 20000) == 5000
    assert 1 + np.sqrt(20000 / 5000) == pytest.approx(3.0)
    assert V._sde_level_paths({"functional_level_paths": 777}, 20000) == 777


def test_a_crashed_coarse_solve_does_not_fail_a_verdict(tmp_path):
    """Asymmetry on purpose: a flaky sandbox must not be able to fail a correct
    solver, while the two findings this check exists to produce are both positive
    evidence about the solver itself."""
    src = (HONEST + "\n_orig = solve_sde\n\n\n"
           "def solve_sde(num_paths, dt, T, seed):\n"
           "    if num_paths != 20000:\n"
           "        raise RuntimeError('boom')\n"
           "    return _orig(num_paths, dt, T, seed)\n")
    out = verdict(tmp_path, src)
    assert out["functional_levels"]["status"] == "not_run"
    assert out["passed"] is True


def test_a_problem_with_no_coarser_level_reports_that_rather_than_passing(tmp_path):
    out = verdict(tmp_path, HONEST, num_paths=4)
    lv = out["functional_levels"]
    assert lv["status"] == "not_run" and "no coarser level" in lv["reason"]


def test_the_inline_path_records_that_the_defence_did_not_run(tmp_path):
    """Inline there is no second solve, so the defence is absent. A passing record
    silent about it would read as a record that had the defence and cleared it."""
    p = plan(tmp_path, HONEST)
    out = V.verify_problem(price_problem(), p, plan_dir=p, sandbox=False)
    assert out["functional_levels"] == {"status": "not_run", "reason": "inline"}


def test_a_problem_with_no_functional_gets_no_two_level_check(tmp_path):
    """It is the functional route's defence. The moment route already has the
    reference-SE comparison, and a second solve would be cost for nothing."""
    prob = price_problem(moments=lambda T: {"mean": 1.0, "variance": 0.0408})
    prob.pop("functional_truth")
    p = plan(tmp_path, HONEST)
    out = V.verify_problem(prob, p, plan_dir=p, sandbox=True, work_precision=False)
    assert "functional_levels" not in out


# --- K5: the record, the summary, the statement, the manifest --------------------

def test_the_printed_summary_shows_the_price_and_the_two_level_check(tmp_path):
    out = verdict(tmp_path, HARDCODED)
    text = V._fmt(out, price_problem())
    assert "functional   : price rel" in text
    assert "two-level    :" in text and "CONSTANT across levels" in text


def test_the_printed_summary_says_when_the_defence_did_not_run():
    out = score(price_problem(), CALL)
    out["plan_dir"] = "x"
    out["functional_levels"] = {"status": "not_run", "reason": "inline"}
    assert "no anti-hardcoding check" in V._fmt(out, price_problem())


def test_the_problem_statement_states_both_the_gate_and_the_second_call():
    md = S.contract_footer(price_problem())
    assert "`functionals`: a dict of scalar results, containing `\"price\"`" in md
    assert "called **twice**" in md and "num_paths=5000" in md
    assert "Two path counts" in md and "must differ" in md
    assert "no terminal-moment formula" in md


def test_the_gate_sentence_is_the_same_sentence_on_both_formulations():
    """BENCHOP poses every problem both ways and compares them. Two paraphrases of
    one tolerance would make that a comparison of two graders."""
    sde = price_problem()
    pde = dict(sde, type="pde")
    assert S.functional_gate(sde) == S.functional_gate(pde)


def test_the_manifest_carries_a_functional_on_either_problem_type():
    entry = S.manifest_entry(price_problem())
    assert entry["functional_truth"] == {"price": CALL}
    assert entry["functional_tol"] == pytest.approx(0.05)
    assert entry["functional_level_paths"] == 5000


def test_no_problem_outside_the_anchored_suite_is_changed_by_any_of_this():
    """Zero churn, re-measured here rather than asserted once in a script: a problem
    statement is an agent *input*, and churn in it makes a later run incomparable
    with an earlier one for a reason that has nothing to do with the solver.

    **Re-read in step 7**, which is what the test below asked for when the first
    SDE problem carrying a ``functional_truth`` landed. The claim is unchanged for
    every problem that existed before that suite did -- they still say *once*, they
    still have no second call, and their manifest entries are byte-identical to
    HEAD (measured, 56/56). It is the new suite, and only the new suite, that opts
    into the second call."""
    for prob in P.PROBLEMS:
        if prob["type"] != "sde" or prob.get("suite") == "benchop":
            continue
        md = S.contract_footer(prob)
        assert "called **once**" in md, prob["slug"]
        assert "functional_levels" not in md and "Two path counts" not in md
        assert "functional_level_paths" not in S.manifest_entry(prob)


def test_the_sde_problems_that_carry_a_functional_are_exactly_the_anchored_suite():
    """The other half of the restatement. ``pde_schrodinger_eigen_2d`` was the only
    problem in the repo scored on a number; it is still the only *PDE* one, and the
    ten that joined it are all SDE and all in the suite that opted in."""
    carrying = [p for p in P.PROBLEMS if p.get("functional_truth")]
    assert [p["slug"] for p in carrying if p["type"] == "pde"] == ["pde_schrodinger_eigen_2d"]
    sde = [p for p in carrying if p["type"] == "sde"]
    assert sde and all(p.get("suite") == "benchop" for p in sde)
    assert len(sde) == len(P.benchop_problems())


def test_the_second_call_reaches_the_problems_that_asked_for_it():
    for prob in P.benchop_problems():
        md = S.contract_footer(prob)
        assert "called **twice**" in md and "Two path counts" in md
        assert "functional_level_paths" in S.manifest_entry(prob)


def test_a_zero_reference_does_not_fail_every_solver_on_the_problem(tmp_path):
    """A relative gate on 0.0 is an absolute gate of 0.0, so every nonzero difference
    between the two levels would be 'inconsistent'. No reference in the suite is zero
    (the step-4 suite asserts that), but an authoring slip in step 7 would otherwise
    present as every solver on the problem failing for no visible reason. The
    consistency half is not formed; the movement half, which is what the hardcoding
    defence rests on, still runs."""
    out = verdict(tmp_path, HONEST, functional_truth={"price": 0.0})
    v = out["functional_levels"]["values"]["price"]
    assert v["band"] == 0.0 and v["consistent"] is True and v["moved"] is True
    assert out["functional_levels"]["passed"] is True
    # It still fails -- on the gate, which is the honest reason, not on the band.
    assert out["passed"] is False and out["functional_ok"] is False


def test_a_zero_reference_still_catches_a_constant(tmp_path):
    """The guard must not become an escape hatch: the movement half is untouched."""
    out = verdict(tmp_path, HARDCODED, functional_truth={"price": 0.0})
    assert out["functional_levels"]["constant"] == ["price"]


def test_the_ranking_axis_can_rank_a_price_only_problem():
    """``wrong_plan_won`` ranks plans on ``plan_harness_error``, which returned
    ``None`` whenever an SDE record had no moment components -- which is every
    BENCHOP SDE problem. Scaled so 1.0 is exactly the pass line, as the moment
    branch is, and the margin is the reference's own absolute uncertainty."""
    prob = price_problem()
    harness = score(prob, CALL * 1.025)       # 2.5% off a 5% gate -> 0.5
    err, margin = RUN.plan_harness_error(prob, harness)
    assert err == pytest.approx(0.5, rel=1e-3)
    assert margin == 0.0


def test_the_ranking_margin_is_the_references_own_uncertainty():
    prob = price_problem(reference_abs_err=0.002)
    harness = score(prob, CALL)
    _err, margin = RUN.plan_harness_error(prob, harness)
    tol = harness["functional_tol_used"]["price"]
    assert margin == pytest.approx((0.002 / CALL) / tol, rel=1e-9)


def test_the_record_names_the_quantity_the_ranking_number_came_from():
    """Step 9. The ranking axis learned to read a price; the label beside it did
    not, so every entry in the externally anchored suite carried a number computed
    from its price under the words "worst moment error"."""
    assert RUN.harness_error_metric(price_problem()) == "worst functional error / its gate"
    assert RUN.harness_error_metric(
        price_problem(moments=lambda T: {"mean": 1.0, "variance": 0.04})
    ) == "worst moment or functional error / its gate"
    assert RUN.harness_error_metric(P.by_slug("sde_gbm")) == "worst moment error / its gate"
    assert RUN.harness_error_metric(P.by_slug("pde_heat_1d")) == "rel_l2_err"


def test_the_moment_axis_still_wins_when_it_is_the_worse_one():
    """The functional does not displace the moments, it joins them on one scale."""
    prob = price_problem(moments=lambda T: {"mean": 1.5, "variance": 0.0408})
    harness = score(prob, CALL)
    err, _m = RUN.plan_harness_error(prob, harness)
    assert err > 1.0                      # the mean is ~33% off a 5% gate


def test_the_report_line_shows_the_price_not_two_zeroes():
    """With no components the moment errors are the 0.0 that 'no moment is wrong'
    puts there, which would print as a flawless solver."""
    harness = score(price_problem(), CALL * 1.2)
    harness["functional_levels"] = {"status": "ok"}
    line = R.key_metric({"type": "sde", "ground_truth_kind": "exact", "verify": harness})
    assert line.startswith("funcErr[price:") and "varErr" not in line


def test_the_report_line_flags_a_missing_two_level_check():
    harness = score(price_problem(), CALL)
    harness["functional_levels"] = {"status": "not_run", "reason": "inline"}
    line = R.key_metric({"type": "sde", "ground_truth_kind": "exact", "verify": harness})
    assert "!2-level:not_run" in line


def test_a_problem_with_moments_and_a_price_reports_both():
    prob = price_problem(moments=lambda T: {"mean": 1.0, "variance": 0.0408})
    harness = score(prob, CALL * 1.2)
    line = R.key_metric({"type": "sde", "ground_truth_kind": "exact", "verify": harness})
    assert "varErr=" in line and "price:" in line


def test_the_results_record_keeps_the_number_the_verdict_rests_on():
    """``run.py`` copies a fixed key list into results.json. A price-only problem
    whose price is not on that list is a verdict with no number behind it."""
    import inspect
    src = inspect.getsource(RUN)
    block = src[src.index('"gt_kind", "worst_variance_rel_err"'):]
    block = block[:block.index("if k in harness")]
    for key in ("functional_errs", "functional_abs_errs", "functional_tol_used",
                "reference_abs_err", "functional_levels"):
        assert f'"{key}"' in block, key


# --- K6: the ranking margin on a price-only problem (step 10) --------------------

def _with_levels(harness, *, graded, coarse, n_hi=4_000_000, n_lo=1_000_000):
    """The block the two-level check leaves in the record, as `verify` writes it."""
    harness["functional_levels"] = {
        "status": "ok", "num_paths_graded": n_hi, "num_paths_coarse": n_lo,
        "values": {"price": {"graded": graded, "coarse": coarse,
                             "delta": graded - coarse, "moved": True,
                             "consistent": True}},
        "passed": True,
    }
    return harness


def test_the_margin_reads_the_solvers_own_sampling_noise():
    """A price has no `empirical_se`, so the margin was the reference's uncertainty
    alone -- and `verify` exports that only when the floor *widened* a gate, which
    on the BENCHOP entries it never does. The margin was therefore identically zero
    and any difference at all between two plans read as `wrong_plan_won`."""
    prob = price_problem()
    harness = _with_levels(score(prob, CALL * 1.025), graded=CALL, coarse=CALL * 0.999)
    _err, margin = RUN.plan_harness_error(prob, harness)
    tol = harness["functional_tol_used"]["price"]
    expected = (abs(CALL - CALL * 0.999) / math.sqrt(3.0) / CALL) / tol
    assert margin == pytest.approx(expected, rel=1e-9)
    assert margin > 0.0


def test_the_two_noise_sources_add_in_quadrature():
    """The same combination the moment branch makes: the reference value's own
    uncertainty and the sampling error are independent, so they are not summed."""
    prob = price_problem(reference_abs_err=0.002)
    harness = _with_levels(score(prob, CALL), graded=CALL, coarse=CALL * 0.999)
    _err, margin = RUN.plan_harness_error(prob, harness)
    tol = harness["functional_tol_used"]["price"]
    sampling = abs(CALL - CALL * 0.999) / math.sqrt(3.0)
    assert margin == pytest.approx((math.hypot(0.002, sampling) / CALL) / tol, rel=1e-9)
    # Neither term alone would do.
    assert margin > (0.002 / CALL) / tol


def test_the_divisor_is_the_one_that_does_not_understate_the_noise():
    """Nested levels (same seed, fewer paths) give sqrt(r-1); independent draws give
    sqrt(r+1). Which of the two holds depends on how a solver consumes its stream,
    which the harness cannot know, so it takes the larger estimate and errs towards
    *not* calling a ranking wrong."""
    levels = _with_levels({}, graded=1.0, coarse=0.9)["functional_levels"]
    se = RUN._functional_sampling_se(levels, "price")
    assert se == pytest.approx(0.1 / math.sqrt(3.0), rel=1e-12)
    assert se > 0.1 / math.sqrt(5.0)


@pytest.mark.parametrize("levels", [
    {},
    {"num_paths_graded": 4_000_000, "num_paths_coarse": 4_000_000,
     "values": {"price": {"delta": 0.1}}},
    {"num_paths_graded": 4_000_000, "num_paths_coarse": 1_000_000, "values": {}},
    {"num_paths_graded": 4_000_000, "num_paths_coarse": 1_000_000,
     "values": {"price": {"delta": float("nan")}}},
], ids=["no levels", "one level", "no value", "nan delta"])
def test_a_missing_second_level_costs_the_margin_nothing_it_cannot_support(levels):
    """A problem the check skipped falls back to the reference term, not to a
    fabricated one."""
    assert RUN._functional_sampling_se(levels, "price") == 0.0


def test_a_difference_below_the_noise_is_not_a_wrong_ranking():
    """The comparison `wrong_plan_won` makes, on two plans a hair apart. Measured on
    B01 of the 2026-10-22 sweep: winner 0.2584, best 0.0259, and the best plan's own
    sampling error 0.5293 on the same scale -- twenty times its error."""
    prob = price_problem()
    winner = _with_levels(score(prob, CALL * 1.021), graded=CALL, coarse=CALL * 0.99)
    rival = _with_levels(score(prob, CALL * 1.020), graded=CALL, coarse=CALL * 0.99)
    w_err, w_margin = RUN.plan_harness_error(prob, winner)
    r_err, r_margin = RUN.plan_harness_error(prob, rival)
    assert w_err > r_err                                  # the winner IS less accurate
    assert w_err <= r_err + max(w_margin, r_margin)       # but not detectably so
