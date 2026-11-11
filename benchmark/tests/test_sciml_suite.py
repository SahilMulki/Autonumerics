"""The SciMLBenchmarks-anchored suite (plan-sciml-benchmark.md step 6).

Nine SDE problems whose *schemes* are compared against `StochasticDiffEq.jl`'s
published algorithm set. Unlike BENCHOP, the external anchor is not an answer key but
a competitor: the truth is mostly a closed form this repo can write down itself, which
is why the suite needs the `pathwise` route and the `ANALYTIC_TRANSCRIPTION` exclusion
rather than a ledger.

  N1  the suite is what step 6 authored, and is kept apart from the other three
  N2  every pathwise entry's truth has the right shape, and every step it will ever be
      solved at is a coarsening of one Brownian grid
  N3  no entry is **vacuous** -- the best possible constant fails its gate. This is the
      §2a.1 finding as a standing test: three of these entries were vacuous as the plan
      first wrote them
  N4  the stiff triple is graded in the transient, with SciML's own parameters
  N5  the brief states the increments, demands a discretisation, and quotes no answer
  N6  the one non-pathwise entry is not handed increments, and its truth is derived
"""
import json
import math

import pytest

from .conftest import RUN, P, R, S, V

SUITE = P.sciml_problems()
PATHWISE = [p for p in SUITE if p.get("ground_truth_kind") == "pathwise"]


# --- N1: the suite, and its separation from the others --------------------------

def test_the_suite_is_the_size_step_six_authored():
    assert len(SUITE) == 9
    assert len(PATHWISE) == 8
    assert [p["id"] for p in SUITE] == [f"M0{i}" for i in range(1, 10)]
    assert all(p["type"] == "sde" for p in SUITE)
    # Tier 3 throughout, for benchop's reason: sde_manual.md has no Roessler-family
    # scheme at all, so every entry is beyond the reference manuals by construction.
    assert {p["tier"] for p in SUITE} == {3}


def test_the_suite_runs_alone_and_cannot_merge_into_another_denominator():
    """Step 6's second acceptance line."""
    assert "sciml" in S.SUITES
    kept = S.apply_suite_filter(list(P.PROBLEMS), only="", suite="sciml")
    assert {p["slug"] for p in kept} == {p["slug"] for p in SUITE}
    # Default (main) must not pick any of them up.
    main = S.apply_suite_filter(list(P.PROBLEMS), only="")
    assert not ({p["slug"] for p in main} & {p["slug"] for p in SUITE})
    # And writing them into the standing main-suite results file is refused.
    with pytest.raises(SystemExit):
        S.guard_results_path(SUITE, None, "benchmark/results/pipeline/{suite}.json")


def test_the_manifest_counts_four_suites_apart(tmp_path):
    path = S.write_manifest(list(P.PROBLEMS), str(tmp_path / "manifest.json"))
    with open(path) as fh:
        manifest = json.load(fh)
    assert manifest["n_by_suite"] == {"main": 49, "no-closed-form": 7,
                                      "benchop": 10, "sciml": 9}
    assert sum(manifest["n_by_suite"].values()) == manifest["n_total"]


@pytest.mark.parametrize("prob", SUITE, ids=lambda p: p["slug"])
def test_each_entry_records_its_suite_and_its_upstream_source(prob):
    assert S.manifest_entry(prob)["suite"] == "sciml"
    # Provenance, as benchop records `benchop_key`: which SciML problem this is, and
    # which notebook (if any) compares schemes on it. Three of the nine are in the
    # library but in no notebook, and that has to stay visible (§2a.3).
    assert prob["sciml_key"]


# --- N2: shapes, and one Brownian grid --------------------------------------------

@pytest.mark.parametrize("prob", PATHWISE, ids=lambda p: p["slug"])
def test_the_truth_is_a_function_of_the_path_with_the_right_shape(prob):
    import numpy as np
    payload = dict(V.sde_pathwise_ladder(prob, num_paths=64)[0], num_paths=64)
    W = V.crn_increments(payload).sum(axis=1)
    exact = V._sde_pathwise_truth(prob, float(prob["T"]), W)
    want = (64,) if prob["state_dimension"] == 1 else (64, prob["state_dimension"])
    assert exact.shape == want, prob["slug"]
    assert np.all(np.isfinite(exact)), prob["slug"]


@pytest.mark.parametrize("prob", PATHWISE, ids=lambda p: p["slug"])
def test_every_step_the_entry_will_be_solved_at_nests(prob):
    """The graded step, the three ladder levels and every declared work-precision step
    must all be coarsenings of `pathwise_fine_steps`. A step that is not would be
    driven by a different Brownian path, so the point would not be on the same curve --
    and the harness reports it `not_nested` rather than drawing it."""
    assert all(lv is not None for lv in V.sde_pathwise_ladder(prob)), prob["slug"]
    for n, dt in prob.get("work_precision_levels") or []:
        assert V.sde_pathwise_crn(prob, int(n), float(dt)) is not None, \
            (prob["slug"], dt)
    # And the save grid divides the graded step grid, or the comparator axis is lost.
    k = prob.get("pathwise_save_points")
    if k:
        nt = max(1, round(prob["T"] / V.sde_verify_dt(prob)))
        assert nt % int(k) == 0, (prob["slug"], nt, k)


@pytest.mark.parametrize("prob", PATHWISE, ids=lambda p: p["slug"])
def test_the_graded_step_is_the_one_the_entry_declares(prob):
    """`sde_verify_dt` caps at 0.04/T, so an entry whose own `dt` is coarser than the
    cap is graded somewhere it never declared -- and then its fine grid would not
    nest."""
    assert V.sde_verify_dt(prob) == prob["dt"], prob["slug"]


# --- N3: no entry is vacuous -------------------------------------------------------

@pytest.mark.parametrize("prob", PATHWISE, ids=lambda p: p["slug"])
def test_the_best_possible_constant_fails_the_gate(prob):
    """§2a.1, as a standing test rather than a finding.

    A `pathwise` entry whose exact terminal state barely varies across paths can be
    answered by a constant -- no increments consumed, nothing simulated -- and the
    constant's relative error is then *inside* the gate. Three of these entries were
    vacuous as the plan first specified them: the stiff triple at SciML's `T = 3`
    (where the terminal state is bit-identically -1), and `additive` and `wave` at the
    default 10% gate.
    """
    import numpy as np
    payload = dict(V.sde_pathwise_ladder(prob, num_paths=400)[0], num_paths=400)
    W = V.crn_increments(payload).sum(axis=1)
    exact = V._sde_pathwise_truth(prob, float(prob["T"]), W)
    best = np.full_like(exact, np.mean(exact)) if exact.ndim == 1 else \
        np.broadcast_to(exact.mean(axis=0), exact.shape)
    err = V._rel_l2_over_paths(best, exact)
    gate = V._pathwise_tol(prob)
    assert err > gate, (
        f"{prob['slug']}: the best constant scores {err:.4g} against a gate of "
        f"{gate:.4g} -- this entry can be answered without simulating")


# --- N4: the stiff triple ----------------------------------------------------------

def test_the_stiff_triple_is_graded_in_the_transient():
    """SciML's `tspan` is (0, 3) for all three. At that horizon every one of them has
    collapsed onto the absorbing point X = -1 and the exact terminal state is a
    constant, so the graded horizon is moved into the transition (§2a.1, §2b)."""
    triple = [p for p in SUITE if p["family"] == "sciml_stiff_quadratic"]
    assert len(triple) == 3
    import numpy as np
    for prob, alpha in zip(triple, (50.0, 500.0, 5000.0), strict=True):
        assert prob["T"] < 3.0, prob["slug"]
        # Near the transition midpoint ln(3)/(2 alpha), where the law is widest.
        assert 0.3 < prob["T"] / (np.log(3.0) / (2 * alpha)) < 3.0, prob["slug"]
        # And the step stays inside the stability limit the notebook reports.
        assert prob["dt"] < prob["T"] / 10, prob["slug"]


def test_the_stiff_triple_is_one_equation_at_three_parameter_sets():
    """Which makes it the suite's own shifted-parameter control for free (§8)."""
    triple = [p for p in SUITE if p["family"] == "sciml_stiff_quadratic"]
    eq = {next(line.strip() for line in p["description"].splitlines()
               if line.strip().startswith("dX ="))
          for p in triple}
    assert len(eq) == 1, f"the three should pose the same equation, got {eq}"
    # and differ only in the parameters and the horizon that follows from them
    assert len({p["T"] for p in triple}) == 3
    assert len({p["dt"] for p in triple}) == 3


# --- N5: the brief -----------------------------------------------------------------

@pytest.mark.parametrize("prob", PATHWISE, ids=lambda p: p["slug"])
def test_the_brief_states_the_increments_and_demands_a_discretisation(prob, tmp_path):
    with open(S.write_problem_md(prob, str(tmp_path))) as fh:
        text = fh.read()
    nt = max(1, round(prob["T"] / V.sde_verify_dt(prob)))
    m = V.sde_noise_dimension(prob)
    shape = f"({prob['num_paths']}, {nt})" if m == 1 else f"({prob['num_paths']}, {nt}, {m})"
    assert shape in text, prob["slug"]
    assert "dW" in text and "already scaled by sqrt(dt)" in text
    assert "discretised scheme is required" in text, prob["slug"]
    assert "excluded" in text, prob["slug"]
    # The call the verifier will actually make.
    dt = V.sde_verify_dt(prob)
    assert (f"solve_sde(num_paths={prob['num_paths']}, dt={dt:.4g}, "
            f"T={prob['T']:.6g}, seed={prob['seed']})") in text, prob["slug"]


@pytest.mark.parametrize("prob", SUITE, ids=lambda p: p["slug"])
def test_the_brief_quotes_no_exact_solution_and_no_reference_value(prob, tmp_path):
    """The whole point of the scheme-matched arm: handing over the closed form would
    make transcription the obvious move, and §0b is about not grading that."""
    with open(S.write_problem_md(prob, str(tmp_path))) as fh:
        text = fh.read()
    for banned in ("exp(0.63155", "0.63155", "arctan(0.1", "atan(0.1",
                   "2t + 1 + W", "(2 * t + 1", "exp(W)", "exp(-2"):
        assert banned not in text, (prob["slug"], banned)
    for value in (prob.get("functional_truth") or {}).values():
        digits = f"{abs(float(value)):.4e}".split("e")[0].replace(".", "")
        assert digits not in text.replace(".", ""), prob["slug"]


# --- N6: the one non-pathwise entry -----------------------------------------------

def test_the_weak_entry_is_not_handed_increments():
    """Deliberate, and the reason is structural: the high-order weak schemes SciML
    compares on this problem (DRI1, PL1WM, RDI1WM, RDI2WM) draw auxiliary randomness
    per step and cannot consume a supplied Gaussian increment. Supplying `dW` would
    exclude the very methods the published page is a comparison between."""
    prob = P.by_slug("sde_sciml_weak_noncommutative")
    assert prob.get("ground_truth_kind", "exact") == "exact"
    assert prob.get("pathwise_exact") is None
    md = S.contract_footer(prob)
    assert "dW" not in md
    # It still declares its noise dimension, which is what the manifest records.
    assert V.sde_noise_dimension(prob) == 2
    assert S.manifest_entry(prob)["noise_dimension"] == 2


def test_the_weak_entrys_moments_are_derived_not_transcribed():
    """SciML publishes one value, `exp(-3)`. It is reproduced here from the problem's
    own coefficients via the closed second-moment system, which also yields two more
    exactly-known observables -- and the first row of that system shows why they are
    needed: `E[X1^2]` decouples entirely, so SciML's own observable cannot see the
    channel structure."""
    import numpy as np
    m = P.sciml_noncommutative_moments(3.0)
    assert m["second_moment_x1"] == float(np.exp(-3.0))
    prob = P.by_slug("sde_sciml_weak_noncommutative")
    assert prob["functional_truth"] == m
    assert set(prob["functional_tol"]) == set(m)
    # Three gates, declared apart, each recorded in the manifest.
    used = S.manifest_entry(prob)["functional_tol_used"]
    assert set(used) == set(m)
    assert len(set(used.values())) == 3


def test_the_weak_entrys_gates_sit_above_the_monte_carlo_noise():
    """§10's prediction 4 said the Monte-Carlo floor rather than the scheme would set
    the resolvable gate here. Measured, it does not: X1 is exactly lognormal, so the
    relative sd of X1(3)^2 is 1.104 and at this entry's path count the relative
    standard error is 0.35% -- the loosest gate is 5.7 sigma clear of it."""
    import numpy as np
    prob = P.by_slug("sde_sciml_weak_noncommutative")
    a1, c11, c12 = -273 / 512, 1 / 4, 1 / 16
    sig2 = c11 ** 2 + c12 ** 2
    mu_d = a1 - 0.5 * sig2
    T = prob["T"]
    e2 = np.exp(2 * mu_d * T + 2 * sig2 * T)
    e4 = np.exp(4 * mu_d * T + 8 * sig2 * T)
    rel_se = float(np.sqrt(e4 - e2 ** 2) / e2 / np.sqrt(prob["num_paths"]))
    assert rel_se < 0.005, rel_se
    assert prob["functional_tol"]["second_moment_x1"] / rel_se > 4.0


# --- N7: what the step-8 smoke test found (plan §7 step 8) -------------------------
#
# Staging the suite and running it end to end with a planted solver -- no pipeline
# call -- turned up four defects, all of them on the path step 10 is about to take and
# none of them visible from the entries alone. Each one gets a test here, because each
# would have produced a *plausible-looking* run that had quietly dropped the thing the
# suite exists to measure.

_PATHWISE_HARNESS = {"status": "ok", "gt_kind": "pathwise", "passed": True,
                     "pathwise_rel_err": 0.0206, "pathwise_tol_used": 0.095,
                     "run": {"num_paths": 1000}}


def test_the_pathwise_route_has_a_per_plan_error_at_all():
    """Defect 1. `plan_harness_error` had a branch for a field, for moments and for a
    functional, and the pathwise route is none of the three -- so it returned None on
    eight of the nine entries, leaving `wrong_plan_won` nothing to rank them on."""
    prob = next(p for p in PATHWISE)
    err, margin = RUN.plan_harness_error(prob, _PATHWISE_HARNESS)
    assert err == pytest.approx(0.0206 / 0.095)
    assert err is not None and margin > 0.0


def test_the_per_plan_margin_is_the_sampling_noise_of_the_statistic():
    """The graded number is an rms over paths, whose standard error is rms/sqrt(2n):
    two plans closer than that are not distinguishable at this path count."""
    prob = next(p for p in PATHWISE)
    _err, margin = RUN.plan_harness_error(prob, _PATHWISE_HARNESS)
    assert margin == pytest.approx(0.0206 / math.sqrt(2 * 1000) / 0.095)
    # Four times the paths halves it, as a standard error must.
    h = {**_PATHWISE_HARNESS, "run": {"num_paths": 16000}}
    assert RUN.plan_harness_error(prob, h)[1] == pytest.approx(margin / 4)


def test_a_pathwise_entry_is_one_at_the_pass_line_exactly():
    prob = next(p for p in PATHWISE)
    h = {**_PATHWISE_HARNESS, "pathwise_rel_err": 0.095}
    assert RUN.plan_harness_error(prob, h)[0] == pytest.approx(1.0)


def test_the_metric_label_names_what_was_measured():
    """Defect 1b, and the same stale-label defect the BENCHOP smoke test found for
    prices: the fallback said "worst moment error" on eight entries with no moment
    gate at all."""
    for p in PATHWISE:
        assert RUN.harness_error_metric(p) == "relative strong error / its gate"
    weak = next(p for p in SUITE if p.get("ground_truth_kind") != "pathwise")
    assert "functional" in RUN.harness_error_metric(weak)


def test_the_report_names_the_four_suites_that_exist():
    """Defect 3. The methodology section described two suites, and a reader of a
    `sciml` report would have had no way to know what denominator it was quoted
    against or that the two arms are never merged."""
    text = R.generate_report([], config={})
    for suite in ("main", "no-closed-form", "benchop", "sciml"):
        assert f"`{suite}`" in text
    assert "scheme-matched" in text and "unrestricted" in text
    assert "ANALYTIC_TRANSCRIPTION" in text


def test_the_work_precision_slope_needs_three_points_to_be_reported():
    """Defect 2's guard. Two points make a slope out of any pair of numbers; the
    exponent is the thing a work-precision diagram is read for, so it is reported only
    once the curve resolves it."""
    pts = [{"evals": 1e3, "rel_err": 1e-1}, {"evals": 1e4, "rel_err": 1e-2}]
    assert R._wp_slope(pts, "evals", "rel_err") is None
    pts.append({"evals": 1e5, "rel_err": 1e-3})
    assert R._wp_slope(pts, "evals", "rel_err") == pytest.approx(-1.0)
    # A degenerate series -- one work value repeated -- resolves nothing either.
    flat = [{"evals": 1e3, "rel_err": e} for e in (1e-1, 1e-2, 1e-3)]
    assert R._wp_slope(flat, "evals", "rel_err") is None


def test_the_work_precision_section_renders_the_series_and_its_slope():
    """Defect 2. `report.py` and `compare.py` contained no occurrence of
    `work_precision` at all: the series reached results.json and nothing displayed
    it. On this suite that is the primary axis, so a sweep would have had to be
    re-run to be read."""
    rec = {"id": "M01", "slug": "sde_sciml_linear_ito", "type": "sde", "tier": 3,
           "family": "sciml_linear_ito", "challenge": "a scheme comparison",
           "has_ground_truth": True, "ground_truth_kind": "pathwise",
           "run": {"status": "completed"}, "pipeline": {"best_score": 10},
           "verify": {"status": "ok", "gt_kind": "pathwise", "passed": True,
                      "work_precision": {"axis": "paths_dt", "points": [
                          {"num_paths": 1000, "dt": 0.04, "paths": 1000, "steps": 25,
                           "stages": 1, "evals": 25000, "rel_err": 1.3e-1,
                           "solve_s": 0.01, "graded": False,
                           "scheme_family": "euler-maruyama",
                           "stages_source": "declared"},
                          {"num_paths": 1000, "dt": 0.008, "paths": 1000, "steps": 125,
                           "stages": 1, "evals": 125000, "rel_err": 5.0e-2,
                           "solve_s": 0.02, "graded": False,
                           "scheme_family": "euler-maruyama",
                           "stages_source": "declared"},
                          {"num_paths": 1000, "dt": 0.0016, "paths": 1000, "steps": 625,
                           "stages": 1, "evals": 625000, "rel_err": 2.1e-2,
                           "solve_s": 0.05, "graded": True,
                           "scheme_family": "euler-maruyama",
                           "stages_source": "declared"}]}}}
    text = R.generate_report([rec], config={})
    assert "## Error vs work" in text
    assert "sde_sciml_linear_ito" in text.split("## Error vs work", 1)[1]
    body = text.split("## Error vs work", 1)[1]
    assert "euler-maruyama (declared)" in body
    assert "625000" in body and "25000" in body
    # The two slopes come from the points themselves, and on a series whose paths and
    # stages are fixed they are each other's negation -- which is the property that
    # says the work axis is being read correctly.
    assert "d log err / d log evals = -0.57" in body
    assert "observed order in dt = 0.57" in body
    assert "| 0.0016 | 625 | 1000 | 1 | 625000 | 2.100e-02 | 0.050 | * |" in body


def test_a_series_of_one_point_renders_no_section():
    rec = {"id": "M01", "slug": "s", "type": "sde", "tier": 3, "family": "f",
           "challenge": "c", "has_ground_truth": True,
           "ground_truth_kind": "pathwise", "run": {"status": "completed"},
           "pipeline": {"best_score": 10},
           "verify": {"status": "ok", "passed": True, "work_precision": {
               "axis": "paths_dt",
               "points": [{"dt": 0.1, "rel_err": 1e-2, "graded": True}]}}}
    assert "## Error vs work" not in R.generate_report([rec], config={})


# --- N8: the arm has to be an experiment, not a label ------------------------------
#
# Found while preparing the step-9 runs, and the worst of the lot: `--arm` was recorded
# on every result and used nowhere else. `setup.py` did not even accept the flag, so
# both arms staged a byte-identical brief -- each telling the solver a discretised
# scheme was required -- and the transcription exclusion applied unconditionally. The
# `unrestricted` arm was therefore the scheme-matched experiment with another name, and
# running "both arms" as §7 steps 9 and 10 ask would have cost twice and measured once
# while meeting its own acceptance line ("records, per arm, with the arm field set").

_DEGENERATE = {"status": "ok", "passed": True, "verified_score": 10,
               "gt_kind": "pathwise",
               "pathwise_ladder": {"status": "ok", "degenerate": True}}


def _brief(problem, arm, tmp_path):
    S.write_problem_md(problem, str(tmp_path), arm=arm)
    return (tmp_path / problem["slug"] / "problem.md").read_text()


@pytest.mark.parametrize("prob", PATHWISE, ids=[p["id"] for p in PATHWISE])
def test_the_two_arms_ask_a_pathwise_problem_for_different_things(prob, tmp_path):
    matched = _brief(prob, "scheme-matched", tmp_path / "a")
    free = _brief(prob, "unrestricted", tmp_path / "b")
    assert matched != free
    assert "A discretised scheme is required" in matched
    assert "A discretised scheme is required" not in free
    assert "Any method is permitted, including the exact solution" in free
    # One paragraph apart and no more: the arm changes the permission, not the problem.
    a, b = matched.splitlines(), free.splitlines()
    assert len(a) == len(b)
    assert sum(1 for x, y in zip(a, b, strict=True) if x != y) == 1


def test_the_scheme_matched_arm_is_the_brief_that_was_already_staged(tmp_path):
    """The default must not move: every measurement taken before the arms existed was
    taken against this text."""
    for prob in SUITE:
        assert _brief(prob, "scheme-matched", tmp_path / "m") == \
               _brief(prob, None, tmp_path / "d")


def test_the_arm_cannot_reach_a_problem_outside_the_pathwise_route(tmp_path):
    """Including M09, which is in this suite but not on that route, and the 66 entries
    of the other three suites -- the arm distinction is meaningless there and a brief
    that moved would make every frozen baseline incomparable."""
    others = [p for p in P.PROBLEMS if p.get("ground_truth_kind") != "pathwise"]
    assert len(others) == 67
    for prob in others:
        assert _brief(prob, "unrestricted", tmp_path / "u") == \
               _brief(prob, "scheme-matched", tmp_path / "s")


def test_a_transcription_is_excluded_in_one_arm_and_passes_in_the_other():
    rec = {"has_ground_truth": True, "ground_truth_kind": "pathwise",
           "run": {"status": "completed"}, "pipeline": {"best_score": 10},
           "verify": _DEGENERATE}
    assert R.compute_verdict({**rec, "arm": "scheme-matched"}) == "ANALYTIC_TRANSCRIPTION"
    assert R.compute_oneshot_verdict({**rec, "arm": "scheme-matched"}) == \
        "ANALYTIC_TRANSCRIPTION"
    # In the unrestricted arm the closed form is the answer the brief asked for, and
    # recognising that the equation is exactly solvable is the capability measured.
    assert R.compute_verdict({**rec, "arm": "unrestricted"}) == "VERIFIED_PASS"
    assert R.compute_oneshot_verdict({**rec, "arm": "unrestricted"}) == "PASS"


def test_a_record_with_no_arm_keeps_the_verdict_it_had():
    """Every result frozen before `--arm` existed carries no arm, and must grade as it
    did -- the exclusion is the default, not the exception."""
    rec = {"has_ground_truth": True, "ground_truth_kind": "pathwise",
           "run": {"status": "completed"}, "pipeline": {"best_score": 10},
           "verify": _DEGENERATE}
    assert R.compute_verdict(rec) == "ANALYTIC_TRANSCRIPTION"
    assert R.compute_oneshot_verdict(rec) == "ANALYTIC_TRANSCRIPTION"


def test_the_unrestricted_arm_still_fails_a_wrong_solver():
    """The permission is to use the closed form, not to be wrong. A degenerate ladder
    on a solver that missed its gate is a constant, and §2b's calibration showed that
    is a real case rather than a hypothetical."""
    wrong = {**_DEGENERATE, "passed": False, "verified_score": 0}
    rec = {"has_ground_truth": True, "ground_truth_kind": "pathwise",
           "run": {"status": "completed"}, "pipeline": {"best_score": 10},
           "verify": wrong, "arm": "unrestricted"}
    assert R.compute_verdict(rec) == "OVERCLAIM"
    assert R.compute_oneshot_verdict(rec) == "FAIL"


@pytest.mark.parametrize("mod_name", ["run", "oneshot", "setup"])
def test_every_entry_point_can_state_an_arm(mod_name):
    """All three, because the brief is written by whichever one runs: `run.py` and
    `oneshot.py` re-stage it on every unit, and `setup.py` stages it by hand."""
    import importlib
    mod = importlib.import_module(mod_name)
    parser_src = __import__("inspect").getsource(mod.main)
    assert "--arm" in parser_src or "add_suite_arguments" in parser_src


def test_staging_threads_the_arm_rather_than_defaulting_it():
    """`write_problem_md`'s arm has no default value that could silently stage the
    wrong experiment: a caller that forgets it gets the scheme-matched text, which is
    also what every pre-arm record was graded against."""
    import inspect
    sig = inspect.signature(S.write_problem_md)
    assert "arm" in sig.parameters
    assert sig.parameters["arm"].default is None


def test_the_two_arms_do_not_share_artifacts_or_unit_keys():
    """Defect 7. Both were keyed on (problem, condition, manuals, model, rep), so the
    second arm wrote its `solver.py` over the first arm's -- and on a scheme benchmark
    the solver source is the only record of which scheme was chosen. The colliding
    `unit_key` was worse: two arms in one results file would have deduped to one unit,
    and `--resume` would have skipped the second arm's work as already done."""
    import oneshot as O
    p = {"slug": "sde_sciml_linear_ito"}
    args = ("c0", "naked", "opus", 1)
    matched = O.plan_dir_for(p, *args, arm="scheme-matched")
    free = O.plan_dir_for(p, *args, arm="unrestricted")
    assert matched != free
    assert O.unit_key(p["slug"], *args, arm="scheme-matched") != \
        O.unit_key(p["slug"], *args, arm="unrestricted")


def test_the_default_arms_paths_and_keys_are_unchanged():
    """Every frozen baseline carries a key with no arm in it, and its artifacts are on
    disk under the old path. Both have to keep matching, or `--resume` re-runs work
    that is already paid for and `baseline_select.py` loses its inputs."""
    import oneshot as O
    p = {"slug": "sde_gbm"}
    args = ("c0", "naked", "opus", 1)
    for arm in (None, "scheme-matched"):
        assert O.plan_dir_for(p, *args, arm=arm) == O.plan_dir_for(p, *args)
        assert O.unit_key(p["slug"], *args, arm=arm) == O.unit_key(p["slug"], *args)
    assert O.unit_key("sde_gbm", *args) == "sde_gbm|c0|naked|opus|1"


def test_a_defaulted_stage_count_is_declared_as_a_lower_bound():
    """Found by the first real pipeline run (§10a). The kernel's SDE vocabulary has no
    Rossler/SRK family, so a plan that writes a stochastic Heun or an SRK scheme can
    only declare `other` -- and `other` carries no cost per step, so the work axis
    falls back to 1 stage. The direction of that fallback is not neutral: it
    *under*-states the work of the plan being measured, which flatters it against a
    comparator whose stages were measured. The report has to say so."""
    pts = [{"dt": d, "steps": s, "paths": 1000, "stages": 1, "evals": e,
            "rel_err": r, "solve_s": 0.1, "graded": g,
            "scheme_family": "other", "stages_source": "default"}
           for d, s, e, r, g in ((0.04, 25, 25000, 1.3e-1, False),
                                 (0.008, 125, 125000, 5e-2, False),
                                 (0.0016, 625, 625000, 2.1e-2, True))]
    rec = {"id": "M01", "slug": "s", "type": "sde", "tier": 3, "family": "f",
           "challenge": "c", "has_ground_truth": True,
           "ground_truth_kind": "pathwise", "run": {"status": "completed"},
           "pipeline": {"best_score": 10},
           "verify": {"status": "ok", "passed": True,
                      "work_precision": {"axis": "paths_dt", "points": pts}}}
    body = R.generate_report([rec], config={}).split("## Error vs work", 1)[1]
    assert "lower bound" in body and "default of 1" in body
    # And it must stay quiet when the count was actually declared.
    for q in pts:
        q["stages_source"] = "declared"
        q["scheme_family"] = "milstein"
        q["stages"] = 2
    clean = R.generate_report([rec], config={}).split("## Error vs work", 1)[1]
    assert "lower bound" not in clean
