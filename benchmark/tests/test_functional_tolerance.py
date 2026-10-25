"""Tolerance plumbing for a scalar quantity of interest (plan-benchop-slv.md step 4).

Four claims, one per section:

**J1** ``max_abs_err`` names an absolute error on both truth routes. It used to hold a
*relative* one on the functional route, so the single field named for an absolute error
meant two incompatible things depending on which route a problem took.

**J2** Both errors are recorded per functional. The relative one grades; the absolute
one is what an external benchmark states its own values and tolerances in, and it cannot
be recovered afterwards, because the record does not carry the reference value.

**J3** The gate is relative, raised per value by whatever absolute uncertainty the
reference number itself carries. It only ever loosens, and it is a property of the
individual value rather than of the problem -- a fixed +/-0.005 is a different relative
precision at 0.0393 than at 37.52.

**J4** All of it reaches the record, the printed summary, the problem statement the
solver is given, and the manifest -- and, for the problems already in the suite, changes
none of them.

No solver runs: every test feeds ``verify.py`` synthetic output, as the rest of this
directory does.
"""
import json

import numpy as np
import pytest

from .conftest import P, S, V, grid_for, problem, result_from, truth_on


def functional_problem(truth, *, tol=1e-3, abs_err=None, field=False):
    """A minimal problem scored on numbers. ``field=False`` is the functional-*only*
    route (no field truth at all), which is the branch that held the naming slip."""
    prob = {"slug": "test_functional", "type": "pde", "dims": 1, "tier": 1,
            "domain": {"x": (0.0, 1.0)}, "axes": ["x"], "t_eval": 1.0,
            "grid_N": 16, "pde_order_check": False, "min_order": 1.0,
            "functional_truth": dict(truth), "functional_tol": tol}
    if abs_err is not None:
        prob["reference_abs_err"] = abs_err
    if field:
        prob["analytic"] = lambda t, x: np.zeros_like(x)
        prob["fields"] = ["u"]
    return prob


def functional_runs(prob, got):
    """One graded run returning ``got`` as its ``functionals`` (and a zero field, so
    the field route can be exercised with the same helper)."""
    grid = grid_for(prob, prob["grid_N"])
    return [(prob["grid_N"], result_from({"u": np.zeros(prob["grid_N"])}, grid,
                                         t_final=prob["t_eval"], functionals=dict(got)))]


CORPUS_FUNCTIONAL = [p for p in P.PROBLEMS if p.get("functional_truth")]


# --- J1: max_abs_err means an absolute error on both routes ----------------------

def test_a_functional_problems_max_abs_err_is_an_absolute_error():
    """The slip itself. A solver off by 0.02 on a reference of 2.0 is 1% relative and
    0.02 absolute; the field used to carry 0.01 under the name ``max_abs_err``."""
    prob = functional_problem({"price": 2.0})
    out = V._score_pde_runs(prob, functional_runs(prob, {"price": 2.02}))
    assert out["rel_l2_err"] == pytest.approx(0.01)
    assert out["max_abs_err"] == pytest.approx(0.02)


def test_the_absolute_error_tracks_the_scale_the_relative_one_divides_out():
    """Same 1% error on a reference 100x larger. A relative field is unmoved; an
    absolute one must move with it, which is how the two are told apart."""
    small = functional_problem({"price": 2.0})
    large = functional_problem({"price": 200.0})
    a = V._score_pde_runs(small, functional_runs(small, {"price": 2.02}))
    b = V._score_pde_runs(large, functional_runs(large, {"price": 202.0}))
    assert b["rel_l2_err"] == pytest.approx(a["rel_l2_err"])
    assert b["max_abs_err"] == pytest.approx(100.0 * a["max_abs_err"])


def test_max_abs_err_is_the_largest_deviation_over_every_functional():
    """``max`` over the quantities of interest, matching the field route's max over
    the grid -- not the primary value's, which would hide the worst one."""
    prob = functional_problem({"a": 1.0, "b": 10.0}, tol=1.0)
    out = V._score_pde_runs(prob, functional_runs(prob, {"a": 1.01, "b": 10.5}))
    assert out["max_abs_err"] == pytest.approx(0.5)


def test_a_field_problems_max_abs_err_is_still_the_grid_maximum():
    """The route that was always right, pinned so the fix did not reach it."""
    prob = problem("pde_heat_1d")
    N = V._pde_grid_N(prob)
    grid = grid_for(prob, N)
    exact = truth_on(prob, grid)["u"]
    bump = np.zeros_like(exact)
    bump[len(bump) // 2] = 0.25
    out = V._score_pde_runs(prob, [(N, result_from({"u": exact + bump}, grid,
                                                   t_final=prob["t_eval"]))])
    assert out["max_abs_err"] == pytest.approx(0.25)


# --- J2: relative and absolute, per functional -----------------------------------

def test_both_errors_are_recorded_for_every_functional():
    prob = functional_problem({"a": 4.0, "b": -50.0}, tol=1.0)
    out = V._score_pde_runs(prob, functional_runs(prob, {"a": 4.04, "b": -49.0}))
    assert out["functional_abs_errs"] == pytest.approx({"a": 0.04, "b": 1.0})
    assert out["functional_errs"] == pytest.approx({"a": 0.01, "b": 0.02})


def test_the_two_errors_agree_with_the_reference_value_that_links_them():
    """Not a tautology in the record: ``functional_truth`` is never written into it,
    so abs/rel is the only place the reference value survives."""
    prob = problem("pde_schrodinger_eigen_2d")
    truth = prob["functional_truth"]["lambda_1"]
    grid = grid_for(prob, prob["grid_N"])
    runs = [(prob["grid_N"], result_from(truth_on(prob, grid), grid, t_final=prob["t_eval"],
                                         functionals={"lambda_1": truth * (1 + 2e-3)}))]
    out = V._score_pde_runs(prob, runs)
    assert "functional_truth" not in out
    ratio = out["functional_abs_errs"]["lambda_1"] / out["functional_errs"]["lambda_1"]
    assert ratio == pytest.approx(abs(truth), rel=1e-9)


def test_the_record_stays_json_serializable():
    prob = functional_problem({"a": 4.0}, abs_err=0.5)
    out = V._score_pde_runs(prob, functional_runs(prob, {"a": 4.04}))
    json.dumps(out)


@pytest.mark.parametrize("got,status", [
    ({}, "contract"),                       # the dict is there, the name is not
    (None, "contract"),                     # no dict at all
    ({"a": "nope"}, "inconclusive"),        # a type slip
    ({"a": float("nan")}, "nonfinite"),
])
def test_every_failure_path_returns_three_values(got, status):
    """``_functional_errs`` grew a return value. A path that forgot to grow with it
    unpacks a 2-tuple into three names and raises inside scoring, where the verdict
    would become ``error`` on a solver that merely returned the wrong thing."""
    prob = functional_problem({"a": 1.0})
    result = {} if got is None else {"functionals": got}
    rel, abs_errs, err = V._functional_errs(prob, result)
    assert rel is None and abs_errs is None
    assert err["status"] == status


def test_a_problem_with_no_functionals_returns_two_empty_dicts():
    assert V._functional_errs({}, {}) == ({}, {}, None)


# --- J3: the per-value absolute floor --------------------------------------------

def test_without_a_declared_uncertainty_every_gate_is_the_declared_one():
    prob = functional_problem({"a": 1.0, "b": 1e6})
    assert V._functional_tols(prob) == {"a": 1e-3, "b": 1e-3}


def test_no_problem_outside_the_anchored_suite_has_a_widened_functional_gate():
    """~~This step is a no-op for the suite as it stands~~ -- **re-stated in step
    7**, which is what the original wording asked for when it said the assertion
    *is* the measurement and that a landing BENCHOP problem should fail here rather
    than let the claim be silently outgrown.

    What is still true, and is the part that mattered: the two suites that carry
    measured baselines are untouched. Every problem in them is held to one gate on
    every value, exactly as before. The widened and per-value gates live only in
    the externally anchored suite, where they were put deliberately and where no
    baseline has been measured yet."""
    assert CORPUS_FUNCTIONAL, "the fixture would otherwise assert nothing"
    older = [p for p in CORPUS_FUNCTIONAL if p.get("suite") != "benchop"]
    assert older, "expected the pre-existing functional problems to still be here"
    for p in older:
        base = V._functional_tol(p)
        assert V._functional_tols(p) == {k: base for k in p["functional_truth"]}, p["slug"]


def test_the_anchored_suite_is_where_the_per_value_gates_actually_are():
    """The other half of the restatement: the mechanism is in use, so it is being
    tested by something real rather than by this file's fixtures alone."""
    anchored = [p for p in CORPUS_FUNCTIONAL if p.get("suite") == "benchop"]
    if not anchored:
        pytest.skip("the anchored suite has not been authored")
    assert any(len(set(V._functional_tols(p).values())) > 1 for p in anchored)
    assert any(p.get("reference_abs_err") for p in anchored)


def test_a_single_declared_uncertainty_is_a_different_gate_at_every_value():
    """BENCHOP's shape, and the reason the floor is per value: HHWeuCallI is published
    to +/-0.005, which is 1.3e-4 relative at 37.52 and 0.13 at 0.0393. One number in,
    two very different gates out -- the large value keeps the declared 1e-3, the small
    one cannot be held to a digit its reference does not have."""
    prob = functional_problem({"deep_otm": 0.0393, "deep_itm": 37.52}, tol=1e-3,
                              abs_err=0.005)
    tols = V._functional_tols(prob)
    assert tols["deep_itm"] == pytest.approx(1e-3)
    assert tols["deep_otm"] == pytest.approx(3.0 * 0.005 / 0.0393)


def test_a_per_name_uncertainty_applies_only_to_the_name_it_names():
    """A Monte-Carlo reference has its own standard error at each value -- measured in
    §2c as spanning 10x across three spots of one problem, against 3x for the relative
    errors -- so the dict form is the one that route needs."""
    prob = functional_problem({"a": 1.0, "b": 1.0}, tol=1e-3,
                              abs_err={"a": 0.01})
    tols = V._functional_tols(prob)
    assert tols["a"] == pytest.approx(0.03)
    assert tols["b"] == pytest.approx(1e-3)


def test_the_floor_never_tightens_the_declared_gate():
    prob = functional_problem({"a": 1.0}, tol=0.05, abs_err=1e-12)
    assert V._functional_tols(prob)["a"] == pytest.approx(0.05)


def test_a_value_inside_its_widened_gate_passes_where_the_declared_one_fails():
    """The behaviour change, end to end. 1% off a reference known only to +/-0.005 on
    a value of 0.0393 is far inside the reference's own uncertainty."""
    prob = functional_problem({"price": 0.0393}, tol=1e-3, abs_err=0.005)
    runs = functional_runs(prob, {"price": 0.0393 * 1.01})
    assert V._score_pde_runs(prob, runs)["passed"] is True
    strict = functional_problem({"price": 0.0393}, tol=1e-3)
    assert V._score_pde_runs(strict, functional_runs(strict, {"price": 0.0393 * 1.01}))[
        "passed"] is False


def test_a_value_outside_even_the_widened_gate_still_fails():
    """The floor loosens the gate; it does not remove it."""
    prob = functional_problem({"price": 0.0393}, tol=1e-3, abs_err=0.005)
    runs = functional_runs(prob, {"price": 0.0393 + 0.05})
    assert V._score_pde_runs(prob, runs)["passed"] is False


def test_the_widening_is_decided_value_by_value_within_one_solve():
    """Both values are 1% off. The small one is inside its reference's uncertainty and
    the large one is not, so the problem fails on the large one alone."""
    prob = functional_problem({"small": 0.0393, "large": 37.52}, tol=1e-3, abs_err=0.005)
    out = V._score_pde_runs(prob, functional_runs(prob, {"small": 0.0393 * 1.01,
                                                         "large": 37.52 * 1.01}))
    assert out["passed"] is False
    assert out["functional_tol_widened"] == ["small"]


def test_no_functional_reference_in_the_suite_is_zero():
    """A relative gate on a truth of exactly 0 is meaningless -- the error is
    ``|val| / 1e-14`` and the absolute floor divides by ``1e-12`` -- so such a value
    has to be scored absolutely or re-posed, never gated by this machinery."""
    for p in CORPUS_FUNCTIONAL:
        for name, ref in p["functional_truth"].items():
            assert abs(float(ref)) > 1e-12, f"{p['slug']}.{name}"


def test_a_declared_uncertainty_is_a_non_negative_number_or_a_dict_of_them():
    """Schema, for the problems step 7 will write. A negative uncertainty would pass
    silently through ``max`` and read as if the key were absent."""
    for p in P.PROBLEMS:
        ra = p.get("reference_abs_err")
        if ra is None:
            continue
        items = ra.items() if isinstance(ra, dict) else [(None, ra)]
        for name, val in items:
            assert float(val) >= 0.0, f"{p['slug']}: {name}={val}"
            if name is not None:
                assert name in (p.get("functional_truth") or {}), f"{p['slug']}: {name}"


# --- J4: the record, the printout, the statement, the manifest -------------------

def test_the_effective_gate_is_recorded_even_when_it_equals_the_declared_one():
    """A reader should not have to know the widening rule, or go back to problems.py,
    to see what a value was held to."""
    prob = functional_problem({"a": 1.0}, tol=2e-3)
    out = V._score_pde_runs(prob, functional_runs(prob, {"a": 1.0}))
    assert out["functional_tol"] == pytest.approx(2e-3)
    assert out["functional_tol_used"] == pytest.approx({"a": 2e-3})
    assert "functional_tol_widened" not in out
    assert "reference_abs_err" not in out


def test_a_widened_gate_records_which_values_moved_and_what_moved_them():
    prob = functional_problem({"small": 0.0393, "large": 37.52}, tol=1e-3, abs_err=0.005)
    out = V._score_pde_runs(prob, functional_runs(prob, {"small": 0.0393, "large": 37.52}))
    assert out["functional_tol_widened"] == ["small"]
    assert out["reference_abs_err"] == pytest.approx({"small": 0.005, "large": 0.005})
    assert out["functional_tol_used"]["small"] > out["functional_tol"]


def test_the_printed_summary_carries_both_errors_and_the_gate_it_used():
    prob = functional_problem({"price": 2.0}, tol=1e-3)
    out = V._score_pde_runs(prob, functional_runs(prob, {"price": 2.02}))
    text = V._fmt(dict(out, slug=prob["slug"], plan_dir="-"), prob)
    line = next(ln for ln in text.splitlines() if "functional" in ln)
    assert "rel 1.000e-02" in line and "abs 2.000e-02" in line
    assert "pass < 0.001" in line and "FAIL" in line
    assert "widened" not in line


def test_the_printed_summary_says_when_a_gate_was_widened():
    prob = functional_problem({"price": 0.0393}, tol=1e-3, abs_err=0.005)
    out = V._score_pde_runs(prob, functional_runs(prob, {"price": 0.0393 * 1.01}))
    line = next(ln for ln in V._fmt(dict(out, slug=prob["slug"], plan_dir="-"),
                                    prob).splitlines() if "functional" in ln)
    assert "widened from 0.001" in line and "-> ok" in line


def test_the_statement_given_to_a_solver_is_unchanged_where_nothing_is_widened():
    """The problem statements are inputs to the agents, so churn in them would make a
    run incomparable with every run before it. Byte-identical for the one functional
    problem the suite has."""
    text = S.contract_footer(problem("pde_schrodinger_eigen_2d"))
    assert ("- **quantity of interest** — `lambda_1` must be within 0.05% (relative) of "
            "the reference value. This gate is independent of the field accuracy: "
            "reporting a number without a solution behind it, or a solution without "
            "the number, fails.") in text


def test_the_statement_names_each_gate_when_they_differ():
    """A solver told only the tightest number would chase a digit the answer key does
    not have; told only the loosest it would under-solve the values that are known."""
    prob = functional_problem({"small": 0.0393, "large": 37.52}, tol=1e-3, abs_err=0.005)
    prob.update(title="T", challenge="c", description="d", family="f", id="T00")
    line = next(ln for ln in S.contract_footer(prob).splitlines()
                if "quantity of interest" in ln)
    assert "`small` within 38.1679%" in line and "`large` within 0.1%" in line


def test_the_manifest_records_the_effective_gates_only_when_they_differ():
    """``manifest.json`` is tracked, so an entry that gained a key on every problem
    would be a diff on every problem."""
    eg = problem("pde_schrodinger_eigen_2d")
    entry = S.manifest_entry(eg)
    assert entry["functional_tol"] == pytest.approx(V._functional_tol(eg))
    assert "functional_tol_used" not in entry and "reference_abs_err" not in entry

    prob = dict(eg, functional_truth={"lambda_1": -8.442746362964},
                reference_abs_err=0.01)
    wide = S.manifest_entry(prob)
    assert wide["functional_tol_used"]["lambda_1"] == pytest.approx(3.0 * 0.01 / 8.442746362964)
    assert wide["reference_abs_err"] == {"lambda_1": 0.01}


# --- J5: the gate may be declared per value, not only floored per value ---------
#
# Added in step 7. The floor above is about how precisely a value is *known*; this
# is about how precisely it can be *computed*. A Monte-Carlo price carries a
# standard error set by the payoff's spread, so BENCHOP's deep-out-of-the-money
# Heston call -- 31x smaller than its in-the-money sibling and priced from the same
# paths -- has 31x the relative noise. One relative gate across both either fails
# the small one or stops testing the large one, which is the trade section 8 of the
# plan names and declines.

def test_a_per_value_tolerance_grades_each_value_at_its_own_gate():
    prob = functional_problem({"otm": 0.9085, "itm": 28.5148},
                              tol={"otm": 1.5e-2, "itm": 1.5e-3})
    tols = V._functional_tols(prob)
    assert tols == {"otm": pytest.approx(1.5e-2), "itm": pytest.approx(1.5e-3)}


def test_the_tight_value_fails_at_an_error_the_loose_one_survives():
    """The whole point: 1% is a pass on one value and a fail on the other."""
    prob = functional_problem({"otm": 1.0, "itm": 1.0},
                              tol={"otm": 1.5e-2, "itm": 1.5e-3})
    out = V._score_pde_runs(prob, functional_runs(prob, {"otm": 1.01, "itm": 1.0005}))
    assert out["passed"]
    worse = V._score_pde_runs(prob, functional_runs(prob, {"otm": 1.01, "itm": 1.01}))
    assert not worse["passed"]


def test_a_name_missing_from_a_per_value_tolerance_falls_back_to_the_default():
    prob = functional_problem({"a": 1.0, "b": 1.0}, tol={"a": 2e-2})
    assert V._functional_tol(prob, "a") == pytest.approx(2e-2)
    assert V._functional_tol(prob, "b") == pytest.approx(V.PDE_L2_TOL)


def test_the_headline_tolerance_is_the_loosest_declared_one():
    """``functional_tol`` stays a float in the record so every reader of it keeps
    working; the per-value gates live in ``functional_tol_used``, which is already
    the field a reader is told to consult."""
    prob = functional_problem({"a": 1.0, "b": 1.0}, tol={"a": 2e-2, "b": 1e-3})
    assert V._functional_tol(prob) == pytest.approx(2e-2)
    out = V._score_pde_runs(prob, functional_runs(prob, {"a": 1.0, "b": 1.0}))
    assert isinstance(out["functional_tol"], float)
    assert out["functional_tol_used"] == {"a": pytest.approx(2e-2),
                                          "b": pytest.approx(1e-3)}


def test_the_reference_floor_still_only_loosens_a_per_value_gate():
    """The two mechanisms compose, and the floor keeps its one-way behaviour: it
    raises the gate on a value whose reference is uncertain and leaves a value
    declared looser than the floor exactly where it was put."""
    prob = functional_problem({"small": 0.0393, "large": 37.52},
                              tol={"small": 5e-1, "large": 1e-3}, abs_err=0.005)
    tols = V._functional_tols(prob)
    assert tols["small"] == pytest.approx(0.5)                     # declared > floor
    assert tols["large"] == pytest.approx(1e-3)                    # floor is smaller
    tight = functional_problem({"small": 0.0393}, tol={"small": 1e-3}, abs_err=0.005)
    assert V._functional_tols(tight)["small"] == pytest.approx(3.0 * 0.005 / 0.0393)


def test_the_statement_names_every_gate_when_they_were_declared_apart():
    """Not only when the floor widened them. A solver told one number for two
    values graded at different ones is being told the wrong question."""
    prob = functional_problem({"otm": 0.9085, "itm": 28.5148},
                              tol={"otm": 1.5e-2, "itm": 1.5e-3})
    prob.update(title="T", challenge="c", description="d", family="f", id="T00")
    line = next(ln for ln in S.contract_footer(prob).splitlines()
                if "quantity of interest" in ln)
    assert "`otm` within 1.5%" in line and "`itm` within 0.15%" in line
    # and no reference-uncertainty sentence, because no floor did this
    assert "absolute\nuncertainty" not in line and "own absolute" not in line


def test_a_single_tolerance_renders_exactly_as_it_always_did():
    """Zero churn: the one problem in the suite carrying a functional must read the
    same as before step 7 touched this."""
    eg = problem("pde_schrodinger_eigen_2d")
    assert "must be within 0.05% (relative) of the reference value." in S.functional_gate(eg)
