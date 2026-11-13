"""The pathwise (strong-error) route and its exclusion, as the harness grades them.

Step 5 of plan-sciml-benchmark.md: H1-H4, H8-H10. Every acceptance line in §4 of that
plan is a test here, and unlike the rest of this directory these tests run **real
solvers through the real sandbox** -- four of them, differing by one decision each --
because the thing being verified is what the harness does with a solver that is
perfectly accurate and not a scheme, and that cannot be shown with synthetic output.

The fixture problems are SciML's own equations (the linear Ito problem from
`SDEProblemLibrary.jl` v1.2.4, and a two-channel variant of it) but they are declared
*here* rather than in `problems.py`: step 6 authors the suite, and a harness change has
to be provable before the entries that depend on it exist.
"""
import os
import textwrap

import numpy as np
import pytest

from .conftest import P, R, S, V

# --- fixture problems ------------------------------------------------------------
# SciML's `prob_sde_linear`, read off the pinned library source: f = 1.01u, g = 0.87u,
# u0 = 1/2, tspan (0,1), analytic u0*exp(0.63155t + 0.87W). 0.63155 = 1.01 - 0.87^2/2.
U0, MU, SIG = 0.5, 1.01, 0.87


def _linear_exact(t, W):
    return U0 * np.exp((MU - 0.5 * SIG ** 2) * t + SIG * W)


LINEAR = {
    "id": "T01", "slug": "test_pathwise_linear", "type": "sde", "tier": 1,
    "family": "sciml_linear_ito", "title": "SciML linear Ito (test fixture)",
    "challenge": "pathwise-exact truth; the route's own acceptance problem",
    "suite": "sciml", "state_dimension": 1,
    "T": 1.0, "dt": 0.01, "num_paths": 4000, "seed": 42,
    "has_ground_truth": True, "ground_truth_kind": "pathwise",
    "pathwise_exact": _linear_exact,
    "pathwise_save_points": 10,
    "description": "# fixture\n",
}


def _two_channel_exact(t, W):
    """Two *independent* GBMs, one per noise channel. A solver that drives both
    components from one increment gets the second one wrong -- which is the defect
    H3 exists to make exercisable."""
    W = np.asarray(W, dtype=float)
    return np.stack([U0 * np.exp((MU - 0.5 * SIG ** 2) * t + SIG * W[:, 0]),
                     U0 * np.exp((MU - 0.5 * SIG ** 2) * t + SIG * W[:, 1])], axis=-1)


TWO_CHANNEL = dict(LINEAR, **{
    "id": "T02", "slug": "test_pathwise_two_channel",
    "title": "Two-channel pathwise (test fixture)",
    "state_dimension": 2, "noise_dimension": 2,
    "pathwise_exact": _two_channel_exact,
    "pathwise_save_points": None,
})


# --- solvers, differing by one decision each ------------------------------------

_EM = '''
import numpy as np

def solve_sde(num_paths, dt, T, seed=42, dW=None):
    Nt = dW.shape[1]
    dt = T / Nt
    x = np.full(num_paths, 0.5)
    hist = []
    for i in range(Nt):
        x = x + 1.01 * x * dt + 0.87 * x * dW[:, i]
        hist.append(x.copy())
    hist = np.stack(hist, axis=1)
    keep = hist[:, (Nt // 10) - 1::Nt // 10] if Nt >= 10 else hist
    return {"terminal_paths": x, "path_values": keep,
            "empirical_mean": float(x.mean()), "empirical_variance": float(x.var(ddof=1))}
'''

# One decision different: the Milstein correction term is present.
_MILSTEIN = _EM.replace(
    "x = x + 1.01 * x * dt + 0.87 * x * dW[:, i]",
    "x = x + 1.01 * x * dt + 0.87 * x * dW[:, i] \\\n            "
    "+ 0.5 * 0.87 * 0.87 * x * (dW[:, i] ** 2 - dt)")

# Not a scheme: the closed form, evaluated. Exact at every dt, consumes the supplied
# increments, varies correctly with the path count -- and simulates nothing.
_TRANSCRIBED = '''
import numpy as np

def solve_sde(num_paths, dt, T, seed=42, dW=None):
    W = dW.sum(axis=1)
    x = 0.5 * np.exp(0.63155 * T + 0.87 * W)
    Nt = dW.shape[1]
    Wc = np.cumsum(dW, axis=1)
    idx = np.arange((Nt // 10) - 1, Nt, Nt // 10)
    t = (idx + 1) * (T / Nt)
    keep = 0.5 * np.exp(0.63155 * t[None, :] + 0.87 * Wc[:, idx])
    return {"terminal_paths": x, "path_values": keep,
            "empirical_mean": float(x.mean()), "empirical_variance": float(x.var(ddof=1))}
'''

# One decision different from _EM: the drift coefficient is wrong.
_WRONG = _EM.replace("1.01 * x * dt", "0.50 * x * dt")

_TWO_CHANNEL_OK = '''
import numpy as np

def solve_sde(num_paths, dt, T, seed=42, dW=None):
    Nt = dW.shape[1]
    dt = T / Nt
    x = np.full((num_paths, 2), 0.5)
    for i in range(Nt):
        x = x + 1.01 * x * dt + 0.87 * x * dW[:, i, :]
    return {"terminal_paths": x, "empirical_mean": list(x.mean(axis=0)),
            "empirical_variance": list(x.var(axis=0, ddof=1))}
'''

# One decision different: channel 0's increment drives both components.
_TWO_CHANNEL_SHARED = _TWO_CHANNEL_OK.replace(
    "0.87 * x * dW[:, i, :]", "0.87 * x * dW[:, i, 0:1]")


def _plan(tmp_path, source, name="plan", scheme_family=None):
    """A plan directory the harness can run: solver.py, plus the frontmatter H4 reads."""
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "solver.py").write_text(textwrap.dedent(source).lstrip("\n"))
    if scheme_family:
        (d / "SOLUTION.md").write_text(f"---\nscheme_family: {scheme_family}\n---\n\n# plan\n")
    return str(d)


def _graded(problem, plan_dir, **kw):
    """The route as `verify_problem` reaches it, sandboxed, work-precision off."""
    return V.verify_problem(problem, os.path.dirname(plan_dir), plan_dir=plan_dir,
                            sandbox=True, work_precision=False, **kw)


# --- H1: the route measures a strong error, at the right order -------------------

def test_h1_pathwise_route_grades_a_strong_error(tmp_path):
    out = _graded(LINEAR, _plan(tmp_path, _EM, scheme_family="euler-maruyama"))
    assert out["status"] == "ok", out
    assert out["gt_kind"] == "pathwise"
    assert 0.0 < out["pathwise_rel_err"] < 0.2, out["pathwise_rel_err"]
    assert out["passed"] is True
    # SciML's own axis, recorded beside the gate and never gating it.
    assert out["pathwise_abs_l2"] is not None
    assert out["pathwise_abs_l2"] > 0.0


@pytest.mark.parametrize("source,family,expected", [
    (_EM, "euler-maruyama", 0.5),
    (_MILSTEIN, "milstein", 1.0),
])
def test_h1_observed_strong_order_matches_the_scheme(tmp_path, source, family, expected):
    """H1's acceptance line: EM reads ~0.5 and Milstein ~1.0 on this problem.

    Checkable against the numbers `crn.py` records for the same two schemes on GBM --
    0.488 and 1.006 -- because this *is* that measurement, taken against an exact
    solution instead of against a finer step.
    """
    plan_dir = _plan(tmp_path, source, scheme_family=family)
    errs = []
    for k in (0, 1, 2):
        dt = V.sde_verify_dt(LINEAR) / 2 ** k
        pt, why = V._wp_solve_sde(LINEAR, plan_dir, LINEAR["num_paths"], dt,
                                  60, 2048, False, pathwise=True)
        assert pt is not None, why
        errs.append(pt["rel_err"])
    orders = [np.log2(errs[i] / errs[i + 1]) for i in range(2)]
    assert all(abs(o - expected) < 0.25 for o in orders), (errs, orders)


def test_h1_a_wrong_scheme_fails_the_gate(tmp_path):
    out = _graded(LINEAR, _plan(tmp_path, _WRONG, scheme_family="euler-maruyama"))
    assert out["status"] == "ok"
    assert out["passed"] is False
    assert out["pathwise_rel_err"] > V._pathwise_tol(LINEAR)


def test_h1_increments_are_nested_coarsenings_of_one_path():
    """Every level is an aggregation of a single draw, not an independent draw.

    This is what makes the levels comparable rather than independently noisy, and it
    is checked structurally rather than by eye: the coarse increments must be exact
    sums of the fine ones.
    """
    levels = V.sde_pathwise_ladder(LINEAR, num_paths=64)
    coarse, fine = V.crn_increments(levels[0]), V.crn_increments(levels[2])
    agg = fine.shape[1] // coarse.shape[1]
    assert np.allclose(coarse, fine.reshape(64, coarse.shape[1], agg).sum(axis=2),
                       rtol=0, atol=0)


def test_h1_a_non_nesting_step_is_reported_not_redrawn():
    """A declared level that is not a coarsening is skipped with a reason."""
    assert V.sde_pathwise_crn(LINEAR, 100, V.sde_verify_dt(LINEAR) / 3) is None


def test_h1_parent_and_child_build_identical_increments():
    """`runner._crn_increments` is a deliberate copy of `verify.crn_increments` (the
    child imports numpy and the stdlib only). Held to equality here rather than by
    hope, since the harness evaluates its truth on the parent's copy and the solver
    integrates the child's."""
    import runner
    for payload in V.sde_pathwise_ladder(TWO_CHANNEL, num_paths=32):
        a, b = V.crn_increments(payload), runner._crn_increments(payload)
        assert a.shape == b.shape
        assert np.array_equal(a, b)


# --- H2: the degenerate-ladder exclusion ----------------------------------------

def test_h2_a_transcribed_solution_is_excluded_not_passed(tmp_path):
    """The finding the plan exists to record, as a test.

    The transcribing solver is *more* accurate than either scheme, consumes the
    increments, and is not a discretization. It must be excluded by name.
    """
    out = _graded(LINEAR, _plan(tmp_path, _TRANSCRIBED, scheme_family="euler-maruyama"))
    assert out["status"] == "ok"
    assert out["passed"] is True                      # it *is* accurate
    assert out["pathwise_rel_err"] < 1e-12            # exactly so
    ladder = out["pathwise_ladder"]
    assert ladder["status"] == "ok", ladder
    assert ladder["degenerate"] is True
    assert ladder["reason"] == "pathwise_exact"

    rec = {"id": LINEAR["id"], "slug": LINEAR["slug"], "has_ground_truth": True,
           "ground_truth_kind": "pathwise",
           "run": {"status": "completed"},
           "pipeline": {"best_score": 10}, "verify": out}
    assert R.compute_verdict(rec) == "ANALYTIC_TRANSCRIPTION"
    # Neither a success nor a defect of the run.
    assert "ANALYTIC_TRANSCRIPTION" not in R._VERIFIED_VERDICTS
    assert "ANALYTIC_TRANSCRIPTION" in R._VERDICT_ORDER
    assert "ANALYTIC_TRANSCRIPTION" in R.VERDICT_LABEL


def test_h2_a_real_scheme_is_not_excluded(tmp_path):
    out = _graded(LINEAR, _plan(tmp_path, _EM, scheme_family="euler-maruyama"))
    ladder = out["pathwise_ladder"]
    assert ladder["status"] == "ok", ladder
    assert ladder["degenerate"] is False
    rec = {"id": LINEAR["id"], "slug": LINEAR["slug"], "has_ground_truth": True,
           "ground_truth_kind": "pathwise", "run": {"status": "completed"},
           "pipeline": {"best_score": 10}, "verify": out}
    assert R.compute_verdict(rec) == "VERIFIED_PASS"


def test_h2_a_wrong_solver_is_still_a_failure_not_an_exclusion(tmp_path):
    """The exclusion must not become a soft landing for being wrong."""
    out = _graded(LINEAR, _plan(tmp_path, _WRONG, scheme_family="euler-maruyama"))
    assert (out.get("pathwise_ladder") or {}).get("degenerate") is False
    rec = {"id": LINEAR["id"], "slug": LINEAR["slug"], "has_ground_truth": True,
           "ground_truth_kind": "pathwise", "run": {"status": "completed"},
           "pipeline": {"best_score": 4}, "verify": out}
    assert R.compute_verdict(rec) == "FAIL"


def test_h2_separates_a_constant_from_a_transcription(tmp_path):
    """§2a.1: the stiff-quadratic problems have a *constant* exact terminal state at
    SciML's horizon, so `-ones(num_paths)` is both perfectly accurate and perfectly
    uninformative. The ladder cannot tell the two degeneracies apart; the record must.
    """
    const = {**LINEAR, "slug": "test_pathwise_constant",
             "pathwise_exact": (lambda t, W: np.full(np.shape(W), -1.0)),
             "pathwise_save_points": None}
    out = _graded(const, _plan(tmp_path, '''
import numpy as np

def solve_sde(num_paths, dt, T, seed=42, dW=None):
    x = -np.ones(num_paths)
    return {"terminal_paths": x, "empirical_mean": -1.0, "empirical_variance": 0.0}
'''))
    assert out["passed"] is True
    assert out["pathwise_ladder"]["degenerate"] is True
    assert out["pathwise_ladder"]["reason"] == "path_independent"


def test_h2_records_the_absence_of_the_check_inline(tmp_path):
    """Inline there is no second solve, so the exclusion check cannot run. A passing
    record with nothing in the field would read as one that had the check."""
    out = V.verify_problem(LINEAR, str(tmp_path),
                           plan_dir=_plan(tmp_path, _EM), sandbox=False,
                           work_precision=False)
    assert out["pathwise_ladder"] == {"status": "not_run", "reason": "inline"}


# --- H3: the noise dimension ----------------------------------------------------

def test_h3_noise_dimension_defaults_to_the_state_dimension():
    for p in P.sde_problems():
        assert V.sde_noise_dimension(p) == p["state_dimension"], p["slug"]
    assert V.sde_noise_dimension(TWO_CHANNEL) == 2


def test_h3_the_brief_states_the_increment_shape():
    """The solver is told the shape it is promised -- H3's acceptance line."""
    body = S.contract_footer(TWO_CHANNEL)
    nt = max(1, round(TWO_CHANNEL["T"] / V.sde_verify_dt(TWO_CHANNEL)))
    assert f"({TWO_CHANNEL['num_paths']}, {nt}, 2)" in body
    assert "dW" in body
    assert "2 independent noise sources" in body
    # Scalar noise says so without a third axis.
    assert f"({LINEAR['num_paths']}, {nt})" in S.contract_footer(LINEAR)


def test_h3_payloads_carry_the_channel_axis():
    for payload in V.sde_pathwise_ladder(TWO_CHANNEL, num_paths=16):
        dW = V.crn_increments(payload)
        assert dW.ndim == 3 and dW.shape[0] == 16 and dW.shape[2] == 2


def test_h3_sharing_one_increment_across_channels_fails(tmp_path):
    """The defect `verifylib/kernel/sde/driver.py` describes and no problem in the
    first three suites can exercise."""
    ok = _graded(TWO_CHANNEL, _plan(tmp_path, _TWO_CHANNEL_OK, name="ok",
                                    scheme_family="euler-maruyama"))
    assert ok["status"] == "ok" and ok["passed"] is True, ok

    shared = _graded(TWO_CHANNEL, _plan(tmp_path, _TWO_CHANNEL_SHARED, name="shared",
                                        scheme_family="euler-maruyama"))
    assert shared["status"] == "ok", shared
    assert shared["passed"] is False
    assert shared["pathwise_rel_err"] > ok["pathwise_rel_err"]
    # And it is a failure, not an exclusion: it is wrong, not exact.
    assert shared["pathwise_ladder"]["degenerate"] is False


# --- H4: the work axis ----------------------------------------------------------

def test_h4_evals_scale_with_the_step_count(tmp_path):
    """H4's acceptance line: four `dt` values, `evals` scaling exactly 1:2:4:8 for a
    one-stage scheme."""
    plan_dir = _plan(tmp_path, _EM, scheme_family="euler-maruyama")
    dt0 = V.sde_verify_dt(LINEAR)
    works = [V._wp_sde_work(LINEAR, plan_dir, LINEAR["num_paths"], dt0 / 2 ** k)
             for k in range(4)]
    assert [w["evals"] // works[0]["evals"] for w in works] == [1, 2, 4, 8]
    assert [w["steps"] for w in works] == [100, 200, 400, 800]
    assert all(w["paths"] == LINEAR["num_paths"] for w in works)
    assert all(w["stages_source"] == "declared" for w in works)
    assert all(w["scheme_family"] == "euler-maruyama" for w in works)


def test_h4_an_undeclared_family_says_so(tmp_path):
    """A curve whose work axis rests on a guess has to admit it, rather than look
    like a measurement."""
    w = V._wp_sde_work(LINEAR, _plan(tmp_path, _EM), LINEAR["num_paths"], 0.01)
    assert w["stages"] == 1
    assert w["stages_source"] == "default"
    assert w["scheme_family"] is None


def test_h4_the_series_carries_work_and_the_comparator_axis(tmp_path):
    """Without this the suite produces a verdict and no table (plan §4a.i)."""
    plan_dir = _plan(tmp_path, _EM, scheme_family="euler-maruyama")
    prob = dict(LINEAR, work_precision_levels=[(LINEAR["num_paths"],
                                                V.sde_verify_dt(LINEAR) / 2)])
    out = V.verify_problem(prob, os.path.dirname(plan_dir), plan_dir=plan_dir,
                           sandbox=True, work_precision=True)
    series = out["work_precision"]
    assert series["axis"] == "paths_dt"
    assert len(series["points"]) >= 2, series
    for pt in series["points"]:
        assert pt["evals"] > 0 and pt["steps"] > 0 and pt["paths"] > 0
        assert pt["rel_err"] > 0
        assert "abs_l2" in pt          # SciML's own axis, per point
    # Error falls as work rises: the curve is a curve, not a flat line.
    assert series["points"][0]["rel_err"] > series["points"][-1]["rel_err"]


# --- H8 / H9 / H10 --------------------------------------------------------------

def test_h8_the_suite_is_registered():
    assert "sciml" in S.SUITES
    assert S.problem_suite({"suite": "sciml"}) == "sciml"
    # And it cannot be merged into another suite's denominator.
    with pytest.raises(SystemExit):
        S.guard_results_path([{"suite": "sciml"}], None, "x/{suite}.json")


def test_h9_the_comparator_artefacts_are_scanned():
    from verifylib.leakage import citation_scan
    for text in ("read benchmark/references/sciml/comparators.json",
                 "see comparators.json for the reference errors",
                 "as produced by harvest.jl"):
        assert citation_scan(text), text
    # And the public literature is not flagged: citing a scheme is not reading a key.
    assert not citation_scan("an SRA1 scheme, after Roessler (2010), as in "
                             "StochasticDiffEq.jl's algorithm set")


def test_h9_the_deny_rules_cover_the_artefacts():
    import json

    from .conftest import BENCHMARK
    with open(os.path.join(BENCHMARK, "pipeline-settings.json")) as fh:
        rules = json.load(fh)["permissions"]["deny"]
    assert "Read(./benchmark/**)" in rules          # covers references/sciml/*
    assert any(r.startswith("Bash(julia") for r in rules), rules


def test_h10_one_column_may_not_pool_two_arms():
    import compare as C
    col = C.Column("k", "mixed", "oneshot")
    col.by_slug["a"] = [{"arm": "unrestricted", "verdict": "PASS"},
                        {"arm": "scheme-matched", "verdict": "PASS"}]
    with pytest.raises(C.ArmConflict):
        col.arm()
    with pytest.raises(C.ArmConflict):
        C.check_arms([col])


def test_h10_records_without_an_arm_read_as_the_default():
    """Every suite but `sciml` has one arm, which is what the earlier runs were."""
    import compare as C
    col = C.Column("k", "old", "pipeline")
    col.by_slug["a"] = {"verdict": "VERIFIED_PASS"}
    assert col.arm() == S.DEFAULT_ARM


def test_h10_two_arms_across_columns_are_labelled_apart():
    import compare as C
    a, b = C.Column("a", "C0", "oneshot"), C.Column("b", "C0", "oneshot")
    a.by_slug["x"] = [{"arm": "scheme-matched"}]
    b.by_slug["x"] = [{"arm": "unrestricted"}]
    C.check_arms([a, b])
    assert a.label == "C0"
    assert b.label == "C0 [unrestricted]"


# --- zero churn outside the new suite -------------------------------------------

def test_no_problem_outside_the_new_suite_is_changed_by_any_of_this():
    """Step 5's acceptance line, in the form `test_sde_functional.py` already uses for
    the suite before this one: a problem statement is an agent *input*, so churn in it
    makes a later run incomparable with an earlier one for a reason that has nothing to
    do with the solver.

    Measured when this landed: all 66 problem statements and the whole manifest hashed
    identically before and after. Asserted here as the invariant that produces that
    result, rather than as the hash it produced, so a legitimate future edit to one
    problem does not read as a regression in this route.
    """
    for prob in P.PROBLEMS:
        if prob.get("suite") == "sciml":
            continue
        md = S.contract_footer(prob)
        entry = S.manifest_entry(prob)
        assert "dW" not in md, prob["slug"]
        assert "path_values" not in md, prob["slug"]
        assert "strong error" not in md, prob["slug"]
        assert "discretised scheme is required" not in md, prob["slug"]
        for key in ("noise_dimension", "pathwise_tol", "pathwise_fine_steps",
                    "pathwise_ladder_dts", "pathwise_save_points"):
            assert key not in entry, (prob["slug"], key)


def test_the_pathwise_route_is_opt_in():
    """The route is used by exactly the suite that asked for it.

    ~~No existing problem is on the new route~~ -- true when step 5 shipped, and step 6
    authored the eight that use it. The claim that survives, and the one that protects
    the other three suites' measured baselines, is that *only* those eight are on it
    and every other SDE problem still has `m = d` with no increments supplied.
    """
    on_route = [p["slug"] for p in P.PROBLEMS
                if p.get("ground_truth_kind") == "pathwise"]
    assert on_route == [p["slug"] for p in P.sciml_problems()
                        if p.get("ground_truth_kind") == "pathwise"]
    assert len(on_route) == 8
    for prob in P.sde_problems():
        if prob.get("suite") == "sciml":
            continue
        assert V.sde_noise_dimension(prob) == prob["state_dimension"], prob["slug"]
        assert prob.get("noise_dimension") is None, prob["slug"]


def test_a_non_pathwise_problem_is_never_handed_increments(tmp_path):
    """The graded call of every other problem is unchanged -- which is what keeps the
    three existing suites' measured baselines comparable with new runs."""
    import runner
    prob = next(p for p in P.sde_problems() if p.get("ground_truth_kind") != "pathwise")
    plan_dir = _plan(tmp_path, '''
import numpy as np

def solve_sde(num_paths, dt, T, seed=42):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(num_paths)
    return {"terminal_paths": x, "empirical_mean": float(x.mean()),
            "empirical_variance": float(x.var(ddof=1))}
''')
    # A solver with no `dW` parameter at all runs: the argument is passed only on the
    # route that grades a path, so this is a signature the harness must still accept.
    run = runner.run_solver(plan_dir=plan_dir, problem=prob, timeout_s=60, mem_mb=2048)
    assert run["status"] == "ok", run
    assert "crn" not in run["run_meta"]


def test_a_pathwise_record_reports_its_distance_from_its_gate():
    """F10, one route later: the per-plan table reads `harness_error`, and a shape it
    does not know prints `-` for a whole suite."""
    val, label = R.harness_error({"gt_kind": "pathwise", "pathwise_rel_err": 0.05,
                                  "pathwise_tol_used": 0.10})
    assert label == "pathwise"
    assert abs(val - 0.5) < 1e-12
    # A quantity of interest still wins where a problem declares both.
    val2, label2 = R.harness_error({"gt_kind": "pathwise", "pathwise_rel_err": 0.05,
                                    "pathwise_tol_used": 0.10,
                                    "functional_errs": {"price": 0.09},
                                    "functional_tol": 0.03})
    assert label2 == "functional"
    assert abs(val2 - 3.0) < 1e-12


# --- the exclusion applies only to a run that would otherwise pass ---------------

def _rec(out, *, best_score=10):
    return {"id": LINEAR["id"], "slug": LINEAR["slug"], "has_ground_truth": True,
            "ground_truth_kind": "pathwise", "run": {"status": "completed"},
            "pipeline": {"best_score": best_score}, "verify": out}


def test_a_degenerate_ladder_does_not_rescue_a_failing_solver(tmp_path):
    """Step 6's calibration found this: a *constant* solver has a degenerate ladder
    too -- a constant does not change with `dt` either -- so reading the exclusion
    before the pass decision reported it `ANALYTIC_TRANSCRIPTION` and threw away the
    fact that it had also missed its gate by up to 60x.

    A wrong solver is a FAIL whatever its ladder says. The exclusion is for the solver
    that is *right* and uninformative.
    """
    const = {**LINEAR, "slug": "test_pathwise_wrong_constant",
             "pathwise_save_points": None}
    out = _graded(const, _plan(tmp_path, '''
import numpy as np

def solve_sde(num_paths, dt, T, seed=42, dW=None):
    x = np.full(num_paths, 99.0)       # a constant, and nowhere near the solution
    return {"terminal_paths": x, "empirical_mean": 99.0, "empirical_variance": 0.0}
'''))
    assert out["passed"] is False
    assert out["pathwise_ladder"]["degenerate"] is True       # it is degenerate
    assert R.compute_verdict(_rec(out, best_score=4)) == "FAIL"
    assert R.compute_oneshot_verdict(_rec(out)) == "FAIL"


def test_the_baseline_verdict_excludes_a_transcription_too(tmp_path):
    """§10's prediction 2 is that the **C0 baseline** is where a transcribed answer is
    most likely -- a single completion with no tools is exactly the setting in which
    writing down the closed form is the obvious move. Until step 6 only the pipeline
    verdict knew about the exclusion, so such a baseline would have been recorded as a
    clean PASS on every problem in the suite."""
    out = _graded(LINEAR, _plan(tmp_path, _TRANSCRIBED))
    assert out["passed"] is True
    assert R.compute_oneshot_verdict(_rec(out)) == "ANALYTIC_TRANSCRIPTION"
    assert R.compute_verdict(_rec(out)) == "ANALYTIC_TRANSCRIPTION"
    # Counted as neither a success nor a failure, on both sides.
    assert "ANALYTIC_TRANSCRIPTION" not in R._ONESHOT_PASS_VERDICTS
    assert "ANALYTIC_TRANSCRIPTION" in R._ONESHOT_VERDICT_ORDER
    assert "ANALYTIC_TRANSCRIPTION" in R.ONESHOT_VERDICT_LABEL
    assert not R.oneshot_passed("ANALYTIC_TRANSCRIPTION")


def test_the_two_verdict_functions_read_the_ladder_the_same_way():
    """One reading, so the pipeline and baseline sides cannot drift apart."""
    for degenerate in (True, False):
        ver = {"status": "ok", "passed": True, "pathwise_rel_err": 0.01,
               "pathwise_ladder": {"status": "ok", "degenerate": degenerate}}
        assert R._is_transcription(ver) is degenerate
        want = "ANALYTIC_TRANSCRIPTION" if degenerate else "VERIFIED_PASS"
        assert R.compute_verdict(_rec(ver)) == want
        want_os = "ANALYTIC_TRANSCRIPTION" if degenerate else "PASS"
        assert R.compute_oneshot_verdict(_rec(ver)) == want_os
    # A ladder that could not be measured never excludes anything.
    for ladder in ({"status": "not_run", "reason": "inline"}, {}, None):
        ver = {"status": "ok", "passed": True, "pathwise_rel_err": 0.01,
               "pathwise_ladder": ladder}
        assert R._is_transcription(ver) is False
        assert R.compute_verdict(_rec(ver)) == "VERIFIED_PASS"
