"""The work-precision series: error against the time that error cost.

A work-precision diagram is how the numerical-analysis literature compares methods,
because "more accurate" is not a claim until "at what cost" is attached. The harness
was already producing both halves and keeping them apart: ``_score_pde_runs``
recorded a per-resolution error for the observed-order check, ``_produce_pde_runs``
recorded a per-resolution clock (step 1), and nothing joined them. These tests cover
the join, the optional sweep past it (``work_precision_levels``), and the property
the whole thing hangs on -- that none of it can move a verdict
(plan-benchop-slv.md §6, step 3).

Like the timing tests, these really run solvers in the sandbox: a clock and a
re-solve at another resolution are not things synthetic output can stand in for.
Errors are made deterministic instead (a known perturbation of the exact solution),
so the curve's *shape* is checkable without depending on a scheme.

Sections: I1 the free points, I2 the levels a problem pays for, I3 measurement that
cannot move a verdict, I4 the trip into the record, I5 (step 9) the SDE side's own
free point -- the coarse solve the two-level anti-hardcoding check already makes.
"""
import argparse
import json
import sys

import numpy as np
import pytest

from .conftest import BENCHMARK, RUN, P, V, problem

sys.path.insert(0, BENCHMARK)
import runner  # noqa: E402


# u_t = 0.1 u_xx, u(x,0)=sin(pi x), scored at t=0.5. The closed form times a known
# 1/N^2 perturbation: a second-order-accurate "scheme" with no scheme in it, so the
# curve must fall by exactly 4x per doubling and the test does not depend on any
# discretization actually being correct.
def _heat(extra=""):
    return f"""
import numpy as np
{extra}

def solve_pde(N):
    _log(N)
    x = np.linspace(0.0, 1.0, N)
    u = np.exp(-0.1 * np.pi ** 2 * 0.5) * np.sin(np.pi * x) * (1.0 + 4.0 / N ** 2)
    return {{"numerical_solution": u, "grid": {{"x": x}}, "t_final": 0.5}}
"""


def _logger(path, extra=""):
    """A ``_log`` that appends one line per call, so a test can count solves and see
    what each was asked for -- the only way to show a level was *reused* rather than
    re-solved."""
    return f"""
def _log(*a):
    with open({str(path)!r}, "a") as fh:
        fh.write(repr(a) + "\\n")
{extra}
"""


# Euler-Maruyama on dX = 0.1 X dt + 0.2 X dW, X0 = 1: a real scheme, so both the
# path count and the step size move the error and the sweep has something to show.
def _gbm(body="    pass"):
    return f"""
import numpy as np


def solve_sde(num_paths, dt, T, seed):
    _log(num_paths, dt, seed)
{body}
    rng = np.random.default_rng(seed)
    x = np.full(int(num_paths), 1.0)
    for _ in range(int(round(T / dt))):
        z = rng.standard_normal(int(num_paths))
        x = x + 0.1 * x * dt + 0.2 * x * np.sqrt(dt) * z
    return {{"terminal_paths": x, "empirical_mean": float(x.mean()),
            "empirical_variance": float(x.var(ddof=1))}}
"""


def plan(tmp_path, *sources):
    (tmp_path / "solver.py").write_text("".join(sources))
    return str(tmp_path)


def solves(log_path):
    """The arguments of every solve the sandbox actually performed."""
    if not log_path.exists():
        return []
    return [eval(line) for line in log_path.read_text().splitlines() if line.strip()]  # noqa: S307


@pytest.fixture
def heat(tmp_path):
    """(plan_dir, log_path) for the deterministic heat solver."""
    log = tmp_path / "solves.log"
    return plan(tmp_path, _logger(log), _heat()), log


@pytest.fixture
def gbm(tmp_path):
    log = tmp_path / "solves.log"
    return plan(tmp_path, _logger(log), _gbm()), log


# --- I1: the graded ladder becomes a curve, at no extra cost ---------------------

def test_the_graded_ladder_alone_emits_a_two_point_curve(heat):
    """The point of hanging the series off the order check: a problem that declares
    no levels at all still gets a work-precision curve, and pays nothing for it."""
    plan_dir, log = heat
    out = V.verify_problem(problem("pde_heat_1d"), plan_dir, plan_dir=plan_dir, sandbox=True)
    assert out["status"] == "ok" and out["passed"] is True, out
    wp = out["work_precision"]
    assert wp["axis"] == "N" and wp["path"] == "sandbox"
    assert [p["N"] for p in wp["points"]] == [64, 128]
    assert all(p["graded"] for p in wp["points"])
    assert len(solves(log)) == 2, "the free points must not cost a third solve"


def test_each_point_joins_the_error_and_the_clock_of_its_own_grid(tmp_path):
    """The join is by resolution, not by position. A solver whose cost rises with N
    makes a positional join visible: the coarse grid has the larger error and the
    smaller time, so a swap would pair them the wrong way round."""
    log = tmp_path / "solves.log"
    slow = "    import time\n    time.sleep(N / 1000.0)\n"
    src = _heat().replace("    x = np.linspace", slow + "    x = np.linspace")
    plan_dir = plan(tmp_path, _logger(log), src)
    out = V.verify_problem(problem("pde_heat_1d"), plan_dir, plan_dir=plan_dir, sandbox=True)
    coarse, fine = out["work_precision"]["points"]
    assert (coarse["N"], fine["N"]) == (64, 128)
    assert coarse["rel_err"] == out["grid_errors"]["64"]
    assert fine["rel_err"] == out["grid_errors"]["128"]
    assert coarse["rel_err"] > fine["rel_err"]
    assert coarse["solve_s"] < fine["solve_s"], "the cost axis is the wrong way round"


def test_the_curve_falls_at_the_solvers_own_order(heat):
    """A 1/N^2 error must read as 4x per doubling. If it does not, the series is
    reporting something other than the error that was graded."""
    plan_dir, _log = heat
    pts = V.verify_problem(problem("pde_heat_1d"), plan_dir, plan_dir=plan_dir,
                           sandbox=True)["work_precision"]["points"]
    assert pts[0]["rel_err"] / pts[1]["rel_err"] == pytest.approx(4.0, rel=1e-6)


def test_the_inline_path_emits_no_series(heat):
    """Same rule as the clock it is built on: a sandboxed time and a warm in-process
    time are different measurement conditions, so the inline path records neither."""
    plan_dir, _log = heat
    out = V.verify_problem(problem("pde_heat_1d"), plan_dir, plan_dir=plan_dir, sandbox=False)
    assert out["status"] == "ok" and out["passed"] is True
    assert "work_precision" not in out


def test_a_graded_sde_run_is_its_own_anchor_point(gbm):
    """A problem graded on moments alone is solved once, so its anchor is the whole
    free curve -- **[rev, step 9]**, which found the free second point that *does*
    exist wherever a problem is scored on a functional (I5)."""
    plan_dir, log = gbm
    prob = problem("sde_gbm")
    out = V.verify_problem(prob, plan_dir, plan_dir=plan_dir, sandbox=True)
    assert out["status"] == "ok", out
    wp = out["work_precision"]
    assert wp["axis"] == "paths_dt"
    assert [(p["num_paths"], p["dt"], p["graded"]) for p in wp["points"]] == [
        (prob["num_paths"], V.sde_verify_dt(prob), True)]
    assert wp["points"][0]["solve_s"] > 0.0
    assert len(solves(log)) == 1


def test_the_sde_point_plots_the_error_the_verdict_rested_on(gbm):
    """Worst checked relative moment error -- the same quantity the gate uses. A
    series that summarized the solver differently would describe a different solver
    than the one that was graded."""
    plan_dir, _log = gbm
    out = V.verify_problem(problem("sde_gbm"), plan_dir, plan_dir=plan_dir, sandbox=True)
    assert out["work_precision"]["points"][0]["rel_err"] == max(
        out["worst_variance_rel_err"], out["worst_mean_rel_err"])


# --- I2: declared levels extend it, and reuse what the ladder already solved ------

def test_a_level_the_ladder_already_covers_is_not_re_solved(tmp_path):
    """Declaring the problem's own grid_N and 2*grid_N must cost nothing: four
    declared levels, two already graded, two extra solves."""
    log = tmp_path / "solves.log"
    plan_dir = plan(tmp_path, _logger(log), _heat())
    prob = dict(problem("pde_heat_1d"), work_precision_levels=[32, 64, 128, 256])
    out = V.verify_problem(prob, plan_dir, plan_dir=plan_dir, sandbox=True)
    pts = out["work_precision"]["points"]
    assert [p["N"] for p in pts] == [32, 64, 128, 256]
    assert [p["graded"] for p in pts] == [False, True, True, False]
    assert sorted(n for (n,) in solves(log)) == [32, 64, 128, 256]


def test_points_come_back_ordered_by_the_work_they_cost(tmp_path):
    """Declared in any order, emitted along the cost axis, so the series reads as a
    curve rather than as a scatter the reader has to sort."""
    log = tmp_path / "solves.log"
    plan_dir = plan(tmp_path, _logger(log), _heat())
    prob = dict(problem("pde_heat_1d"), work_precision_levels=[256, 16, 32])
    out = V.verify_problem(prob, plan_dir, plan_dir=plan_dir, sandbox=True)
    assert [p["N"] for p in out["work_precision"]["points"]] == [16, 32, 64, 128, 256]


def test_an_sde_sweep_moves_both_axes_and_orders_by_paths_times_steps(gbm):
    """``(num_paths, dt)`` is two knobs on one cost: the ordering key is the product
    of the path count and the step count, which is what the solver actually does."""
    plan_dir, log = gbm
    prob = dict(problem("sde_gbm"),
                work_precision_levels=[(50000, 0.0025), (5000, 0.05), (20000, 0.01)])
    out = V.verify_problem(prob, plan_dir, plan_dir=plan_dir, sandbox=True)
    pts = out["work_precision"]["points"]
    assert [(p["num_paths"], p["dt"]) for p in pts] == [
        (5000, 0.05), (20000, 0.01), (50000, 0.01), (50000, 0.0025)]
    assert [p["solve_s"] for p in pts] == sorted(p["solve_s"] for p in pts)
    assert len(solves(log)) == 4


def test_the_seed_is_held_fixed_across_an_sde_sweep(gbm):
    """Moving the sample and the work together would measure two things at once;
    the curve's whole value is that only the work changes."""
    plan_dir, log = gbm
    prob = dict(problem("sde_gbm"), work_precision_levels=[(5000, 0.05), (20000, 0.01)])
    V.verify_problem(prob, plan_dir, plan_dir=plan_dir, sandbox=True)
    calls = solves(log)
    assert len(calls) == 3
    assert {seed for _n, _dt, seed in calls} == {prob["seed"]}
    assert {(n, dt) for n, dt, _s in calls} == {
        (5000, 0.05), (20000, 0.01), (prob["num_paths"], V.sde_verify_dt(prob))}


def test_declared_levels_are_validated_rather_than_trusted(tmp_path):
    """A zero or negative level is a problem-authoring slip, and asking the sandbox
    to solve at N=0 would record a crash against the solver instead."""
    prob = dict(problem("pde_heat_1d"), work_precision_levels=[64, 0, -8, 64, 32])
    assert V._wp_pde_levels(prob) == [32, 64]
    sde = dict(problem("sde_gbm"),
               work_precision_levels=[(1000, 0.1), (0, 0.1), (1000, 0.0), (1000, 0.1)])
    assert V._wp_sde_levels(sde) == [(1000, 0.1)]


def test_every_declared_level_in_the_corpus_is_well_formed():
    """Whatever problems.py declares now or later: PDE levels are positive ints, SDE
    levels are positive (num_paths, dt) pairs, and the key is never a bare number."""
    for prob in P.PROBLEMS:
        levels = prob.get("work_precision_levels")
        if not levels:
            continue
        assert isinstance(levels, (list, tuple)), prob["slug"]
        if prob["type"] == "pde":
            assert V._wp_pde_levels(prob) == sorted({int(n) for n in levels}), prob["slug"]
        else:
            assert len(V._wp_sde_levels(prob)) == len({tuple(x) for x in levels}), prob["slug"]


# --- I3: it is measurement, so it can never move a verdict -----------------------

def test_a_level_that_crashes_is_skipped_and_the_verdict_stands(tmp_path):
    """The graded ladder decided this solver passes. A crash at a resolution nobody
    grades on is recorded, with its reason, and changes nothing."""
    log = tmp_path / "solves.log"
    guard = "    if N not in (64, 128):\n        raise RuntimeError('boom')\n"
    src = _heat().replace("    x = np.linspace", guard + "    x = np.linspace")
    plan_dir = plan(tmp_path, _logger(log), src)
    prob = dict(problem("pde_heat_1d"), work_precision_levels=[32, 256])
    out = V.verify_problem(prob, plan_dir, plan_dir=plan_dir, sandbox=True)
    assert out["status"] == "ok" and out["passed"] is True and out["verified_score"] == 10
    wp = out["work_precision"]
    assert [p["N"] for p in wp["points"]] == [64, 128]
    assert [(s["N"], s["reason"]) for s in wp["skipped"]] == [(32, "exception"),
                                                              (256, "exception")]


def test_a_level_that_times_out_is_skipped_and_the_verdict_stands(tmp_path):
    """The expensive failure mode, and the one a 4N level invites: the budget is per
    solve, so an unaffordable level costs its timeout and nothing else."""
    log = tmp_path / "solves.log"
    slow = "    import time\n    time.sleep(0 if N < 256 else 30)\n"
    src = _heat().replace("    x = np.linspace", slow + "    x = np.linspace")
    plan_dir = plan(tmp_path, _logger(log), src)
    prob = dict(problem("pde_heat_1d"), work_precision_levels=[256])
    out = V.verify_problem(prob, plan_dir, plan_dir=plan_dir, sandbox=True, timeout_s=2)
    assert out["status"] == "ok" and out["passed"] is True
    assert out["work_precision"]["skipped"] == [{"N": 256, "reason": "timeout"}]


def test_a_legacy_no_argument_solver_is_not_swept(tmp_path):
    """``solve_pde()`` ignores N, so every level would return the identical solve --
    one point drawn four times, which reads as a flat curve rather than as the
    absent measurement it is."""
    log = tmp_path / "solves.log"
    src = _heat().replace("def solve_pde(N):", "def solve_pde():").replace("_log(N)", "_log(64)")
    src = src.replace("N ** 2", "64 ** 2").replace("0.0, 1.0, N", "0.0, 1.0, 64")
    plan_dir = plan(tmp_path, _logger(log), src)
    prob = dict(problem("pde_heat_1d"), work_precision_levels=[32, 128])
    out = V.verify_problem(prob, plan_dir, plan_dir=plan_dir, sandbox=True)
    assert out["status"] == "ok" and out["order_checked"] is False
    wp = out["work_precision"]
    assert [p["N"] for p in wp["points"]] == [None]
    assert {s["reason"] for s in wp["skipped"]} == {"fixed_resolution"}


def test_an_sde_solver_that_ignores_the_level_is_recorded_as_ignoring_it(tmp_path):
    """The SDE analogue of the order check. A hardcoded path count would otherwise
    contribute the same error at every level and draw a flat curve that looks like
    a method insensitive to sampling."""
    log = tmp_path / "solves.log"
    plan_dir = plan(tmp_path, _logger(log), _gbm("    num_paths = 50000"))
    prob = dict(problem("sde_gbm"), work_precision_levels=[(5000, 0.01)])
    out = V.verify_problem(prob, plan_dir, plan_dir=plan_dir, sandbox=True)
    assert out["status"] == "ok"
    assert out["work_precision"]["skipped"] == [
        {"num_paths": 5000, "dt": 0.01, "reason": "level_ignored"}]


def test_the_sweep_cannot_change_a_single_field_of_the_verdict(tmp_path):
    """The property the whole design rests on, checked the blunt way: grade the same
    solver with the sweep on and off, and require the two records to differ in
    exactly one key."""
    log = tmp_path / "solves.log"
    guard = "    if N not in (64, 128):\n        raise RuntimeError('boom')\n"
    src = _heat().replace("    x = np.linspace", guard + "    x = np.linspace")
    plan_dir = plan(tmp_path, _logger(log), src)
    prob = dict(problem("pde_heat_1d"), work_precision_levels=[32, 256])
    swept = V.verify_problem(prob, plan_dir, plan_dir=plan_dir, sandbox=True)
    plain = V.verify_problem(prob, plan_dir, plan_dir=plan_dir, sandbox=True,
                             work_precision=False)
    assert swept["work_precision"] != plain["work_precision"]
    for key in set(swept) | set(plain):
        if key in ("work_precision", "timing"):   # timing carries its own clock
            continue
        assert swept[key] == plain[key], key


def test_turning_the_sweep_off_keeps_the_free_points_and_skips_the_paid_ones(tmp_path):
    """``--no-work-precision`` is about compute, not about the record: the graded
    ladder's own points are already paid for, so they survive it."""
    log = tmp_path / "solves.log"
    plan_dir = plan(tmp_path, _logger(log), _heat())
    prob = dict(problem("pde_heat_1d"), work_precision_levels=[32, 256])
    out = V.verify_problem(prob, plan_dir, plan_dir=plan_dir, sandbox=True,
                           work_precision=False)
    assert [p["N"] for p in out["work_precision"]["points"]] == [64, 128]
    assert "skipped" not in out["work_precision"]
    assert sorted(n for (n,) in solves(log)) == [64, 128]


def test_a_failing_solver_still_gets_a_curve(tmp_path):
    """A wrong answer's error-vs-time curve is a measurement, not a reward: it is
    what shows a solver converging steadily to the wrong thing."""
    log = tmp_path / "solves.log"
    src = _heat().replace("(1.0 + 4.0 / N ** 2)", "0.5")
    plan_dir = plan(tmp_path, _logger(log), src)
    out = V.verify_problem(problem("pde_heat_1d"), plan_dir, plan_dir=plan_dir, sandbox=True)
    assert out["status"] == "ok" and out["passed"] is False
    # Halving the field halves the relative L2, at every resolution: the curve is
    # flat, which is the signature of an error that is not discretization error.
    assert [p["rel_err"] for p in out["work_precision"]["points"]] == pytest.approx([0.5, 0.5])


def test_an_unscorable_extra_level_is_skipped_rather_than_raised(tmp_path):
    """``_pde_eval_run`` raises on output it cannot read. Inside the sweep that must
    become a reason, because an exception here would turn a graded PASS into an
    ``error`` status on the strength of a number no verdict uses."""
    log = tmp_path / "solves.log"
    bad = '    if N == 256:\n        return {"grid": {"x": x}, "t_final": 0.5,\n' \
          '                "numerical_solution": np.zeros(3)}\n'
    src = _heat().replace("    return {", bad + "    return {")
    plan_dir = plan(tmp_path, _logger(log), src)
    prob = dict(problem("pde_heat_1d"), work_precision_levels=[256])
    out = V.verify_problem(prob, plan_dir, plan_dir=plan_dir, sandbox=True)
    assert out["status"] == "ok" and out["passed"] is True
    assert [s["N"] for s in out["work_precision"]["skipped"]] == [256]


# --- I4: the series reaches the record and the reader ----------------------------

def test_the_series_survives_the_trip_into_results_json(tmp_path):
    """``run.py`` writes the verify dict straight into results.json."""
    log = tmp_path / "solves.log"
    plan_dir = plan(tmp_path, _logger(log), _heat())
    prob = dict(problem("pde_heat_1d"), work_precision_levels=[32])
    out = V.verify_problem(prob, plan_dir, plan_dir=plan_dir, sandbox=True)
    assert json.loads(json.dumps(out["work_precision"])) == out["work_precision"]


def test_the_summary_prints_the_series_and_names_what_it_could_not_measure(tmp_path):
    """A number stored and never shown is a number nobody checks -- and a short
    curve must never be mistaken for a cheap one."""
    log = tmp_path / "solves.log"
    guard = "    if N == 256:\n        raise RuntimeError('boom')\n"
    src = _heat().replace("    x = np.linspace", guard + "    x = np.linspace")
    plan_dir = plan(tmp_path, _logger(log), src)
    prob = dict(problem("pde_heat_1d"), work_precision_levels=[32, 256])
    text = V._fmt(V.verify_problem(prob, plan_dir, plan_dir=plan_dir, sandbox=True), prob)
    assert "work-precision" in text and "N=32" in text and "N=128" in text
    assert "* = swept, not graded" in text
    assert "not measured" in text and "N=256: exception" in text


def test_both_runners_sweep_by_default_and_under_one_setting():
    """A baseline swept and a pipeline not swept would have no common error-vs-time
    axis, which is the comparison the sweep exists for. Same argument as the sandbox
    in step 2, so the same single source of truth."""
    def kw(**over):
        return RUN._verify_kwargs(argparse.Namespace(**over))

    assert kw(work_precision=True)["work_precision"] is True
    assert kw(work_precision=False)["work_precision"] is False
    # and the default, for a caller that does not know the flag exists
    assert kw()["work_precision"] is True


def test_a_transposed_return_is_oriented_by_the_run_not_by_the_problem():
    """``_run_and_read_sde`` accepts a ``(d, num_paths)`` return and transposes it,
    and it recognized that shape by comparing against the *problem's* path count.
    A swept level varies that count, so the comparison failed and the array was
    scored component-for-path -- silently, with plausible numbers. The orientation
    now keys off the run that was actually executed."""
    prob = problem("sde_gbm_2d_high_corr")
    n = 777                                   # a swept level, not the problem's own
    assert n != prob["num_paths"]
    paths = np.vstack([np.full(n, 1.0), np.full(n, 2.0)])      # (d, n): transposed
    result = {"terminal_paths": paths, "empirical_mean": 0.0, "empirical_variance": 0.0}
    meta = {"num_paths": n, "dt": 0.01, "T": 1.0, "seed": 1}
    comps, _meta, _res, err = V._run_and_read_sde(prob, "/nonexistent",
                                                  injected=(result, meta))
    assert err is None
    assert [(c[0], c[1]) for c in comps] == [("X", 1.0), ("Y", 2.0)]


@pytest.mark.parametrize("mod", ["run", "oneshot"])
def test_both_runners_expose_the_flag_on_the_command_line(mod, capsys):
    """Recorded in each runner's config too, so a sweep taken with it off is
    identifiable rather than silently pooled with one taken with it on."""
    import importlib

    with pytest.raises(SystemExit):
        importlib.import_module(mod).main(["--help"])
    assert "--no-work-precision" in capsys.readouterr().out


def test_the_sweep_can_override_the_sde_configuration_but_never_the_seed(tmp_path):
    """``run_solver``'s overrides, at the level below verify.py: the sweep needs to
    move num_paths and dt, and must not be able to move the sample."""
    log = tmp_path / "solves.log"
    plan_dir = plan(tmp_path, _logger(log), _gbm())
    prob = problem("sde_gbm")
    run = runner.run_solver(plan_dir, prob, sde_num_paths=1234, sde_dt=0.25)
    assert run["status"] == "ok"
    assert run["run_meta"]["num_paths"] == 1234 and run["run_meta"]["dt"] == 0.25
    assert run["run_meta"]["seed"] == prob["seed"]
    assert solves(log) == [(1234, 0.25, prob["seed"])]


# --- I5: the free second point on the SDE side (step 9) --------------------------
#
# Step 5 gave every functional-scored SDE problem a second sandboxed solve -- a
# quarter of the paths, same step, same seed -- to prove the reported number is an
# estimate and not a constant. Step 9's staging smoke test found that solve was
# timed, scored and then dropped: the externally anchored suite, which declares no
# `work_precision_levels`, emitted a one-point "curve" for every entry, so a
# step-10 sweep would have paid for ten curves and recorded none of them.

from .test_sde_functional import PRICE_PROBLEM, price_problem  # noqa: E402

_PRICE_SOLVER = """
import numpy as np


def solve_sde(num_paths, dt, T, seed):
    _log(num_paths, dt, seed)
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(int(num_paths))
    s = np.exp(-0.5 * 0.2 ** 2 * T + 0.2 * np.sqrt(T) * z)
    return {"terminal_paths": s, "empirical_mean": float(s.mean()),
            "empirical_variance": float(s.var(ddof=1)),
            "functionals": {"price": float(np.maximum(s - 1.0, 0.0).mean())}}
"""


@pytest.fixture
def priced(tmp_path):
    log = tmp_path / "solves.log"
    return plan(tmp_path, _logger(log), _PRICE_SOLVER), log


def test_a_functional_sde_problem_gets_its_coarse_level_for_free(priced):
    plan_dir, log = priced
    prob = price_problem()
    out = V.verify_problem(prob, plan_dir, plan_dir=plan_dir, sandbox=True)
    n_lo = V._sde_level_paths(prob, prob["num_paths"])
    assert [(p["num_paths"], p["dt"], p["graded"]) for p in out["work_precision"]["points"]] == [
        (n_lo, prob["dt"], False), (prob["num_paths"], prob["dt"], True)]
    # The whole claim: two points, two solves -- the two the grade already made.
    assert len(solves(log)) == 2
    assert all(p["solve_s"] > 0.0 for p in out["work_precision"]["points"])


def test_the_free_point_plots_the_coarse_levels_own_error(priced):
    """Not the graded one's. A curve whose two points carried the same error would
    be a straight line through a measurement that was never taken."""
    plan_dir, _log = priced
    out = V.verify_problem(price_problem(), plan_dir, plan_dir=plan_dir, sandbox=True)
    free, graded = out["work_precision"]["points"]
    assert free["rel_err"] == out["functional_levels"]["worst_functional_rel_err"]
    assert graded["rel_err"] == out["worst_functional_rel_err"]
    assert free["rel_err"] != graded["rel_err"]


def test_turning_the_sweep_off_keeps_the_sde_free_point(priced):
    """`--no-work-precision` declines to *pay* for levels. It cannot decline a point
    that costs nothing, any more than it can on the PDE side."""
    plan_dir, _log = priced
    out = V.verify_problem(price_problem(), plan_dir, plan_dir=plan_dir, sandbox=True,
                           work_precision=False)
    assert len(out["work_precision"]["points"]) == 2


def test_a_moment_gated_problem_reports_the_coarse_level_as_skipped(priced):
    """The graded point's error is max(mean, variance, functional) and the coarse
    level scores the functional alone, so plotting them together would draw a curve
    out of two different summaries of the solver."""
    plan_dir, _log = priced
    prob = price_problem(moments=lambda T: {"mean": 1.0,
                                            "variance": float(np.exp(0.04) - 1.0)})
    out = V.verify_problem(prob, plan_dir, plan_dir=plan_dir, sandbox=True)
    wp = out["work_precision"]
    assert [p["graded"] for p in wp["points"]] == [True]
    assert wp["skipped"] and "functional" in wp["skipped"][0]["reason"]
    assert out["functional_levels"]["status"] == "ok"  # the check itself still ran


def test_the_free_point_is_not_re_solved_when_a_declared_level_names_it(priced):
    plan_dir, log = priced
    n_lo = V._sde_level_paths(PRICE_PROBLEM, PRICE_PROBLEM["num_paths"])
    prob = price_problem(work_precision_levels=[(n_lo, PRICE_PROBLEM["dt"])])
    out = V.verify_problem(prob, plan_dir, plan_dir=plan_dir, sandbox=True)
    assert len(out["work_precision"]["points"]) == 2
    assert len(solves(log)) == 2
