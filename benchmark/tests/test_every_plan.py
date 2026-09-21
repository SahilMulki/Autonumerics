"""Findings §11.1 and F6: the harness grades every plan, and only the file the
kernel scored.

The single most informative measurement in the no-closed-form findings was made
by hand three times -- ``verify.py --plan-dir`` on the plans the conductor did
*not* pick -- and each time it produced the finding. ``grade_every_plan`` makes it
a recorded statistic (``wrong_plan_won``), and the solver hash the kernel writes
into the metrics block is what stops a drifted file from being graded as the plan.
"""
import json
import os
import shutil
import sys

import numpy as np

from verifylib.kernel import run
from verifylib.kernel.metrics import render_block

from .conftest import REPO, RUN, R, V
from .test_cross_layer import _canary_problem, _spec

CANARY = os.path.join(REPO, "verifylib", "tests", "canaries", "kernel")

SCALED = """
import numpy as np
from honest import solve_pde as _honest

def solve_pde(N, override=None):
    out = _honest(N, override)
    u = np.asarray(out["numerical_solution"]) * (1.0 + {factor})
    out["numerical_solution"], out["fields"] = u, {{"u": u}}
    return out
"""


def _workspace(tmp_path, plans, best):
    """``plans``: ``{name: solver_text}``; every plan gets a kernel-scored
    SOLUTION.md (so the metrics block carries the hash) and STATE.md names ``best``."""
    ws = tmp_path / "ws"
    (ws / "plans").mkdir(parents=True)
    (ws / "problem_spec.json").write_text(json.dumps(_spec()))
    dirs = {}
    for name, text in plans.items():
        plan = ws / "plans" / name
        plan.mkdir()
        shutil.copy(os.path.join(CANARY, "honest.py"), plan / "honest.py")
        (plan / "solver.py").write_text(text)
        m = run(_spec(), str(plan), {})
        (plan / "SOLUTION.md").write_text(
            f"---\nscheme_family: fd\n---\n\n{render_block(m)}\n\n"
            f"<review score={m['score']}>\nScore: {m['score']}/10\n</review>\n")
        dirs[name] = str(plan)
        if str(plan) not in sys.path:
            sys.path.insert(0, str(plan))
    scores = "\n".join(f"  {n}:\n    iter: 1\n    score: {json.loads(json.dumps(0))}\n"
                       for n in plans)
    (ws / "STATE.md").write_text(f"---\nphase: done\nproblem_type: pde\nbest_plan: {best}\n"
                                 f"plans:\n{scores}---\n")
    return ws, dirs


# --- F6 ---------------------------------------------------------------------------

def test_the_kernel_records_the_solver_hash_and_the_block_renders_it(tmp_path):
    ws, dirs = _workspace(tmp_path, {"1-honest": "from honest import solve_pde\n"}, "1-honest")
    with open(os.path.join(dirs["1-honest"], "SOLUTION.md")) as fh:
        block = fh.read()
    assert "solver_sha256: " in block
    assert V.scored_solver_sha256(dirs["1-honest"]) == V.solver_sha256(dirs["1-honest"])
    assert V.solver_drift(dirs["1-honest"]) is None


def test_a_solver_edited_after_scoring_is_unverified_not_graded(tmp_path):
    """The Cahn-Hilliard case: a refine cycle cut short mid-edit left a 9 that
    belonged to the earlier file. The harness must not grade the new one as the
    plan."""
    ws, dirs = _workspace(tmp_path, {"1-honest": "from honest import solve_pde\n"}, "1-honest")
    with open(os.path.join(dirs["1-honest"], "solver.py"), "a") as fh:
        fh.write("\n# a second cycle, cut short\n")
    drift = V.solver_drift(dirs["1-honest"])
    assert drift and drift["scored"] != drift["on_disk"]
    out = V.verify_problem(_canary_problem(), str(ws))
    assert out["status"] == "error" and "changed after scoring" in out["error"]
    assert out["solver_drift"] == drift
    rec = {"has_ground_truth": True, "ground_truth_kind": "exact",
           "run": {"status": "completed"}, "pipeline": {"best_score": 10}, "verify": out}
    assert R.compute_verdict(rec) == "UNVERIFIED"


def test_a_plan_without_a_recorded_hash_is_graded_as_before(tmp_path):
    plan = tmp_path / "p"
    plan.mkdir()
    shutil.copy(os.path.join(CANARY, "honest.py"), plan / "honest.py")
    shutil.copy(os.path.join(CANARY, "honest.py"), plan / "solver.py")
    (plan / "SOLUTION.md").write_text("---\nscheme: ftcs\n---\n")
    assert V.solver_drift(str(plan)) is None


# --- §11.1 ------------------------------------------------------------------------

def test_every_plan_is_graded_and_the_wrong_winner_is_named(tmp_path):
    """Three plans: honest, 0.5% off, 3% off. The conductor 'picked' the 0.5% one;
    the harness finds the honest plan better by more than the reference error, so
    ``wrong_plan_won`` is True. Every plan carries the kernel's own block score."""
    ws, dirs = _workspace(tmp_path, {
        "1-honest": "from honest import solve_pde\n",
        "2-close": SCALED.format(factor=0.005),
        "3-off": SCALED.format(factor=0.03),
    }, best="2-close")
    problem = {**_canary_problem(), "reference_error": 1e-9}
    winner_verify = V.verify_problem(problem, str(ws))
    rec = {"has_ground_truth": True, "ground_truth_kind": "exact",
           "pipeline": {"best_plan": "2-close"}}
    out = RUN.grade_every_plan(problem, str(ws), rec, winner_verify, None, rerun_kernel=False)
    assert set(out["plans"]) == {"1-honest", "2-close", "3-off"}
    assert out["plans"]["2-close"]["is_winner"] is True
    errs = {n: e["harness"]["rel_l2_err"] for n, e in out["plans"].items()}
    assert errs["1-honest"] < errs["2-close"] < errs["3-off"]
    assert out["best_plan_by_harness"] == "1-honest"
    assert out["wrong_plan_won"] is True
    for name, entry in out["plans"].items():
        assert entry["kernel"]["score"] is not None and entry["kernel"]["provenance"]
        assert entry["kernel"]["solver_sha256"] == V.solver_sha256(dirs[name])
        assert entry["verdict"] in R._VERDICT_ORDER
    # The 3% plan fails the 1% gate on its own and the kernel scored it below 10:
    assert out["plans"]["3-off"]["harness"]["passed"] is False
    assert out["plans"]["3-off"]["verdict"] in ("FAIL", "UNDERCLAIM")


def test_wrong_plan_won_is_false_when_the_winner_is_the_most_accurate(tmp_path):
    ws, _ = _workspace(tmp_path, {
        "1-honest": "from honest import solve_pde\n",
        "2-close": SCALED.format(factor=0.005),
    }, best="1-honest")
    problem = {**_canary_problem(), "reference_error": 1e-9}
    rec = {"has_ground_truth": True, "ground_truth_kind": "exact",
           "pipeline": {"best_plan": "1-honest"}}
    out = RUN.grade_every_plan(problem, str(ws), rec, None, None)
    assert out["wrong_plan_won"] is False


def test_a_margin_inside_the_reference_error_is_not_a_wrong_winner(tmp_path):
    """Two plans within the reference's own error of each other are a tie, not
    a wrong pick."""
    ws, _ = _workspace(tmp_path, {
        "1-honest": "from honest import solve_pde\n",
        "2-close": SCALED.format(factor=0.005),
    }, best="2-close")
    problem = {**_canary_problem(), "reference_error": 0.02}
    rec = {"has_ground_truth": True, "ground_truth_kind": "exact",
           "pipeline": {"best_plan": "2-close"}}
    out = RUN.grade_every_plan(problem, str(ws), rec, None, None)
    assert out["wrong_plan_won"] is False


def test_a_drifted_plan_is_graded_unverified_inside_the_per_plan_block(tmp_path):
    ws, dirs = _workspace(tmp_path, {
        "1-honest": "from honest import solve_pde\n",
        "2-close": SCALED.format(factor=0.005),
    }, best="1-honest")
    with open(os.path.join(dirs["2-close"], "solver.py"), "a") as fh:
        fh.write("\n# drifted\n")
    rec = {"has_ground_truth": True, "ground_truth_kind": "exact",
           "pipeline": {"best_plan": "1-honest"}}
    out = RUN.grade_every_plan(_canary_problem(), str(ws), rec, None, None)
    drifted = out["plans"]["2-close"]
    assert drifted["harness"]["status"] == "error" and drifted["harness"]["solver_drift"]
    assert drifted["verdict"] == "UNVERIFIED"
    assert out["wrong_plan_won"] is False       # only graded plans compete


def test_the_report_renders_the_every_plan_section():
    rec = {"id": "T1", "slug": "p", "type": "pde", "tier": 1, "family": "f", "title": "t",
           "challenge": "", "has_ground_truth": True, "ground_truth_kind": "reference",
           "run": {"status": "completed", "wall_seconds": 1.0},
           "pipeline": {"best_score": 10, "best_plan": "3-fd4", "n_plans": 2},
           "verify": {"status": "ok", "passed": False, "rel_l2_err": 0.07},
           "harness_by_plan": {
               "winner": "3-fd4", "wrong_plan_won": True,
               "best_plan_by_harness": "1-spectral", "best_harness_err": 2e-3,
               "winner_harness_err": 0.07,
               "plans": {
                   "1-spectral": {"harness": {"status": "ok", "passed": True,
                                              "rel_l2_err": 2e-3},
                                  "kernel": {"score": 9, "provenance": "manufactured_partial"},
                                  "is_winner": False, "verdict": "UNDERCLAIM"},
                   "3-fd4": {"harness": {"status": "ok", "passed": False, "rel_l2_err": 0.07},
                             "kernel": {"score": 10, "provenance": "manufactured"},
                             "is_winner": True, "verdict": "OVERCLAIM"}}}}
    md = R.generate_report([rec], {})
    assert "## Every plan, graded" in md
    assert "`wrong_plan_won`: 1/1" in md
    assert "Wrong plan won:** 1/1" in md
    assert "3-fd4 *" in md and "1-spectral" in md
    assert "kernel provenance (per plan)" in md


def test_run_records_the_wall_clock_beside_monotonic(tmp_path, monkeypatch):
    """A laptop asleep mid-run advances the wall clock and not the monotonic one
    (findings §8); both are recorded so the gap is visible."""
    monkeypatch.setattr(RUN, "LOGS_DIR", str(tmp_path))
    monkeypatch.setattr(RUN, "conductor_command",
                        lambda *a, **k: [sys.executable, "-c", "print('hi')"])
    monkeypatch.setattr(RUN, "isolated_settings", lambda *a, **k: None)
    got = RUN.run_conductor("slug", 30, True, None, None)
    assert got["status"] == "completed"
    assert "wall_clock_seconds" in got and "started_at" in got and "ended_at" in got
    assert abs(got["wall_clock_seconds"] - got["wall_seconds"]) < 5.0
    assert np.isfinite(got["wall_clock_seconds"])
