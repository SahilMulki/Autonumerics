"""The solve clock: what a work-precision axis has to measure, and what it must not.

``run_solver`` has always reported ``wall_s`` -- the whole subprocess, including
interpreter startup, the numpy import and the solver module's own top-level code.
On a fast problem that is mostly overhead, so an error-vs-time curve plotted
against it would be a picture of the harness rather than of the scheme.
``solve_s`` times the ``solve_pde`` / ``solve_sde`` call alone
(plan-benchop-slv.md §6, step 1); keeping both is what makes their difference
readable as the overhead.

These tests do run a solver, in the sandbox, because a clock is the one thing that
cannot be checked on synthetic output.
"""
import os
import sys

import numpy as np
import pytest

from .conftest import BENCHMARK, V, problem

sys.path.insert(0, BENCHMARK)
import runner  # noqa: E402

SLEEP = 0.3   # long enough to separate from the clock's own noise, short enough to pay for

# Sleeps where the import happens, solves instantly: wall_s sees it, solve_s must not.
SLOW_IMPORT = f"""
import time

import numpy as np

time.sleep({SLEEP})


def solve_pde(N):
    x = np.linspace(0.0, 1.0, N)
    return {{"numerical_solution": np.sin(np.pi * x), "grid": {{"x": x}}, "t_final": 0.5}}
"""

# The mirror image: instant import, slow solve. Both clocks must see this one.
SLOW_SOLVE = f"""
import time

import numpy as np


def solve_pde(N):
    time.sleep({SLEEP})
    x = np.linspace(0.0, 1.0, N)
    return {{"numerical_solution": np.sin(np.pi * x), "grid": {{"x": x}}, "t_final": 0.5}}
"""

# u_t = 0.1 u_xx with u(x,0)=sin(pi x): the closed form, so the graded run passes
# and the clock is exercised on the ordinary verdict path rather than a crash.
HEAT_1D = """
import numpy as np


def solve_pde(N):
    x = np.linspace(0.0, 1.0, N)
    u = np.exp(-0.1 * np.pi ** 2 * 0.5) * np.sin(np.pi * x)
    return {"numerical_solution": u, "grid": {"x": x}, "t_final": 0.5}
"""

# Exact GBM simulation: no scheme bias, so the moments land inside the gate.
GBM = """
import numpy as np


def solve_sde(num_paths, dt, T, seed):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(int(num_paths))
    x = np.exp((0.1 - 0.5 * 0.2 ** 2) * T + 0.2 * np.sqrt(T) * z)
    return {"terminal_paths": x, "empirical_mean": float(x.mean()),
            "empirical_variance": float(x.var(ddof=1))}
"""

FAKE_PDE = {"type": "pde"}
FAKE_SDE = {"type": "sde", "num_paths": 1000, "dt": 0.01, "T": 1.0, "seed": 42}


def plan(tmp_path, source):
    (tmp_path / "solver.py").write_text(source)
    return str(tmp_path)


# --- G1: the child times the call, not the process -------------------------------

def test_the_clock_starts_after_the_import(tmp_path):
    """The whole reason for a second clock. A solver whose module-level code is slow
    and whose solve is instant must report a small ``solve_s`` and a large
    ``wall_s`` -- if ``solve_s`` saw the import, the two would be the same number
    and the overhead would be unmeasurable."""
    run = runner.run_solver(plan(tmp_path, SLOW_IMPORT), FAKE_PDE, pde_N=16)
    assert run["status"] == "ok", run.get("error")
    assert run["solve_s"] < SLEEP / 2
    assert run["wall_s"] >= SLEEP


def test_the_clock_does_see_the_solve(tmp_path):
    """The mirror image, so the first test cannot be passed by a clock that is
    simply always zero."""
    run = runner.run_solver(plan(tmp_path, SLOW_SOLVE), FAKE_PDE, pde_N=16)
    assert run["status"] == "ok", run.get("error")
    assert run["solve_s"] >= SLEEP
    assert run["wall_s"] >= run["solve_s"]


def test_the_overhead_is_the_difference_and_it_is_real(tmp_path):
    """``wall_s - solve_s`` is the number step 1 exists to expose: on a solve this
    cheap the subprocess and the numpy import dominate it, which is precisely why
    ``wall_s`` alone cannot carry an error-vs-time axis."""
    run = runner.run_solver(plan(tmp_path, HEAT_1D), FAKE_PDE, pde_N=16)
    assert run["status"] == "ok", run.get("error")
    assert 0.0 <= run["solve_s"] <= run["wall_s"]
    assert run["wall_s"] - run["solve_s"] > run["solve_s"]


@pytest.mark.parametrize("fake,source,pde_N", [
    (FAKE_PDE, HEAT_1D, 16),
    (FAKE_SDE, GBM, None),
])
def test_both_contracts_carry_the_clock_in_run_meta(tmp_path, fake, source, pde_N):
    """``run_meta`` is what crosses the process boundary and what the verifier
    stores, so the number has to be there and not only on the parent's return."""
    run = runner.run_solver(plan(tmp_path, source), fake, pde_N=pde_N)
    assert run["status"] == "ok", run.get("error")
    assert run["run_meta"]["solve_s"] == run["solve_s"]
    assert isinstance(run["run_meta"]["solve_s"], float)


# --- G2: the clock reaches the verdict record ------------------------------------

def test_a_graded_pde_records_one_clock_per_grid(tmp_path):
    """The order check already solves at N and 2N. Each gets its own clock, and the
    headline pair is the graded (fine) run -- the same arrangement ``rel_l2_err``
    has with ``grid_errors``."""
    prob = problem("pde_heat_1d")
    p = plan(tmp_path, HEAT_1D)
    out = V.verify_problem(prob, p, plan_dir=p, sandbox=True)
    assert out["status"] == "ok" and out["passed"] is True, out

    timing = out["timing"]
    assert timing["path"] == "sandbox"
    assert [r["N"] for r in timing["runs"]] == [out["grid_N"], out["fine_grid_N"]]
    assert timing["solve_s"] == timing["runs"][-1]["solve_s"]
    assert timing["wall_s"] == timing["runs"][-1]["wall_s"]
    assert all(r["solve_s"] <= r["wall_s"] for r in timing["runs"])


def test_a_graded_sde_records_the_clock_beside_the_run_parameters(tmp_path):
    """The SDE route has no second grid, so the clock rides in ``run`` next to the
    (num_paths, dt, T, seed) the run was made at -- the block a reader already
    consults to know what was measured."""
    prob = problem("sde_gbm")
    p = plan(tmp_path, GBM)
    out = V.verify_problem(prob, p, plan_dir=p, sandbox=True)
    assert out["status"] == "ok" and out["passed"] is True, out
    assert out["run"]["solve_s"] > 0.0
    assert set(out["run"]) >= {"num_paths", "dt", "T", "seed", "solve_s"}


def test_the_record_stays_json_serializable(tmp_path):
    """``run.py`` writes the verify dict straight into results.json."""
    import json

    prob = problem("pde_heat_1d")
    p = plan(tmp_path, HEAT_1D)
    out = V.verify_problem(prob, p, plan_dir=p, sandbox=True)
    assert json.loads(json.dumps(out["timing"])) == out["timing"]


# --- G3: nothing is reported that was not measured --------------------------------

def test_the_inline_path_records_no_timing(tmp_path):
    """Sandboxed and inline are different measurement conditions -- a fresh
    interpreter per solve against a warm in-process one -- so an inline number
    under the same key would invite a comparison that measures the harness.
    Step 2 of the plan resolves this by routing both through the sandbox; until
    then the honest record is no record."""
    prob = problem("pde_heat_1d")
    p = plan(tmp_path, HEAT_1D)
    out = V.verify_problem(prob, p, plan_dir=p, sandbox=False)
    assert out["status"] == "ok" and out["passed"] is True, out
    assert "timing" not in out


def test_a_crash_carries_no_solve_time(tmp_path):
    """There is no completed solve to time, and a 0.0 would read as an instant one."""
    run = runner.run_solver(plan(tmp_path, "def solve_pde(N):\n    raise RuntimeError('x')\n"),
                            FAKE_PDE, pde_N=16)
    assert run["status"] == "crashed" and run["reason"] == "exception"
    assert "solve_s" not in run
    assert run["wall_s"] > 0.0


def test_a_timed_out_solver_still_reports_its_wall_clock(tmp_path):
    """The timeout path never hears from the child, so only the process clock
    survives -- and it must, because it is what the budget is spent against."""
    src = "import time\n\n\ndef solve_pde(N):\n    time.sleep(30)\n"
    run = runner.run_solver(plan(tmp_path, src), FAKE_PDE, timeout_s=1, pde_N=16)
    assert run["status"] == "crashed" and run["reason"] == "timeout"
    assert "solve_s" not in run
    assert run["wall_s"] >= 1.0


def test_the_summary_prints_both_clocks(tmp_path):
    """``verify.py --slug ... --sandbox`` is read by a person; a number stored and
    never shown is a number nobody checks."""
    prob = problem("pde_heat_1d")
    p = plan(tmp_path, HEAT_1D)
    out = V.verify_problem(prob, p, plan_dir=p, sandbox=True)
    text = V._fmt(out, prob)
    assert "solve time" in text
    assert "overhead" in text


def test_the_npz_scratch_is_still_cleaned_up(tmp_path):
    """Unrelated to the clock, and checked here because this is the only place that
    runs the sandbox end to end: a timing change must not leak scratch files."""
    scratch = tmp_path / "scratch"
    run = runner.run_solver(plan(tmp_path, HEAT_1D), FAKE_PDE, pde_N=16,
                            scratch_dir=str(scratch))
    assert run["status"] == "ok"
    assert os.listdir(scratch) == []
    assert np.isfinite(run["result"]["numerical_solution"]).all()
