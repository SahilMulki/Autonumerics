"""One solver, two execution paths, one verdict.

Until 2026-10-17 the pipeline was graded by importing ``solver.py`` inline and the
baseline by running it in a sandboxed subprocess. Comparing their *times* was
therefore comparing two harnesses (plan-benchop-slv.md §6, step 2), so both sides
now go through ``runner.run_solver``. That is only sound if the boundary is
faithful, and measured over the 34 solved workspaces it was not: five returned a
correct answer inline and ``inconclusive`` sandboxed, and the one functional-scored
problem failed every baseline for not returning a number it had returned.

These tests pin the three things that had to be true before the routing could be
flipped: the boundary carries everything the scorer reads, the two paths agree on
the verdict, and there is one budget rather than two defaults in two files.
"""
import argparse
import os
import sys

import numpy as np
import pytest

from .conftest import BENCHMARK, RUN, V, problem

sys.path.insert(0, BENCHMARK)
import oneshot as ONESHOT  # noqa: E402
import runner  # noqa: E402

# Every key `verify.py` and `verifylib.audit` read off a solver result. Enumerated
# from the source, not recalled: `grep -n 'result\.get(\|result\[' benchmark/verify.py
# verifylib/audit.py`. A key added to that list without being added here is a key
# the sandbox will silently drop.
PDE_CONTRACT_KEYS = {"grid", "fields", "numerical_solution", "t_final", "functionals"}
SDE_CONTRACT_KEYS = {"terminal_paths", "empirical_mean", "empirical_variance",
                     "functionals"}  # functionals: SDE side since step 5

# Returns every PDE contract key at once, including both result shapes -- which is
# what five of the archived winners do.
FULL_PDE = """
import numpy as np


def solve_pde(N):
    x = np.linspace(0.0, 1.0, N)
    u = np.exp(-0.1 * np.pi ** 2 * 0.5) * np.sin(np.pi * x)
    return {"numerical_solution": u, "fields": {"u": u}, "grid": {"x": x},
            "t_final": 0.5, "functionals": {"peak": float(u.max())}}
"""

# sin(pi x) sin(pi y) with the true ground-state eigenvalue: the functional is exact
# and the field is not, so the verdict is interesting rather than a flat pass, and
# any divergence between the paths shows up in two places at once.
EIGEN = """
import numpy as np


def solve_pde(N):
    x = np.linspace(0.0, 1.0, N)
    y = np.linspace(0.0, 1.0, N)
    X, Y = np.meshgrid(x, y, indexing="ij")
    return {"numerical_solution": np.sin(np.pi * X) * np.sin(np.pi * Y),
            "grid": {"x": x, "y": y}, "t_final": None,
            "functionals": {"lambda_1": -8.442746362964}}
"""

HEAT_EXACT = """
import numpy as np


def solve_pde(N):
    x = np.linspace(0.0, 1.0, N)
    u = np.exp(-0.1 * np.pi ** 2 * 0.5) * np.sin(np.pi * x)
    return {"numerical_solution": u, "grid": {"x": x}, "t_final": 0.5}
"""

HEAT_WRONG = """
import numpy as np


def solve_pde(N):
    x = np.linspace(0.0, 1.0, N)
    return {"numerical_solution": 0.5 * np.sin(np.pi * x), "grid": {"x": x},
            "t_final": 0.5}
"""

GBM = """
import numpy as np


def solve_sde(num_paths, dt, T, seed):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(int(num_paths))
    x = np.exp((0.1 - 0.5 * 0.2 ** 2) * T + 0.2 * np.sqrt(T) * z)
    return {"terminal_paths": x, "empirical_mean": float(x.mean()),
            "empirical_variance": float(x.var(ddof=1))}
"""

# The same solver with the key the SDE contract gained in step 5. `GBM` is kept
# without it because the parity cases below are the shapes that broke in step 2, and
# a functional-scored SDE problem is a different shape with its own tests.
GBM_PRICE = GBM.rstrip()[:-1] + """,
            "functionals": {"price": float(np.maximum(x - 1.0, 0.0).mean())}}
"""

FAKE_PDE = {"type": "pde"}
FAKE_SDE = {"type": "sde", "num_paths": 1000, "dt": 0.01, "T": 1.0, "seed": 42}


def plan(tmp_path, source, name="solver.py"):
    (tmp_path / name).write_text(source)
    return str(tmp_path)


# --- H1: the boundary carries everything the scorer reads -------------------------

def test_every_pde_contract_key_survives_the_boundary(tmp_path):
    run = runner.run_solver(plan(tmp_path, FULL_PDE), FAKE_PDE, pde_N=16)
    assert run["status"] == "ok", run.get("error")
    assert set(run["result"]) == PDE_CONTRACT_KEYS


def test_every_sde_contract_key_survives_the_boundary(tmp_path):
    run = runner.run_solver(plan(tmp_path, GBM_PRICE), FAKE_SDE)
    assert run["status"] == "ok", run.get("error")
    assert set(run["result"]) == SDE_CONTRACT_KEYS


def test_a_solver_returning_both_result_shapes_keeps_both(tmp_path):
    """The regression that cost five workspaces. ``fields`` is the systems contract
    and ``numerical_solution`` the scalar one; a solver may fill in each, and the
    boundary kept only the first, so a *scalar* problem then looked for a key that
    was no longer there and the verdict became ``inconclusive``."""
    run = runner.run_solver(plan(tmp_path, FULL_PDE), FAKE_PDE, pde_N=16)
    result = run["result"]
    assert "numerical_solution" in result and "fields" in result
    assert np.allclose(result["numerical_solution"], result["fields"]["u"])


def test_a_functional_survives_the_boundary(tmp_path):
    """The regression that cost every baseline on the one functional-scored problem:
    the boundary dropped ``functionals``, and the parent then failed the solver for
    not returning what it had in fact returned."""
    run = runner.run_solver(plan(tmp_path, FULL_PDE), FAKE_PDE, pde_N=16)
    want = float(np.exp(-0.1 * np.pi ** 2 * 0.5) * np.sin(np.pi * np.linspace(0, 1, 16)).max())
    assert run["result"]["functionals"]["peak"] == pytest.approx(want, rel=1e-12)


@pytest.mark.parametrize("value,expected", [
    ("float('nan')", float("nan")),      # non-finite is the parent's verdict to reach
    ("'not a number'", "'not a number'"),  # a type slip stays a type slip
    ("True", 1.0),                        # float() accepts it inline, so it does here
])
def test_a_functional_the_parent_must_judge_is_not_judged_in_the_child(tmp_path, value,
                                                                       expected):
    """``_functional_errs`` resolves a number, a non-number (``inconclusive``) and a
    missing name (``contract``) differently. The child must not collapse any of
    those into a crash by calling ``float()`` itself and letting it raise."""
    src = ("import numpy as np\n\n\ndef solve_pde(N):\n"
           "    x = np.linspace(0.0, 1.0, N)\n"
           "    return {'numerical_solution': x, 'grid': {'x': x}, 't_final': 0.5,\n"
           f"            'functionals': {{'q': {value}}}}}\n")
    run = runner.run_solver(plan(tmp_path, src), FAKE_PDE, pde_N=16)
    assert run["status"] == "ok", run.get("error")
    got = run["result"]["functionals"]["q"]
    if isinstance(expected, float) and np.isnan(expected):
        assert np.isnan(got)
    else:
        assert got == expected


def test_a_relative_plan_dir_resolves_the_same_in_the_child(tmp_path, monkeypatch):
    """The child is launched with ``cwd=benchmark/`` while the parent may be anywhere,
    so a relative ``plan_dir`` that imports fine inline resolved against the wrong
    directory in the child."""
    plan(tmp_path, HEAT_EXACT)
    monkeypatch.chdir(tmp_path)
    run = runner.run_solver(".", FAKE_PDE, pde_N=16)
    assert run["status"] == "ok", run.get("error")


# --- H2: the two paths reach the same verdict -------------------------------------

PARITY_CASES = [
    ("pde_heat_1d", HEAT_EXACT, "a correct solver"),
    ("pde_heat_1d", HEAT_WRONG, "a wrong solver"),
    ("pde_schrodinger_eigen_2d", EIGEN, "an exact functional on an inexact field"),
    ("sde_gbm", GBM, "an SDE"),
]


@pytest.mark.parametrize("slug,source,_label", PARITY_CASES,
                         ids=[c[2].replace(" ", "-") for c in PARITY_CASES])
def test_the_two_paths_return_the_same_verdict(tmp_path, slug, source, _label):
    """The claim step 2 rests on. Measured over all 34 solved workspaces as well;
    these four are the shapes that broke, kept here so a regression is caught
    without a workspace."""
    prob = problem(slug)
    p = plan(tmp_path, source)
    a = V.verify_problem(prob, p, plan_dir=p, sandbox=False)
    b = V.verify_problem(prob, p, plan_dir=p, sandbox=True)
    for key in ("status", "passed", "verified_score", "primary_field", "gt_kind",
                "order_ok", "converged", "constraint_violation"):
        assert a.get(key) == b.get(key), f"{key}: inline {a.get(key)} vs sandbox {b.get(key)}"
    for key in ("rel_l2_err", "worst_variance_rel_err", "worst_mean_rel_err"):
        if a.get(key) is not None:
            assert b.get(key) == pytest.approx(a[key], rel=1e-9, abs=1e-12), key
    assert a.get("functional_errs") == b.get("functional_errs")


def test_the_sandbox_is_the_only_path_that_times_itself(tmp_path):
    """Parity of verdict, not of instrumentation: only the sandboxed run carries a
    clock, which is what keeps a sandboxed time from being compared with an inline
    one (step 1)."""
    prob = problem("pde_heat_1d")
    p = plan(tmp_path, HEAT_EXACT)
    assert "timing" not in V.verify_problem(prob, p, plan_dir=p, sandbox=False)
    assert "timing" in V.verify_problem(prob, p, plan_dir=p, sandbox=True)


# --- H3: one budget, not two defaults ---------------------------------------------

def test_both_runners_default_to_the_one_shared_budget():
    """Two defaults in two files drift, and a pipeline and a baseline measured under
    different budgets are not comparable."""
    assert V.DEFAULT_SOLVER_TIMEOUT_S == 300
    assert V.DEFAULT_SOLVER_MEM_MB == 4096
    for module in (RUN, ONESHOT):
        with open(os.path.join(BENCHMARK, module.__name__ + ".py")) as fh:
            source = fh.read()
        assert "V.DEFAULT_SOLVER_TIMEOUT_S" in source, module.__name__
        assert "V.DEFAULT_SOLVER_MEM_MB" in source, module.__name__


def test_the_shared_budget_clears_the_slowest_measured_solve():
    """71.1 s on `pde_cahn_hilliard_2d` at N=128, measured 2026-10-17 over the 34
    solved workspaces. The budget has to leave room for a slower machine, because
    the failure mode is a correct solver recorded CRASHED."""
    assert V.DEFAULT_SOLVER_TIMEOUT_S >= 4 * 71.1


def test_run_py_defaults_to_grading_in_the_sandbox():
    """Exact, not a subset: this is the full set of conditions a solve is executed
    under, and a condition added here without being added to the recorded config is
    a sweep that cannot be told apart from one taken before it existed."""
    args = argparse.Namespace()
    kw = RUN._verify_kwargs(args)
    assert kw == {"sandbox": True, "timeout_s": V.DEFAULT_SOLVER_TIMEOUT_S,
                  "mem_mb": V.DEFAULT_SOLVER_MEM_MB, "work_precision": True}


def test_every_plan_is_graded_under_the_winners_conditions(tmp_path, monkeypatch):
    """Grading the winner sandboxed and its rivals inline would make
    ``wrong_plan_won`` a comparison between two harnesses rather than two plans."""
    ws = tmp_path / "ws"
    for name in ("1-a", "2-b"):
        (ws / "plans" / name).mkdir(parents=True)
        (ws / "plans" / name / "solver.py").write_text(HEAT_EXACT)

    seen = []

    def fake_verify(problem, workspace_dir, **kw):
        seen.append(kw)
        return {"status": "ok", "passed": True, "verified_score": 10, "rel_l2_err": 1e-9}

    monkeypatch.setattr(RUN.V, "verify_problem", fake_verify)
    want = {"sandbox": True, "timeout_s": 123, "mem_mb": 456}
    rec = {"pipeline": {"best_plan": None}, "has_ground_truth": True,
           "ground_truth_kind": "exact"}
    RUN.grade_every_plan(problem("pde_heat_1d"), str(ws), rec, None, None,
                         rerun_kernel=False, verify_kw=want)
    assert seen, "no plan was graded"
    for kw in seen:
        assert {k: kw[k] for k in want} == want
