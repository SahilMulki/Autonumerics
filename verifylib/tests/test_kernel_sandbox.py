"""Criterion 11: a hanging or memory-hungry solver yields ``crashed``, not a hung
session.

The kernel imports and executes solver code up a three-level ladder -- in 3-D the
top level is 32x the base -- so every failure mode has to come back as data. The
measured precedent is on the guardrails side: an unbounded Stop hook fired 8 times
and the run died at max-turns. A hung top-level solve burns the same budget with
less to show for it.
"""
import numpy as np
import pytest

from verifylib.kernel import sandbox

SOLVER = """
import numpy as np

def solve_pde(N, override=None):
    ov = override or {}
    x = np.linspace(0.0, 1.0, N)
    u = ov["ic"]((x,)) if "ic" in ov else np.sin(np.pi * x)
    u = np.asarray(u, dtype=float) * float(ov.get("params", {}).get("scale", 1.0))
    return {"numerical_solution": u, "grid": {"x": x}, "t_final": 1.0, "dt": 0.01,
            "invariant_trace": {"energy": np.ones(4)},
            "snapshots": [{"t": 0.98, "fields": {"u": u * 0.9}},
                          {"t": 0.99, "fields": {"u": u * 0.95}},
                          {"t": 1.00, "fields": {"u": u}}]}
"""


def _write(tmp_path, text):
    (tmp_path / "solver.py").write_text(text)
    return str(tmp_path)


def test_the_whole_return_contract_survives_the_process_boundary(tmp_path):
    """`benchmark/runner.py` extracts the field, the grid and t_final and drops
    everything else -- including the three keys the kernel exists to consume."""
    got = sandbox.run(_write(tmp_path, SOLVER), "pde", {"N": 9})
    assert got["status"] == "ok"
    result = got["result"]
    assert set(result) >= {"grid", "fields", "t_final", "dt", "invariant_trace",
                           "snapshots"}
    assert list(result["invariant_trace"]) == ["energy"]
    assert [s["t"] for s in result["snapshots"]] == [0.98, 0.99, 1.0]
    assert result["snapshots"][-1]["fields"]["u"].shape == (9,)


def test_an_override_reaches_the_solver_as_a_rebuilt_callable(tmp_path):
    """Callables cannot cross a process boundary, so ``ic`` travels as the spec
    expression it is built from and is rebuilt in the child."""
    plan = _write(tmp_path, SOLVER)
    payload = {"ic": {"kind": "expr", "exprs": {"u": "np.cos(np.pi*x)"},
                      "params": {}, "spatial_variables": ["x"], "multi": False,
                      "fixed_t": 0.0}}
    got = sandbox.run(plan, "pde", {"N": 9}, override=payload)
    assert got["status"] == "ok"
    assert np.allclose(got["result"]["fields"]["u"], np.cos(np.pi * np.linspace(0, 1, 9)))
    scaled = sandbox.run(plan, "pde", {"N": 9}, override={"params": {"scale": 2.0}})
    assert np.allclose(scaled["result"]["fields"]["u"],
                       2.0 * np.sin(np.pi * np.linspace(0, 1, 9)))


def test_an_array_override_travels_through_the_npz(tmp_path):
    """The route ``translation_invariance`` takes: a shifted initial condition is
    genuinely an array, not an expression."""
    want = np.arange(9, dtype=float)
    got = sandbox.run(_write(tmp_path, SOLVER), "pde", {"N": 9},
                      override={"ic": {"kind": "array", "key": "ov_ic"}},
                      in_arrays={"ov_ic": want})
    assert got["status"] == "ok"
    assert np.allclose(got["result"]["fields"]["u"], want)


@pytest.mark.parametrize("body,reason", [
    ("", "no_solve_fn"),
    ("def solve_pde(N, override=None):\n    raise RuntimeError('boom')\n", "exception"),
    ("def solve_pde(N, override=None):\n    return 42\n", "bad_schema"),
    ("def solve_pde(N, override=None):\n    return {'numerical_solution': [1.0]}\n",
     "bad_schema"),
    ("import pandas\ndef solve_pde(N, override=None):\n    return {}\n", "blocked_import"),
])
def test_every_failure_mode_comes_back_as_a_reason(tmp_path, body, reason):
    got = sandbox.run(_write(tmp_path, body), "pde", {"N": 9})
    assert got["status"] == "crashed"
    assert got["reason"] == reason, got.get("error")
    assert got["reason"] in sandbox.CRASH_REASONS


def test_a_hanging_solver_is_killed_rather_than_waited_on(tmp_path):
    plan = _write(tmp_path, "import time\n"
                            "def solve_pde(N, override=None):\n"
                            "    time.sleep(60)\n")
    got = sandbox.run(plan, "pde", {"N": 9}, timeout_s=2)
    assert got["status"] == "crashed" and got["reason"] == "timeout"
    assert got["wall_s"] < 20, "the timeout must bound the wait, not merely report it"


def test_a_missing_solver_is_reported_not_raised(tmp_path):
    got = sandbox.run(str(tmp_path), "pde", {"N": 9})
    assert got["status"] == "crashed" and got["reason"] == "no_solve_fn"


def test_a_leaking_solver_never_runs(tmp_path):
    """The leakage scan happens *before* import, so an answer-key read cannot
    execute even once."""
    plan = _write(tmp_path, "import json\n"
                            "def solve_pde(N, override=None):\n"
                            "    return json.load(open('benchmark/results/exact.json'))\n")
    got = sandbox.run(plan, "pde", {"N": 9})
    assert got["status"] == "crashed" and got["reason"] == "leakage"


def test_the_import_policy_matches_what_the_solver_agents_are_granted():
    """Library parity, so the kernel measures the same code the pipeline runs:
    solver-pde.md grants scipy.sparse, solver-sde.md grants numpy only."""
    assert "scipy" in sandbox.deny_imports("sde")
    assert "scipy" not in sandbox.deny_imports("pde")
    assert "sympy" in sandbox.deny_imports("pde")


def test_the_crn_increments_are_rebuilt_identically_at_every_level():
    """The finest level is 160 MB at 50 000 paths; it is rebuilt from a seed rather
    than shipped through a file, so every level must be an exact coarsening."""
    fine = sandbox._crn_increments({"seed": 3, "num_paths": 5, "Nt_fine": 8, "T": 1.0,
                                    "aggregate": 1, "m": None})
    coarse = sandbox._crn_increments({"seed": 3, "num_paths": 5, "Nt_fine": 8, "T": 1.0,
                                      "aggregate": 2, "m": None})
    assert fine.shape == (5, 8) and coarse.shape == (5, 4)
    assert np.allclose(coarse, fine.reshape(5, 4, 2).sum(axis=2))
    assert np.allclose(fine.sum(axis=1), coarse.sum(axis=1)), "same Brownian path"
