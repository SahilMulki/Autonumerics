"""Shared fixtures for the benchmark harness tests.

``benchmark/`` is a directory of scripts, not a package: ``verify.py`` puts its own
directory on ``sys.path`` and imports ``problems`` bare, and every other module
does the same. The tests import them the same way, once, here.

The harness is the layer that grades the *pipeline*, so these tests never run
``claude`` and never touch ``workspace/``: they feed ``verify.py`` synthetic solver
output built from the harness's own ground truth (a reference field with noise
added, an exact eigenfunction with a wrong eigenvalue beside it) and check what
the verdict says. Everything is deterministic and, apart from the ``slow``-marked
reference rebuilds, runs in seconds.
"""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
BENCHMARK = os.path.dirname(HERE)
REPO = os.path.dirname(BENCHMARK)
for path in (BENCHMARK, REPO):
    if path not in sys.path:
        sys.path.insert(0, path)

import problems as P  # noqa: E402
import report as R  # noqa: E402
import run as RUN  # noqa: E402
import setup as S  # noqa: E402
import verify as V  # noqa: E402

__all__ = ["P", "R", "RUN", "S", "V", "REPO", "BENCHMARK"]


def problem(slug):
    for p in P.PROBLEMS:
        if p["slug"] == slug:
            return p
    raise KeyError(slug)


def grid_for(prob, N, *, endpoint=True):
    """1-D axes on the problem's domain, one per declared axis."""
    axes = prob.get("axes") or ["x", "y", "z"][:prob["dims"]]
    return {name: np.linspace(float(prob["domain"][name][0]), float(prob["domain"][name][1]),
                              int(N), endpoint=endpoint) for name in axes}


def truth_on(prob, grid):
    """The harness's own field truth evaluated on ``grid`` -- ``{field: array}``."""
    axes = prob.get("axes") or list(grid)
    coords = np.meshgrid(*[grid[a] for a in axes], indexing="ij")
    fn = V._pde_truth(prob)
    out = fn(prob.get("t_eval"), *coords)
    return out if isinstance(out, dict) else {"u": out}


def result_from(fields, grid, *, t_final=None, functionals=None):
    """A solver return dict of the shape ``verify.py`` scores."""
    out = {"grid": dict(grid), "t_final": t_final}
    if list(fields) == ["u"]:
        out["numerical_solution"] = np.asarray(fields["u"], dtype=float)
    else:
        out["fields"] = {k: np.asarray(v, dtype=float) for k, v in fields.items()}
    if functionals is not None:
        out["functionals"] = dict(functionals)
    return out


def record(prob, *, run_status="completed", verify=None, best_score=10, kernel=None):
    """A results.json record of the shape ``run.process`` writes."""
    rec = {"id": prob["id"], "slug": prob["slug"], "type": prob["type"], "tier": prob["tier"],
           "family": prob["family"], "title": prob["title"], "challenge": prob["challenge"],
           "has_ground_truth": prob["has_ground_truth"],
           "ground_truth_kind": prob.get("ground_truth_kind", "exact"),
           "run": {"status": run_status, "returncode": 0, "timed_out": False,
                   "wall_seconds": 1.0, "log": None},
           "pipeline": {"phase": "done", "best_plan": "1-plan", "best_score": best_score,
                        "n_plans": 1, "total_iters": 1, "plans": {}},
           "verify": verify if verify is not None else {"status": "ok", "passed": True,
                                                        "rel_l2_err": 1e-4}}
    if kernel is not None:
        rec["kernel"] = kernel
    return rec


@pytest.fixture(scope="session")
def ncf_problems():
    return P.no_closed_form_problems()
