"""The two layers on one solver.

The kernel scores a plan against its own spec; the harness grades the same
``solver.py`` against harness-owned truth. Running one solver through both is
what calibrates the verdict table: a solver both layers agree on is the
VERIFIED_PASS reference point, and a seeded defect the kernel sees must come out
as FAIL or UNDERCLAIM -- never OVERCLAIM, which is reserved for the kernel being
*wrong*.
"""
import json
import os
import shutil
import sys

import numpy as np
import pytest

from verifylib.kernel import run

from .conftest import REPO, R, V, record

CANARY = os.path.join(REPO, "verifylib", "tests", "canaries", "kernel")


def _canary_problem():
    """The honest canary's PDE, ``u_t = alpha u_xx - k u`` on [0, 1] with
    ``u(x, 0) = sin(pi x)``, as a benchmark problem with its closed form."""
    alpha, k = 0.05, 0.3
    return {"id": "T00", "slug": "canary_reaction_diffusion", "type": "pde", "tier": 1,
            "family": "canary", "title": "canary", "challenge": "",
            "dims": 1, "axes": ["x"], "domain": {"x": [0.0, 1.0]}, "t_eval": 0.5,
            "has_ground_truth": True, "grid_N": 33, "min_order": 1.8,
            "analytic": lambda t, x: np.exp(-(alpha * np.pi ** 2 + k) * t) * np.sin(np.pi * x)}


def _plan(tmp_path, solver):
    shutil.copy(os.path.join(CANARY, "honest.py"), tmp_path / "honest.py")
    shutil.copy(os.path.join(CANARY, solver), tmp_path / "solver.py")
    (tmp_path / "SOLUTION.md").write_text("---\nscheme: finite-difference-ftcs\n---\n")
    if str(tmp_path) not in sys.path:
        sys.path.insert(0, str(tmp_path))          # `from honest import ...` inline
    return str(tmp_path)


def _spec():
    with open(os.path.join(CANARY, "spec.json")) as fh:
        return json.load(fh)


def _both(tmp_path, solver):
    plan = _plan(tmp_path, solver)
    kernel = run(_spec(), plan, {})
    verify = V.verify_pde(_canary_problem(), plan)
    verify.pop("_run_result", None)
    rec = record(_canary_problem(), best_score=kernel["score"], verify=verify,
                 kernel={"status": "ok", "score": kernel["score"],
                         "provenance": kernel["provenance"],
                         "tier": kernel["certification"]["tier"]})
    return kernel, verify, R.compute_verdict(rec)


def test_the_honest_canary_passes_both_layers(tmp_path):
    kernel, verify, verdict = _both(tmp_path, "honest.py")
    assert (kernel["score"], kernel["provenance"]) == (10, "manufactured")
    assert verify["status"] == "ok" and verify["passed"] is True, verify
    assert verify["order_ok"] is True and verify["observed_order"] > 1.8
    assert verdict == "VERIFIED_PASS"


@pytest.mark.parametrize("solver", ["over_diffusion.py", "fake_second_order.py",
                                    "wrong_bc.py", "trivial_collapse.py"])
def test_a_seeded_defect_the_kernel_sees_is_never_an_overclaim(tmp_path, solver):
    """The kernel scores each below 10, so whatever the harness says the verdict
    cannot be OVERCLAIM -- that word is for the kernel certifying something the
    harness rejects."""
    kernel, verify, verdict = _both(tmp_path, solver)
    assert kernel["score"] < 10, (solver, kernel["certification"]["reason"])
    assert verdict in ("FAIL", "UNDERCLAIM", "CONSTRAINT_VIOLATION"), (solver, verdict, verify)


def test_the_gross_defects_fail_the_harness_too(tmp_path):
    """Two of the four are wrong enough that the independent check fails them
    outright; the kernel and the harness then agree on FAIL."""
    for solver in ("wrong_bc.py", "trivial_collapse.py"):
        kernel, verify, verdict = _both(tmp_path, solver)
        assert verify["passed"] is False, (solver, verify)
        assert verdict == "FAIL", (solver, verdict)


def test_an_overclaim_is_what_the_table_calls_a_kernel_ten_the_harness_rejects(tmp_path):
    """Synthetic: the honest solver's harness verdict with the kernel score forced
    to 10 on a failing check. The verdict logic itself, on real verify output."""
    plan = _plan(tmp_path, "trivial_collapse.py")
    verify = V.verify_pde(_canary_problem(), plan)
    verify.pop("_run_result", None)
    assert verify["passed"] is False
    rec = record(_canary_problem(), best_score=10, verify=verify)
    assert R.compute_verdict(rec) == "OVERCLAIM"
