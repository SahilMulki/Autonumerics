"""Criterion 9: the pipeline execution path is leakage-scanned.

This is the wiring bug the plan was written to avoid. ``runner.run_solver`` is only
reached when ``sandbox=True``, which is the one-shot baseline; the pipeline path is
``sandbox=False`` and imports the solver inline through ``verify.import_solver``.
A scan wired only into the runner -- as originally specified -- would never have
seen a pipeline-produced solver. Both paths are tested here for that reason.
"""
import os
import sys

import pytest

from .conftest import REPO, read_text

sys.path.insert(0, os.path.join(REPO, "benchmark"))
sys.path.insert(0, REPO)

verify = pytest.importorskip("verify", reason="benchmark/ not importable")
runner = pytest.importorskip("runner", reason="benchmark/ not importable")

CANARY = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "canaries", "hardcoded_exact.py")
CLEAN = "import numpy as np\n\n\ndef solve_pde(N, override=None):\n    return {'grid': {}}\n"


def _plan(tmp_path, source):
    (tmp_path / "solver.py").write_text(source)
    return str(tmp_path)


def test_pipeline_path_refuses_a_leaking_solver(tmp_path):
    """sandbox=False -- the path every pipeline run takes."""
    plan_dir = _plan(tmp_path, read_text(CANARY))
    with pytest.raises(verify.OracleLeak, match="benchmark"):
        verify.import_solver(plan_dir)


def test_pipeline_path_imports_a_clean_solver(tmp_path):
    module = verify.import_solver(_plan(tmp_path, CLEAN))
    assert hasattr(module, "solve_pde")


def test_sandbox_path_refuses_a_leaking_solver(tmp_path):
    """sandbox=True -- the one-shot baseline path."""
    plan_dir = _plan(tmp_path, read_text(CANARY))
    with pytest.raises(runner.BlockedImport, match="leakage"):
        runner._import_solver(os.path.join(plan_dir, "solver.py"))


def test_sandbox_path_imports_a_clean_solver(tmp_path):
    plan_dir = _plan(tmp_path, CLEAN)
    assert runner._import_solver(os.path.join(plan_dir, "solver.py"))


def test_certification_runs_the_ledger_audit():
    """verify.ledger_audit is what turns a re-audit finding into a score cap."""
    assert callable(verify.ledger_audit)


def test_ledger_audit_caps_on_a_shortened_horizon(tmp_path):
    import json

    import numpy as np

    (tmp_path / "problem_spec.json").write_text(json.dumps({
        "spatial_variables": ["x"], "domain": {"bounds": {"x": [0.0, 1.0]}},
        "parameters": {"t_final": 0.5},
        "requirements": [{"id": "R2", "kind": "domain", "quote": "t in [0, 0.5]",
                          "status": "mapped", "spec_path": "parameters.t_final"}],
    }))
    out = {"_run_result": {"numerical_solution": np.zeros(9),
                           "grid": {"x": np.linspace(0, 1, 9)}, "t_final": 0.25}}
    folded = verify.ledger_audit(str(tmp_path), str(tmp_path), out)
    assert folded["score_cap"] == 3
    assert folded["ledger_violations"][0]["path"] == "t_final"


def test_ledger_audit_never_crashes_a_good_run(tmp_path):
    """An audit that raises must not fail a run it was only inspecting."""
    (tmp_path / "problem_spec.json").write_text("{not json")
    assert "ledger_audit_error" in verify.ledger_audit(str(tmp_path), str(tmp_path), {})
