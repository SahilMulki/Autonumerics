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


def test_a_declared_threshold_the_harness_does_not_run_at_reports_without_capping(tmp_path):
    """The SS10j fix, end to end through `verify.ledger_audit`.

    The formulator declares `num_paths=50000`; the harness runs the problem at 1000
    and hands the solver that number as a call argument. The disagreement is real and
    must be recorded -- under its own key, naming the ledger field -- and it must not
    move the solver's score, because the solver never chose either value.

    This is the exact shape that capped seven of nine SciML entries, every one of
    which the harness had independently graded 10/10.
    """
    import json

    (tmp_path / "problem_spec.json").write_text(json.dumps({
        "state_dimension": 1, "time_interval": {"T0": 0.0, "T": 1.0},
        "evaluation_thresholds": {"num_paths": 50000, "seed": 42},
        "requirements": [{"id": "R5", "kind": "evaluation",
                          "quote": "solve_sde(num_paths=1000, dt=0.0016, T=1, seed=42)",
                          "status": "mapped",
                          "spec_path": "evaluation_thresholds.num_paths"}],
    }))
    out = {"run": {"num_paths": 1000, "dt": 0.0016, "T": 1.0, "seed": 42}}
    folded = verify.ledger_audit(str(tmp_path), str(tmp_path), out)

    assert "score_cap" not in folded, folded
    assert "ledger_violations" not in folded, folded
    defects = folded["ledger_spec_defects"]
    assert len(defects) == 1, defects
    assert defects[0]["severity"] == "spec"
    assert defects[0]["path"] == "evaluation_thresholds.num_paths"
    assert "R5" in defects[0]["message"], "the finding still cites the ledger entry"

    # The same run with a ledger that declares what the harness runs is silent.
    (tmp_path / "problem_spec.json").write_text(json.dumps({
        "state_dimension": 1, "time_interval": {"T0": 0.0, "T": 1.0},
        "evaluation_thresholds": {"num_paths": 1000, "seed": 42},
    }))
    assert verify.ledger_audit(str(tmp_path), str(tmp_path), out) == {}


def test_the_spec_defect_is_recorded_per_plan():
    """A finding that caps nothing still has to reach the record, or it is invisible
    to every reader of a results file -- and a finding nobody can see is the state
    `score_cap` was already in when SS10j found it."""
    import run as RUN

    assert "ledger_spec_defects" in RUN.RECORDED_HARNESS, \
        "the per-plan record drops the finding, so nothing downstream can see it"
    # The cap it replaces is still recorded too: it remains live for PDE output
    # findings, which do read the solver's own returned field.
    assert "score_cap" in RUN.RECORDED_HARNESS


def test_ledger_audit_never_crashes_a_good_run(tmp_path):
    """An audit that raises must not fail a run it was only inspecting."""
    (tmp_path / "problem_spec.json").write_text("{not json")
    assert "ledger_audit_error" in verify.ledger_audit(str(tmp_path), str(tmp_path), {})
