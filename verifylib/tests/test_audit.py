"""Criterion 10: a shortened horizon is a hard gate, a relaxed eps is a warning."""
import numpy as np
import pytest

from verifylib.audit import audit_parameters, audit_run, scan_parameter_bindings
from verifylib.findings import ERROR, WARNING
from verifylib.findings import SPEC as SPEC_SEVERITY  # `SPEC` below is a PDE fixture

SPEC = {
    "spatial_variables": ["x"],
    "domain": {"bounds": {"x": [0.0, 1.0]}},
    "boundary_conditions": {"type": "dirichlet", "values": {"x=0": 0.0, "x=1": 0.0}},
    "parameters": {"alpha": 0.1, "t_final": 0.5, "eps": 1e-3},
    "requirements": [
        {"id": "R1", "kind": "domain", "quote": "x in [0, 1], t in [0, 0.5]",
         "status": "mapped", "spec_path": "domain.bounds.x, parameters.t_final"},
        {"id": "R2", "kind": "parameter", "quote": "eps = 1e-3",
         "status": "mapped", "spec_path": "parameters.eps"},
        {"id": "R3", "kind": "boundary_condition", "quote": "u(0,t) = u(1,t) = 0",
         "status": "mapped", "spec_path": "boundary_conditions"},
    ],
}


def _run(t_final=0.5, lo=0.0, hi=1.0, edge=0.0):
    x = np.linspace(lo, hi, 33)
    u = np.sin(np.pi * np.linspace(0, 1, 33))
    u[0] = u[-1] = edge
    return {"numerical_solution": u, "grid": {"x": x}, "t_final": t_final}


def test_a_correct_run_is_silent():
    assert audit_run(SPEC, result=_run()) == []


# --- output-derived: hard gates ----------------------------------------------

def test_shortened_horizon_is_a_hard_gate():
    """The canary: T/2 reported as T. Every error metric gets easier."""
    findings = audit_run(SPEC, result=_run(t_final=0.25))
    assert [f.severity for f in findings] == [ERROR]
    assert "t_final" in findings[0].path
    assert "R1" in findings[0].message, "a finding must name the ledger entry it breaks"


@pytest.mark.parametrize("lo,hi", [(0.0, 0.5), (0.2, 1.0)])
def test_shrunken_domain_is_a_hard_gate(lo, hi):
    findings = audit_run(SPEC, result=_run(lo=lo, hi=hi))
    assert findings and all(f.severity == ERROR for f in findings)


def test_violated_dirichlet_edge_is_a_hard_gate():
    findings = audit_run(SPEC, result=_run(edge=0.4))
    assert any(f.severity == ERROR and "boundary_conditions" in f.path for f in findings)


def test_expression_valued_boundary_is_skipped_not_guessed():
    spec = {**SPEC, "boundary_conditions": {"type": "dirichlet",
                                            "values": {"x=0": "np.sin(t)"}}}
    assert audit_run(spec, result=_run()) == []


# --- SDE ---------------------------------------------------------------------

SDE_SPEC = {
    "state_dimension": 1, "time_interval": {"T0": 0.0, "T": 1.0},
    "evaluation_thresholds": {"num_paths": 50000, "seed": 42},
    "requirements": [{"id": "R5", "kind": "evaluation",
                      "quote": "solve_sde(num_paths=50000, dt=0.01, T=1, seed=42)",
                      "status": "mapped", "spec_path": "evaluation_thresholds.num_paths"}],
}


def test_sde_run_meta_reports_a_spec_defect_and_never_caps_the_solver():
    """~~`test_sde_run_meta_gates`~~ **corrected, sciml SS10j.**

    The old test asserted `[ERROR]` for each of the three fields, which is what made
    the defect durable: `run_meta` is the *harness's own call* -- `runner.run_solver`
    builds `spec["call"]` from `problems.py` and then constructs `run_meta` out of
    that same dict -- so none of these fields is a choice the solver made, and an
    `ERROR` here caps a score for someone else's number.

    A mismatch is still a real finding and still reported; it is attributed to the
    ledger field it came from rather than to the run. On the nine SciML entries the
    old severity took seven solvers the harness had graded 10/10 down to 3, for a
    `num_paths` the formulator had invented.
    """
    good = {"num_paths": 50000, "dt": 0.01, "T": 1.0, "seed": 42}
    assert audit_run(SDE_SPEC, run_meta=good) == []
    for key, bad, where in (("num_paths", 5000, "evaluation_thresholds.num_paths"),
                            ("seed", 7, "evaluation_thresholds.seed"),
                            ("T", 0.5, "time_interval.T")):
        findings = audit_run(SDE_SPEC, run_meta={**good, key: bad})
        assert [f.severity for f in findings] == [SPEC_SEVERITY], \
            f"{key} should be reported, not gated"
        # The finding points at where the defect is, not where it was noticed.
        assert findings[0].path == where, findings[0].path
        assert "has no say in it" in findings[0].message
        # Both numbers are named, so a reader does not have to go and find them.
        assert f"{key}={bad}" in findings[0].message, findings[0].message


def test_a_step_coarser_than_the_horizon_is_also_not_the_solvers_fault():
    """Both `dt` and `T` come from the problem definition, so a `dt > T` reading is a
    broken problem rather than a misbehaving solver (SS10j)."""
    findings = audit_run(SDE_SPEC, run_meta={"num_paths": 50000, "seed": 42,
                                             "T": 1.0, "dt": 2.0})
    assert [f.severity for f in findings] == [SPEC_SEVERITY]
    assert "not from the solver" in " ".join(findings[0].message.split())


def test_the_pde_output_checks_still_gate():
    """The re-attribution is scoped to the SDE run parameters. A PDE output finding
    reads the solver's own returned field -- its grid, horizon and Dirichlet edges --
    which is conduct the solver *is* answerable for, so those still cap (SS10j).
    """
    # The three PDE output findings, each still an ERROR: a shortened horizon, a
    # shrunken domain, and a violated Dirichlet edge.
    assert [f.severity for f in audit_run(SPEC, result=_run(t_final=0.25))] == [ERROR]
    assert all(f.severity == ERROR
               for f in audit_run(SPEC, result=_run(lo=0.0, hi=0.5)))
    assert any(f.severity == ERROR and "boundary_conditions" in f.path
               for f in audit_run(SPEC, result=_run(edge=0.4)))
    # And a correct run stays silent, so the widening added no new noise.
    assert audit_run(SPEC, result=_run()) == []


# --- source-derived: warnings only -------------------------------------------

def test_relaxed_parameter_warns_and_never_gates():
    """The flagship example, honestly scoped. eps is not recoverable from run
    output, so this reads solver.py -- and a source-derived finding never caps."""
    source = "eps = 1e-2\n\ndef solve_pde(N):\n    return {}\n"
    findings = audit_parameters(SPEC, source)
    assert [f.severity for f in findings] == [WARNING]
    assert "R2" in findings[0].message


def test_a_correctly_bound_parameter_is_silent():
    assert audit_parameters(SPEC, "eps = 1e-3\nalpha = 0.1\n") == []


def test_keyword_binding_is_seen():
    assert scan_parameter_bindings("solve(eps=0.5)", {"eps": 1e-3}) == {"eps": [0.5]}


def test_a_nondimensionalised_solver_is_not_flagged():
    """The parameter never appears literally -- the documented false-positive mode,
    and the reason this warns rather than gates."""
    source = "def solve_pde(N):\n    peclet = 1000.0\n    return {}\n"
    assert audit_parameters(SPEC, source) == []


def test_unparseable_solver_does_not_crash_the_audit():
    assert scan_parameter_bindings("def broken(:\n", {"eps": 1.0}) == {}
