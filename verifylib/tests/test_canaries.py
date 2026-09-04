"""The two canaries retained from Layer 7 (§9).

Seven of the nine originally proposed test the numerical kernel, which is not
being built here. These two test *this plan's* layers, and the rule they exist to
enforce is "build the test harness before adding the thing it tests" -- a guard
with no failing case behind it is a guard nobody knows is broken.
"""
import os

from verifylib import gate
from verifylib.audit import audit_run
from verifylib.findings import ERROR

from .conftest import read_text

CANARIES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "canaries")

SPEC = {
    "spatial_variables": ["x"],
    "domain": {"bounds": {"x": [0.0, 1.0]}},
    "parameters": {"alpha": 0.1, "t_final": 0.5},
    "requirements": [{"id": "R2", "kind": "domain", "quote": "x in [0, 1],   t in [0, 0.5]",
                      "status": "mapped", "spec_path": "domain.bounds.x, parameters.t_final"}],
}


def test_hardcoded_exact_is_caught_before_it_runs():
    """Both tells fire: the executable read of a benchmark/ path, and the prose
    citation. Either alone would be enough; the archived incidents took one each."""
    path = os.path.join(CANARIES, "hardcoded_exact.py")
    findings = gate.check_solver_text(read_text(path), path=path)
    assert [f.severity for f in findings] == [ERROR, ERROR], [f.render() for f in findings]
    messages = " ".join(f.message for f in findings)
    assert "benchmark/" in messages and "problems.py" in messages
    assert "open('benchmark/results/exact_heat_1d.json')" in messages


def test_shortened_horizon_is_a_hard_gate():
    """The numbers look fine; the horizon does not. Caught from the run's own
    output, which is what makes it a gate rather than a warning."""
    import numpy as np

    module = {}
    exec(compile(read_text(os.path.join(CANARIES, "shortened_horizon.py")),  # noqa: S102
                 "shortened_horizon.py", "exec"), module)
    result = module["solve_pde"](33)
    assert np.isfinite(result["numerical_solution"]).all(), "the output itself looks healthy"

    findings = audit_run(SPEC, result=result)
    assert [f.severity for f in findings] == [ERROR]
    assert "t_final" in findings[0].path
    assert "R2" in findings[0].message, "must name the ledger entry it breaks"


def test_an_honest_solver_passes_both():
    """The canaries are only meaningful if the guard is silent on correct work."""
    import numpy as np

    x = np.linspace(0.0, 1.0, 33)
    honest = {"numerical_solution": np.exp(-0.1 * np.pi ** 2 * 0.5) * np.sin(np.pi * x),
              "grid": {"x": x}, "t_final": 0.5}
    assert audit_run(SPEC, result=honest) == []
    assert gate.check_solver_text("import numpy as np\n\ndef solve_pde(N):\n    return {}\n") == []
