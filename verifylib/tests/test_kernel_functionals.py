"""Declared scalar quantities of interest: transported, measured, and gated.

The gap these close was found by running the pipeline, not by reading the code.
``benchmark/verify.py`` grew a ``functionals`` return key so an eigenvalue problem
could be scored on its eigenvalue as well as its eigenfunction, and the kernel's
sandbox -- which documents itself as speaking *the whole* solver contract -- silently
dropped it. The formulator noticed and wrote it into its own spec as a limitation:
"The kernel sandbox does not transport `functionals`, so this gate cannot be evaluated
by cli.py." A declared gate nobody can evaluate is exactly what plan §6 calls worse
than no gate, because it reads as evidence.
"""
import pytest

from verifylib.kernel import functionals, sandbox
from verifylib.kernel.score import certify

SOLVER = """
import numpy as np

def solve_pde(N, override=None):
    x = np.linspace(0.0, 1.0, N)
    u = np.sin(np.pi * x)
    # A converging eigenvalue: exact value 2.0, error ~ 1/N^2.
    return {"numerical_solution": u, "grid": {"x": x}, "t_final": 0.0,
            "functionals": {"lambda_1": 2.0 - 1.0 / (N * N)}}
"""

MESSY = """
import numpy as np

def solve_pde(N, override=None):
    x = np.linspace(0.0, 1.0, N)
    return {"numerical_solution": np.sin(np.pi * x), "grid": {"x": x}, "t_final": 0.0,
            "functionals": {"good": 1.5, "an_array": np.ones(3), "a_string": "nope",
                            "not_finite": float("nan")}}
"""


def _write(tmp_path, text):
    (tmp_path / "solver.py").write_text(text)
    return str(tmp_path)


def _runs(*values, key="lambda_1"):
    """A ladder whose levels reported ``values``, coarse to fine."""
    return [{"N": 8 * 2 ** i, "result": {"functionals": {key: v}}}
            for i, v in enumerate(values)]


# --- transport ---------------------------------------------------------------


def test_functionals_cross_the_process_boundary(tmp_path):
    got = sandbox.run(_write(tmp_path, SOLVER), "pde", {"N": 9})
    assert got["status"] == "ok"
    assert got["result"]["functionals"] == pytest.approx({"lambda_1": 2.0 - 1 / 81})


def test_a_non_scalar_functional_is_dropped_rather_than_crashing_the_solve(tmp_path):
    """A malformed entry must reach the checks as ``not_reported``, which is a
    measurement, rather than as ``crashed``, which would blame the solver for the
    wrong thing."""
    got = sandbox.run(_write(tmp_path, MESSY), "pde", {"N": 9})
    assert got["status"] == "ok"
    assert got["result"]["functionals"] == {"good": 1.5}


def test_a_solver_reporting_no_functionals_still_round_trips(tmp_path):
    plain = "import numpy as np\n\ndef solve_pde(N, override=None):\n" \
            "    x = np.linspace(0.0, 1.0, N)\n" \
            "    return {'numerical_solution': x, 'grid': {'x': x}, 't_final': 0.0}\n"
    got = sandbox.run(_write(tmp_path, plain), "pde", {"N": 9})
    assert got["status"] == "ok"
    assert "functionals" not in got["result"]


# --- the error estimate ------------------------------------------------------


def test_three_levels_give_richardson_and_two_give_the_raw_difference():
    """Second-order convergence toward 2.0: 1.9, 1.975, 1.99375."""
    rel3, detail3 = functionals.estimate_error([1.9, 1.975, 1.99375])
    assert detail3["observed_order"] == pytest.approx(2.0, abs=1e-9)
    # e_est = |d21| / (2^2 - 1) = 0.01875/3
    assert rel3 == pytest.approx((0.01875 / 3.0) / 1.99375, rel=1e-9)

    rel2, detail2 = functionals.estimate_error([1.975, 1.99375])
    assert detail2["observed_order"] is None
    assert rel2 == pytest.approx(0.01875 / 1.99375, rel=1e-9)
    assert rel2 > rel3, "two levels must not report a smaller error than three"


def test_an_apparent_order_above_the_clamp_cannot_understate_the_error():
    """A too-high order *understates* the error, which is the direction that turns a
    fail into a pass -- so it is clamped (plan §7a's correction)."""
    rel, detail = functionals.estimate_error([1.0, 1.5, 1.5 + 1e-12])
    assert detail["observed_order"] <= functionals.MAX_INFERRED_ORDER


def test_successive_levels_at_round_off_are_converged():
    rel, detail = functionals.estimate_error([2.0, 2.0, 2.0])
    assert detail["at_roundoff"] is True
    assert rel == 0.0


def test_one_level_measures_nothing():
    rel, _ = functionals.estimate_error([2.0])
    assert rel is None


# --- the declared check ------------------------------------------------------


def test_a_converged_functional_is_clean():
    thresholds = {"functionals": {"lambda_1": {"rel_tol": 1e-2, "gate": True}}}
    got = functionals.run_declared(thresholds, _runs(1.9, 1.975, 1.99375))
    assert got["outcomes"]["lambda_1"] == "clean"
    assert got["functionals_ok"] is True
    assert got["gate_failed"] == []


def test_a_functional_short_of_its_tolerance_is_stalled_and_gates():
    thresholds = {"functionals": {"lambda_1": {"rel_tol": 1e-6, "gate": True}}}
    got = functionals.run_declared(thresholds, _runs(1.9, 1.975, 1.99375))
    assert got["outcomes"]["lambda_1"] == "stalled"
    assert got["gate_failed"] == ["lambda_1"]
    assert got["functionals_ok"] is False


def test_an_ungated_functional_is_measured_but_never_sinks_the_run():
    thresholds = {"functionals": {"lambda_1": {"rel_tol": 1e-9, "gate": False}}}
    got = functionals.run_declared(thresholds, _runs(1.9, 1.975, 1.99375))
    assert got["outcomes"]["lambda_1"] == "stalled"
    assert got["gate_failed"] == []
    assert got["functionals_ok"] is True


def test_a_declared_functional_the_solver_never_reported_is_not_a_pass():
    thresholds = {"functionals": {"lambda_1": {"rel_tol": 1e-3, "gate": True}}}
    runs = [{"N": 8, "result": {}}, {"N": 16, "result": {}}]
    got = functionals.run_declared(thresholds, runs)
    assert got["outcomes"]["lambda_1"] == "not_reported"
    assert got["not_reported_gated"] == ["lambda_1"]
    # Not a failure either: nothing was measured, so nothing is claimed.
    assert got["functionals_ok"] == "unavailable"


def test_declaring_nothing_is_a_no_op():
    got = functionals.run_declared({}, _runs(1.9, 1.975))
    assert got["functionals_ok"] == "unavailable"
    assert got["outcomes"] == {}


def test_a_malformed_declaration_is_ignored_rather_than_crashing():
    assert functionals.declared({"functionals": ["lambda_1"]}) == {}
    assert functionals.declared({"functionals": {"a": "not-a-dict"}}) == {}


# --- the score consequence ---------------------------------------------------


def _evidence(**over):
    base = {"kind": "pde", "path": "B", "crashed": False, "has_reference": False,
            "reference_outcome": None, "tier_b": True, "surrogate_ok": "unavailable",
            "d1_outcome": "clean", "operator_validated": True, "converged": True,
            "converged_measurable": True, "order_ok": True, "invariants_ok": True,
            "non_gate_invariants_ok": True, "constraints_ok": True, "temporal_ok": True,
            "asymptotic": True, "shrinking": True, "gate_violation": False,
            "not_reported_invariants": [], "interpolated": False, "any_test_ran": True,
            "e_fine": 1e-4, "rel_err_tol": 1e-2, "chaotic": False, "agent_cap": None}
    base.update(over)
    return base


def test_an_unevaluable_gated_functional_caps_the_score_at_nine():
    clean = certify(_evidence())
    capped = certify(_evidence(not_reported_functionals=["lambda_1"]))
    assert clean["score"] > capped["score"]
    assert capped["score"] == 9
    assert any(c["source"] == "not_reported_functional" for c in capped["caps"])


def test_the_cap_is_absent_when_nothing_is_declared():
    got = certify(_evidence(not_reported_functionals=[]))
    assert not any(c["source"] == "not_reported_functional" for c in got["caps"])
