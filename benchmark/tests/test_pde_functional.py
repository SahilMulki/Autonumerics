"""The PDE route's anti-hardcoding defence (plan-benchop-slv.md step 7).

plan-benchop-slv.md section 5 rests the whole memorization argument for the PDE
route on one sentence: *"When functionals are the only truth, ``_pde_eval_run``
makes the functional error the primary error, so the 2-grid order check measures
convergence of the price itself. A hardcoded constant has zero convergence and
fails."* Measured 2026-10-20, every clause of that was true except the one that
mattered: ``_pde_order_check`` asked for a **field** ground truth, so on a
functional-only problem the second grid was never run and a hardcoded constant
scored 10.

  L1  the check runs at all on a problem whose only truth is a number
  L2  a constant fails, and fails for the right reason -- the superconvergence
      waiver hands it a perfect observed order, so movement has to be its own test
  L3  an honest solver is not caught by it
  L4  nothing changes for a problem that has a field

Sections L2 and L3 run a real ``solve_pde`` in a real subprocess, because what is
under test is the two-grid ladder and a stub would assume it away.
"""
import numpy as np
import pytest

from .conftest import P, V

REF = 0.908502728459621


def price_problem(**over):
    """A problem whose entire ground truth is one number -- the BENCHOP shape."""
    prob = {"slug": "test_pde_price", "type": "pde", "tier": 3, "dims": 1,
            "axes": ["x"], "domain": {"x": [0.0, 1.0]}, "t_eval": 1.0,
            "grid_N": 32, "min_order": 1.0, "has_ground_truth": True,
            "closed_form": False, "analytic": None,
            "functional_truth": {"price": REF}, "functional_tol": 1e-2}
    prob.update(over)
    return prob


SOLVER = '''
import numpy as np

def solve_pde(N):
    x = np.linspace(0.0, 1.0, N)
    return {{"grid": {{"x": x}}, "numerical_solution": np.zeros(N),
            "t_final": 1.0, "functionals": {{"price": {price}}}}}
'''
HARDCODED = SOLVER.format(price=repr(REF))
CONVERGENT = SOLVER.format(price=f"{REF!r} + 0.02 / N")
BIASED = SOLVER.format(price=f"{REF!r} * 1.5 + 0.02 / N")


def plan(tmp_path, source):
    (tmp_path / "solver.py").write_text(source)
    return str(tmp_path)


def verdict(tmp_path, source, **over):
    prob = price_problem(**over)
    p = plan(tmp_path, source)
    return V.verify_problem(prob, p, plan_dir=p, sandbox=True, work_precision=False)


# --- L1: the check runs on a problem whose only truth is a number ---------------

def test_a_functional_only_problem_is_order_checked():
    assert V._pde_order_check(price_problem()) is True
    assert V._pde_functional_only(price_problem()) is True


def test_a_problem_can_still_turn_the_check_off():
    assert V._pde_order_check(price_problem(pde_order_check=False)) is False


def test_a_problem_with_no_truth_at_all_is_not_order_checked():
    """Nothing to measure an error against is still nothing to measure."""
    prob = price_problem()
    prob.pop("functional_truth")
    assert V._pde_order_check(prob) is False
    assert V._pde_functional_only(prob) is False


# --- L2: a constant fails, and movement is its own test -------------------------

def test_a_hardcoded_price_fails_a_functional_only_problem(tmp_path):
    """The defect this file exists for. The solver returns the published value to
    fifteen digits, so it is *more* accurate than any real solver -- and it cannot
    have computed it, because halving the grid did not change it."""
    out = verdict(tmp_path, HARDCODED)
    assert out["status"] == "ok", out.get("error")
    assert out["order_checked"] is True
    assert out["rel_l2_err"] == pytest.approx(0.0, abs=1e-12)
    assert out["functional_constant"] == ["price"]
    assert out["passed"] is False and out["verified_score"] == 1


def test_the_order_floor_alone_would_not_have_caught_it(tmp_path):
    """Why movement is checked separately. A constant has zero error at both grids,
    which trips the superconvergence waiver -- the waiver exists so a spectral
    scheme sitting on its round-off floor is not failed for having no observable
    order, and it hands a hardcoded constant exactly the same free pass."""
    out = verdict(tmp_path, HARDCODED)
    assert out["observed_order"] == float("inf")
    assert out["order_ok"] is True        # the order check is happy; the record is not


def test_the_two_grid_errors_are_recorded_for_a_constant(tmp_path):
    out = verdict(tmp_path, HARDCODED)
    assert out["grid_errors"] == {"32": 0.0, "64": 0.0}


# --- L3: an honest solver is not caught -----------------------------------------

def test_a_convergent_price_passes_and_shows_its_order(tmp_path):
    out = verdict(tmp_path, CONVERGENT)
    assert out["status"] == "ok", out.get("error")
    assert out["functional_constant"] == []
    assert out["observed_order"] == pytest.approx(1.0, abs=1e-6)
    assert out["passed"] is True and out["verified_score"] == 10


def test_a_convergent_but_wrong_price_still_fails_on_accuracy(tmp_path):
    """Movement is necessary, not sufficient. This one moves and converges, to the
    wrong number."""
    out = verdict(tmp_path, BIASED)
    assert out["functional_constant"] == []
    assert out["passed"] is False
    assert out["rel_l2_err"] == pytest.approx(0.5, rel=1e-3)


def test_the_check_is_recorded_even_when_it_finds_nothing(tmp_path):
    """A passing record silent about the defence would read as one that had it."""
    assert "functional_constant" in verdict(tmp_path, CONVERGENT)


# --- L4: a problem with a field is unaffected -----------------------------------

def test_a_field_problem_does_not_get_the_movement_requirement():
    """Nobody memorizes a surface, so the field is its own defence -- and a field
    solver that happens to report an identical functional at two grids must not be
    failed for it."""
    prob = price_problem(analytic=lambda t, x: np.zeros_like(x), fields=["u"])
    assert V._pde_functional_only(prob) is False
    assert V._pde_functional_constant(prob, [{"functional_values": {"price": 1.0}},
                                             {"functional_values": {"price": 1.0}}]) == []


def test_the_constant_rule_needs_both_grids():
    assert V._pde_functional_constant(price_problem(), [{"functional_values": {}}]) == []


def test_every_shipped_functional_problem_keeps_the_verdict_it_had():
    """Zero churn on the suite as it stands: the one problem in it that carries a
    functional also carries a field, so none of this reaches it."""
    carrying = [p for p in P.PROBLEMS if p.get("functional_truth")]
    assert carrying, "expected at least one functional problem in the suite"
    for prob in carrying:
        if prob["type"] != "pde":
            continue
        assert V._pde_functional_only(prob) is False, prob["slug"]
