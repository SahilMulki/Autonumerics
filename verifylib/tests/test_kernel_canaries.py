"""The seeded-defect suite (§9c): each defect, and the tier that must catch it.

The kernel is the single point of failure for every score in the pipeline, so a
test that only asserts it runs is worth very little. Each case below seeds one
specific defect into a correct solver and names the tier the plan says must see
it. The negative control matters as much as the rest: an honest solver must
trigger none of them, or the guards are over-firing and nobody would know.

These run real solves in the sandbox, so they are slower than the rest of the
suite -- around a second each.
"""
import json
import os
import shutil

import pytest

from verifylib.kernel import run

CANARY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "canaries", "kernel")


def _spec(name="spec.json", **edits):
    with open(os.path.join(CANARY, name)) as fh:
        spec = json.load(fh)
    for path, value in edits.items():
        node, *rest = path.split(".")
        target = spec[node]
        for key in rest[:-1]:
            target = target[key]
        target[rest[-1]] = value
    return spec


def _run(tmp_path, solver, spec=None, config=None):
    """Copy one canary in as ``solver.py`` and evaluate it."""
    shutil.copy(os.path.join(CANARY, "honest.py"), tmp_path / "honest.py")
    shutil.copy(os.path.join(CANARY, solver), tmp_path / "solver.py")
    return run(spec or _spec(), str(tmp_path), config or {})


# --- the negative control -----------------------------------------------------

def test_an_honest_solver_triggers_none_of_them(tmp_path):
    """And reaches 10 the only way a problem with no closed form can: two
    independent circularity breaks."""
    m = _run(tmp_path, "honest.py")
    assert m["score"] == 10, m["certification"]["reason"]
    assert m["provenance"] == "manufactured"
    assert m["summary"]["d1_outcome"] == "clean"
    assert m["summary"]["operator_validated"] is True
    assert m["checks"]["tier_b"] is True
    assert all(v is True for v in m["d2"]["outcomes"].values()), m["d2"]["outcomes"]
    assert m["d1"]["nontrivial"] is True


# --- one row of §9c each ------------------------------------------------------

def test_a_trivial_collapse_is_caught_by_the_nontrivial_guard(tmp_path):
    """``u == 0`` satisfies the operator exactly, so D1's slope test is perfectly
    happy -- having eliminated the only phenomenon the problem is about."""
    m = _run(tmp_path, "trivial_collapse.py")
    assert m["d1"]["nontrivial"] is False
    assert m["score"] == 3, m["certification"]["reason"]


def test_a_phase_lag_is_caught_by_the_invariants(tmp_path):
    """The residual is *identically zero* on a translate of a solution to a
    translation-invariant problem -- one of D1's structural blind spots."""
    m = _run(tmp_path, "phase_lag.py")
    assert m["d2"]["outcomes"]["symmetry"] is False
    assert m["score"] <= 5, m["certification"]["reason"]


def test_over_diffusion_is_caught_by_the_ladder(tmp_path):
    """Numerical viscosity proportional to dx: the residual stays small because
    every derivative on a coarse grid is small and mutually consistent."""
    m = _run(tmp_path, "over_diffusion.py")
    order = m["detail"]["richardson"]["observed_order"]
    assert order < 1.5, f"expected first-order convergence, measured {order}"
    assert m["checks"]["order_ok"] is False
    assert m["score"] <= 7


def test_a_wrong_boundary_condition_is_caught_by_d2(tmp_path):
    """A stencil needs ghost points, so the residual is interior-only and a BC
    violation lives exactly where it cannot be computed."""
    m = _run(tmp_path, "wrong_bc.py")
    assert m["d2"]["outcomes"]["boundary_consistency"] is False
    assert m["score"] == 3
    assert "boundary_consistency" in m["d2"]["gate_failed"]


def test_a_first_order_scheme_claiming_second_order_is_caught_by_tier_c(tmp_path):
    """The error falls cleanly, so nothing looks unstable. Only the measured order
    separates it from the honest control."""
    m = _run(tmp_path, "fake_second_order.py")
    order = m["detail"]["richardson"]["observed_order"]
    assert order == pytest.approx(1.0, abs=0.2), order
    assert m["checks"]["order_ok"] is False
    assert m["score"] <= 7


def test_a_sign_flipped_operator_term_is_caught_by_operator_validation(tmp_path):
    """Both the solver and the residual descend from one declaration, so a wrong
    operator makes them wrong *consistently*. §5d is the only thing that sees it,
    and without it D1 stops being a circularity break."""
    spec = _spec()
    spec["verification"]["operator"]["terms"]["reaction"] = "-k*u"
    m = _run(tmp_path, "honest.py", spec=spec)
    assert m["summary"]["operator_validated"] is False
    assert m["d1"]["validation_route"] in ("mms", "degenerate", "none")
    assert m["score"] <= 9, "a D1 pass against an unvalidated operator is not a break"
    assert m["provenance"] != "manufactured"


def test_mc_under_resolution_is_its_own_outcome_not_a_solver_bug(tmp_path):
    """§14's three-way outcome. Reporting it as a pass, or as a defect, are both
    wrong: the fix is more paths and the feedback must say exactly that."""
    shutil.copy(os.path.join(CANARY, "sde_honest.py"), tmp_path / "solver.py")
    spec = _spec("sde_spec.json", **{"evaluation_thresholds.num_paths": 60})
    m = run(spec, str(tmp_path), {})
    assert m["checks"]["resolved"] is False
    assert m["score"] == 6
    assert "not a solver bug" in m["certification"]["reason"]


def test_the_same_sde_solver_at_full_path_count_scores_ten(tmp_path):
    """The control for the case above: the defect was the path count, not the code."""
    shutil.copy(os.path.join(CANARY, "sde_honest.py"), tmp_path / "solver.py")
    m = run(_spec("sde_spec.json"), str(tmp_path), {})
    assert m["score"] == 10 and m["checks"]["resolved"] is True


# --- the agent's one authority ------------------------------------------------

def test_an_agent_cap_lowers_a_ten_and_is_recorded(tmp_path):
    m = _run(tmp_path, "honest.py", config={
        "agent_cap": {"score": 3, "reason": "SOLUTION.md describes a scheme solver.py "
                                            "does not implement"}})
    assert m["score"] == 3
    assert m["agent_cap"]["reason"].startswith("SOLUTION.md")
    assert m["certification"]["bound_by"] == "agent_cap"


def test_an_agent_cannot_raise_a_score(tmp_path):
    """Criterion 3. The rejection is recorded rather than silently ignored."""
    m = _run(tmp_path, "over_diffusion.py", config={
        "agent_cap": {"score": 10, "reason": "looks fine to me"}})
    assert m["score"] <= 7
    assert any(c["source"] == "agent_cap_rejected" for c in m["certification"]["caps"])
