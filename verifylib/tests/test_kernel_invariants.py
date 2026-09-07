"""D2: the closed registry, its evaluators, and the two conventions that bite.

The registry has to be closed or a gated invariant nobody implements reads as
evidence that a check passed. The evaluators have to get two conventions right --
a duplicated wrap node, and ``tol: 0.0`` -- and both were found by running against
real workspace artifacts rather than by inspection.
"""
import glob
import json
import os

import numpy as np
import pytest

from verifylib.kernel.invariants import (
    UNIMPLEMENTED,
    PdeContext,
    SdeContext,
    duplicated_endpoint,
    evaluate_invariant,
    gateable,
    integral,
    run_declared,
)

from .conftest import WORKSPACE


def _grid(n=65, lo=0.0, hi=1.0):
    return np.linspace(lo, hi, n)


# --- the registry -------------------------------------------------------------

def test_every_registry_name_has_an_evaluator():
    """A name is only allowed into the registry once it can be measured."""
    assert UNIMPLEMENTED == (), f"declared but not implemented: {UNIMPLEMENTED}"


def test_every_gated_invariant_in_the_workspace_is_evaluable():
    """The Phase-0 gate: the registry must not reject a spec that validates today.
    Measured over ``workspace/``: 24 gated entries, all of them either a registry
    name or an entry carrying its own ``expr``."""
    checked = 0
    for path in sorted(glob.glob(os.path.join(WORKSPACE, "*", "problem_spec.json"))):
        with open(path) as fh:
            spec = json.load(fh)
        verification = spec.get("verification") or {}
        for key in ("invariants", "constraints"):
            for entry in verification.get(key) or []:
                if isinstance(entry, dict) and entry.get("gate"):
                    assert gateable(entry), f"{path}: {entry.get('name')}"
                    checked += 1
    assert checked >= 20, "workspace not staged?"


def test_an_unknown_gated_name_is_not_gateable_but_an_expr_rescues_it():
    assert not gateable({"name": "physically_reasonable", "gate": True})
    assert gateable({"name": "physically_reasonable", "gate": True, "expr": "X > 0"})
    assert gateable({"name": "positivity", "gate": True})


def test_an_unrecognised_invariant_reports_not_reported_never_pass():
    """§6's third rule, and the reason the registry has to be closed at all."""
    ctx = PdeContext({"u": np.ones(9)}, [_grid(9)])
    outcome, _ = evaluate_invariant({"name": "physically_reasonable", "gate": True}, ctx)
    assert outcome == "not_reported"
    assert outcome is not True


# --- the two conventions ------------------------------------------------------

def test_a_duplicated_wrap_node_is_not_the_endpoint_convention():
    """Two different facts. ``periodic`` is what the BCs say; ``endpoint_exclusive``
    is what the returned array looks like. Reading one for both rolls a full
    N-array by a fraction of a cell and reports a 0.76 translation-invariance
    violation on three correct spectral schemes."""
    assert duplicated_endpoint([True], [False], 1) == [True]   # periodic, inclusive
    assert duplicated_endpoint([True], [True], 1) == [False]   # periodic, exclusive
    assert duplicated_endpoint([False], [False], 1) == [False]  # Dirichlet


def test_periodic_quadrature_does_not_double_count_the_wrap_node():
    """The rectangle rule is spectrally accurate on a periodic grid -- but only over
    the *independent* nodes. Summing the full N-array of a duplicated-endpoint grid
    counts one physical point twice and over-states the integral by exactly one
    cell, which on a conserved quantity reads as a drift."""
    L = 2 * np.pi
    x = np.linspace(0.0, L, 65)              # inclusive: x[64] duplicates x[0]
    u = 1.0 + np.sin(x) ** 2                 # exact integral over a period: 1.5 L
    exact, dx = 1.5 * L, float(x[1] - x[0])
    trimmed = integral(u, [x], periodic=[True], duplicated=[True])
    untrimmed = float(u.sum() * dx)          # the naive rectangle rule
    assert abs(trimmed - exact) < 1e-10
    assert abs(untrimmed - exact) == pytest.approx(u[0] * dx, rel=1e-9)


def test_a_tolerance_of_zero_does_not_fail_an_exactly_conserved_quantity():
    """Real specs write ``tol: 0.0``. The manual's snippets test ``drift < tol``,
    which reports a violation for a drift of exactly zero -- a perfectly conserving
    scheme failing its own conservation check. Measured on pde_heat_1d."""
    ctx = PdeContext({"u": np.ones(9)}, [_grid(9)], initial=np.ones(9),
                     trace={"energy": np.ones(20)})
    assert evaluate_invariant({"name": "energy_decay", "gate": False, "tol": 0.0},
                              ctx)[0] is True
    assert evaluate_invariant({"name": "mass_conservation", "tol": 0.0}, ctx)[0] is True


def test_a_tolerance_of_zero_also_forgives_a_converged_linear_solve():
    """1e-10, not machine epsilon. Measured on
    ``pde_advection_1d/3-crank-nicolson-fd4``: Crank-Nicolson is unitary and
    conserves the discrete L2 norm exactly, and still reports an energy drift of
    2.0e-12 -- its sparse solve's residual, accumulated over a thousand steps."""
    trace = np.ones(1000)
    trace[500] = 1.0 + 2.0e-12
    ctx = PdeContext({"u": np.ones(9)}, [_grid(9)], initial=np.ones(9),
                     trace={"energy": trace})
    assert evaluate_invariant({"name": "energy_decay", "tol": 0.0}, ctx)[0] is True


def test_the_floor_never_loosens_a_tolerance_the_formulator_chose():
    """Applied as ``max(tol, ROUNDOFF)``, never ``tol + ROUNDOFF``. The smallest
    declared tolerance in ``workspace/`` is exactly 1e-10, and it must stay at 1e-10.
    """
    from verifylib.kernel.invariants import ROUNDOFF, _within
    assert _within(ROUNDOFF, 0.0) is True
    assert _within(1.01 * ROUNDOFF, 0.0) is False
    assert _within(1.01 * ROUNDOFF, ROUNDOFF) is False, "a declared 1e-10 stays 1e-10"
    assert _within(1.5e-8, 1e-8) is False


# --- the evaluators -----------------------------------------------------------

@pytest.mark.parametrize("entry,good,bad", [
    ({"name": "positivity", "tol": 1e-10}, np.array([0.0, 1.0]), np.array([-0.5, 1.0])),
    ({"name": "boundary_consistency", "bc_values": [0.0], "tol": 1e-8},
     np.sin(np.pi * _grid()), np.sin(np.pi * _grid()) + 0.5),
    ({"name": "maximum_principle", "bc_values": [0.0], "tol": 1e-10},
     0.5 * np.sin(np.pi * _grid()), 3.0 * np.sin(np.pi * _grid())),
])
def test_each_evaluator_separates_a_good_field_from_a_bad_one(entry, good, bad):
    axes = [_grid(len(good))]
    initial = np.sin(np.pi * _grid(len(good)))
    assert evaluate_invariant(entry, PdeContext({"u": good}, axes, initial=initial))[0] is True
    assert evaluate_invariant(entry, PdeContext({"u": bad}, axes, initial=initial))[0] is False


def test_symmetry_sees_a_shifted_profile():
    x = _grid()
    even = np.sin(np.pi * x)
    shifted = np.sin(np.pi * np.clip(x - 0.08, 0.0, 1.0))
    entry = {"name": "symmetry", "axis": 0, "parity": "even", "tol": 1e-6}
    assert evaluate_invariant(entry, PdeContext({"u": even}, [x]))[0] is True
    assert evaluate_invariant(entry, PdeContext({"u": shifted}, [x]))[0] is False


def test_a_trace_invariant_without_a_trace_is_not_reported():
    """Manual §8: mark it ``not_reported``, do not fail the plan -- but it cannot
    then be certified above 9."""
    ctx = PdeContext({"u": np.ones(9)}, [_grid(9)])
    outcome, drift = evaluate_invariant(
        {"name": "energy_bounded", "gate": True, "bound": 100.0, "requires_trace": True}, ctx)
    assert outcome == "not_reported"
    assert drift is not None, "the final field is still reported as a diagnostic"


def test_a_spec_supplied_predicate_is_evaluated_generically():
    ctx = SdeContext(np.array([1.0, 2.0, 3.0]), num_paths=3)
    assert evaluate_invariant({"name": "finite", "expr": "np.isfinite(X)"}, ctx)[0] is True
    assert evaluate_invariant({"name": "shape", "expr": "X.shape == (num_paths,)"},
                              ctx)[0] is True
    bad = SdeContext(np.array([1.0, np.nan]), num_paths=2)
    assert evaluate_invariant({"name": "finite", "expr": "np.isfinite(X)"}, bad)[0] is False


def test_the_not_reported_split_only_caps_on_gated_entries():
    """§6 caps the score for a **gated** invariant the kernel cannot evaluate. An
    ungated one is a diagnostic the spec asked for and did not get -- worth
    reporting, never worth a cap."""
    verification = {"invariants": [
        {"name": "energy_bounded", "gate": True, "requires_trace": True, "bound": 1.0},
        {"name": "gaussianity", "gate": False},
    ]}
    got = run_declared(verification, PdeContext({"u": np.ones(9)}, [_grid(9)]))
    assert sorted(got["not_reported"]) == ["energy_bounded", "gaussianity"]
    assert got["not_reported_gated"] == ["energy_bounded"]
