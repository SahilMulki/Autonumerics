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


def test_a_registry_name_beside_an_expr_is_evaluated_by_the_registry():
    """The name wins. ``{"name": "positivity", "expr": "X > -5"}`` tests ``X > 0``
    -- the registry's definition -- and the expression is ignored. A formulator
    who wants a custom predicate must give it a name the registry does not own;
    otherwise the declared gate and the evaluated one silently differ."""
    import numpy as np

    from verifylib.kernel.invariants import SdeContext, evaluate_invariant

    ctx = SdeContext(np.array([1.0, -1.0, 2.0]))
    shadowed, _ = evaluate_invariant({"name": "positivity", "expr": "X > -5", "gate": True}, ctx)
    assert shadowed is False                      # X > 0 fails on the -1.0 path
    custom, _ = evaluate_invariant({"name": "floor_minus_five", "expr": "X > -5", "gate": True}, ctx)
    assert custom is True


def test_the_martingale_check_reads_the_process_the_spec_declares():
    """[REV 50]: ``_martingale``'s registered keys were ``("ci_mult",
    "reference_value_at_T")`` and a declared ``expr`` was silently ignored, so the
    kernel always measured ``|E[X_T] - X_0|``. On two live Path-A plans that is not
    zero and is not supposed to be, and both carried ``invariants_ok: False`` on the
    artifact: ``sde_gbm_2d_high_corr`` at 0.10475 against the predicted
    ``e^(mu1*T) - 1 = 0.10517`` (0.27 SE), and ``sde_bm_with_drift`` at 0.50258
    against ``mu*T = 0.5`` (1.92 SE) on a pathwise-exact solver.

    Exact terminal laws here, so the only thing under test is which process the
    evaluator reads.
    """
    rng = np.random.default_rng(42)
    mu, sigma, T = 0.5, 0.3, 1.0
    X = 1.0 + mu * T + sigma * rng.standard_normal(50000)
    ctx = SdeContext(X, params={"X_0": 1.0, "mu": mu, "sigma": sigma},
                     num_paths=len(X), x0=1.0, T=T)
    entry = {"name": "martingale", "gate": False, "tol": 0.02}
    prose_only, drift = evaluate_invariant(entry, ctx)
    assert prose_only is False and drift == pytest.approx(mu * T, abs=0.01)
    declared, drift = evaluate_invariant({**entry, "expr": "X - mu*T"}, ctx)
    assert declared is True and drift < 0.01


def test_a_declared_martingale_expression_may_discount_every_component():
    """``sde_gbm_2d_high_corr``'s own note says "Same for Y with mu2", which the
    column-0 reduction never tested -- the fifth site of Finding 11's ``[:, 0]``."""
    rng = np.random.default_rng(7)
    mu, sigma, T, rho = 0.1, 0.3, 1.0, 0.95
    Z1 = rng.standard_normal(50000)
    Z2 = rho * Z1 + np.sqrt(1 - rho ** 2) * rng.standard_normal(50000)
    XY = np.stack([np.exp((mu - 0.5 * sigma ** 2) * T + sigma * Z1),
                   np.exp((mu - 0.5 * sigma ** 2) * T + sigma * Z2)], axis=1)
    ctx = SdeContext(XY, params={"X_0": 1.0, "mu1": mu, "mu2": mu},
                     num_paths=len(XY), x0=1.0, T=T)
    entry = {"name": "martingale", "gate": False, "tol": None, "ci_mult": 2.0}
    assert evaluate_invariant(entry, ctx)[0] is False       # |E[X_T] - X_0| = e^mu - 1
    both, drift = evaluate_invariant(
        {**entry, "expr": "np.exp(-np.array([mu1, mu2])*T) * X"}, ctx)
    assert both is True and drift < 0.01
    # A spurious drift in the *second* component alone must be seen, which is what
    # the column-0 reduction could not do.
    skewed = XY.copy()
    skewed[:, 1] *= 1.2
    assert evaluate_invariant(
        {**entry, "expr": "np.exp(-np.array([mu1, mu2])*T) * X"},
        SdeContext(skewed, params=ctx.params, num_paths=len(skewed), x0=1.0, T=T))[0] is False


def test_an_unusable_martingale_expression_is_not_reported_rather_than_ignored():
    """The fails-open shape §6 exists to refuse, in the direction [REV 50] found:
    declared, *read*, and discarded by the evaluator that claims to use it. An
    expression that will not evaluate must not silently fall back to a different
    check whose answer the formulator never asked for."""
    ctx = SdeContext(np.ones(100), params={"X_0": 1.0}, num_paths=100, x0=1.0, T=1.0)
    outcome, reason = evaluate_invariant(
        {"name": "martingale", "expr": "np.exp(-nope*T) * X"}, ctx)
    assert outcome == "not_reported"
    assert "did not evaluate" in reason and "nope" in reason
    wrong_shape, reason = evaluate_invariant(
        {"name": "martingale", "expr": "np.mean(X)"}, ctx)
    assert wrong_shape == "not_reported" and "one value per path" in reason


def test_expr_is_a_predicate_everywhere_except_the_martingale():
    """The two rules side by side, because they look contradictory and are not.

    For every other registry name the name wins and ``expr`` is ignored -- see
    ``test_a_registry_name_beside_an_expr_is_evaluated_by_the_registry``. ``expr``
    on a ``martingale`` entry is not a *predicate* at all: the registry still owns
    the check (``E[M_T] = M_0`` within ``ci_mult`` standard errors) and the
    expression supplies **which process M is**, which no other field can state.
    ``reference_value_at_T`` gives the target, not the process.
    """
    ctx = SdeContext(np.array([1.0, -1.0, 2.0]), params={"a": 1.0}, num_paths=3, x0=1.0, T=1.0)
    # `positivity` tests X > 0 whatever the expr says.
    assert evaluate_invariant({"name": "positivity", "expr": "X > -5"}, ctx)[0] is False
    # `martingale` evaluates the expr as the process, and `0*X` is a martingale at 0.
    assert evaluate_invariant(
        {"name": "martingale", "expr": "0*X", "reference_value_at_T": 0.0}, ctx)[0] is True


# --- headroom against the bar that was applied ([REV 64]) ---------------------

def test_a_statistical_invariant_reports_headroom_against_its_own_bar():
    """``_martingale`` judges ``drift < ci_mult*se + tol``; ``_headroom`` divided by
    ``tol`` alone.

    ``sde_gbm_2d_high_corr`` is the live case: it declares no ``tol`` for its
    martingale, so ``effective_tol`` floors at ``ROUNDOFF`` and a drift of 5.6e-4
    that **passed** -- it is inside ``2*se`` -- was reported at **-6.75 decades** of
    headroom, which reads as a six-decade failure on a plan that scored 10.
    ``worst_headroom`` is a ranking anchor, so the wrong number moves ranks with
    nothing to show for it.
    """
    from verifylib.kernel.invariants import (
        ROUNDOFF, SdeContext, effective_bar, effective_tol, run_declared,
    )

    rng = np.random.default_rng(0)
    mu, T, M = 0.1, 1.0, 50000
    X = np.exp((mu - 0.5 * 0.04) * T + 0.2 * np.sqrt(T) * rng.standard_normal((M, 1)))
    entry = {"name": "martingale", "gate": False, "ci_mult": 2.0,
             "expr": "np.exp(-mu*T) * X[..., 0]"}
    ctx = SdeContext(X, params={"mu": mu}, num_paths=M, x0=1.0, T=T)
    out = run_declared({"invariants": [entry]}, ctx)

    assert out["outcomes"]["martingale"] is True
    assert effective_tol(entry) == ROUNDOFF, "no declared tol: the floor is all there is"
    bar = effective_bar(entry, ctx)
    assert bar > 100 * ROUNDOFF, bar
    # The number that matters: a check that passed does not report negative headroom.
    assert out["headroom"]["martingale"] > 0.0, out["headroom"]
    assert out["worst_headroom"] > 0.0
    # And it is the real ratio, not a constant.
    assert out["headroom"]["martingale"] == pytest.approx(
        float(np.log10(bar / abs(out["drifts"]["martingale"]))), rel=1e-9)
    assert out["bars"]["martingale"] == pytest.approx(bar)


def test_a_declared_tolerance_still_decides_where_there_is_no_statistical_term():
    """The fallback is unchanged: an invariant whose evaluator compares against the
    declared tolerance keeps reporting against it, so [REV 64] moves exactly the
    entries whose bar was wrong and no others."""
    from verifylib.kernel.invariants import SdeContext, effective_bar, run_declared

    X = np.full((1000, 1), 2.0)
    entry = {"name": "estimator_stability", "gate": False, "tol": 1e-2}
    ctx = SdeContext(X, num_paths=1000, x0=2.0, T=1.0)
    assert effective_bar(entry, ctx) == 1e-2
    out = run_declared({"invariants": [entry]}, ctx)
    assert out["bars"]["estimator_stability"] == 1e-2


def test_every_invariant_with_a_statistical_bar_declares_one():
    """The parity assertion. The registry already states which entries take a
    ``ci_mult``; an evaluator that reads one and is missing from ``_STATISTICAL_BAR``
    reports headroom against a bar it never applied, which is the defect [REV 64]
    fixed and the one a new statistical invariant would reintroduce silently."""
    import inspect

    from verifylib.kernel.invariants import _EVALUATORS, _STATISTICAL_BAR, INVARIANTS

    by_registry = {name for name, inv in INVARIANTS.items() if "ci_mult" in inv.params}
    by_source = {name for name, fn in _EVALUATORS.items()
                 if "ci_mult" in inspect.getsource(fn)}
    assert by_registry == by_source == set(_STATISTICAL_BAR), (by_registry, by_source)
