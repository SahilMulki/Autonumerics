"""D1: the slope test, the stencil guard, the trivial guards, operator validation.

The one thing worth restating: there is no magnitude threshold here and there
cannot be. A solver's residual is dominated by its own truncation error and does
not approach zero at fixed resolution, so every test below is about the *rate*.
"""
import json
import os

import numpy as np
import pytest

from verifylib.kernel.residual import (
    INSENSITIVITY_FACTOR,
    RESIDUAL_ROUNDOFF,
    SLOPE_SLACK,
    STALL_P,
    insensitive,
    residual_on_fields,
    slope_test,
    stencil_pair,
    trivial_guards,
    validate_operator,
)
from verifylib.operator import Stencil
from verifylib.reference import resolve_operator

from .conftest import requires

CANARY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "canaries", "kernel")


def _canary_spec():
    with open(os.path.join(CANARY, "spec.json")) as fh:
        return json.load(fh)


# --- the slope test -----------------------------------------------------------

def test_a_residual_falling_at_the_design_rate_is_clean():
    outcome, detail = slope_test([("h", 1e-2), ("h/2", 2.5e-3)], p_design=2.0)
    assert outcome == "clean" and detail["p_res"] == pytest.approx(2.0)


def test_a_residual_that_does_not_fall_is_stalled():
    assert slope_test([("h", 1e-2), ("h/2", 9.9e-3)], p_design=2.0)[0] == "stalled"


def test_a_residual_falling_too_slowly_is_named_rather_than_forced_into_a_neighbour():
    """The plan's four-row table leaves the band between `clean` and `stalled`
    unnamed. Routing it to `stalled` would gate a slowly-converging solver at 3;
    routing it to `clean` would let it certify."""
    outcome, detail = slope_test([("h", 1e-2), ("h/2", 5e-3)], p_design=4.0)
    assert outcome == "slow"
    assert STALL_P < detail["p_res"] < 4.0 - SLOPE_SLACK


def test_the_roundoff_branch_is_a_pass_not_a_stall():
    """Manual §4's not_at_roundoff guard, on the residual. The KS spec's own note
    documents a spectral scheme collapsing to round-off on its ladder."""
    outcome, detail = slope_test([("h", 1e-12), ("h/2", 1e-13)], p_design=4.0)
    assert outcome == "clean" and detail["reason"] == "round-off floor"
    assert detail["roundoff_floor"] == RESIDUAL_ROUNDOFF


def test_one_grid_gives_no_slope():
    assert slope_test([("h", 1e-3)], p_design=2.0)[0] == "unavailable"


# --- the stencil-insensitivity guard -----------------------------------------

def test_two_stencils_that_disagree_mean_the_check_is_the_limit():
    assert insensitive(1e-4, 1.5e-4) is True
    assert insensitive(1e-4, INSENSITIVITY_FACTOR * 1e-4 * 1.01) is False
    assert insensitive(1e-13, 5e-13) is True, "both at round-off: nothing to resolve"


def test_the_stencil_pair_is_declared_or_defaulted_never_guessed():
    pair, choice, name = stencil_pair(None, None, [False])
    assert choice == "default" and name == "fd8+fd6"
    assert [s.order for s in pair] == [8, 6]
    pair, choice, name = stencil_pair("spectral", None, [True])
    assert choice == "declared" and name == "fd8+spectral"
    assert [s.order for s in pair] == [8, "spectral"]


# --- the residual on real solver output ---------------------------------------

def test_the_residual_of_an_exact_field_is_small_and_falls():
    """Substituting an exact solution of ``u_t = alpha*u_xx - k*u`` into the
    declared operator: the residual measures the kernel's own truncation error, so
    it must fall at the kernel's stencil order, not the solver's."""
    spec = _canary_spec()
    operator, _ = resolve_operator(spec)
    params = spec["parameters"]
    alpha, k = params["alpha"], params["k"]
    medians = []
    for N in (65, 129):
        x = np.linspace(0.0, 1.0, N)
        t = 0.5
        decay = -(alpha * np.pi ** 2 + k)
        u = np.exp(decay * t) * np.sin(np.pi * x)
        u_t = decay * u
        median, _stats, _mask = residual_on_fields(
            operator, {"u": u}, {"x": x}, ["x"], [float(x[1] - x[0])], params,
            time_derivs={"u": (u_t, decay * u_t)}, stencil=Stencil(8), t=t)
        medians.append(median)
    assert medians[-1] < 1e-8
    assert slope_test(list(zip(("h", "h/2"), medians, strict=True)), 2.0)[0] == "clean"


# --- §5d: validating the operator itself --------------------------------------

def test_the_operator_is_validated_through_the_mms_probe():
    spec = _canary_spec()
    operator, _ = resolve_operator(spec)
    route, ok, detail = validate_operator(spec, operator)
    assert route == "mms" and ok is True, detail


def test_a_sign_flipped_operator_term_is_caught():
    """The seeded defect §9c names for this route. Both the solver and the residual
    descend from this one declaration, so if it is wrong they are wrong
    consistently -- which is exactly why it has to be checked separately."""
    spec = _canary_spec()
    spec["verification"]["operator"]["terms"]["diffusion"] = "+alpha*lap(u)"
    operator, _ = resolve_operator(spec)
    route, ok, detail = validate_operator(spec, operator)
    assert route == "mms" and ok is False, detail
    assert detail["median"] > detail["tol"]


def test_a_dropped_operator_term_is_caught():
    spec = _canary_spec()
    del spec["verification"]["operator"]["terms"]["reaction"]
    operator, _ = resolve_operator(spec)
    _route, ok, _detail = validate_operator(spec, operator)
    assert ok is False


def test_the_reference_check_passing_is_itself_the_validation():
    """Path A: a wrong operator makes a correct closed form fail, so there is
    nothing further to do."""
    spec = _canary_spec()
    operator, _ = resolve_operator(spec)
    route, ok, _ = validate_operator(spec, operator, reference_outcome="validated")
    assert (route, ok) == ("reference", True)


def test_an_unauditable_operator_reports_none_rather_than_true():
    spec = _canary_spec()
    del spec["verification"]["mms_probe"]
    del spec["verification"]["degenerate_limit"]
    operator, _ = resolve_operator(spec)
    route, ok, detail = validate_operator(spec, operator)
    assert (route, ok) == ("none", False)
    assert "unaudited" in detail["reason"]


def test_a_real_workspace_operator_validates_through_the_degenerate_limit():
    """Kuramoto-Sivashinsky has no closed form at all, so the degenerate limit is
    the only route -- and it is the one that makes D1 a circularity break there."""
    spec = requires("pde_kuramoto_sivashinsky")
    operator, _ = resolve_operator(spec)
    route, ok, detail = validate_operator(spec, operator)
    assert ok is True, detail
    assert route in ("mms", "degenerate")


# --- §5f: the trivial-attractor guards ----------------------------------------

def test_a_collapsed_field_is_caught_however_small_its_residual():
    """A residual rewards any field sitting on a stable steady state of the
    operator, whether or not it is the state the initial condition evolves to."""
    initial = np.sin(np.pi * np.linspace(0, 1, 65))
    assert trivial_guards(initial * 0.5, initial)[0] is True
    assert trivial_guards(initial * 1e-9, initial)[0] is False
    assert trivial_guards(np.zeros(65), initial)[0] is False


def test_a_guard_with_no_initial_field_is_not_reported():
    assert trivial_guards(np.ones(9), None)[0] == "not_reported"
