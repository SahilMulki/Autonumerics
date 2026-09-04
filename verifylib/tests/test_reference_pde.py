"""Criteria 2-6: the PDE reference check.

These tests carry the measurements the plan's §4c/§4f tables are built on. If one
starts failing, either the implementation regressed or the plan's numbers are stale
-- both are worth knowing.
"""
import json

import numpy as np
import pytest

from verifylib.reference import (
    TOL,
    check_boundary_conditions,
    check_initial_condition,
    check_operator_residual,
    check_reference,
    residual_field,
    resolve_operator,
)

from .conftest import requires


def _op(spec):
    operator, _ = resolve_operator(spec)
    assert operator is not None, "no operator resolved -- test cannot run"
    return operator


def _median(slug, expr=None, N=257):
    spec = requires(slug)
    exprs = {"u": expr} if expr else None
    scaled, mask = residual_field(spec, _op(spec), N=N, exprs=exprs)
    return float(np.median(scaled[mask])), float(np.max(scaled[mask]))


# --- criterion 3: a corrupted closed form separates by orders -----------------

@pytest.mark.parametrize("slug,bad,label", [
    ("pde_heat_1d", ("np.pi**2", "1.0"), "pi**2 dropped"),
    ("pde_heat_1d", ("np.sin(np.pi * x)", "np.sin(2*np.pi * x)"), "mode doubled"),
    ("pde_heat_2d", ("np.pi**2", "1.0"), "pi**2 dropped"),
    ("pde_wave_1d", ("np.sin(np.pi * x)", "np.sin(2*np.pi * x)"), "mode doubled"),
])
def test_corrupted_formula_separates_by_seven_orders(slug, bad, label):
    spec = requires(slug)
    good = spec["analytic_solution"]["expression"]
    wrong = good.replace(*bad)
    assert wrong != good, f"perturbation {label} did not apply"
    m_good, _ = _median(slug)
    m_bad, _ = _median(slug, wrong)
    assert m_good < TOL, f"{slug} correct form should validate, got {m_good:.2e}"
    assert m_bad > 1e-2, f"{slug} {label} should fail, got {m_bad:.2e}"
    assert m_bad / m_good > 1e7, f"separation only {m_bad / m_good:.1e}x"


# --- criterion 2: the smooth cases validate ----------------------------------

@pytest.mark.parametrize("slug", ["pde_heat_1d", "pde_heat_2d", "pde_wave_1d",
                                  "pde_wave_2d", "pde_laplace_2d"])
def test_smooth_specs_validate(slug):
    spec = requires(slug)
    outcome, detail = check_operator_residual(spec, _op(spec))
    assert outcome == "validated", f"{slug}: {outcome} (median {detail['median']:.2e})"


def test_single_term_operator_is_not_degenerate():
    """A single-term operator (lap_u) makes residual/max|term| identically 1
    without sub-term decomposition. Laplace is the regression test."""
    med, _ = _median("pde_laplace_2d")
    assert med < 1e-9, f"laplace median {med:.2e} -- sub-term decomposition broken?"


# --- the source-term path (§4d) ----------------------------------------------

@pytest.mark.parametrize("slug", ["pde_poisson_2d", "pde_helmholtz_2d",
                                  "pde_monge_ampere_2d", "pde_anisotropic_diffusion"])
def test_source_bearing_specs_validate(slug):
    """These were called blocked on the ground that the source existed only as
    prose inside ``governing_equation``. It does not: ``source_term`` is a real
    evaluable field, and ``residual_operator`` already references it as ``f``."""
    spec = requires(slug)
    outcome, detail = check_operator_residual(spec, _op(spec))
    assert outcome.startswith("validated"), f"{slug}: {outcome}"
    assert detail["median"] < TOL


def test_cahn_hilliard_is_a_manufactured_solution_not_a_defect():
    """It was called the plan's first real spec defect on a 0.247 residual. That
    residual is against the *homogeneous* operator: the claimed solution is
    manufactured and its source sits in the top-level ``source_term`` field.
    Binding it drops the median to ~1e-8.

    A false accusation of a spec defect is the worst failure this check can have,
    because §16 notes it is invisible as a false positive -- it looks exactly like
    a caught bug. This test pins the correction."""
    spec = requires("pde_cahn_hilliard_2d")
    homogeneous = spec["verification"]["mms_probe"]["operator_check"]
    bare, _ = residual_field(spec, {"kind": "scalar",
                                    "terms": {"op": homogeneous}, "source": None}, N=257)
    assert float(np.median(bare)) > 1e-2, "the homogeneous residual should be large"
    outcome, detail = check_operator_residual(spec, _op(spec))
    assert outcome.startswith("validated"), f"got {outcome}"
    assert detail["median"] < TOL


# --- criterion 4: non-smooth solutions validate off the singularity ----------

@pytest.mark.parametrize("slug", ["pde_burgers_inviscid", "pde_stefan_1d_similarity",
                                  "pde_convection_diffusion_bl", "pde_black_scholes_call"])
def test_localized_singularity_validates_off_it(slug):
    """Median below tol, max above it -- never a formulator-set waiver flag, and
    never a thresholded singular-set fraction (black_scholes keeps ~4% of the
    domain above tol at N=1025 while being exactly right)."""
    spec = requires(slug)
    outcome, detail = check_operator_residual(spec, _op(spec))
    assert outcome == "validated_off_singularity", f"{slug}: {outcome}"
    assert detail["max"] > TOL, "no singularity detected -- test is not exercising the path"


# --- criterion 5: under-resolution refines rather than failing ---------------

@pytest.mark.parametrize("slug", ["pde_advection_1d", "pde_fokker_planck_ou"])
def test_under_resolved_probe_refines(slug):
    """Both are correct but under-resolved at a coarse probe grid. The median must
    fall with refinement, and the check must not report `failed`."""
    coarse, _ = _median(slug, N=129)
    fine, _ = _median(slug, N=513)
    assert fine < coarse, f"{slug}: median {coarse:.2e} -> {fine:.2e}, not converging"
    spec = requires(slug)
    outcome, _ = check_operator_residual(spec, _op(spec))
    assert outcome != "failed", f"{slug} is correct but reported {outcome}"


# --- test 2: the formula at the initial time ---------------------------------

def test_initial_condition_catches_a_wrong_branch():
    """The residual cannot see initial data: widening the Gaussian gives a
    different but still exact solution of u_t + a u_x = 0."""
    spec = requires("pde_advection_1d")
    wide = spec["analytic_solution"]["expression"].replace("-100.0", "-50.0")
    assert wide != spec["analytic_solution"]["expression"]
    med, _ = _median("pde_advection_1d", wide, N=513)
    assert med < 1e-3, "the residual should stay small -- that is the whole point"
    corrupted = json.loads(json.dumps(spec))
    corrupted["analytic_solution"]["expression"] = wide
    assert check_initial_condition(corrupted)[0] == "failed"


def test_backward_parabolic_condition_is_read_at_maturity():
    """pde_black_scholes_call states its condition as a payoff at expiry, not at
    t = 0. Reading it at t = 0 reports a 3.5% mismatch on an exactly correct spec."""
    spec = requires("pde_black_scholes_call")
    outcome, detail = check_initial_condition(spec)
    assert outcome == "validated", f"{outcome}: {detail}"
    assert detail["matched_at"] == "T_maturity"


def test_benign_transcription_gap_is_reported_not_failed():
    """pde_advection_1d's initial_condition is a plain Gaussian while its solution
    is the periodic wrap; they differ by 1.2e-04 near the seam. Real, benign, and
    two orders below the problem's own accuracy target."""
    spec = requires("pde_advection_1d")
    outcome, detail = check_initial_condition(spec)
    assert outcome == "inconsistent", f"{outcome}: {detail}"
    assert 1e-6 < detail["max_rel_error"] < 1e-3


# --- test 3: the formula on the boundary -------------------------------------

@pytest.mark.parametrize("slug", ["pde_heat_1d", "pde_heat_2d", "pde_laplace_2d",
                                  "pde_poisson_2d"])
def test_boundary_conditions_validate(slug):
    assert check_boundary_conditions(requires(slug))[0] == "validated"


def test_boundary_check_catches_a_wrong_bc_selection():
    spec = json.loads(json.dumps(requires("pde_laplace_2d")))
    values = spec["boundary_conditions"]["values"]
    values["y=1"] = "np.sin(2*np.pi * x)"
    assert check_boundary_conditions(spec)[0] == "failed"


def test_moving_boundary_faces_are_skipped_not_guessed():
    """pde_stefan_1d_similarity keys a face "x=s(t)". Guessing at it would be
    worse than saying the check does not reach there."""
    spec = requires("pde_stefan_1d_similarity")
    _, detail = check_boundary_conditions(spec)
    assert "x=s(t)" not in (detail.get("faces") or {})


# --- the systems path (§4d) --------------------------------------------------

NAVIER_STOKES_OPERATOR = {
    "fields": ["u", "v"],
    "equations": {
        "u": {"terms": {"u_t": "dt(u)", "advection": "u*u_x + v*u_y",
                        "pressure": "p_x", "viscous": "-nu*lap_u"}, "source": "0"},
        "v": {"terms": {"v_t": "dt(v)", "advection": "u*v_x + v*v_y",
                        "pressure": "p_y", "viscous": "-nu*lap_v"}, "source": "0"},
    },
    "combine": "rms",
}


def test_systems_operator_validates_taylor_green():
    """The multi-field path, on a real system with a real closed form. The legacy
    operator_check cannot do this: it says ``adv_u``, a composite that means
    something different in every system that uses it."""
    spec = json.loads(json.dumps(requires("pde_navier_stokes_2d")))
    spec["verification"]["operator"] = NAVIER_STOKES_OPERATOR
    operator, provenance = resolve_operator(spec)
    assert provenance == "verification.operator" and operator["kind"] == "system"
    outcome, detail = check_operator_residual(spec, operator)
    assert outcome.startswith("validated"), f"{outcome}: {detail}"
    assert detail["median"] < TOL


def test_systems_operator_catches_a_sign_flipped_field():
    spec = json.loads(json.dumps(requires("pde_navier_stokes_2d")))
    spec["verification"]["operator"] = NAVIER_STOKES_OPERATOR
    fields = spec["analytic_solution"]["fields"]
    fields["p"] = "-1*(" + fields["p"] + ")"
    outcome, detail = check_operator_residual(spec, resolve_operator(spec)[0])
    assert outcome == "failed", f"{outcome} (median {detail['median']:.2e})"


def test_legacy_system_operator_reports_unavailable_rather_than_crashing():
    """``adv_u`` is not a generic helper. Saying so is the correct outcome."""
    outcome, detail = check_reference(requires("pde_navier_stokes_2d"))
    assert outcome == "unavailable"
    assert "verification.operator.equations" in detail["reason"]


# --- §14 C9: chaotic problems short-circuit ----------------------------------

def test_chaotic_problem_short_circuits():
    outcome, detail = check_reference({"chaotic": True, "spatial_variables": ["x"]})
    assert outcome == "unavailable" and detail["short_circuit"]
