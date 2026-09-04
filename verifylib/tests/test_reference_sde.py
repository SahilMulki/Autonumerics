"""Criterion 7: the SDE reference check.

The claimed moments against the moment ODE the spec independently declares. A
mis-recalled OU mean and a correctly transcribed OU moment ODE will not agree, and
that is the failure being hunted.
"""
import json

import pytest

from verifylib.reference import check_moment_ode_consistency, check_reference

from .conftest import requires

SCALAR = ["sde_bm_with_drift", "sde_cir_feller_violated", "sde_ornstein_uhlenbeck"]
VECTOR = ["sde_gbm_2d_high_corr", "sde_multichannel_stiff_m13"]


@pytest.mark.parametrize("slug", SCALAR + VECTOR)
def test_tier_a_specs_are_consistent(slug):
    """All five, not three. Forward-integrating the declared moment ODE needs
    nothing from the claimed moments, so it works at any state dimension --
    including multichannel_stiff_m13's p12, which appears in no claimed moment and
    therefore could not be differentiated."""
    outcome, detail = check_moment_ode_consistency(requires(slug))
    assert outcome == "validated", f"{slug}: {outcome} {detail}"
    assert detail["max_rel_mismatch"] < 1e-9, detail


@pytest.mark.parametrize("slug", VECTOR)
def test_vector_state_is_supported(slug):
    """These carry 5 coupled moments including cross terms."""
    spec = requires(slug)
    assert len(spec["verification"]["moment_ode"]["state"]) == 5
    assert check_moment_ode_consistency(spec)[0] == "validated"


@pytest.mark.parametrize("slug,field,old,new,index", [
    ("sde_ornstein_uhlenbeck", "mean_expression", "-theta * t", "-0.5 * theta * t", None),
    ("sde_ornstein_uhlenbeck", "variance_expression", "2*theta", "theta", None),
    ("sde_bm_with_drift", "mean_expression", "mu", "2*mu", None),
    ("sde_gbm_2d_high_corr", "variance_expression", "sigma1**2", "0.5*sigma1**2", 0),
    ("sde_multichannel_stiff_m13", "mean_X", "F12*t", "2*F12*t", None),
])
def test_a_mis_recalled_moment_is_caught(slug, field, old, new, index):
    """Measured separation: every corruption lands at >= 2.5e-01 against a correct
    spec's <= 2.1e-10."""
    spec = json.loads(json.dumps(requires(slug)))
    target = spec["analytic_moments"][field]
    if index is None:
        assert old in target
        spec["analytic_moments"][field] = target.replace(old, new)
    else:
        assert old in target[index]
        spec["analytic_moments"][field][index] = target[index].replace(old, new)
    outcome, detail = check_moment_ode_consistency(spec)
    assert outcome == "failed", f"{slug} {field}: {outcome}"
    assert detail["max_rel_mismatch"] > 1e-2, detail


def test_covariance_is_matched_against_the_declared_covariance():
    """The trap: ``cross_from`` is a covariance ("mxy - m1x*m1y") while
    ``cross_moment.expression`` is the raw E[XY]. Pairing those by name reports a
    1.0 mismatch on a perfectly consistent spec."""
    spec = requires("sde_gbm_2d_high_corr")
    _, detail = check_moment_ode_consistency(spec)
    assert detail["per_observable"]["covariance"] < 1e-9


def test_a_closure_approximation_is_not_a_reference():
    spec = json.loads(json.dumps(requires("sde_ornstein_uhlenbeck")))
    spec["verification"]["moment_ode"]["closes_exactly"] = False
    outcome, detail = check_moment_ode_consistency(spec)
    assert outcome == "unavailable" and "closure" in detail["reason"]


def test_spec_without_analytic_moments_is_unavailable():
    outcome, _ = check_reference(requires("sde_ginzburg_landau_s6"))
    assert outcome == "unavailable"


def test_check_reference_dispatches_sde():
    outcome, detail = check_reference(requires("sde_ornstein_uhlenbeck"))
    assert outcome == "validated" and detail["tests"] == {"moment_ode": "validated"}
