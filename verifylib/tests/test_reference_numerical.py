"""Findings F5: a discretized closed form is a *numerical* closed form -- flagged,
never demoted -- and ``validated_off_singularity`` is A- when the check was waived
at a declared feature.

Both measured on the staged Burgers spec (the Cole-Hopf quadrature the formulator
wrote as a 2,010-character expression over ``np.linspace(..., 4001)``) and on
Black-Scholes, whose kink keeps 8.5% of the domain above tol at N = 1025 while the
formula is exactly right.
"""
import json

import pytest

from verifylib import reference as ref
from verifylib.kernel import score as S
from verifylib.reference import check_reference, numerical_closed_form, reference_findings

from .conftest import requires


@pytest.mark.parametrize("expr, flagged, label", [
    ("np.sin(x)*np.exp(-t)", False, None),
    ("np.sum(np.linspace(0, 1, 4001)*x[..., None], axis=-1)", True, "linspace(4001)"),
    ("(lambda y: y**2)(x)", True, "lambda"),
    ("np.linspace(0, 1, 5)[0]*x", False, None),          # constructed, not reduced
    ("np.sum(x)", False, None),                          # reduced, not constructed
])
def test_numerical_closed_form_flags_a_discretization_inside_the_expression(expr, flagged, label):
    got, detail = numerical_closed_form(expr)
    assert got is flagged
    if flagged:
        assert detail["label"] == label


def test_the_burgers_quadrature_is_flagged_with_its_node_count_and_not_demoted():
    spec = requires("pde_burgers_viscous_1d")
    outcome, detail = check_reference(spec)
    assert outcome in ("validated", "validated_off_singularity"), (outcome, detail.get("reason"))
    assert detail["analytic_numerical"]["u"]["label"] == "linspace(4001)"
    assert detail["analytic_numerical"]["u"]["nodes"] == [4001]
    found = reference_findings(spec)
    assert any("numerical closed form" in f.message and f.severity == "warning" for f in found)
    assert not [f for f in found if f.severity == "error"]


def test_the_fraction_above_tol_is_reported_at_the_finest_probe():
    """Both staged cases shrink with the probe (BS 23% -> 8.5%, Burgers 15% -> 5.4%),
    which is why a fixed-level fraction cannot separate them."""
    for slug in ("pde_black_scholes_call", "pde_burgers_viscous_1d"):
        outcome, detail = check_reference(requires(slug))
        assert outcome == "validated_off_singularity", (slug, outcome)
        assert detail["finest_N"] == ref.PROBE_N[-1]
        assert detail["frac_above_tol_finest"] < detail["frac_above_tol"]
        assert detail["frac_above_tol_finest"] < ref.OFF_SINGULARITY_MAX_FRAC


def test_a_declared_layer_demotes_the_off_singularity_pass_to_a_minus():
    spec = json.loads(json.dumps(requires("pde_burgers_viscous_1d")))
    spec["verification"]["structural_facts"] = {"layer": "internal layer of width ~2 nu"}
    outcome, detail = check_reference(spec)
    assert outcome == "unvalidated_at_feature"
    assert detail["demoted_from"] == "validated_off_singularity"
    assert "layer" in detail["reason"]
    found = reference_findings(spec)
    assert any("unvalidated_at_feature" in f.message for f in found)
    assert not [f for f in found if f.severity == "error"]     # A-, not null


def test_a_gross_fraction_demotes_without_a_declaration(monkeypatch):
    monkeypatch.setattr(ref, "OFF_SINGULARITY_MAX_FRAC", 0.01)
    outcome, detail = check_reference(requires("pde_black_scholes_call"))
    assert outcome == "unvalidated_at_feature"
    assert "finest probe" in detail["reason"]


def test_unvalidated_at_feature_is_priced_as_a_minus_and_validates_the_operator():
    evidence = {"kind": "pde", "path": "A", "has_reference": True,
                "reference_outcome": "unvalidated_at_feature", "converged": True,
                "order_ok": True, "invariants_ok": True, "constraints_ok": True,
                "temporal_ok": True, "e_fine": 1e-4, "rel_err_tol": 1e-2,
                "converged_measurable": True, "any_test_ran": True}
    got = S.certify(evidence)
    assert (got["score"], got["provenance"], got["tier"]) == (9, "analytic_unvalidated",
                                                              "A_unavailable")
    assert "declared feature" in got["reason"]
    from verifylib.kernel.residual import validate_operator
    route, ok, _ = validate_operator({}, {"kind": "scalar", "terms": {}},
                                     reference_outcome="unvalidated_at_feature")
    assert (route, ok) == ("reference", True)


def test_the_driver_reports_a_bare_reference_token_and_carries_the_exception(monkeypatch):
    from verifylib.kernel import driver

    def boom(spec):
        raise ValueError("no such helper")
    monkeypatch.setattr(ref, "check_reference", boom)
    spec = {"analytic_solution": {"expression": "np.sin(x)"}}
    outcome, detail = driver._reference_outcome(spec)
    assert outcome == "unavailable"
    assert "ValueError" in detail["reason"]
    assert S.tier_of({"has_reference": True, "reference_outcome": outcome}) == "A_unavailable"
