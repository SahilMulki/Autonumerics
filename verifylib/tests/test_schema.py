"""Criterion 1, and the spec-shape half of criterion 2.

The spec is the chokepoint (§6a): the archived Heston leak entered at
``problem_spec.json`` and propagated downstream by hand-transcription. Everything
here fires there.
"""
import glob
import os

import pytest

from verifylib import schema
from verifylib.findings import ERROR, errors

from .conftest import FIXTURES, WORKSPACE, read_json


def _specs(pattern="*"):
    return sorted(glob.glob(os.path.join(WORKSPACE, pattern, "problem_spec.json")))


def _staged(pattern="*"):
    paths = _specs(pattern)
    if not paths:
        pytest.skip("workspace/ is not staged")
    return paths


# --- criterion 1 -------------------------------------------------------------

def test_heston_leak_spec_is_rejected():
    """A 2063-character def with Gauss-Legendre quadrature in an expression field."""
    spec = read_json(os.path.join(FIXTURES, "leak_heston_spec.json"))
    findings = schema.check_expressions(spec)
    assert any(f.path == "analytic_solution.expression" for f in findings)


def test_every_real_expression_field_parses():
    """Zero false positives over the staged workspace. A guard that fires on a
    correct spec is a guard that gets switched off."""
    for path in _staged():
        bad = schema.check_expressions(read_json(path))
        assert not bad, f"{os.path.basename(os.path.dirname(path))}: {[f.render() for f in bad]}"


def test_prose_is_allowed_in_guidance_but_a_program_is_not():
    """``drift_expression`` on a vector problem is legitimately a description of a
    matrix action. A ``def`` in the same field is still the Heston shape."""
    prose = {"drift_expression": "X @ F.T  (row-major paths, X shape (num_paths, 2))"}
    assert not schema.check_expressions(prose)
    program = {"drift_expression": "def f(x):\n    return x * 2\n"}
    assert schema.check_expressions(program)


def test_gated_constraint_must_be_evaluable_but_a_note_need_not_be():
    gated = {"verification": {"constraints": [{"expr": "X ~ Normal(0, 1)", "gate": True}]}}
    assert schema.check_expressions(gated)
    note = {"verification": {"constraints": [{"expr": "X ~ Normal(0, 1)", "gate": False}]}}
    assert not schema.check_expressions(note)


def test_placeholder_text_is_not_an_expression():
    """``"unknown"`` parses as a Python name, so ast.parse does not catch it."""
    spec = {"analytic_solution": {"expression": "unknown"}}
    assert schema.check_prose_as_formula(spec)


# --- the operator declaration (§4d) ------------------------------------------

def test_operator_declaration_is_required_on_a_new_pde_spec():
    spec = {"spatial_variables": ["x"], "verification": {}}
    findings = schema.check_operator_declaration(spec)
    assert findings and findings[0].severity == "error"


def test_null_operator_needs_a_reason():
    """A recorded gap, never a silent one."""
    bare = {"spatial_variables": ["x"], "verification": {"operator": None}}
    assert schema.check_operator_declaration(bare)
    excused = {"spatial_variables": ["x"],
               "verification": {"operator": None,
                                "operator_note": "Caputo derivative is non-local"}}
    assert not schema.check_operator_declaration(excused)


def test_the_schema_gate_agrees_with_the_reference_check_on_legacy_sources():
    """Both legacy sources, or the gate contradicts the check it gates for.
    pde_anisotropic_diffusion carries only residual_operator, and the reference
    check validates it at 2.5e-10 -- erroring on it here would block a spec that
    passes the very test the field exists to enable."""
    spec = {"spatial_variables": ["x", "y"],
            "verification": {"residual_operator": "-(A_xx*u_xx + A_yy*u_yy) - f"}}
    assert [f.severity for f in schema.check_operator_declaration(spec)] == ["warning"]


def test_legacy_operator_check_warns_rather_than_blocks():
    """19 of 22 staged PDE specs predate the field. They still run on the fallback,
    and saying so is a warning, not a gate."""
    spec = {"spatial_variables": ["x"],
            "verification": {"mms_probe": {"operator_check": "u_t - alpha*lap_u"}}}
    findings = schema.check_operator_declaration(spec)
    assert [f.severity for f in findings] == ["warning"]


@pytest.mark.parametrize("operator,ok", [
    ({"fields": ["u"], "terms": {"a": "dt(u)"}, "source": "0"}, True),
    ({"fields": ["u", "v"], "equations": {"u": {"terms": {"a": "dt(u)"}},
                                          "v": {"terms": {"b": "dt(v)"}}},
      "combine": "rms"}, True),
    ({"fields": ["u"], "terms": {"a": "dt(u)"},
      "equations": {"u": {"terms": {}}}}, False),          # both forms at once
    ({"fields": ["u", "v"], "equations": {"u": {"terms": {"a": "dt(u)"}}}}, False),  # v missing
    ({"fields": ["u"], "terms": {}}, False),                # empty
])
def test_operator_shape(operator, ok):
    spec = {"spatial_variables": ["x"], "verification": {"operator": operator}}
    assert bool(schema.check_operator_declaration(spec)) != ok


# --- ledger shape ------------------------------------------------------------

def test_real_ledgers_pass_shape_validation():
    for path in _staged():
        bad = schema.check_ledger(read_json(path))
        assert not bad, f"{path}: {[f.render() for f in bad]}"


@pytest.mark.parametrize("entry", [
    {"id": "R1", "kind": "parameter", "quote": "eps = 1e-3", "status": "mapped"},
    {"id": "R1", "kind": "nonsense", "quote": "x", "status": "mapped", "spec_path": "a"},
    {"id": "R1", "kind": "parameter", "quote": "", "status": "mapped", "spec_path": "a"},
    {"id": "R1", "kind": "parameter", "quote": "x", "status": "dropped"},
])
def test_malformed_ledger_entries_are_caught(entry):
    assert schema.check_ledger({"requirements": [entry]})


def test_operator_ledger_entry_is_a_warning_not_a_gate():
    """§4g's common-mode mitigation. The formulator writes the operator and the
    analytic solution in one pass, so the ledger is the only check that reaches
    outside the spec -- but 19 of 22 staged specs predate the rule."""
    spec = {"spatial_variables": ["x"],
            "requirements": [{"id": "R1", "kind": "equation", "quote": "u_t = u_xx",
                              "status": "mapped", "spec_path": "governing_equation"}]}
    findings = schema.check_operator_ledgered(spec)
    assert [f.severity for f in findings] == ["warning"]
    spec["requirements"][0]["spec_path"] = "governing_equation, verification.operator"
    assert not schema.check_operator_ledgered(spec)


# --- the invariant registry (plan-no-closed-form §6) -------------------------

def test_a_gated_invariant_the_kernel_cannot_evaluate_is_a_schema_error():
    """A gate nobody implements is worse than no gate, because it reads as
    evidence that a check passed. Before this, ``{"name": "physically_reasonable",
    "gate": true}`` passed the schema and matched nothing in the kernel."""
    spec = {"spatial_variables": ["x"],
            "verification": {"invariants": [
                {"name": "physically_reasonable", "gate": True}]}}
    found = schema.check_invariant_names(spec)
    assert [f.severity for f in found] == [ERROR]
    assert "not in the kernel registry" in found[0].message


def test_an_ungated_invariant_stays_free_form():
    """Eight distinct names are in use that way; requiring an implementation for a
    diagnostic nobody gates on would buy nothing."""
    spec = {"spatial_variables": ["x"],
            "verification": {"invariants": [{"name": "front_position", "gate": False}]}}
    assert schema.check_invariant_names(spec) == []


def test_a_gated_entry_with_its_own_expression_is_accepted():
    """Every gated SDE constraint in ``workspace/`` takes this route."""
    spec = {"state_dimension": 1,
            "verification": {"constraints": [
                {"name": "finite", "expr": "np.isfinite(X)", "gate": True}]}}
    assert schema.check_invariant_names(spec) == []


def test_every_staged_spec_still_passes_the_registry_check():
    """The Phase-0 gate: adding a check must not invalidate a spec that validates
    today."""
    for path in sorted(glob.glob(os.path.join(WORKSPACE, "*", "problem_spec.json"))):
        found = errors(schema.check_invariant_names(read_json(path)))
        assert not found, f"{path}: {[f.render() for f in found]}"


# --- the chaotic flag (plan-no-closed-form §8a) ------------------------------

_CHAOTIC = {
    "spatial_variables": ["x"],
    "time_interval": {"T": 50.0},
    "chaotic": True,
    "chaotic_T_ref": 5.0,
    "requirements": [{"id": "R1", "kind": "other", "status": "mapped",
                      "quote": "This equation is chaotic and has no closed-form solution",
                      "spec_path": "chaotic, chaotic_T_ref"}],
}


def test_a_well_formed_chaotic_declaration_passes():
    assert schema.check_chaotic(_CHAOTIC) == []


def test_chaotic_without_a_ledger_quote_is_an_error():
    """It waives the Tier-C order check, which is precisely the threat shape Layer 2
    exists for: a formulator that sets it escapes Tier C entirely."""
    spec = {**_CHAOTIC, "requirements": []}
    found = errors(schema.check_chaotic(spec))
    assert found and "requirements ledger entry" in found[0].message


def test_chaotic_without_a_declared_reference_horizon_is_an_error():
    spec = {k: v for k, v in _CHAOTIC.items() if k != "chaotic_T_ref"}
    found = errors(schema.check_chaotic(spec))
    assert found and "chaotic_T_ref" in found[0].path


def test_a_reference_horizon_beyond_the_problems_own_is_an_error():
    found = errors(schema.check_chaotic({**_CHAOTIC, "chaotic_T_ref": 500.0}))
    assert found and "exceeds the problem's own horizon" in found[0].message


def test_a_reference_horizon_without_the_flag_is_an_error():
    """A stray relaxation: the window only means something once the order check has
    been waived."""
    found = errors(schema.check_chaotic({"chaotic_T_ref": 5.0}))
    assert found and "without chaotic: true" in found[0].message


def test_no_staged_spec_declares_chaotic_yet_and_none_is_broken_by_the_check():
    for path in sorted(glob.glob(os.path.join(WORKSPACE, "*", "problem_spec.json"))):
        assert errors(schema.check_chaotic(read_json(path))) == [], path
