"""Criterion 1, and the spec-shape half of criterion 2.

The spec is the chokepoint (§6a): the archived Heston leak entered at
``problem_spec.json`` and propagated downstream by hand-transcription. Everything
here fires there.
"""
import glob
import os

import pytest

from verifylib import schema

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
