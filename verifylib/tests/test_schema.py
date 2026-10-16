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


def test_a_multi_component_sde_must_declare_its_noise_dimension():
    """Phase 2a. The field existed, the sandbox honoured it and nothing validated
    it, so a correct 2-noise solver got a rank-2 increment array and the crash was
    reported against the solver."""
    base = {"state_dimension": 2, "sde_family": "gbm_2d_correlated"}
    assert [f.check for f in schema.check_noise_dimension(base)] == ["noise_dimension"]
    assert schema.check_noise_dimension({**base, "noise_dimension": None})
    assert schema.check_noise_dimension({**base, "noise_dimension": 0})
    assert schema.check_noise_dimension({**base, "noise_dimension": 2}) == []
    assert schema.check_noise_dimension({**base, "noise_dimension": 13}) == []
    # A scalar state takes a scalar channel, so there is nothing to mismatch.
    assert schema.check_noise_dimension({"state_dimension": 1, "sde_family": "ou"}) == []
    # And this is an SDE check only.
    assert schema.check_noise_dimension({"spatial_variables": ["x"], "state_dimension": 2}) == []


def test_the_noise_dimension_is_an_error_not_a_warning():
    """``check_invariant_names``' reasoning, reused: there is no legacy population to
    grandfather. Of the two specs in the repo that mentioned the field, one declared
    13 and the other left it out -- and the omission was the formulator gap."""
    found = schema.check_noise_dimension({"state_dimension": 2, "sde_family": "x"})
    assert [f.severity for f in found] == ["error"]
    assert "cannot be inferred" in found[0].message


def test_every_live_sde_spec_declares_what_the_crn_ladder_needs():
    """The artifacts, not the rule. ``sde_multichannel_stiff_m13`` declares 13,
    ``sde_gbm_2d_high_corr`` declares 2, and the scalar specs declare nothing
    because nothing is needed."""
    for path in sorted(glob.glob(os.path.join(WORKSPACE, "sde_*", "problem_spec.json"))):
        spec = read_json(path)
        assert schema.check_noise_dimension(spec) == [], (path, spec.get("noise_dimension"))


# --- criterion 7's per-family half ([REV 63]) ---------------------------------

def test_a_spec_that_admits_milstein_must_declare_its_milstein_floor():
    """Criterion 7 reads "every spec declares a floor for **every scheme family its
    plans use**". Until [REV 63] ``expected_strong_order_milstein`` appeared only as
    prose inside the other error's message, so this spec passed the check and then
    had its Milstein plan judged against Euler-Maruyama's floor -- a correct solver
    measuring 1.0 against 0.5 is a two-sided miss of 0.5. It is how
    ``sde_cir_feller_violated``'s floor came to be fixed by hand."""
    admits = {"state_dimension": 1, "sde_family": "cir", "milstein_eligible": True,
              "verification": {"expected_strong_order": 0.5}}
    found = schema.check_strong_order_declared(admits)
    assert [f.severity for f in found] == ["error"]
    assert "expected_strong_order_milstein" in found[0].message \
        or "milstein" in str(found[0])
    assert schema.check_strong_order_declared(
        {**admits, "verification": {"expected_strong_order": 0.5,
                                    "expected_strong_order_milstein": 1.0}}) == []


def test_a_spec_where_no_milstein_plan_can_exist_needs_no_milstein_floor():
    """The inverse false positive, which is why the trigger is the spec's own
    statement and not the mere presence of a ``verification`` block. Both reasons
    live: ``zero_diffusion_derivative`` means the correction vanishes identically so
    the Euler floor *is* the right one, and ``levy_area`` means
    ``driver._milstein_claim`` caps the plan instead."""
    for reason in ("zero_diffusion_derivative", "levy_area"):
        spec = {"state_dimension": 1, "sde_family": "ou", "milstein_eligible": False,
                "milstein_ineligible_reason": reason,
                "verification": {"expected_strong_order": 1.0}}
        assert schema.check_strong_order_declared(spec) == [], reason


def test_a_spec_that_says_nothing_about_milstein_is_held_to_the_floor():
    """Silence is not an exemption: `expected_orders` will read
    ``expected_strong_order`` for a Milstein plan on this spec, so the spec has to
    either declare the floor or declare that no such plan can exist."""
    quiet = {"state_dimension": 1, "sde_family": "gbm",
             "verification": {"expected_strong_order": 0.5}}
    assert [f.severity for f in schema.check_strong_order_declared(quiet)] == ["error"]


def test_the_required_family_keys_are_the_ones_the_order_study_reads():
    """The parity assertion the check's docstring promises. ``schema`` deliberately
    does not import the kernel, so the two mappings are held together here: a family
    the order study can select a floor for, with no rule requiring that floor, is
    criterion 7's hole reopening under a new scheme name."""
    from verifylib.kernel.sde.crn import FAMILY_ORDER_KEY

    assert set(schema.FAMILY_ORDER_REQUIRED) == set(FAMILY_ORDER_KEY.values())


def test_every_live_sde_spec_declares_a_floor_for_every_family_it_admits():
    """The artifacts, not the rule. Measured across all 39 SDE specs under
    ``workspace*/``: 8 carry a ``verification`` block, the two that admit a Milstein
    plan (``cir_feller_violated``, ``ginzburg_landau_s6``) both declare
    ``expected_strong_order_milstein: 1.0``, and the other five say
    ``milstein_eligible: false`` with a reason."""
    admits = 0
    for path in _specs("sde_*"):
        spec = read_json(path)
        assert schema.check_strong_order_declared(spec) == [], path
        if spec.get("milstein_eligible") is not False:
            admits += 1
            assert (spec["verification"]["expected_strong_order_milstein"] > 0), path
    assert admits == 2, admits


# --- prose in an expr field ([REV 66]) ----------------------------------------

def test_prose_in_an_invariant_expr_is_refused():
    """Finding 9's shape, one field over. ``sde_bm_with_drift`` declared
    ``expr: "X ~ Normal(X_0 + mu*T, sigma**2*T)"`` on an ungated ``gaussianity``
    constraint: ``~`` is not Python, so the evaluator reported ``not_reported`` with
    ``SyntaxError: invalid syntax`` on every run, for ever. The spec read as though
    it declared a distributional check and nothing could evaluate one.

    ``operator.is_program``'s docstring uses this very string as its example of text
    that parses as neither an expression nor a program, so the repo already knew the
    shape; nothing refused it at the chokepoint."""
    spec = {"state_dimension": 1, "sde_family": "bm", "verification": {
        "constraints": [{"name": "gaussianity", "gate": False,
                         "expr": "X ~ Normal(X_0 + mu*T, sigma**2*T)"}]}}
    found = schema.check_invariant_expressions(spec)
    assert [f.severity for f in found] == ["error"]
    assert "gaussianity" in found[0].path
    assert "terminal_law" in found[0].message, "the message names the machine-readable home"


def test_a_program_in_an_invariant_expr_is_refused_too():
    """The archived ``pde_heston_2d`` leak was a 2063-character ``def`` in an
    expression field; the same refusal applies here."""
    spec = {"state_dimension": 1, "verification": {"invariants": [
        {"name": "martingale", "expr": "def f(x):\n    return x\n"}]}}
    assert [f.severity for f in schema.check_invariant_expressions(spec)] == ["error"]


def test_a_real_predicate_and_a_martingale_process_both_pass():
    """Both live uses of ``expr``: a boolean predicate, and the martingale *process*
    (which ``_martingale`` evaluates rather than routing to ``_from_expr``). Either
    one is a single expression, which is all this check asks."""
    spec = {"state_dimension": 2, "verification": {
        "constraints": [{"name": "finite", "expr": "np.isfinite(X)", "gate": True},
                        {"name": "support", "expr": "X > 0"}],
        "invariants": [{"name": "martingale", "expr": "np.exp(-mu1*T) * X[..., 0]"},
                       {"name": "no_expr_at_all", "note": "prose belongs here"}]}}
    assert schema.check_invariant_expressions(spec) == []


def test_every_live_spec_carries_only_evaluable_exprs():
    """The artifacts. Measured before the fix: exactly one instance across all 39
    SDE specs and every PDE spec under ``workspace*/``."""
    for path in _specs():
        assert schema.check_invariant_expressions(read_json(path)) == [], path


# --- a tolerance for a check nothing runs ([REV 71]) --------------------------

def test_a_tolerance_on_an_unevaluable_entry_is_flagged():
    """Found while measuring [REV 64]; the same class as [REV 66]. An ungated entry
    that is neither a registry name nor carries an `expr` is legal — the formulator
    guidance offers "leave it, knowingly, as a free-form note" — but a `tol` on it
    says a comparison was intended, and the evaluator reports it `not_reported` for
    ever instead.

    `sde_gbm_2d_high_corr`'s `component_symmetry` is the live case, and its own note
    describes a real defect: *"a mismatch indicates the correlated-increment
    construction was applied asymmetrically (a common Cholesky bug)"*.
    """
    spec = {"state_dimension": 2, "verification": {"invariants": [
        {"name": "component_symmetry", "gate": False, "tol": 0.05,
         "note": "the two marginals are identically distributed"}]}}
    found = schema.check_tolerance_without_a_check(spec)
    assert [f.severity for f in found] == ["warning"], "legal, but it must say so"
    assert "component_symmetry" in found[0].message
    assert "never applied" in found[0].message


@pytest.mark.parametrize("entry", [
    # A registry name: the kernel evaluates it, so the tol is applied.
    {"name": "positivity", "tol": 1e-10},
    # An `expr`: evaluated generically whatever the name says.
    {"name": "whatever_i_like", "tol": 0.05, "expr": "X > 0"},
    # No tol: the entry is a note for the reader and says so by omission.
    {"name": "component_symmetry", "note": "the two marginals agree"},
])
def test_an_applied_tolerance_is_not_flagged(entry):
    assert schema.check_tolerance_without_a_check(
        {"state_dimension": 2, "verification": {"invariants": [entry]}}) == []


def test_the_three_live_entries_this_names_are_warnings_and_not_errors():
    """Measured across all 39 SDE specs and every PDE spec under `workspace*/`:
    three distinct entries, all ungated. They must not break `check-spec`, which is
    why this is a warning — writing the predicate is formulator work with a grade
    attached, since `component_symmetry` failing would take a live plan from 10 to
    the invariants row's 5."""
    flagged = {}
    for path in _specs():
        spec = read_json(path)
        from_this_check = schema.check_tolerance_without_a_check(spec)
        for f in from_this_check:
            assert f.severity != ERROR, path
            flagged[os.path.basename(os.path.dirname(path))] = f.path
        # ...and this check contributes nothing to `check_spec`'s error list, so
        # adding it cannot have broken `check-spec` on any existing spec. (Several
        # PDE specs carry unrelated pre-existing errors, which is not this check's
        # business and is why the assertion is scoped to its own `check` name.)
        mine = [f for f in errors(schema.check_spec(spec))
                if f.check == "tolerance_without_a_check"]
        assert mine == [], path
    assert set(flagged) == {"pde_kdv_zabusky_kruskal", "pde_stefan_1d_similarity",
                            "sde_gbm_2d_high_corr"}, flagged
