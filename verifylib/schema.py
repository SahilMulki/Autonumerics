"""Spec-shape guards: expression-not-program, the evidence rule, ledger shape.

The spec is the chokepoint (§6a). The archived Heston leak entered at
``problem_spec.json`` -- a 2063-character ``def`` with Gauss-Legendre quadrature
sitting in ``analytic_solution.expression``, plus a ``notes`` line telling the
evaluator to transcribe it verbatim -- and propagated downstream by hand. Block it
at the spec and the copy never happens, which is why these checks live here rather
than on the artifacts that copy from it.

None of this needs judgement. Every check is a parse or a set membership.
"""

from __future__ import annotations

import re

from .findings import Finding, error, warning
from .operator import assert_is_expression, is_program

# --- which fields must hold an evaluable expression --------------------------
#
# A declarative path list rather than hand-written traversal: the shapes are
# irregular (str | list[str] | dict[field -> str]) and new ones keep appearing,
# so the walker handles the shapes and this list only names the paths.
#
# ``*`` matches one path segment (a dict key or a list index). Fields that are
# deliberately *not* here: ``initial_condition`` and ``boundary_conditions.values``,
# which are prose on multi-field systems by design ("u = sin(x)cos(y); v = ..."),
# and ``governing_equation``, which is prose everywhere.
#
# Two classes, because measurement forced the split. ANSWER_KEY paths are read and
# evaluated by the pipeline, so they must parse -- no exceptions. GUIDANCE paths are
# written *for the solver agent to read*, and on vector-valued problems the useful
# content is genuinely prose: ``sde_multichannel_stiff_m13``'s drift is
# ``"X @ F.T  (row-major paths, X shape (num_paths, 2), F = [[F11, F12], ...])"``,
# which describes a matrix action a scalar expression cannot express. Requiring
# those to parse hard-fails 3 of 6 real SDE specs on correct content.
#
# What §4a is actually defending against is a *program* smuggled into a field the
# evaluator is instructed to transcribe. So guidance fields keep that half of the
# check and drop the other: prose is allowed, a program is not, anywhere.

ANSWER_KEY_PATHS = (
    "analytic_solution.expression",
    "analytic_solution.fields.*",
    "analytic_moments.mean_expression",
    "analytic_moments.variance_expression",
    "analytic_moments.mean_expression.*",
    "analytic_moments.variance_expression.*",
    "analytic_moments.mean_X",
    "analytic_moments.mean_Y",
    "analytic_moments.variance_X",
    "analytic_moments.variance_Y",
    "verification.mms_probe.exact",
    "verification.mms_probe.exact.*",
    "verification.mms_probe.source",
    "verification.mms_probe.source.*",
    "verification.mms_probe.operator_check",
    "verification.degenerate_limit.exact",
    "verification.degenerate_limit.exact.*",
    "verification.residual_operator",
    "verification.moment_ode.rhs.*",
    "verification.moment_ode.initial.*",
    "verification.moment_ode.mean_from",
    "verification.moment_ode.variance_from",
    "verification.moment_ode.mean_from.*",
    "verification.moment_ode.variance_from.*",
    "verification.moment_ode.cross_from",
    "verification.cross_moment.expression",
    "verification.cross_moment.correlation_of_terminal",
    "verification.dynkin_test_functions.*",
    # D3 / [REV 20]: the machine-readable generator goes under its own **evaluated**
    # path rather than promoting a GUIDANCE field. `drift_expression` and
    # `diffusion_matrix_expression` are classified as guidance precisely because a
    # program there is a leak vector and prose there is normal and correct -- on a
    # vector problem the useful content genuinely is prose ("X @ F.T (row-major
    # paths...)"). Making those evaluated would hard-fail 3 of 6 real SDE specs on
    # correct content, which is the shape `check_expressions` exists to refuse.
    # Here `check_expressions` can police the generator exactly as it polices
    # `verification.residual_operator`, and the prose fields stay untouched.
    "verification.generator.drift",
    "verification.generator.drift.*",
    "verification.generator.diffusion_matrix.*.*",
    "verification.operator.source",
    "verification.operator.terms.*",
    "verification.operator.equations.*.source",
    "verification.operator.equations.*.terms.*",
)

#: Read by an agent, not evaluated by the pipeline. A program here is still a leak
#: vector; prose here is normal and correct.
GUIDANCE_PATHS = (
    "drift_expression",
    "drift_expression.*",
    "diffusion_expression",
    "diffusion_expression.*",
    "diffusion_derivative_expression",
    "diffusion_matrix_expression",
)

#: ``verification.constraints`` / ``invariants`` straddle the line: an entry with
#: ``gate: true`` is evaluated and must parse; an ungated one is a note. Measured
#: over ``workspace/``: all 11 gated entries parse, and the one that does not
#: (``"X ~ Normal(X_0 + mu*T, sigma**2*T)"``) is ungated -- a distributional
#: statement, not a check.
_GATED_EXPR_KEYS = ("constraints", "invariants")


def _children(node):
    if isinstance(node, dict):
        return node.items()
    if isinstance(node, list):
        return enumerate(node)
    return ()


def _collect(node, segments, prefix, out):
    if not segments:
        if isinstance(node, str):
            out.append((prefix, node))
        return
    head, rest = segments[0], segments[1:]
    for key, child in _children(node):
        if head == "*" or head == str(key):
            _collect(child, rest, f"{prefix}.{key}" if prefix else str(key), out)


def _match(spec, patterns) -> list[tuple[str, str]]:
    seen: dict[str, str] = {}
    for pattern in patterns:
        found: list[tuple[str, str]] = []
        _collect(spec, pattern.split("."), "", found)
        for path, text in found:
            seen.setdefault(path, text)
    return sorted(seen.items())


def _gated_expressions(spec) -> list[tuple[str, str]]:
    """``constraints`` / ``invariants`` entries carrying ``gate: true``."""
    out = []
    for key in _GATED_EXPR_KEYS:
        for i, entry in enumerate((spec.get("verification") or {}).get(key) or []):
            if isinstance(entry, dict) and entry.get("gate") and isinstance(entry.get("expr"), str):
                out.append((f"verification.{key}[{i}].expr", entry["expr"]))
    return out


def iter_expression_fields(spec) -> list[tuple[str, str]]:
    """Every ``(dotted_path, text)`` the pipeline evaluates, so every one that must
    parse as a single expression."""
    return _match(spec, ANSWER_KEY_PATHS) + _gated_expressions(spec)


def iter_guidance_fields(spec) -> list[tuple[str, str]]:
    """Fields an agent reads. Prose is allowed here; a program is not."""
    return _match(spec, GUIDANCE_PATHS)


def check_expressions(spec) -> list[Finding]:
    """§4a. One line, zero judgement, no false negatives."""
    out = []
    for path, text in iter_expression_fields(spec):
        try:
            assert_is_expression(text, field=path)
        except ValueError as exc:
            out.append(error("expression", path, str(exc)))
    for path, text in iter_guidance_fields(spec):
        if is_program(text):
            out.append(error(
                "expression", path,
                f"a {len(text)}-character program in a field an agent is told to read. "
                "This is the shape the archived Heston leak took: executable code parked "
                "in a spec field and transcribed downstream by hand",
            ))
    return out


# --- the quote route ---------------------------------------------------------
#
# §4b's fallback: where validation cannot run, a verbatim quote from problem.md is
# accepted in its place. Two things make a quote worth anything -- it has to be
# present, and it has to actually contain a formula. Prose describing a solution
# is not a quote of one.

#: A quoted source text has to carry arithmetic. These are the cheapest tells that
#: something is a formula rather than a sentence about a formula.
_FORMULA_TOKENS = re.compile(r"[-+*/^=]|\bexp\b|\bsin\b|\bcos\b|\berf\b|\blog\b|\bsqrt\b")

#: Prose posing as a formula, in a field the schema requires to be evaluable.
#: These all parse as Python names or calls, so ast.parse does not catch them.
PROSE_BLOCKLIST = (
    "unknown",
    "none",
    "n_a",
    "tbd",
    "see_notes",
    "see_above",
    "standard",
    "known",
    "as_given",
    "omitted",
)


def check_source_text(spec) -> list[Finding]:
    """Validate the quote when one is supplied. Not *requiring* one is deliberate:
    whether a quote is needed depends on the reference check's outcome, which
    ``reference.py`` decides and ``gate.py`` combines."""
    out = []
    sol = spec.get("analytic_solution")
    if not isinstance(sol, dict):
        return out
    text = sol.get("analytic_solution_source_text") or spec.get("analytic_solution_source_text")
    if text is None:
        return out
    path = "analytic_solution.analytic_solution_source_text"
    if not isinstance(text, str) or not text.strip():
        out.append(error("evidence", path, "quote route declared but the source text is empty"))
    elif not _FORMULA_TOKENS.search(text):
        out.append(error(
            "evidence", path,
            "the quoted source text contains no formula tokens -- a sentence describing "
            "a solution is not a quotation of one",
        ))
    return out


def check_prose_as_formula(spec) -> list[Finding]:
    """An expression field holding a placeholder word. ``ast.parse`` accepts these
    (they are valid Python names); nothing downstream can evaluate them."""
    out = []
    for path, text in iter_expression_fields(spec):
        if text.strip().lower().replace(" ", "_").replace("/", "_") in PROSE_BLOCKLIST:
            out.append(error("expression", path, f"placeholder text {text!r}, not an expression"))
    return out


# --- verification.operator (§4d) ---------------------------------------------

def _operator_shape_findings(op, path) -> list[Finding]:
    out = []
    if not isinstance(op, dict):
        return [error("operator", path, f"must be an object or null, got {type(op).__name__}")]
    fields = op.get("fields")
    if not isinstance(fields, list) or not all(isinstance(f, str) for f in fields):
        out.append(error("operator", f"{path}.fields", "must be a list of field names"))
    has_terms, has_equations = "terms" in op, "equations" in op
    if has_terms == has_equations:
        out.append(error(
            "operator", path,
            "declare exactly one of 'terms' (scalar) or 'equations' (system)",
        ))
        return out
    if has_terms:
        if not isinstance(op["terms"], dict) or not op["terms"]:
            out.append(error("operator", f"{path}.terms", "must be a non-empty object of "
                                                          "{term_name: expression}"))
    else:
        eqs = op["equations"]
        if not isinstance(eqs, dict) or not eqs:
            out.append(error("operator", f"{path}.equations", "must be a non-empty object keyed "
                                                              "by field name"))
        else:
            for name, eq in eqs.items():
                if not isinstance(eq, dict) or not isinstance(eq.get("terms"), dict):
                    out.append(error("operator", f"{path}.equations.{name}.terms",
                                     "each equation needs a 'terms' object"))
            if isinstance(fields, list):
                missing = [f for f in fields if f not in eqs]
                if missing:
                    out.append(error("operator", f"{path}.equations",
                                     f"no equation for declared field(s) {missing}"))
        combine = op.get("combine", "rms")
        if combine not in ("rms", "max"):
            out.append(error("operator", f"{path}.combine", f"unknown combine {combine!r}; "
                                                            "expected 'rms' or 'max'"))
    return out


def check_operator_declaration(spec) -> list[Finding]:
    """§4d. Required on every PDE spec, both paths -- with or without a closed form.

    A spec that genuinely cannot express its operator (``fractional_diffusion``'s
    Caputo derivative) declares ``"operator": null`` with a reason. That is a
    recorded gap; an absent field is a silent one, and the difference is the whole
    point of requiring it.
    """
    if not is_pde(spec):
        return []
    verification = spec.get("verification")
    if not isinstance(verification, dict):
        return [error("operator", "verification", "PDE specs must carry a verification block")]
    if "operator" not in verification:
        # Both legacy sources, because reference.resolve_operator consults both. An
        # earlier version looked only at operator_check and so errored on
        # pde_anisotropic_diffusion, whose residual_operator the reference check uses
        # to validate it at 2.5e-10 -- a gate contradicting the check it gates for.
        legacy = ((verification.get("mms_probe") or {}).get("operator_check")
                  or verification.get("residual_operator"))
        if isinstance(legacy, str) and legacy.strip():
            # §4d: 19 of 22 existing PDE specs carry operator_check in exactly the
            # required form, so the reference check still runs. But operator_check is
            # the *homogeneous* operator -- it has nowhere to put a source term and no
            # per-equation form -- which is why Poisson, Helmholtz, Monge-Ampere and
            # the two systems cannot be checked on this path. A warning, not a gate:
            # these specs predate the field.
            return [warning(
                "operator", "verification.operator",
                "absent; falling back to mms_probe.operator_check. That form cannot carry "
                "a source term or a per-equation system, so a spec with either gets no "
                "reference check. Declare verification.operator instead",
            )]
        return [error(
            "operator", "verification.operator",
            "required on every PDE spec. Declare the residual form "
            "{fields, terms, source} (or {fields, equations, combine} for a system), "
            "or null with verification.operator_note giving the reason it cannot be expressed",
        )]
    op = verification["operator"]
    if op is None:
        if not (verification.get("operator_note") or "").strip():
            return [error("operator", "verification.operator_note",
                          "operator is null, so a reason is required")]
        return []
    return _operator_shape_findings(op, "verification.operator")


# --- the invariant registry (plan-no-closed-form §6) --------------------------

def check_invariant_names(spec) -> list[Finding]:
    """A gated invariant the kernel cannot evaluate is worse than no gate.

    ``_GATED_EXPR_KEYS`` above validates only entries carrying an ``expr``, so
    ``{"name": "physically_reasonable", "gate": true}`` used to pass the schema,
    match nothing in the kernel, and read downstream as a satisfied gate. The
    registry in ``kernel/invariants.py`` is closed and importable, so this is a set
    membership -- and it is an **error**, not a warning, because unlike the operator
    rules there is no legacy population to grandfather: measured over ``workspace/``,
    all 24 gated entries name a registry entry or carry an ``expr``.

    Ungated entries stay free-form notes. Eight distinct names are in use that way
    (``front_position``, ``monotone_in_S``, ...) and requiring an implementation for
    a diagnostic nobody gates on would buy nothing.
    """
    verification = spec.get("verification")
    if not isinstance(verification, dict):
        return []
    # Imported here rather than at module scope: this module must keep importing on
    # an interpreter without numpy (E32), and a lazy import inside the one function
    # that needs the registry is cheaper to reason about than a try/except at the top.
    from .kernel.invariants import INVARIANTS, gateable

    out = []
    for key in _GATED_EXPR_KEYS:
        for i, entry in enumerate(verification.get(key) or []):
            if not isinstance(entry, dict) or not entry.get("gate"):
                continue
            if gateable(entry):
                continue
            out.append(error(
                "invariant", f"verification.{key}[{i}].name",
                f"gated invariant {entry.get('name')!r} is not in the kernel registry "
                f"and carries no 'expr'. A gate nothing implements reads as evidence "
                f"that a check passed. Known names: {sorted(INVARIANTS)}; or supply an "
                f"evaluable 'expr'; or set gate: false and keep it as a diagnostic",
            ))
    return out


# --- the noise dimension (plan §3 Phase 2a) ----------------------------------

def check_noise_dimension(spec) -> list[Finding]:
    """A multi-component SDE must declare how many noise channels it has.

    ``sandbox._crn_increments`` already honours the number -- ``shape = (paths,
    fine) if not crn.get("m") else (paths, fine, m)`` -- and ``ladder_spec`` already
    takes it. Nothing passed it, and nothing validated it, so the CRN ladder handed
    a ``(paths, Nt)`` array to a solver following ``solver-sde.md``'s
    ``(num_paths, Nt, m)`` contract. Measured against the real
    ``sde_gbm_2d_high_corr`` plan: ``IndexError: too many indices for array``, and
    the kernel reported it as *"the CRN level at dt=0.02 crashed"* -- a solver defect
    in a review, on a correct solver.

    **It is not derivable, which is why it is required rather than defaulted.**
    ``noise_dimension = state_dimension if drift is a list else 1`` fails in both
    directions on this very set: ``sde_multichannel_stiff_m13`` has 2 states, a
    *string* drift and **13** channels, so the rule gives 1 and it still crashes;
    both oscillators are ``dX = Y dt; dY = -X dt + 0.3 dW`` with 2 states and **1**
    channel, so the rule gives 2 and would hand a correct solver an array of the
    wrong shape -- converting a working plan into a crash. Noise dimension is a
    property of the equation that only the formulator knows.

    Scoped to ``state_dimension > 1`` because that is where a mismatch can exist: a
    scalar state takes a scalar channel and ``m=None`` is already right. An
    **error** rather than a warning for the reason ``check_invariant_names`` gives:
    there is no legacy population to grandfather -- of the two specs in the repo
    that mention the field, one declares ``13`` and the other declares ``null``, and
    the ``null`` is itself the formulator gap.
    """
    if is_pde(spec):
        return []
    state = spec.get("state_dimension")
    if not isinstance(state, int) or isinstance(state, bool) or state <= 1:
        return []
    declared = spec.get("noise_dimension")
    if isinstance(declared, int) and not isinstance(declared, bool) and declared >= 1:
        return []
    return [error(
        "noise_dimension", "noise_dimension",
        f"state_dimension is {state} and noise_dimension is {declared!r}. The CRN "
        f"ladder builds the Brownian increments from it -- (num_paths, Nt) for a "
        f"single channel, (num_paths, Nt, m) for m of them, which is "
        f"solver-sde.md's contract -- so an undeclared one hands the solver an "
        f"array of the wrong rank and the crash is reported against the solver. It "
        f"cannot be inferred: sde_multichannel_stiff_m13 has 2 states and 13 "
        f"channels, and sde_stochastic_oscillator has 2 states and 1. Declare the "
        f"number of independent Brownian channels the equation drives",
    )]


def check_strong_order_declared(spec) -> list[Finding]:
    """An SDE spec with a ``verification`` block must declare its strong-order floor.

    Criterion 7, and the thing that keeps D2's ``kernel_default`` branch a genuine
    fallback rather than the normal case. ``crn.expected_orders`` defaults to **0.5**
    when nothing is declared, which is Euler-Maruyama's order under *multiplicative*
    noise and wrong by a factor of two whenever the noise is additive: six of the
    stale specs are additive or noise-free (``sde_bm_standard``,
    ``sde_linear_additive``, ``sde_ou_stiff``, both oscillators,
    ``sde_quintic_random_ic``), where Euler-Maruyama has strong order **1**.

    Under step 7's 1c -- the order study gating on every certifying plan -- a
    defaulted 0.5 against a measured 1.0 is a two-sided miss of 0.5, and a correct
    solver loses two points for a field nobody filled in. D2 makes the band
    one-sided in exactly that case so the default cannot hard-fail anyone; this
    check is the other half, so the one-sided path stays rare.

    Scoped to specs that carry a ``verification`` block: a spec without one has not
    been formulated for this kernel at all and the finding would be noise.

    **The per-family half is enforced too**, since [REV 63]. Criterion 7 reads
    "every spec declares a floor for every scheme family its plans use", and until
    that revision ``expected_strong_order_milstein`` appeared only as *prose inside
    this error's own message* -- so a spec with a Milstein plan and no
    ``_milstein`` key passed the check and then had its Milstein plan judged
    against the Euler floor. That is not hypothetical: it is how
    ``sde_cir_feller_violated``'s floor came to be fixed by hand (0.5 -> 1.0)
    rather than by a check, and ``sde_ginzburg_landau_s6``'s own note already
    spells the failure out -- *"do not fail a Milstein plan for measuring 1.0
    against the 0.5 default"*.

    The spec cannot see the plans, so the trigger is the spec's own statement about
    whether such a plan is possible: ``milstein_eligible: false`` says no Milstein
    plan can exist here (either the correction vanishes identically, in which case
    the Euler floor is the right one, or the Levy area blocks it, in which case
    ``driver._milstein_claim`` caps the plan), and anything else says one can.
    Measured across all 39 SDE specs in ``workspace*/``: 8 carry a ``verification``
    block, and every one of them already satisfies this.
    """
    if is_pde(spec):
        return []
    verification = spec.get("verification")
    if not isinstance(verification, dict) or not verification:
        return []
    value = verification.get("expected_strong_order")
    if not (isinstance(value, (int, float)) and not isinstance(value, bool)
            and value > 0):
        return [error(
            "expected_strong_order", "verification.expected_strong_order",
            f"absent or not a positive number ({value!r}). The kernel defaults to 0.5, "
            f"which is Euler-Maruyama's order under multiplicative noise and wrong by a "
            f"factor of two under additive noise, so the order study then judges a "
            f"correct solver against the wrong floor. Declare the strong order the "
            f"scheme family this problem expects actually has, and add "
            f"expected_strong_order_milstein beside it wherever a Milstein plan is "
            f"possible",
        )]
    return _check_family_order_floors(spec, verification)


#: ``{family order key: what makes it required}``. Mirrors
#: ``kernel.sde.crn.FAMILY_ORDER_KEY``, which is the consumer: that mapping says
#: which key the order study *reads* for a given ``scheme_family``, and this one
#: says when the spec has to have written it. Kept here rather than imported so
#: ``schema`` stays free of the kernel, and held to the consumer by
#: ``test_schema``'s parity assertion.
FAMILY_ORDER_REQUIRED = {
    "expected_strong_order_milstein": (
        "milstein_eligible",
        "a Milstein plan is possible on this problem. Without this key the order "
        "study judges it against expected_strong_order -- Euler-Maruyama's floor -- "
        "and a correct Milstein solver measuring 1.0 against a 0.5 floor is a "
        "two-sided miss of 0.5. Set milstein_eligible: false with a reason if no "
        "Milstein plan can exist here",
    ),
}


def check_tolerance_without_a_check(spec) -> list[Finding]:
    """A declared ``tol`` on an entry nothing can evaluate.

    [REV 71], found while measuring [REV 64] and the same class as [REV 66]. An
    ungated invariant that is neither a registry name nor carries an ``expr`` is
    **legal**: ``agents/formulator.md`` explicitly offers "leave it, knowingly, as a
    free-form note for the reader". A ``tol`` on it is the tell that comparison was
    intended anyway, and the evaluator reports those entries ``not_reported`` for
    ever while their notes describe real checks:

    * ``sde_gbm_2d_high_corr``'s ``component_symmetry`` (``tol: 0.05``) — *"a
      mismatch indicates the correlated-increment construction was applied
      asymmetrically (a common Cholesky bug)"*;
    * ``pde_kdv_zabusky_kruskal``'s ``kdv_third_invariant`` (``tol: 0.001``);
    * ``pde_stefan_1d_similarity``'s ``enthalpy_balance`` (``tol: 0.02``).

    A **warning**, not an error, for two reasons. The knowingly-unchecked note stays
    legal, which is the formulator's call to make; and writing the predicate is
    formulator work with a grade attached — ``component_symmetry`` is ungated, so an
    ``expr`` that failed would take that plan from 10 to the invariants row's 5. The
    warning names the entry so that decision gets made rather than defaulted.

    Measured across all 39 SDE specs and every PDE spec in ``workspace*/``: three
    entries, the three above.
    """
    # Imported here for `check_invariant_names`' reason: this module must keep
    # importing on an interpreter without numpy.
    from .kernel.invariants import INVARIANTS

    out = []
    for holder, label in ((spec.get("verification"), "verification."), (spec, "")):
        if not isinstance(holder, dict):
            continue
        for group in ("invariants", "constraints"):
            for i, entry in enumerate(holder.get(group) or []):
                if not isinstance(entry, dict) or entry.get("tol") is None:
                    continue
                name = entry.get("name")
                if name in INVARIANTS or isinstance(entry.get("expr"), str):
                    continue
                out.append(warning(
                    "tolerance_without_a_check", f"{label}{group}[{i}].tol",
                    f"{name!r} declares tol {entry['tol']!r} and is neither a name in "
                    f"the invariant registry nor carries an `expr`, so the evaluator "
                    f"reports it not_reported on every run and the tolerance is never "
                    f"applied. Give it an `expr` (a single expression, which may "
                    f"reduce to one boolean), map it onto a registry name, or drop "
                    f"the `tol` to say the entry is a note for the reader",
                ))
    return out


def _check_family_order_floors(spec, verification) -> list[Finding]:
    """[REV 63]: one floor per scheme family the problem admits."""
    out = []
    for key, (gate, why) in FAMILY_ORDER_REQUIRED.items():
        if spec.get(gate) is False:
            continue
        value = verification.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
            continue
        out.append(error(
            "expected_strong_order_family", f"verification.{key}",
            f"absent or not a positive number ({value!r}), and {gate} is "
            f"{spec.get(gate)!r} -- {why}",
        ))
    return out


def check_invariant_expressions(spec) -> list[Finding]:
    """A declared ``expr`` on an invariant or a constraint is a single expression.

    [REV 66], and Finding 9's shape one field over. ``sde_bm_with_drift`` declares

        {"name": "gaussianity", "expr": "X ~ Normal(X_0 + mu*T, sigma**2*T)"}

    which is **prose**: ``~`` is not Python, so the evaluator reports
    ``not_reported`` with ``SyntaxError: invalid syntax`` on every run, for ever.
    The spec reads as though it declares a distributional check and nothing can
    evaluate one -- which is exactly what ``check_stability_check`` refuses for
    ``evaluation_thresholds.stability_check``, and the reason that check exists.

    Two shapes are refused, both already named elsewhere in this repo:
    :func:`operator.is_program`'s docstring uses *this very string* as its example
    of text that "parses as neither" an expression nor a program, and a multi-line
    program in an expression field is the archived ``pde_heston_2d`` leak.

    An ungated prose entry costs no score, so this is about the reader: a permanent
    ``not_reported`` in the metrics teaches whoever reads them to skip the field,
    and the one that matters is the gated one two lines below it. Measured across
    all 39 SDE specs and every PDE spec in ``workspace*/``: one instance, the one
    above.
    """
    out = []
    groups = [("verification.{}", (spec.get("verification") or {}))]
    groups.append(("{}", spec))
    for template, holder in groups:
        if not isinstance(holder, dict):
            continue
        for group in ("invariants", "constraints"):
            for i, entry in enumerate(holder.get(group) or []):
                if not isinstance(entry, dict):
                    continue
                expr = entry.get("expr")
                if not isinstance(expr, str) or not expr.strip():
                    continue
                try:
                    assert_is_expression(expr, field="expr")
                except ValueError as exc:
                    where = template.format(group)
                    out.append(error(
                        "invariant_expression",
                        f"{where}[{i}].expr ({entry.get('name') or '?'})",
                        f"{exc}. The evaluator reports this not_reported with a "
                        f"SyntaxError on every run, so the spec reads as declaring a "
                        f"check that nothing can evaluate. Write a predicate the "
                        f"evaluator can apply, declare the claim in a field that is "
                        f"machine-readable (verification.terminal_law states a "
                        f"distributional law), or move the sentence to `note`",
                    ))
    return out


def check_stability_check(spec) -> list[Finding]:
    """``evaluation_thresholds.stability_check`` is an object or ``null``, never prose.

    Finding 9 / plan §3 5a, per D5. ``project_manual.md`` defines three values --
    ``{"type": "finite"}``, ``{"type": "domain", "abs_bound": b}`` and ``null`` --
    and the one stale spec that carries the field wrote a **sentence** in it. A
    string there is not a check the kernel can evaluate, and silently ignoring it is
    the fails-open shape the finding is about: the spec reads as though it declares a
    stability criterion and nothing measures one.

    ``null`` stays legal and means "not a stability problem", which is what all seven
    live specs declare.
    """
    if is_pde(spec):
        return []
    thresholds = spec.get("evaluation_thresholds")
    if not isinstance(thresholds, dict) or "stability_check" not in thresholds:
        return []
    value = thresholds["stability_check"]
    if value is None or isinstance(value, dict):
        out = []
        if isinstance(value, dict) and value.get("type") not in (None, "finite", "domain"):
            out.append(error(
                "stability_check", "evaluation_thresholds.stability_check.type",
                f"{value.get('type')!r} is not a type the kernel implements. "
                f"project_manual.md defines 'finite' and 'domain'"))
        return out
    return [error(
        "stability_check", "evaluation_thresholds.stability_check",
        f"is {type(value).__name__} ({value!r:.60}). It must be null or an object: "
        f'{{"type": "finite"}} or {{"type": "domain", "abs_bound": b}}. On a spec '
        f"with has_analytic_solution: false this field *is* the scoreable criterion "
        f"-- the moment tolerances do not apply -- so prose here leaves the problem "
        f"with nothing to grade",
    )]


# --- the chaotic flag (plan-no-closed-form §8a) ------------------------------

def check_chaotic(spec) -> list[Finding]:
    """``chaotic: true`` waives the Tier-C order check, so it is a self-relaxation
    vector and needs the same treatment as any other.

    The consumer already shipped -- ``reference.check_reference`` short-circuits on
    it (guardrails §14 C9) -- with no producer side at all: it appears in no spec,
    in no template, and in ``formulator.md`` nowhere. This is the producer side.

    Two requirements, both because of what the flag *buys*. It must be quoted from
    ``problem.md`` through the ledger, because a formulator that can set it escapes
    Tier C entirely; and ``chaotic_T_ref`` must be **declared**, because an
    evaluator choosing the predictable window at scoring time chooses its own bar.
    """
    out = []
    chaotic = spec.get("chaotic")
    if chaotic is not None and not isinstance(chaotic, bool):
        out.append(error("chaotic", "chaotic",
                         f"must be true or false, got {type(chaotic).__name__}"))
        return out
    t_ref = spec.get("chaotic_T_ref")
    if not chaotic:
        if t_ref is not None:
            out.append(error("chaotic", "chaotic_T_ref",
                             "declared without chaotic: true. The reference window only "
                             "means something when the order check has been waived"))
        return out

    if not isinstance(t_ref, (int, float)) or isinstance(t_ref, bool) or t_ref <= 0:
        out.append(error(
            "chaotic", "chaotic_T_ref",
            "chaotic: true waives the Tier-C order check at full T, so the horizon "
            "Tier C runs at instead must be declared here -- inside the predictable "
            "window, and chosen by the formulator rather than by the evaluator at "
            "scoring time"))
    else:
        horizon = (spec.get("time_interval") or {}).get("T")
        if isinstance(horizon, (int, float)) and float(t_ref) > float(horizon):
            out.append(error("chaotic", "chaotic_T_ref",
                             f"{t_ref} exceeds the problem's own horizon T = {horizon}"))

    reqs = [r for r in (spec.get("requirements") or []) if isinstance(r, dict)]
    if not any("chaotic" in (r.get("spec_path") or "") and (r.get("quote") or "").strip()
               for r in reqs):
        out.append(error(
            "chaotic", "requirements",
            "chaotic: true waives a scoring gate, so it needs a requirements ledger "
            "entry naming 'chaotic' in its spec_path and quoting the clause of "
            "problem.md that establishes it. Without the quote the waiver rests on "
            "the formulator's own judgement, which is exactly what the ledger exists "
            "to check"))
    return out


def check_unshadowable(spec) -> list[Finding]:
    """``unshadowable: true``: the reporting horizon is past what any resolution
    shadows, so the kernel reports ``converged: unshadowable`` (capped at 8) when
    the reference-horizon ladder converges and the full-``T`` pair difference
    does not, instead of sending the solver off to refine.

    Declared, never inferred (findings F1 [rev2]): "the full-T differences do not
    shrink at these grids" is not the fact "cannot shrink at any grid" -- an
    under-resolved scheme on a coarse ladder shows the same signature, and there
    refinement is the whole answer. So the flag is a self-relaxation vector with
    the same gate as ``chaotic``: it needs ``chaotic: true`` (and so a
    ``chaotic_T_ref``), and a ledger entry naming it with a quote.
    """
    out = []
    flag = spec.get("unshadowable")
    if flag is None:
        return out
    if not isinstance(flag, bool):
        out.append(error("chaotic", "unshadowable",
                         f"must be true or false, got {type(flag).__name__}"))
        return out
    if not flag:
        return out
    if spec.get("chaotic") is not True:
        out.append(error(
            "chaotic", "unshadowable",
            "declared without chaotic: true. An unshadowable horizon is a chaotic "
            "problem's horizon; the reference-horizon ladder it presupposes only "
            "exists under the chaotic flag"))
    reqs = [r for r in (spec.get("requirements") or []) if isinstance(r, dict)]
    if not any("unshadowable" in (r.get("spec_path") or "")
               and (r.get("quote") or "").strip() for r in reqs):
        out.append(error(
            "chaotic", "requirements",
            "unshadowable: true loosens what `converged` means at t_final, so it needs "
            "a requirements ledger entry naming 'unshadowable' in its spec_path and "
            "quoting the clause of problem.md that establishes the horizon is past "
            "any shadowing time. An inference may loosen a test only when a person "
            "has signed it"))
    return out


def check_graded_n(spec) -> list[Finding]:
    """``evaluation_thresholds.graded_N``: the resolution the statement grades at.

    Findings F4: a stated grading resolution is a *requirement* (``kind:
    evaluation``), not an ambiguity, and the kernel reads this key to know which
    ladder level it is certifying when the ladder is not built from it.
    """
    thresholds = spec.get("evaluation_thresholds") or {}
    if not isinstance(thresholds, dict) or "graded_N" not in thresholds:
        return []
    value = thresholds.get("graded_N")
    if value is None:
        return []
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0 \
            or int(value) != value:
        return [error("threshold", "evaluation_thresholds.graded_N",
                      f"must be a positive integer resolution, got {value!r}")]
    return []


# --- requirements ledger shape (§5) ------------------------------------------

LEDGER_KINDS = frozenset({
    "equation", "parameter", "domain", "initial_condition",
    "boundary_condition", "evaluation", "method", "other",
})
LEDGER_STATUSES = frozenset({"mapped", "ambiguous", "dropped"})


def check_ledger(spec) -> list[Finding]:
    """Shape only. Whether the *quote* is verbatim is not machine-checkable here;
    whether the solver honoured it is ``audit.py``."""
    reqs = spec.get("requirements")
    if not isinstance(reqs, list) or not reqs:
        return [error("ledger", "requirements", "a requirements ledger is required on every spec")]
    out = []
    for i, entry in enumerate(reqs):
        p = f"requirements[{i}]"
        if not isinstance(entry, dict):
            out.append(error("ledger", p, "each entry must be an object"))
            continue
        rid = entry.get("id")
        p = f"requirements[{rid or i}]"
        if not rid:
            out.append(error("ledger", p, "missing 'id'"))
        if not (entry.get("quote") or "").strip():
            out.append(error("ledger", p, "missing 'quote' -- the verbatim clause from problem.md"))
        kind, status = entry.get("kind"), entry.get("status")
        if kind not in LEDGER_KINDS:
            out.append(error("ledger", p, f"kind {kind!r} not one of {sorted(LEDGER_KINDS)}"))
        if status not in LEDGER_STATUSES:
            out.append(error("ledger", p, f"status {status!r} not one of {sorted(LEDGER_STATUSES)}"))
        if status in ("mapped", "ambiguous") and not (entry.get("spec_path") or "").strip():
            out.append(error("ledger", p, f"status {status!r} requires a spec_path"))
        if status in ("ambiguous", "dropped") and not (entry.get("note") or "").strip():
            out.append(error("ledger", p, f"status {status!r} requires a note saying why"))
    return out


def check_operator_ledgered(spec) -> list[Finding]:
    """§4g's common-mode mitigation, and the only thing that catches it.

    The formulator writes the analytic solution *and* the operator declaration in
    one pass from one reading. If it believes the equation is ``u_t = alpha*u_x``
    it writes both consistently and the residual is ~0 -- the reference check
    cannot see this, because both of its inputs are wrong the same way. The ledger
    is the only place the operator is compared against ``problem.md`` itself, so an
    equation entry must point at the operator declaration.

    A warning, not a gate: 19 of 22 existing PDE specs predate the rule, and a
    false hard-fail here is invisible as a false positive (§5b).
    """
    if not is_pde(spec):
        return []
    reqs = [r for r in (spec.get("requirements") or []) if isinstance(r, dict)]
    equation_entries = [r for r in reqs if r.get("kind") == "equation"]
    if not equation_entries:
        return [warning("ledger", "requirements",
                        "no 'equation' ledger entry: the governing equation was never quoted "
                        "from problem.md, so verification.operator is unaudited")]
    if not any("operator" in (r.get("spec_path") or "") for r in equation_entries):
        return [warning(
            "ledger", "requirements",
            "no equation ledger entry names verification.operator in its spec_path. The "
            "operator and the analytic solution are written by the same agent in the same "
            "pass, so the ledger is the only check that reaches outside the spec (§4g)",
        )]
    return []


# --- composition -------------------------------------------------------------

def is_pde(spec) -> bool:
    """PDE specs carry spatial variables; SDE specs carry a state dimension."""
    if "spatial_variables" in spec or "spatial_dimension" in spec:
        return True
    return not ("sde_family" in spec or "state_dimension" in spec)


def check_spec(spec) -> list[Finding]:
    """Every static check on a spec. The reference check (§4c/§4e) is not here --
    it evaluates the formula on a grid, which is ``reference.py``."""
    return [
        *check_expressions(spec),
        *check_prose_as_formula(spec),
        *check_source_text(spec),
        *check_operator_declaration(spec),
        *check_invariant_names(spec),
        *check_noise_dimension(spec),
        *check_strong_order_declared(spec),
        *check_stability_check(spec),
        *check_invariant_expressions(spec),
        *check_tolerance_without_a_check(spec),
        *check_chaotic(spec),
        *check_unshadowable(spec),
        *check_graded_n(spec),
        *check_ledger(spec),
        *check_operator_ledgered(spec),
    ]
