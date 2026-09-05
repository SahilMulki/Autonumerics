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
        *check_ledger(spec),
        *check_operator_ledgered(spec),
    ]
