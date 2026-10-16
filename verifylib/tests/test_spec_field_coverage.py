"""Every declared spec field has a consumer, or a recorded reason it has none.

Plan §4b [REV 29]: the SDE findings share a shape the audit never named --
**a spec field that is declared and never read**. Six of the fourteen are instances
(``cross_moment``, ``noise_dimension``, ``*_value_at_T``, ``stability_check``,
``empirical_mean``/``empirical_variance``, ``milstein_eligible``), and a gated one
is worse than absent because it reads downstream as evidence that a check passed.

This is a stronger structural guard than rubric parity, which covers one finding.
It catches the next instance too, which is the point: a new field lands with either
a consumer or a line in :data:`UNREAD` saying why not, and nothing lands silently.

[REV 46] adds the other direction -- a metrics key the kernel computes and nothing
reads -- which was live on ``weak_order_out_of_band``.

[REV 65]: "read by the kernel" now means **read by the kernel's code**. Every
search here ran against the raw file text, so a field mentioned once in a docstring
or a ``#:`` comment counted as having a consumer -- and this file is heavily
commented by design, which made the guard weakest exactly where the prose is
densest. Measured on switching: five fields declared in live SDE specs turned out
to have prose mentions and no consumer (``linear``, ``stiff``,
``evaluation_thresholds.required_components``, ``output_contract.returns`` and
``output_contract.imports``), all five genuinely unread and now recorded as such.
``required_components`` is the instructive one: ``_compare``'s docstring names it
while saying, in the same sentence, that the kernel checks every declared moment
regardless -- an accurate description of not reading it, which the text search
could not tell from a consumer.
"""
import ast
import glob
import json
import os
import re

from .conftest import WORKSPACE

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _kernel_files():
    for path in sorted(glob.glob(os.path.join(REPO, "verifylib", "**", "*.py"),
                                 recursive=True)):
        if f"{os.sep}tests{os.sep}" not in path:
            yield path


def _strip_docstrings(tree):
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            body = list(node.body)
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                # A function cannot have an empty body, so the docstring that was
                # the whole body becomes a `pass`.
                node.body = body[1:] or [ast.Pass()]
    return tree


def _kernel_source():
    """Every non-test **statement** of ``verifylib/``, concatenated.

    [REV 65]: docstrings and comments removed, via ``ast.unparse``, which drops
    comments outright. String *values* survive, because a literal in a dict or a
    comparison is code -- it is how half the kernel names a spec field.
    """
    return "\n".join(ast.unparse(_strip_docstrings(ast.parse(open(p).read())))
                      for p in _kernel_files())


def _declared_keys(node, prefix="", out=None):
    """``{key: {dotted paths it appears at}}`` over one spec."""
    out = {} if out is None else out
    if isinstance(node, dict):
        for key, value in node.items():
            out.setdefault(key, set()).add(prefix + key)
            _declared_keys(value, f"{prefix}{key}.", out)
    elif isinstance(node, list):
        for value in node:
            _declared_keys(value, prefix, out)
    return out


#: Subtrees whose keys are **values**, not field names. ``parameters`` holds the
#: problem's constants and they reach the kernel through an evaluation namespace, so
#: ``kappa`` never appears in kernel source and should not.
VALUE_SUBTREES = ("parameters.", "requirements.")

#: Keys assembled at the call site, which a literal search cannot see.
#: ``reference._moment_observables`` builds ``f"{kind}_{comp}_from"`` and
#: ``f"{kind}_value_at_T"``; ``driver._claimed`` builds ``f"{kind}_expression"``.
CONSTRUCTED = (re.compile(r"^(mean|variance)_[XYZ]?_?(from|expression|value_at_T)$"),
               re.compile(r"^(mean|variance)_[XYZ]$"))

#: Declared, read by nothing, and **why** -- the half of [REV 29] that makes this a
#: guard rather than a freeze. Each entry is a decision, and two of them name the
#: step that will remove them.
UNREAD = {
    # --- guidance for the solver agent, by design -----------------------------
    "implementation_notes": "guidance prose for the solver agent (schema.GUIDANCE_PATHS' class)",
    "output_contract": "guidance prose: the return shape the solver must produce",
    "called_with": "inside output_contract; guidance prose",
    "import_safety": "inside output_contract; guidance prose",
    "step_rounding": "inside output_contract; guidance prose",
    "noise_structure": "a classification for the solver agent to read; the kernel "
                       "reads the coefficients themselves",
    "milstein_correction": "guidance prose: the correction term the solver should write",
    "diffusion_derivative": "guidance prose, paired with milstein_correction",
    "commutative_noise": "a property the solver agent needs for the Levy-area "
                         "question; the kernel's Milstein check reads "
                         "milstein_eligible instead (plan §3 2d)",
    "cholesky_lower": "guidance: how to build the correlated increments. The kernel "
                      "grades the resulting correlation (cross_moment), not the "
                      "construction",
    # --- implemented in substance, not by reading the list --------------------
    "cross_checks":
        "the checks it names are implemented by reference.check_moment_ode_consistency, "
        "which decides what to compare from the declared moment_ode and "
        "analytic_moments rather than from a spec-supplied list of check names",
    # --- not applicable to how the kernel measures -----------------------------
    "T0": "the kernel integrates [0, T]; every live spec declares T0 = 0 and a "
          "nonzero one would have to shift every reference expression too, so "
          "reading it halfway would be worse than not reading it",
    "burn_in": "the §17 stationary route solves the invariant density from the "
               "coefficients analytically, so there is no simulation to burn in. It "
               "is guidance for a solver estimating the invariant law by long "
               "simulation",
    # --- surfaced by [REV 65], when "read" came to mean "read by the code" -----
    "returns": "inside output_contract; guidance prose",
    "imports": "inside output_contract; guidance prose",
    "linear": "a classification for the solver agent and for the reader, like "
              "noise_structure; the kernel reads the coefficient expressions and "
              "measures what they do rather than trusting the label",
    "stiff": "likewise a classification. Stiffness changes which scheme a plan "
             "should choose, which is the plan-creator's decision; the kernel "
             "grades the scheme that was written",
    "required_components":
        "_compare's docstring says why in the same breath as naming it: the field "
        "narrows the set the spec cares about, and the kernel checks every declared "
        "moment of every component regardless. Reading it could only ever make the "
        "comparison weaker than the spec's own declarations",
}

#: A key whose name ends -- or begins -- with one of these is prose by convention.
#: Both ends, because ``sde_ginzburg_landau_s6`` writes ``order_note`` and
#: ``note_on_convergence_study`` and both are paragraphs for a human reader.
NOTE_WORDS = ("note", "notes", "derivation", "reason", "rationale")


def _is_prose(key):
    return any(key == w or key.endswith("_" + w) or key.startswith(w + "_")
               for w in NOTE_WORDS)


def _exempt(key, paths):
    if any(p.startswith(sub) for p in paths for sub in VALUE_SUBTREES):
        return True
    if _is_prose(key) or key in UNREAD:
        return True
    return any(pattern.match(key) for pattern in CONSTRUCTED)


def _census(pattern):
    found, source = {}, _kernel_source()
    for path in sorted(glob.glob(os.path.join(WORKSPACE, pattern, "problem_spec.json"))):
        _declared_keys(json.load(open(path)), out=found)
    return {key: sorted(paths) for key, paths in found.items()
            if not re.search(r"\b" + re.escape(key) + r"\b", source)}


def test_no_sde_spec_field_is_declared_and_unread_without_a_reason():
    """Criterion 11. The failure message is the useful part: a new field that
    nothing reads names itself here, and the fix is a consumer or a line in
    ``UNREAD``."""
    orphans = {k: v for k, v in _census("sde_*").items() if not _exempt(k, v)}
    assert orphans == {}, (
        "declared in a live SDE spec and read by nothing in verifylib/: "
        f"{orphans}. Give it a consumer, or record why it has none in "
        "test_spec_field_coverage.UNREAD")


def test_the_recorded_exemptions_are_all_still_unread():
    """The other direction, so the table cannot go stale. A field that acquires a
    consumer must lose its exemption, or the next reader believes a live check is
    still missing -- which is how ``noise_dimension`` sat in a plan as "the field
    does not exist" while the spec declared it."""
    source = _kernel_source()
    declared = set()
    for path in sorted(glob.glob(os.path.join(WORKSPACE, "*", "problem_spec.json"))):
        declared |= set(_declared_keys(json.load(open(path))))
    stale = sorted(k for k in UNREAD
                   if k in declared and re.search(r"\b" + re.escape(k) + r"\b", source))
    assert stale == [], (f"these are now read by the kernel and must come out of "
                         f"UNREAD: {stale}")


def test_the_six_findings_this_guard_was_written_for_are_closed_or_recorded():
    """The findings [REV 29] generalised, named individually so the guard's own
    coverage is visible rather than implied."""
    source = _kernel_source()
    for field, finding in (("cross_moment", "Finding 3"),
                           ("noise_dimension", "Finding 2"),
                           ("value_at_T", "Finding 6"),
                           ("empirical_mean", "Finding 12"),
                           ("empirical_variance", "Finding 12")):
        assert re.search(r"\b" + field + r"\b", source), (field, finding)
    # The last two closed at step 10, so they must no longer be exempt: the
    # exemption table is the record of what is *still* missing.
    for field in ("stability_check", "milstein_eligible"):
        assert field not in UNREAD, field
        assert re.search(r"\b" + field + r"\b", source), field


def test_no_metrics_key_the_kernel_computes_goes_unread():
    """[REV 46]'s direction. ``crn.guards`` computed ``weak_order_out_of_band`` and
    nothing read it, so the weak order was decorative and its own docstring's
    "flagged rather than fatal" was flagged to nobody."""
    source = _kernel_source()
    for key in ("weak_order_out_of_band", "se_order", "median_order", "roundoff_floor",
                "abstained", "pinned", "skewness", "excess_kurtosis"):
        hits = len(re.findall(r"\b" + key + r"\b", source))
        # [REV 65]: `source` is now statements only, so a docstring mention no
        # longer satisfies this -- which is what it used to do. Two occurrences of
        # a *code* identifier is "written somewhere and read somewhere else".
        assert hits >= 2, (f"{key} is computed in one place and read in none "
                           f"({hits} occurrence(s) outside docstrings)")


def test_a_docstring_mention_is_not_a_consumer():
    """[REV 65]'s own rule, asserted against the thing it was wrong about. Before
    this, ``required_components`` passed every check in this file on the strength of
    one sentence in a docstring that was explaining why the kernel does *not* read
    it."""
    source = _kernel_source()
    # The exact sentence that used to satisfy the search, verbatim from
    # `sde/driver.py:_compare`.
    assert "narrows the set the" not in source, "docstrings still reach the search"
    assert "#:" not in source, "`#:` comments still reach the search"
    # ...while string literals, which are how the kernel names spec fields, survive.
    assert "'expected_strong_order'" in source or '"expected_strong_order"' in source
    assert "milstein_eligible" in source


def test_the_unread_table_names_only_fields_some_live_spec_declares():
    """An exemption for a field no spec carries is dead weight that reads as a known
    gap. Two fields here are PDE-side or stale-spec-side by design, so the assertion
    is scoped to the SDE census this file guards."""
    declared = set()
    for path in sorted(glob.glob(os.path.join(WORKSPACE, "*", "problem_spec.json"))):
        _declared_keys(json.load(open(path)), out=(d := {}))
        declared |= set(d)
    assert set(UNREAD) <= declared, sorted(set(UNREAD) - declared)
