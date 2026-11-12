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
#:
#: ``evaluation_thresholds.functionals`` joined them with F0/F2. Its keys are the
#: problem's own quantity names -- ``price_K1``, ``no_touch_S75`` -- and the kernel
#: reads the whole map generically: ``functionals.declared`` takes every entry,
#: ``functionals._series`` carries each name up the ladder, and the SDE driver calls
#: both. A literal search for ``price_K1`` in kernel source can never succeed and
#: should not; what matters is that *something* consumes the map, and until F0
#: carried ``functionals`` across the SDE sandbox boundary nothing did, which is
#: what nine names in this failure message were telling us.
#: ``verification.structural_facts`` is the same shape one step further: its *keys*
#: are the formulator's own names for the problem's structural features, and the
#: value is prose for the solver agent. `pool.check_boundary_facts` reads the names
#: it knows -- `pool.BOUNDARY_FACTS` -- and deliberately treats the rest as
#: description rather than instruction, because a key that implies a technique and a
#: key that merely classifies the problem are not interchangeable.
#:
#: Entered as a subtree after four runs coined four names in two days:
#: `variance_boundary` (B04), `feller_violated` (B05), `degenerate_boundary` (B07)
#: and, under `functional_notes`, `expected_ordering` (B06). Listing them one at a
#: time makes this test flap on every run without telling anyone anything new; the
#: question worth asking is whether `pool.BOUNDARY_FACTS` should grow, and that is a
#: design decision, not a census entry.
#: ``verification.pathwise_reference.`` and ``verification.pathwise_exact_solution.``
#: are the same decision one suite later, and for a stronger reason than convenience
#: [sciml 10d]. Every sciml formulator that meets a *pathwise* strong error finds no
#: `evaluation_thresholds` key that accepts one (the gap the `pathwise_exact_solution`
#: entry below documents), and writes the reference it derived into a block it coins
#: on the spot. Seven runs have now coined seven spellings -- `W_t`,
#: `terminal_from_increments`, `exact_solution_expression`, `W_definition`,
#: `strong_rel_err_definition`, `strong_rel_err_max`, `measured_reference_points` --
#: inside a `workspace/` directory the next run rewrites. Listing leaves one at a time
#: pins an agent's wording and makes this test fail on every re-run without naming
#: anything new; the finding is the *block*, and the repair is a threshold key for a
#: pathwise strong error, not a census entry per leaf.
VALUE_SUBTREES = ("parameters.", "requirements.",
                  "evaluation_thresholds.functionals.",
                  "verification.structural_facts.",
                  "verification.pathwise_reference.",
                  "verification.pathwise_exact_solution.")

#: Keys assembled at the call site, which a literal search cannot see.
#: ``reference._moment_observables`` builds ``f"{kind}_{comp}_from"`` and
#: ``f"{kind}_value_at_T"``; ``driver._claimed`` builds ``f"{kind}_expression"``.
CONSTRUCTED = (re.compile(r"^(mean|variance)_[XYZ]?_?(from|expression|value_at_T)$"),
               re.compile(r"^(mean|variance)_[XYZ]$"))

#: Declared, read by nothing, and **why** -- the half of [REV 29] that makes this a
#: guard rather than a freeze. Each entry is a decision, and two of them name the
#: step that will remove them.
UNREAD = {
    # --- invented by a formulator, 2026-11-11: the pathwise vocabulary gap -----
    # These three are one finding, not three. The sciml suite grades a *pathwise*
    # relative strong error, `evaluation_thresholds` has no key that accepts such a
    # bound, and both formulators that met the problem said so in their own
    # `ambiguities` (M01's R4/R17, M06's R19/R21: "genuine vocabulary gap"). Having
    # no key, M06's formulator wrote the whole thing into fields it coined -- it even
    # went and *measured* the convergence table itself while writing the spec. So the
    # evidence the kernel needed to rank those plans was computed, written down, and
    # read by nothing. See plan-sciml-benchmark.md §10a; the repair is a threshold key
    # for a pathwise strong error, which is a cross-suite change and not these tests'.
    "pathwise_exact_solution": (
        "sde_sciml_stiffquad_mild, inside analytic_moments: "
        "'X(t) = np.tanh(np.arctanh(X_0) - alpha*t + beta*W(t))'. The closed form "
        "problem.md says exists without printing it, transcribed by the formulator so "
        "the spec would at least *carry* it. Unread because the kernel's analytic "
        "route is moment-based: it evaluates `mean_expression` and "
        "`variance_expression`, and has no consumer for a pathwise map against which "
        "terminal states could be compared path by path. A consumer for this field is "
        "the pathwise-strong-error check the suite wants and the kernel lacks"),
    "mean_check_applies": (
        "sde_sciml_stiffquad_mild, inside analytic_moments: 'abs(mean) = 0.0488 > "
        "near_zero_mean_threshold = 0.01, so the mean tolerance is live and must not "
        "be skipped'. A note to a reader about a decision `compare_moment` already "
        "makes from `near_zero`, so it is a (correct) restatement rather than a "
        "declaration -- unread, and nothing is missing"),
    "measured_reference_errors": (
        "sde_sciml_stiffquad_mild, inside verification: a six-level relative "
        "strong-error table the formulator measured itself at 20 000 paths "
        "(Euler-Maruyama 2.637e-2 down to 1.640e-3, Milstein 2.519e-2 down to "
        "7.799e-4, 'a clean factor of 2.00 per halving, i.e. strong order 1.00'). "
        "This is the most instructive unread field in the registry: it is exactly the "
        "quantity the suite grades on and the quantity no `evaluation_thresholds` key "
        "accepts, so a formulator with nowhere to put it put it in prose. Not to be "
        "consumed as declared -- a solver-adjacent agent's own measurement cannot "
        "grade that agent -- but it is the field that says what key is missing"),
    "terminal_moments_at_T": (
        "sde_sciml_weak_noncommutative, top level: the three graded functionals' "
        "exact values at T (0.049787068367863944, 11.213075341441485, "
        "3668.4138609506863) with the closed forms they came from and a cross-check "
        "against solve_ivp on the moment ODE at 3.8e-15 / 3.7e-12. The fifth "
        "instance of the same gap and the clearest: the problem's own statement "
        "both denies and asserts a terminal-moment reference (the formulator logged "
        "it as R22), so having derived exact references for the quantities the "
        "problem is graded on, it had nowhere declared to put them. The kernel's "
        "functional route takes `functionals[].reference` per functional; a block of "
        "values keyed by functional name is the same information in an undeclared "
        "shape, so the repair here is the prompt and the corpus, not a consumer"),
    # --- invented by a formulator, 2026-11-12: the same gap, four more names ---
    # The containers and threshold keys the subtree exemption above does not cover.
    # One finding with the three fields at the top of this registry, counted here
    # because each is a *declared* key rather than a leaf of a coined block.
    "pathwise_reference": (
        "sde_sciml_cubic and sde_sciml_nltest, under `verification`: the block each "
        "formulator coined to hold the pathwise reference -- the exact solution "
        "expression, the W it is driven by, the strong-error definition, its bound, "
        "and the points the formulator measured. Unread because nothing in the "
        "kernel takes a pathwise reference as a spec field; `benchmark/verify.py` "
        "computes the strong error itself from the problem's own ground truth. The "
        "leaves are exempt as a subtree; this is the name of the block"),
    "pathwise_strong_rel_err_max": (
        "sde_sciml_wave, under `evaluation_thresholds`: the formulator's bound on "
        "the relative strong error, written into the one place a threshold belongs "
        "-- which has no such key, so nothing reads it. The clearest statement yet "
        "of the missing key: a formulator put the right quantity in the right "
        "section under a name it had to invent"),
    "pathwise_strong_rel_err_definition": (
        "sde_sciml_wave, beside the bound above: the prose defining what that bound "
        "is a bound on. Carried for the same reason and read by nothing for the same "
        "reason"),
    "path_values": (
        "sde_sciml_stiffquad_det, under `output_contract`: the per-path terminal "
        "states the contract asks the solver to return. Unread *by this census*, "
        "which scans kernel source only -- `benchmark/verify.py`, `setup.py` and "
        "`runner.py` all read it, so this is a scoping artifact rather than a gap, "
        "and recorded as such so the next reader does not go looking for a consumer "
        "that already exists"),
    # --- invented by a formulator, 2026-11-06 ---------------------------------
    # Two fields the BENCHOP SDE run coined for itself that are not prose and not
    # covered by a VALUE_SUBTREES entry. Recorded rather than consumed, with what
    # each one would be good for, because a consumer is a design decision and this
    # test's job is only to stop them going unnoticed.
    "state_components": (
        "sde_heston_dnt: [\"S\", \"V\"], the state vector's component names. Worth a "
        "consumer: `surrogates.component_of` takes `applies_to_component` as a bare "
        "index and `dynkin._initial_state` reads only X_0/Y_0/Z_0, so a spec that "
        "names its components in domain terms (S_0, V_0) cannot be matched to "
        "positions -- which is how B06 lost its generator check. This field is "
        "exactly the missing map, but using it to reorder anything is the positional "
        "mispairing `component_of` exists to prevent, so it needs a design pass"),
    "expected_ordering": (
        "sde_hhw_call_corr, inside verification.functional_notes: "
        "'price_S75 < price_S100 < price_S125'. This is a *valid* "
        "`functionals.cross_check` claim written under a key the formulator invented, "
        "because `verification.expected_cross_check` was documented nowhere. The "
        "prompt now names that field; this stays until the corpus is re-made"),

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
    # --- a 2-D Kolmogorov route the scalar surrogate declines -------------------
    # The formulator proposed one and wrote its own note calling it a "best-effort
    # 2-D extension of the documented 1-D Kolmogorov route". `surrogates.kolmogorov`
    # is scalar: it refuses a per-dimension `bounds` and refuses any spec whose
    # `state_dimension != 1`, so the whole block is unreachable and these fields
    # have nowhere to be read. Recorded rather than given consumers, because the
    # consumer is a 2-D Kolmogorov solver and that is a feature, not a fix.
    #
    # The finding is now stronger than when it was first written on 2026-10-31. The
    # formulator proposed a 2-D Kolmogorov block on **two consecutive runs of the
    # same problem, under different field names** -- `state_vars` / `read_at` on the
    # Oct-31 run, `evaluate_at` / `functional_names` / `Ny` on the Nov-01 one -- so
    # this exemption churns with every re-run while the underlying waste stays put.
    # Nothing tells the formulator the route cannot be run, so it keeps paying for
    # the design and keeps spelling it differently. The fix is a signal to the
    # formulator, not a line in this table; until then the table tracks whichever
    # spelling the current workspace happens to carry.
    # Retired 2026-11-05, and the churn this predicted is what retired them. The note
    # above said the exemption "churns with every re-run while the underlying waste
    # stays put", and tracked whichever spelling the current workspace carried --
    # `state_vars`/`read_at` on the Oct-31 run, `evaluate_at`/`functional_names`/`Ny`
    # on Nov-01. A third `--fresh` run of the same problem the same afternoon
    # (12-implementation.md §7) proposed **no** 2-D Kolmogorov block at all, so the
    # third spelling aged out like the first two.
    #
    # Which is the useful datum: across three runs the formulator proposed this
    # unreachable route twice and skipped it once, so it is not a stable property of
    # the problem and there is nothing here to exempt. The waste is real when it
    # happens and the fix is still a signal to the formulator that `surrogates.
    # kolmogorov` refuses any spec with `state_dimension != 1`, not a line in this
    # table chasing field names. If a fourth run invents a fourth spelling it will
    # meet the guard and this paragraph.
    # --- the replication knobs: a decision deferred, with its evidence ---------
    # These four are the declared place to configure a second solve at a different
    # path count, and nothing re-solves. Measured 2026-10-29 by re-solving the
    # eleven BENCHOP B01-B04 plans at two further seeds: per-plan harness error
    # carries 9% to 97% relative spread, median ~35%, and on one problem the shipped
    # plan clears the external gate at two seeds of three. So a replication gate is
    # worth building and these are the fields it will read -- but it costs a second
    # graded-scale solve per plan, which is a budget decision and not a bug fix.
    # Retired 2026-11-05, and the argument kept because only the *exemption* went
    # stale. `num_paths_replicate`, `num_paths_secondary` and `replication_check` are
    # the declared place to configure a second solve at a different path count, and
    # nothing re-solves. No live spec declares any of the three any more -- the
    # BENCHOP corpus turned over -- so the entries had become the dead weight this
    # file's own guard test exists to catch: an exemption for a field no spec carries
    # reads as a known gap in a check that is not missing.
    #
    # The evidence has not aged, only the spelling. Measured 2026-10-29 by re-solving
    # the eleven BENCHOP B01-B04 plans at two further seeds: per-plan harness error
    # carries 9% to 97% relative spread, median ~35%, and on one problem the shipped
    # plan clears the external gate at two seeds of three. So a replication gate is
    # still worth building and these are still the fields it will read; it costs a
    # second graded-scale solve per plan, which is a budget decision rather than a bug
    # fix. A formulator that declares them again will meet the guard and reopen the
    # question on this paragraph, which is what the table is for.
    # --- structural facts the kernel reaches by another route -----------------
    # Retired 2026-10-31. `second_call_num_paths` and `attainable_boundary` were
    # declared only by the October Opus `sde_heston_call_feller` spec, which a
    # `--fresh` re-run replaced with one that declares neither -- so both exemptions
    # had become the dead weight this table's own guard test exists to catch. Removed
    # rather than kept as documentation: a formulator that reinvents either should
    # meet the guard and have the question reopened, not find a stale pass waiting.
    # `second_call_num_paths` has a live successor that *is* read --
    # `graded_num_paths`, which the memory projection in `sde/driver.py` consumes.
    # `feller_condition` was retired on the same date and for the same reason: no live
    # spec declares it. Its argument also stands -- it was guidance telling the
    # plan-creator whether the variance process can reach zero and therefore which
    # scheme to propose, and the kernel grades the resulting moments and the declared
    # domain invariant rather than the classification. What the current corpus states
    # instead is `verification.structural_facts.boundary` ("Feller condition violated:
    # 2*kappa*eta = 0.22188 < sigma^2 = 1, V attains zero"), in prose, which is the
    # same fact one field over.
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


#: Entries whose declaring spec lives in ``workspace/`` under a slug a pipeline run
#: regenerates, so the field is present only while that run's spec is staged. They
#: are exempt from the liveness check below: an agent re-run on the same problem
#: coins what it coins, and a recorded reason that outlives its example is still the
#: answer to "why does this field have no consumer" the next time one appears.
REGENERATED = {"pathwise_exact_solution", "mean_check_applies",
               "measured_reference_errors", "terminal_moments_at_T",
               # [sciml 10d] the four the 2026-11-12 runs coined, same reasoning
               "pathwise_reference", "pathwise_strong_rel_err_max",
               "pathwise_strong_rel_err_definition", "path_values"}


def test_the_unread_table_names_only_fields_some_live_spec_declares():
    """An exemption for a field no spec carries is dead weight that reads as a known
    gap. Two fields here are PDE-side or stale-spec-side by design, so the assertion
    is scoped to the SDE census this file guards.

    **[rev, sciml 10a]** ``REGENERATED`` is excluded. The fields in it were coined by
    a formulator inside a `workspace/` spec that the next run on that problem
    rewrites, so pinning their liveness makes this test fail whenever the pipeline is
    re-run and the agent words things differently -- the third stale-census failure
    in this suite in two days, all three from asserting equality against a directory
    that runs mutate (a fourth followed the next day, from M09's
    ``terminal_moments_at_T``). The registry's purpose is the *reason*, and a reason keeps its
    value after its example is overwritten."""
    declared = set()
    for path in sorted(glob.glob(os.path.join(WORKSPACE, "*", "problem_spec.json"))):
        _declared_keys(json.load(open(path)), out=(d := {}))
        declared |= set(d)
    assert (set(UNREAD) - REGENERATED) <= declared, sorted(
        (set(UNREAD) - REGENERATED) - declared)
    # REGENERATED is an exemption list, not a dumping ground: every name in it has to
    # be in the registry it exempts.
    assert REGENERATED.issubset(UNREAD), sorted(REGENERATED - set(UNREAD))


# --- the third direction: read by the ranker, never carried to it -------------
#
# `11-sde-accuracy-fixes.md` §1. The two directions above are about the *kernel*:
# a spec field nothing reads, and a metrics key nothing reads. This one is about
# the reporting boundary between the kernel and selection, and it was open the
# whole time the other two were guarded.
#
# `ranking.py` takes a `summary` dict. The only summary it is ever handed in
# production is `review.parse_metrics(SOLUTION.md)` -- the rendered `<metrics>`
# block, which `metrics.render_block` filters through the `BLOCK_KEYS` whitelist.
# So a key `ranking.py` reads that is not on that whitelist is read from a dict
# that cannot contain it: the component silently returns None and its weight is
# renormalised onto the others.
#
# Measured when this test was written: four keys were in that state --
# `functional_rel_error`, `functional_rel_tol`, `order_band` and `kind`. The first
# two are the whole of F3, so `ranking.accuracy` had never once scored the quantity
# a BENCHOP problem is graded on, on any of the thirty plans on disk; it fell
# through to a state moment instead. It had happened twice, to two separate
# findings' fields, and nothing could see it, because adding a key to one list and
# not the other moves ranks rather than scores and no acceptance criterion reads
# ranks.

#: Keys ``ranking.py`` reads through a variable rather than a literal, which the
#: AST walk below cannot see. One entry, in :func:`ranking.headroom_decades`, which
#: builds ``keys`` and then calls ``summary.get(key)`` in a loop.
RANKING_INDIRECT_READS = ("estimated_rel_error_graded", "estimated_rel_error_T")


def _ranking_summary_reads():
    """Every ``summary`` key :mod:`verifylib.ranking` reads by literal name."""
    path = os.path.join(REPO, "verifylib", "ranking.py")
    with open(path, encoding="utf-8") as fh:
        tree = _strip_docstrings(ast.parse(fh.read()))
    found = set()

    def _name(node):
        return node.id if isinstance(node, ast.Name) else None

    for node in ast.walk(tree):
        # `summary.get("x")`, and the same read off a pool member (`s.get("x")`).
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get" and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
                and _name(node.func.value) in ("summary", "s")):
            found.add(node.args[0].value)
        # `summary["x"]`
        if (isinstance(node, ast.Subscript) and _name(node.value) in ("summary", "s")
                and isinstance(node.slice, ast.Constant)
                and isinstance(node.slice.value, str)):
            found.add(node.slice.value)
    return found


def test_ranking_inputs_are_all_on_the_metrics_block():
    """Every key the ranker reads is a key the block carries.

    The failure this exists to catch is structural and silent: `ranking.py` and
    `BLOCK_KEYS` are two lists that have to agree, and until now nothing made them.
    """
    from verifylib.kernel.metrics import BLOCK_KEYS

    reads = _ranking_summary_reads() | set(RANKING_INDIRECT_READS)
    missing = sorted(reads - set(BLOCK_KEYS))
    assert not missing, (
        "ranking.py reads summary keys that `metrics.BLOCK_KEYS` does not carry, so "
        "they are never in the dict it is handed and the components that read them "
        f"return None on every plan: {missing}. Add them to BLOCK_KEYS (and to "
        "`references/project_manual.md`'s block specification), or stop reading "
        "them.")
    # The walk has to actually find things; a regex-shaped refactor of `ranking.py`
    # that hid every read would make this test vacuously green.
    assert len(reads) >= 18, f"the AST walk found only {len(reads)} reads"


def test_the_indirect_ranking_reads_are_still_indirect():
    """:data:`RANKING_INDIRECT_READS` is a hand-maintained escape hatch, so it needs
    its own guard: an entry that has become a literal read is covered by the walk and
    should be dropped, and an entry naming a key `ranking.py` no longer reads at all
    is dead weight that reads as a known gap."""
    path = os.path.join(REPO, "verifylib", "ranking.py")
    with open(path, encoding="utf-8") as fh:
        source = ast.unparse(_strip_docstrings(ast.parse(fh.read())))
    direct = _ranking_summary_reads()
    for key in RANKING_INDIRECT_READS:
        assert key not in direct, f"{key} is read by literal now; drop the exemption"
        assert re.search(r"\b" + key + r"\b", source), (
            f"{key} is no longer read by ranking.py at all; drop the exemption")


#: The block's second home. ``verification_manual.md`` §2 says the fields are
#: specified **once**, in ``project_manual.md`` -- *The `<metrics>` Block* -- and
#: emitted by ``BLOCK_KEYS``; it says so because the two copies had already drifted
#: once and "a schema with two homes is a schema with none". Saying it was not
#: enough: measured when this test was written, the manual carried 24 of the then-38
#: keys, missing `graded_N`, `mean_rel_err`, `variance_rel_err`, `solver_sha256` and
#: ten others. A prose rule about one list needs a test, like any other.
MANUAL = os.path.join(REPO, "references", "project_manual.md")
_MANUAL_BLOCK_RE = re.compile(
    r"## The `<metrics>` Block.*?```\n<metrics>\n(.*?)\n</metrics>\n```", re.S)


def _manual_block_keys():
    with open(MANUAL, encoding="utf-8") as fh:
        text = fh.read()
    match = _MANUAL_BLOCK_RE.search(text)
    assert match, f"no `<metrics>` example block found in {MANUAL}"
    out = []
    for line in match.group(1).splitlines():
        line = line.split("#", 1)[0].strip()
        if line and ":" in line:
            out.append(line.split(":", 1)[0].strip())
    return out


def test_the_manual_and_block_keys_name_the_same_fields_in_the_same_order():
    """The example block in ``project_manual.md`` *is* the specification, so it has to
    be the whole specification. Order too: the manual is what a reader learns the
    block from, and a field in a different place reads as a different field."""
    from verifylib.kernel.metrics import BLOCK_KEYS

    manual, keys = _manual_block_keys(), list(BLOCK_KEYS)
    assert sorted(set(manual)) == sorted(set(keys)), (
        "`references/project_manual.md`'s `<metrics>` block and "
        "`metrics.BLOCK_KEYS` disagree.\n"
        f"  in BLOCK_KEYS, not in the manual: {[k for k in keys if k not in manual]}\n"
        f"  in the manual, not in BLOCK_KEYS: {[k for k in manual if k not in keys]}")
    assert manual == keys, (
        "same fields, different order; the manual is read top to bottom\n"
        f"  manual: {manual}\n  BLOCK_KEYS: {keys}")
