"""Path A, with defective solvers on it. Plan §4b's "Path-A fixture suite".

``test_kernel_canaries.py`` already runs the *honest* SDE solver through
``sde_spec.json`` -- so the Path-A fixture existed -- and no **defective** solver
was ever run on Path A at all. That is what let Finding 1 survive the life of
`rubric_sde_analytic`: on the path 21 of the 23 SDE problems take, the only thing
measured was whether two terminal moments landed inside their tolerances, so a
solver that drew from the exact terminal law and never integrated the SDE scored
10, certified, provenance ``analytic``.

Every canary here names the finding it pins and the check that must catch it. The
defective ones run at reduced ``num_paths`` (§4d): they fail by orders of magnitude
rather than marginally, so a few thousand paths separates them, and the 50 000-path
configuration is reserved for the honest controls where the margin is the thing
being measured.
"""
import json
import os
import shutil

import pytest

from verifylib.kernel import run

from .test_kernel_canaries import CANARY, _spec


def _plan(tmp_path, solver, frontmatter=None):
    shutil.copy(os.path.join(CANARY, solver), tmp_path / "solver.py")
    if frontmatter:
        (tmp_path / "SOLUTION.md").write_text(f"---\n{frontmatter}\n---\n")
    return str(tmp_path)


def _run(tmp_path, solver, spec="sde_spec.json", frontmatter=None, config=None, **over):
    return run(_spec(spec, **over), _plan(tmp_path, solver, frontmatter), config or {})


# --- Finding 1: the fabrication hole ------------------------------------------

def test_a_fabricated_sampler_does_not_certify_on_path_a(tmp_path):
    """Criterion 1. It scored **10, certified, provenance analytic** -- every moment
    it reports is a sample from the exact law, so no tolerance can see it. Dynkin is
    the only check in the manual that establishes the paths came from the SDE."""
    m = _run(tmp_path, "sde_fabricated_sampler.py",
             **{"evaluation_thresholds.num_paths": 4000})
    assert m["checks"]["moments_ok"] is True, "the moments pass -- that is the point"
    assert m["checks"]["dynkin_ok"] is False, m["detail"].get("dynkin_ok")
    assert m["score"] <= 3, m["certification"]
    assert m["certified"] is False
    # Criterion 1 is about the *reason*, not only the number: before the ceiling fix
    # this cap surfaced as "D1 stalled with a validated operator", PDE vocabulary
    # naming neither Dynkin nor the order study.
    reason = m["certification"]["reason"].lower()
    assert "dynkin" in reason or "order study" in reason, m["certification"]["reason"]


def test_the_honest_solver_still_certifies_with_dynkin_on_path_a(tmp_path):
    """The control for the test above, and criterion 9's named guard: 1b and 1c both
    enabled must leave the honest Path-A solver at 10."""
    m = _run(tmp_path, "sde_honest.py")
    assert m["checks"]["dynkin_ok"] is True, m["detail"].get("dynkin_ok")
    assert m["checks"]["orders_ok"] is True, m["detail"].get("orders_ok")
    assert m["score"] == 10 and m["provenance"] == "analytic"


def test_dynkin_runs_on_a_certifying_path_a_plan_and_not_on_a_failing_one(tmp_path):
    """1b's cost rule. ~7x a graded solve, so it is gated on `would_certify` -- but
    the gate is `would_certify`, not `path == "B"`, which is what §24 had."""
    certifying = _run(tmp_path, "sde_honest.py")
    assert certifying["config"]["run_dynkin"]["value"] is True
    failing = _run(tmp_path, "sde_honest.py", **{"evaluation_thresholds.num_paths": 60})
    assert failing["checks"]["resolved"] is False
    assert failing["config"]["run_dynkin"]["value"] is False
    assert failing["checks"]["dynkin_ok"] == "not_run"


# --- Finding 1 / 1c: the order study gates on Path A --------------------------

def test_the_crn_ladder_runs_on_a_certifying_plan_whatever_the_floor(tmp_path):
    """1c. `run_crn_ladder` carried `and expected_strong_order >= 1.0`, which
    exempted every Euler-Maruyama plan in the set -- measuring a floor of 0.5 and
    getting 0.0 is exactly how a solver that ignores `dW` is caught."""
    m = _run(tmp_path, "sde_honest.py", spec="sde_spec_gbm.json",
             **{"evaluation_thresholds.num_paths": 4000})
    assert m["config"]["expected_strong_order"]["value"] == 0.5
    assert m["config"]["run_crn_ladder"]["value"] is True


def test_a_milstein_plan_with_no_correction_is_capped_by_the_order_study(tmp_path):
    """Criterion 2 and D1. The moments cannot see it -- §24's own argument,
    "Euler-Maruyama and Milstein have the same weak order" -- and the strong order
    reads 0.504 against the spec's declared Milstein floor of 1.0."""
    m = _run(tmp_path, "sde_milstein_no_correction.py", spec="sde_spec_gbm.json",
             frontmatter="scheme_family: milstein")
    assert m["config"]["expected_strong_order"]["value"] == 1.0
    assert m["summary"]["order_band"] == "two-sided"
    assert m["checks"]["moments_ok"] is True, "the moments pass -- that is the point"
    assert m["checks"]["orders_ok"] is False
    assert m["summary"]["observed_order"] == pytest.approx(0.5, abs=0.2)
    assert m["score"] == 8, m["certification"]
    # D1 item 2: the reason names which of the two defects it saw.
    assert "0.5" in m["certification"]["reason"] or "does not clear" in \
        m["certification"]["reason"], m["certification"]["reason"]


def test_correct_milstein_on_the_same_spec_still_scores_ten(tmp_path):
    """The control. The two-sided band a declared `scheme_family` selects must pass
    the scheme it was declared for."""
    m = _run(tmp_path, "sde_milstein_honest.py", spec="sde_spec_gbm.json",
             frontmatter="scheme_family: milstein")
    assert m["checks"]["orders_ok"] is True, m["detail"]["orders_ok"]
    assert m["summary"]["observed_order"] == pytest.approx(1.0, abs=0.2)
    assert m["score"] == 10


def test_the_post_step_correction_is_not_failed_by_the_order_study(tmp_path):
    """A correction to the plan, measured. §4a lists the post-step correction as a
    defect the strong-order gate must catch; it is an O(dt) perturbation of an O(dt)
    scheme, so the rate is unchanged (0.998 against the honest scheme's 0.998) and
    only the error constant doubles. Asserting the non-failure is what keeps a future
    "fix" from tightening the band until this fails -- which would cost the correct
    Milstein plan beside it."""
    m = _run(tmp_path, "sde_milstein_post_step.py", spec="sde_spec_gbm.json",
             frontmatter="scheme_family: milstein")
    assert m["checks"]["orders_ok"] is not False, m["detail"]["orders_ok"]
    assert m["summary"]["observed_order"] == pytest.approx(1.0, abs=0.25)


# --- [REV 28]: a defaulted floor must not fail a correct solver ---------------

def test_a_correct_additive_noise_plan_is_not_failed_by_a_defaulted_floor(tmp_path):
    """Criteria 10 and 13, and the window §7 exists to avoid. Euler-Maruyama has
    strong order **1** whenever the noise is additive, and six of the stale specs are
    additive or noise-free and default to 0.5. Under 1c with a two-sided band around
    the default, a correct solver on each measures ~0.99 and misses by 0.49."""
    m = _run(tmp_path, "sde_honest.py", spec="sde_spec_defaulted_floor.json")
    assert m["config"]["expected_strong_order"]["source"] == "kernel_default"
    assert m["summary"]["order_band"].startswith("one-sided")
    assert m["summary"]["observed_order"] == pytest.approx(1.0, abs=0.2)
    assert m["checks"]["orders_ok"] is True, m["detail"]["orders_ok"]
    assert m["score"] == 10


# --- Finding 2 and 3: the multi-noise problems --------------------------------

def test_the_ladder_completes_on_two_components_and_recovers_the_correlation(tmp_path):
    """Criterion 3 and [REV 28]. The ladder used to hand this solver a
    `(num_paths, Nt)` array against `solver-sde.md`'s `(num_paths, Nt, m)` contract
    and raise `IndexError`, reported as "the CRN level at dt=0.02 crashed"."""
    m = _run(tmp_path, "sde_two_component.py", spec="sde_spec_2d.json")
    assert m["config"]["noise_dimension"]["value"] == 2
    assert m["config"]["noise_dimension"]["source"] == "spec"
    assert not isinstance(m["checks"]["orders_ok"], str), m["detail"]["orders_ok"]
    assert m["detail"]["orders_ok"]["components"] == 2, "both components, Finding 11"
    corr = m["detail"]["moments_ok"]["cross_moment"]["correlation"]
    assert corr["empirical"] == pytest.approx(0.896, abs=0.02), corr
    assert m["checks"]["cross_moment_ok"] is True
    assert m["score"] == 10, m["certification"]


def test_a_solver_that_drops_the_correlation_is_capped_at_eight(tmp_path):
    """Criterion 4. The spec's own note: *"A solver that drops the correlation
    entirely (independent increments) still passes both per-component moment
    checks, so this check is the only thing that detects it."* Through `moments_ok`,
    not through the gate -- `cross_moment.gate` stays `false`."""
    m = _run(tmp_path, "sde_drops_correlation.py", spec="sde_spec_2d.json")
    corr = m["detail"]["moments_ok"]["cross_moment"]["correlation"]
    assert abs(corr["empirical"]) < 0.05, corr
    assert m["checks"]["cross_moment_ok"] is False
    assert m["checks"]["moments_ok"] is False
    assert m["checks"]["variance_ok"] is True, "both marginals are exactly right"
    assert m["score"] == 8, m["certification"]
    entry = (_spec("sde_spec_2d.json")["verification"]["cross_moment"])
    assert entry["gate"] is False, "[REV 22]: the gate flip is dropped"


# --- Finding 4: the distributional tier ---------------------------------------

def test_the_standardized_shape_check_passes_a_correct_solver(tmp_path):
    """Criterion 16 and D4. The un-standardized test at 50 000 paths resolves the
    scheme's own O(dt) bias and rejects correct solvers; standardizing both sides
    leaves it measuring shape, which is what the first two moments cannot see."""
    m = _run(tmp_path, "sde_gbm_euler.py", spec="sde_spec_gbm.json")
    assert m["checks"]["terminal_law_ok"] is True, m["detail"]["terminal_law_ok"]
    assert m["summary"]["ks_statistic"] < m["summary"]["ks_critical"]
    assert m["detail"]["terminal_law_ok"]["standardized"] is True
    assert m["score"] == 10
    ou = _run(tmp_path, "sde_honest.py")
    assert ou["checks"]["terminal_law_ok"] is True, ou["detail"]["terminal_law_ok"]
    assert ou["score"] == 10


def test_the_shape_check_rejects_the_right_moments_from_the_wrong_family():
    """What the tier is *for*: Finding 4's "the check that separates 'right moments'
    from 'right process'". A lognormal sample matched to a Gaussian law's first two
    moments passes every moment check by construction, and the standardized
    statistic rejects it by two orders of magnitude."""
    import numpy as np

    from verifylib.kernel.sde import laws

    law, _ = laws.build({"law": "normal", "mean": 0.4463, "variance": 0.2814 ** 2}, {})
    rng = np.random.default_rng(4)
    honest = law.mean + law.sd * rng.standard_normal(50000)
    # A lognormal with the same mean and variance.
    s2 = np.log(1 + (law.sd / law.mean) ** 2)
    wrong = np.exp(np.log(law.mean) - 0.5 * s2 + np.sqrt(s2) * rng.standard_normal(50000))
    assert abs(wrong.mean() - honest.mean()) < 0.01
    assert abs(wrong.var(ddof=1) - honest.var(ddof=1)) < 0.01
    d_ok, crit, _ = laws.ks_standardized(honest, law)
    d_bad, _, _ = laws.ks_standardized(wrong, law)
    assert d_ok < crit < d_bad
    assert d_bad > 15 * crit, (d_bad, crit)      # measured 18.6x


def test_rademacher_increments_are_invisible_on_additive_noise(tmp_path):
    """A measured correction to §4a, which lists this defect under "the
    distributional check (3b)". On a linear additive-noise SDE the terminal state is
    a **sum** of the increments, so the CLT makes the law Gaussian to within the
    test's resolution: 0.00233 against a bar of 0.00607 at the spec's dt, and the
    Gaussian-increment control sits at 0.00295. A check that rejected this would be
    wrong. The defect is real and the terminal law is not where it shows."""
    m = _run(tmp_path, "sde_uniform_increments.py")
    assert m["checks"]["terminal_law_ok"] is True, m["detail"]["terminal_law_ok"]
    assert m["summary"]["ks_statistic"] < m["summary"]["ks_critical"]


def test_rademacher_increments_are_caught_on_multiplicative_noise(tmp_path):
    """...and here is where the same defect does show, and **only** the shape check
    sees it. Measured on GBM, correct-Milstein control beside it:

    =======================  ===========  ============  ===========  ==========
    check                    correct      Rademacher    bar          verdict
    =======================  ===========  ============  ===========  ==========
    first two moments        pass         pass          5% / 10%     blind
    `observed_order`         0.980        0.980         1.0 +/- 0.4  blind
    **standardized KS**      **0.0045**   **0.0197**    **0.0068**   **caught**
    =======================  ===========  ============  ===========  ==========

    The order study's blindness is **structural**, not incidental, and it is the
    reason the shape tier is the only instrument here. The CRN ladder and the Dynkin
    ladder both **supply their own increments** -- that is what §18's common random
    numbers *are*: the sandbox child rebuilds `dW` from a seed and hands it in. So
    the order study measures the solver's *scheme* driven by Gaussian noise and never
    sees its increment law at all. Dynkin is blind for a second reason too: its
    generator identity holds in expectation for *any* increment law with the right
    first two moments.

    The graded run is the only one that uses the solver's own draws, so its terminal
    distribution is the only window onto them -- which is exactly the evidence class
    Finding 4 says was missing.

    Multiplicative noise is what keeps that window open: the terminal state is a
    product whose factors carry `dW**2` explicitly, so the increment law survives to
    `T`. On additive noise it is a sum and the CLT closes it -- see
    `test_rademacher_increments_are_invisible_on_additive_noise`.
    """
    m = _run(tmp_path, "sde_milstein_rademacher.py", spec="sde_spec_gbm.json",
             frontmatter="scheme_family: milstein")
    assert m["checks"]["terminal_law_ok"] is False, m["detail"]["terminal_law_ok"]
    assert m["summary"]["ks_statistic"] > 2 * m["summary"]["ks_critical"]
    # The checks that are blind to it, asserted so the claim above is pinned rather
    # than described. The order study reads the *same* value as the correct scheme.
    assert m["checks"]["orders_ok"] is True
    assert m["summary"]["observed_order"] == pytest.approx(1.0, abs=0.1)
    honest = _run(tmp_path, "sde_milstein_honest.py", spec="sde_spec_gbm.json",
                  frontmatter="scheme_family: milstein")
    assert honest["summary"]["observed_order"] == pytest.approx(
        m["summary"]["observed_order"], abs=0.02), "the ladder supplies its own dW"
    assert honest["checks"]["dynkin_ok"] is True and honest["score"] == 10
    # ...and it is graded, through `moments_ok`, exactly as the cross moment is.
    assert m["checks"]["moments_ok"] is False
    assert m["score"] == 8, m["certification"]
    # Dynkin does not even run here, and that is §24's cost rule working: with the
    # plan already capped at 8 an expensive probe cannot change the outcome.
    assert m["checks"]["dynkin_ok"] == "not_run"


# --- Finding 10: the clip scan ------------------------------------------------

def test_a_state_clip_is_surfaced_on_the_solver_that_does_it(tmp_path):
    """Finding 10, end to end rather than on a synthetic string."""
    from verifylib.gate import check_solver_file

    path = os.path.join(CANARY, "sde_clips_positivity.py")
    found = [f for f in check_solver_file(path) if f.check == "state_clip"]
    assert len(found) == 1 and found[0].severity == "warning", found
    honest = os.path.join(CANARY, "sde_honest.py")
    assert [f for f in check_solver_file(honest) if f.check == "state_clip"] == []


# --- Finding 5: a random initial condition ------------------------------------

def test_a_random_initial_condition_is_not_a_dynkin_failure(tmp_path):
    """Criterion 5, and the rule §4a states: *a false hard-fail on correct work is
    invisible as a false positive.* `x0` defaulted to 0.0, so for `phi = X**2` the
    identity was formed with `phi(0) = 0` where it needs `E[phi(X_0)] =
    sigma_bar**2 = 0.1111`, and the residual carried that offset at every level."""
    m = _run(tmp_path, "sde_random_ic_honest.py", spec="sde_spec_random_ic.json")
    assert m["checks"]["dynkin_ok"] is not False, m["detail"].get("dynkin_ok")
    state = m["detail"]["dynkin_ok"].get("initial_state") or {}
    assert state.get("initial_condition") == "normal", state
    assert state.get("variance") == pytest.approx(1.0 / 9.0, rel=1e-6)


def test_a_spec_with_a_random_ic_and_no_declared_law_abstains(tmp_path):
    """The other half of criterion 5: where the law is *not* declared the check
    reports `unavailable` with the random initial condition as its stated reason.
    Never `False` -- the kernel cannot form the identity, which is a property of what
    the spec declares and not of the solver."""
    from verifylib.kernel.sde import dynkin

    spec = _spec("sde_spec_random_ic.json")
    del spec["initial_condition"]
    phi0, detail = dynkin.initial_moments(spec, ["X", "X**2"])
    assert phi0 is None and detail["random_initial_condition"] is True
    assert dynkin.check(spec, ["X"], [], detail)[0] == "unavailable"


# --- the lower band edge, found live by 1b (2026-10-15) ----------------------

def test_a_slowly_converging_residual_abstains_rather_than_failing(tmp_path):
    """**Found by shipping 1b**, on `sde_cir_feller_violated/2-milstein`: enabling
    Dynkin on Path A took a live plan from 10 to 3, and the solver is correct.

    `sde_cir_truncated.py` is that plan, and `sde_spec_cir.json` its spec, at the
    live configuration. The scheme is the standard full-truncation Milstein; the
    problem is a CIR with `2*kappa*theta = 1.0` against `sigma**2 = 4`, so zero is
    attainable, `sqrt(X)` is non-Lipschitz where the paths spend their time, and
    every truncated scheme converges slowly. Measured against the exact
    noncentral-chi-squared law, `E[tanh(X_T)]` errs by 0.0161 / 0.0103 / 0.0061 at
    `Nt = 200 / 800 / 12800` -- converging, at order ~0.2.

    Dynkin's Richardson cannot remove an order-0.2 bias from a three-level ladder, so
    it reported the leftover as a generator violation at z = 5.1. The outcome must be
    the skip, and the z must still be recorded: abstaining is not silence.

    The guard against this reopening Finding 1's hole is
    `test_a_residual_that_does_not_move_with_dt_still_fails` and
    `test_a_fabricated_sampler_does_not_certify_on_path_a`.

    **[REV 60]: the abstention does not certify either.** The two halves are
    separate questions and this test now pins both. The solver is not called wrong
    -- 9, not the 3 a Dynkin *failure* earns -- and it is not called verified, because
    nothing in the run establishes that these paths came from this SDE and the exact
    terminal moments provably cannot: §24's own argument is that two processes can
    share them. An honest "could not reach a verdict" buys a report, not a 10.
    """
    spec = _spec("sde_spec_cir.json")
    # Dropped so this test measures the Dynkin branch alone; the declared law is
    # what `test_full_truncation_moves_mass_to_an_atom_the_ks_test_sees` runs on.
    spec["verification"].pop("terminal_law")
    m = run(spec, _plan(tmp_path, "sde_cir_truncated.py"), {})
    assert m["checks"]["moments_ok"] is True, m["detail"]["moments_ok"]
    assert m["checks"]["dynkin_ok"] == "unresolved", m["detail"]["dynkin_ok"]
    detail = m["detail"]["dynkin_ok"]
    assert detail["unresolved_functions"] == ["X", "np.tanh(X)"], detail
    assert "not in an asymptotic regime" in detail["reason"]
    # Every level's residual is shrinking -- the condition the abstention rests on.
    for seq in detail["residuals"]:
        assert all(abs(b) < abs(a) for a, b in zip(seq, seq[1:], strict=False)), seq
    # The diagnostic survives: abstaining is not silence.
    assert m["summary"]["dynkin_z"] > 2 * m["summary"]["dynkin_bar"]
    # Not failed -- a Dynkin *failure* is 3, and this is not one.
    assert m["score"] == 9, m["certification"]
    assert m["certified"] is False
    assert m["certification"]["bound_by"] == "rubric"
    assert "could not reach a verdict" in m["certification"]["reason"]
    # ...and the moments that *did* pass are still reported as passing, so the
    # reason names what is missing rather than what is broken.
    assert m["checks"]["variance_ok"] is True
    # More seeds cannot turn a non-verdict into one, so replication must not fire.
    assert detail["replication"]["replicated"] is False
    assert "not a verdict" in detail["replication"]["decision"]


def test_full_truncation_moves_mass_to_an_atom_the_ks_test_sees(tmp_path):
    """The same solver, with the exact terminal law declared -- and now it **fails**,
    correctly. Full truncation clips the state at zero, so 12% of the paths land
    *exactly* at 0 where the true noncentral-chi-squared density has no atom at all.
    The first two moments cannot see it: the mean is 4.96% out against a 5% tolerance
    and the variance 1.8% against 10%. The standardized KS statistic is 0.295 against
    a 0.0061 bar -- **48x** -- which is Finding 4's third defect class, "a clipped
    state that moves mass to a point and leaves the mean inside tolerance", caught on
    a real scheme rather than on a canary written to be caught.

    This is a property of the scheme and not of this implementation, which is why it
    is asserted as a *finding* rather than fixed: the live spec declares no
    `terminal_law`, so nothing sees it there, and declaring one is formulator work."""
    m = run(_spec("sde_spec_cir.json"), _plan(tmp_path, "sde_cir_truncated.py"), {})
    assert m["checks"]["terminal_law_ok"] is False, m["detail"]["moments_ok"]
    assert m["summary"]["ks_statistic"] > 20 * m["summary"]["ks_critical"]
    law = m["detail"]["moments_ok"]["terminal_law"]["per_component"][0]
    assert law["law"] == "noncentral_chi2"
    # The mean and the variance are both inside tolerance -- that is the point.
    assert m["detail"]["moments_ok"]["mean_ok"] is True
    assert m["detail"]["moments_ok"]["variance_ok"] is True
    # Tier 1 stays report-only (D11): recorded with its sigma distance, no verdict.
    assert law["skewness"]["outcome"] is None and law["skewness"]["graded"] is False


# --- [REV 60]: a 10 needs generator evidence, by whichever door it is missing --

def test_a_dynkin_the_budget_refused_does_not_certify(tmp_path):
    """The budget door, end to end. `DYNKIN_MAX_BYTES` refuses rather than quietly
    running a weaker check — which is right, because fewer paths widen the residual's
    interval and could turn a real inconsistency into a pass. But the refusal
    returned a skip and **no rubric row or cap read a skipped Dynkin**, so a
    would-certify Path-A plan too large for the budget certified at 10 with no
    generator evidence at all. `_dynkin`'s own docstring named the risk: *"it does
    not certify today, so the ladder is not run there; the guard is here because 'it
    does not certify' is luck rather than design."*

    The same honest solver, same spec, same everything — only the budget moves.
    """
    from verifylib.kernel.sde import driver as sde_driver

    full = _run(tmp_path, "sde_honest.py")
    assert full["checks"]["dynkin_ok"] is True
    assert full["score"] == 10 and full["certified"] is True

    original = sde_driver.DYNKIN_MAX_BYTES
    try:
        sde_driver.DYNKIN_MAX_BYTES = 1000          # 1 kB: nothing fits
        refused = _run(tmp_path, "sde_honest.py")
    finally:
        sde_driver.DYNKIN_MAX_BYTES = original

    assert refused["checks"]["dynkin_ok"] == "unavailable"
    assert "GB budget" in refused["detail"]["dynkin_ok"]["reason"]
    # Everything else is identical, and that is the point: the only thing missing is
    # the generator check.
    assert refused["checks"]["moments_ok"] is True
    assert refused["checks"]["resolved"] is True
    assert refused["checks"]["orders_ok"] == full["checks"]["orders_ok"]
    assert refused["score"] == 9, refused["certification"]
    assert refused["certified"] is False
    assert refused["provenance"] == "analytic", "the tier did not change, the evidence did"
    assert "could not run" in refused["certification"]["reason"]


def test_a_generator_that_cannot_be_built_does_not_certify_either(tmp_path):
    """The second door. The generator is built from the spec's own drift and
    diffusion, and `observables_payload` requires them to **evaluate** rather than
    merely to be strings ([REV 8]). Prose there — which is the live shape on
    `sde_multichannel_stiff_m13` — is a property of the spec rather than of the
    solver, and still not evidence about the solver."""
    spec = _spec("sde_spec.json")
    spec["drift_expression"] = ("theta * (mu - X)  (mean-reverting, theta and mu from "
                               "parameters)")
    m = run(spec, _plan(tmp_path, "sde_honest.py"), {})
    assert m["checks"]["dynkin_ok"] == "unavailable"
    assert "generator cannot be built" in m["detail"]["dynkin_ok"]["reason"]
    assert m["checks"]["moments_ok"] is True
    assert m["score"] == 9 and m["certified"] is False
    assert "could not run" in m["certification"]["reason"]
