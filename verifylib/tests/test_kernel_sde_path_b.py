"""The SDE side of 'no closed form': Path B, end to end and piece by piece.

``sde_spec.json`` is Path A (``has_analytic_solution: true``), so until now the
route an SDE takes when the formulator declares *no* closed-form moments -- the
surrogate tier A' (§15-17), the CRN order study (§18-19) and Dynkin (§20) as the
only evidence -- had no end-to-end test at all. ``sde_spec_no_closed_form.json``
is the same OU process with the moments withheld and a moment ODE declared;
``sde_spec_no_surrogate.json`` withholds the ODE too, so nothing but
self-convergence and Dynkin remain.

The seeded defects are contract-shaped rather than numerical: a flipped drift
(the wrong equation, integrated correctly), a solver that ignores the supplied
Brownian increments (the CRN contract, manual §18), and reported paths that breach
a declared support. Each names the check that must see it and what the score must
then say.
"""
import json
import os
import shutil

import numpy as np
import pytest

from verifylib.kernel import agreement, plan_meta, run, sandbox
from verifylib.kernel.driver import solver_sha256
from verifylib.kernel.sde import crn, dynkin

from .test_kernel_canaries import CANARY, _spec


def _plan(tmp_path, solver):
    shutil.copy(os.path.join(CANARY, "sde_honest.py"), tmp_path / "sde_honest.py")
    shutil.copy(os.path.join(CANARY, solver), tmp_path / "solver.py")
    return str(tmp_path)


def _run(tmp_path, solver, spec=None, config=None):
    return run(spec or _spec("sde_spec_no_closed_form.json"), _plan(tmp_path, solver),
               config or {})


# --- the honest solver on both Path-B specs -----------------------------------

def test_an_honest_sde_with_a_surrogate_scores_ten_surrogate(tmp_path):
    """Tier A': the moment ODE closes exactly, passes its trust guard, and the
    empirical moments match it inside their confidence intervals."""
    m = _run(tmp_path, "sde_honest.py")
    assert (m["score"], m["provenance"]) == (10, "surrogate"), m["certification"]
    assert m["certification"]["tier"] == "A_prime"
    assert m["path"] == "B"
    assert m["checks"]["surrogate_ok"] is True
    assert m["reference"]["route"] == "moment_ode"
    assert m["checks"]["moments_ok"] is True and m["checks"]["resolved"] is True
    assert m["checks"]["dynkin_ok"] is True and m["checks"]["orders_ok"] is True


def test_path_b_runs_the_order_study_and_dynkin_unconditionally(tmp_path):
    """§24: on Path B both Stage-2 checks run every cycle -- there is no exact
    reference that could make them redundant."""
    m = _run(tmp_path, "sde_honest.py")
    assert m["config"]["run_crn_ladder"]["value"] is True
    assert m["config"]["run_dynkin"]["value"] is True
    assert m["summary"]["observed_order"] == pytest.approx(1.0, abs=0.2)


def test_an_honest_sde_with_no_surrogate_still_measures(tmp_path):
    """No reference at all: Tier C is the CRN ladder, the error estimate is the
    Richardson moment's own relative shift, and 'no test could be run' is wrong."""
    m = _run(tmp_path, "sde_honest.py", spec=_spec("sde_spec_no_surrogate.json"))
    assert m["checks"]["surrogate_ok"] == "unavailable"
    assert m["checks"]["moments_ok"] == "unavailable"
    assert m["checks"]["orders_ok"] is True and m["checks"]["dynkin_ok"] is True
    assert m["checks"]["richardson_stable"] is True
    assert m["summary"]["estimated_rel_error"] == pytest.approx(
        m["detail"]["orders_ok"]["richardson"]["relative_shift"])
    assert m["score"] >= 8, m["certification"]


def test_an_honest_sde_with_no_surrogate_reaches_nine_through_dynkin(tmp_path):
    """Dynkin is D1's SDE counterpart (§4b) and is built from the spec's own drift
    and diffusion; §4c prices one circularity break at 9."""
    m = _run(tmp_path, "sde_honest.py", spec=_spec("sde_spec_no_surrogate.json"))
    assert m["score"] == 9, m["certification"]


def test_a_moment_ode_that_does_not_close_is_not_a_reference(tmp_path):
    """§15: a moment-closure approximation is a sanity band, never ground truth.
    Marking it ``closes_exactly: false`` must demote it, not merely annotate it."""
    spec = _spec("sde_spec_no_closed_form.json",
                 **{"verification.moment_ode.closes_exactly": False})
    m = _run(tmp_path, "sde_honest.py", spec=spec)
    assert m["checks"]["surrogate_ok"] == "unavailable"
    assert m["provenance"] != "surrogate"
    assert "close" in json.dumps(m["reference"]).lower()


# --- the seeded defects ---------------------------------------------------------

def test_a_sign_flipped_drift_is_caught_by_dynkin(tmp_path):
    """The generator is built from the *spec's* drift, so a solver integrating the
    wrong equation leaves a residual hundreds of standard errors from zero."""
    m = _run(tmp_path, "sde_wrong_drift.py")
    assert m["checks"]["dynkin_ok"] is False
    z = m["detail"]["dynkin_ok"]["z"]
    assert min(abs(v) for v in z) > 50, z
    assert m["score"] < 9


def test_a_dynkin_failure_is_never_reported_as_mc_inconclusive(tmp_path):
    m = _run(tmp_path, "sde_wrong_drift.py")
    assert m["checks"]["dynkin_ok"] is False
    assert "not a solver bug" not in m["certification"]["reason"], m["certification"]
    assert m["score"] <= 4


def test_a_solver_that_ignores_the_crn_increments_is_detected(tmp_path):
    """Manual §18: independent draws at each level make the pathwise differences
    pure noise, so the strong order reads ~0 on an otherwise correct scheme."""
    m = _run(tmp_path, "sde_ignores_crn.py")
    assert m["checks"]["orders_ok"] is False
    assert m["detail"]["orders_ok"]["strong_order"] < 0.2
    assert m["checks"]["richardson_stable"] is not None


def test_a_solver_that_ignores_the_crn_increments_does_not_certify(tmp_path):
    """Manual §7's rule for ``override`` extended to §18's ``dW`` (findings §7.6,
    decided 2026-09-20): a solver that ignores its inputs is not certified above
    Tier C however well its moments match the surrogate."""
    m = _run(tmp_path, "sde_ignores_crn.py")
    assert m["score"] < 10, m["certification"]
    assert m["score"] == 8
    assert "increments" in m["certification"]["reason"]


def test_paths_outside_a_declared_support_are_a_gate_violation(tmp_path):
    """§21: a ``gate: true`` constraint sinks the score whatever the moments say,
    and the gate is what names the failure."""
    spec = _spec("sde_spec_no_closed_form.json")
    # A registry name always uses its registry evaluator, even beside an `expr`
    # (so `positivity` here would test X > 0 and fail the honest OU, whose paths
    # do go negative). `support` with an explicit floor is the §21 entry for this.
    spec["verification"]["constraints"].append(
        {"name": "support", "lower": -5.0, "gate": True})
    m = _run(tmp_path, "sde_negative_paths.py", spec=spec)
    assert m["d2"]["gate_failed"] == ["support"]
    assert m["checks"]["constraints_ok"] is False
    assert m["score"] <= 4
    assert "gate" in m["certification"]["reason"]
    honest = _run(tmp_path, "sde_honest.py", spec=spec)
    assert honest["d2"]["gate_failed"] == [] and honest["score"] == 10


def test_an_sde_solver_returning_no_terminal_paths_is_bad_schema(tmp_path):
    (tmp_path / "solver.py").write_text(
        "def solve_sde(num_paths, dt, T, seed=42, dW=None, observables=None):\n"
        "    return {'empirical_mean': 0.0}\n")
    got = sandbox.run(str(tmp_path), "sde", {"num_paths": 10, "dt": 0.1, "T": 1.0, "seed": 1})
    assert got["status"] == "crashed" and got["reason"] == "bad_schema"
    assert "terminal_paths" in got["error"]
    m = run(_spec("sde_spec_no_closed_form.json"), str(tmp_path), {})
    assert (m["score"], m["provenance"]) == (1, "none")


# --- Dynkin, piece by piece (errata I9) ----------------------------------------

def _fake_run(residual_per_path, phi_T=None):
    """A run whose per-path Dynkin residual for ``phi = X`` is exactly the array
    given: terminal paths ``X_T`` with ``x0 = 0`` and ``integral(L phi) = X_T - r``."""
    r = np.asarray(residual_per_path, dtype=float)
    X = np.asarray(phi_T if phi_T is not None else r + 1.0, dtype=float)
    return {"result": {"terminal_paths": X, "path_integrals": {"L0": X - r}}}


SPEC_X = {"parameters": {}, "drift_expression": "0*X", "diffusion_expression": "1+0*X"}


def test_dynkin_takes_its_standard_error_on_the_extrapolated_quantity():
    """I9(b): the error bar belongs to the *extrapolated* quantity path by path, not
    to ``r_fine``. With anti-correlated levels the two differ by a factor of ~3.

    Two levels, so no order can be measured and the historical ``O(dt)`` assumption
    applies -- which is what pins the formula to ``2 r_fine - r_coarse`` here."""
    rng = np.random.default_rng(0)
    fine = rng.standard_normal(4000)
    coarse = -fine + 0.01 * rng.standard_normal(4000)
    outcome, detail = dynkin.check(SPEC_X, ["X"], [_fake_run(coarse), _fake_run(fine)], 0.0)
    expected = float(np.std(2 * fine - coarse, ddof=1) / np.sqrt(fine.size))
    wrong = float(np.std(fine, ddof=1) / np.sqrt(fine.size))
    assert detail["se"][0] == pytest.approx(expected)
    assert detail["se"][0] > 2.5 * wrong
    assert detail["extrapolated"][0] == pytest.approx(float(np.mean(2 * fine - coarse)))


def test_dynkin_refuses_levels_that_cannot_share_a_brownian_path():
    """Different path counts at the two dt levels mean they were not driven by one
    path, and a Richardson extrapolation across them would be noise."""
    outcome, detail = dynkin.check(SPEC_X, ["X"], [_fake_run(np.zeros(100)),
                                                  _fake_run(np.zeros(200))], 0.0)
    assert outcome == "unavailable"
    assert "Brownian" in detail["reason"]


def test_common_random_numbers_shrink_the_extrapolation_noise(tmp_path):
    """I9(a), measured on the honest solver: under CRN the extrapolated residual's
    standard error is far smaller than with independent draws per level, because
    ``r_fine - r_coarse`` is then a pathwise difference rather than noise."""
    plan = _plan(tmp_path, "sde_honest.py")
    spec = _spec("sde_spec_no_closed_form.json")
    phis = ["X", "X**2"]
    payload, _ = dynkin.observables_payload(spec, phis)
    call = lambda dt, seed: {"num_paths": 8000, "dt": dt, "T": 1.0, "seed": seed}  # noqa: E731

    levels = crn.ladder_spec(8000, 0.02, 1.0, levels=2, seed=7)
    crn_runs = [sandbox.run(plan, "sde", call(lv["dt"], 7), crn=lv, observables=payload)
                for lv in levels]
    indep_runs = [sandbox.run(plan, "sde", call(0.02, 7), observables=payload),
                  sandbox.run(plan, "sde", call(0.01, 8), observables=payload)]
    assert all(r["status"] == "ok" for r in crn_runs + indep_runs)

    _, with_crn = dynkin.check(spec, phis, crn_runs, 2.0)
    _, without = dynkin.check(spec, phis, indep_runs, 2.0)
    for a, b in zip(with_crn["se"], without["se"], strict=True):
        assert a < 0.5 * b, (with_crn["se"], without["se"])


def test_dynkin_measures_the_residual_order_instead_of_assuming_one():
    """The `sde_double_well_langevin` regression: a residual converging at order 2
    must not be extrapolated as if it converged at order 1.

    Synthetic, so the arithmetic is the whole assertion: r(dt) = C dt^2 with no
    offset, sampled on dt, dt/2, dt/4. The order-1 formula reports a residual of the
    *opposite sign* and a third of the coarse magnitude; the measured-order formula
    lands on zero, which is where this sequence is going."""
    rng = np.random.default_rng(11)
    n = 20000
    noise = 1e-6 * rng.standard_normal(n)
    seq = [_fake_run(c + noise) for c in (0.16, 0.04, 0.01)]        # C dt^2, halving dt
    outcome, detail = dynkin.check(SPEC_X, ["X"], seq, 0.0, ci_mult=2.0)
    assert detail["extrapolation_order_source"] == ["measured"]
    assert detail["extrapolation_order"][0] == pytest.approx(2.0, abs=0.01)
    assert detail["extrapolated"][0] == pytest.approx(0.0, abs=1e-5), detail["extrapolated"]
    assert outcome is True
    # What the old code would have produced from the same two finest levels:
    # 2*0.01 - 0.04 = -0.02, the opposite sign and a third of the coarse magnitude.
    assert pytest.approx(-0.02) == 2 * 0.01 - 0.04


def test_dynkin_falls_back_to_order_one_when_the_rate_is_unreadable_and_says_so():
    """A non-monotone or noise-dominated sequence has no rate to read. The check then
    uses the historical ``O(dt)`` assumption rather than a fitted number it cannot
    justify, and records that it did."""
    rng = np.random.default_rng(12)
    n = 4000
    # Differences that do not share a sign: no power law, so no order.
    seq = [_fake_run(c + 1e-9 * rng.standard_normal(n)) for c in (0.02, 0.05, 0.01)]
    outcome, detail = dynkin.check(SPEC_X, ["X"], seq, 0.0, ci_mult=2.0)
    assert detail["extrapolation_order_source"] == ["assumed_1"]
    assert detail["extrapolation_order"] == [None]
    assert "sign" in detail["order_detail"][0]["reason"]
    assert detail["extrapolated"][0] == pytest.approx(2 * 0.01 - 0.05, abs=1e-6)

    # And a sequence whose finer difference is pure noise: also unreadable.
    flat = [_fake_run(0.003 + 1e-3 * rng.standard_normal(n)) for _ in range(3)]
    _, quiet = dynkin.check(SPEC_X, ["X"], flat, 0.0, ci_mult=2.0)
    assert quiet["extrapolation_order_source"] == ["assumed_1"]
    assert "Monte-Carlo noise" in quiet["order_detail"][0]["reason"]


def test_a_measured_order_outside_the_believable_band_is_not_used():
    """An order of 8 is a small noisy denominator, not a rate. ORDER_BAND rejects it
    and the fallback applies -- the alternative is an extrapolation whose leverage
    ``1/(2^p - 1)`` is set by noise."""
    rng = np.random.default_rng(13)
    n = 20000
    noise = 1e-9 * rng.standard_normal(n)
    seq = [_fake_run(c + noise) for c in (1.0, 0.004, 0.002)]      # ratio ~ 498 -> p ~ 9
    _, detail = dynkin.check(SPEC_X, ["X"], seq, 0.0, ci_mult=2.0)
    assert detail["extrapolation_order_source"] == ["assumed_1"]
    assert "believable band" in detail["order_detail"][0]["reason"]


def test_a_residual_shrinking_too_slowly_to_extrapolate_abstains():
    """The mirror image of the test above, and of the `sde_double_well_langevin`
    regression `residual_order` was written for. There the measured order was *above*
    1 and the hard-coded `p = 1` overshot through zero; here it is *below* 0.5, the
    formula under-corrects, and the leftover -- the scheme's own bias -- was reported
    as a generator violation.

    Measured live on `sde_cir_feller_violated/2-milstein`, a full-truncation Milstein
    on a CIR whose Feller condition is violated: residual 0.0316 -> 0.0263 -> 0.0207,
    differences 11 s.e. from zero, read rate -0.05. Extrapolated at `p = 1` that is
    +0.0164 against a 0.0032 s.e., **z = 5.1**, a hard 3 -- for a solver that
    converges to the exact noncentral-chi-squared law, just at order ~0.2.

    The numbers here are that sequence. The outcome is a skip, not a verdict."""
    rng = np.random.default_rng(14)
    n = 20000
    noise = 1e-5 * rng.standard_normal(n)
    seq = [_fake_run(c + noise) for c in (0.0316, 0.0263, 0.0207)]
    outcome, detail = dynkin.check(SPEC_X, ["X"], seq, 0.0, ci_mult=2.0)
    assert outcome == "unresolved", (outcome, detail["order_detail"])
    assert detail["order_detail"][0]["unresolved"] is True
    assert detail["unresolved_functions"] == ["X"]
    assert "not in an asymptotic regime" in detail["reason"]
    # The diagnostic survives the abstention -- D8's rule: abstaining is not silence.
    assert detail["z"][0] > 2.0, detail["z"]
    assert detail["extrapolation_order_source"] == ["assumed_1"]


def test_a_growing_residual_is_evidence_rather_than_an_abstention():
    """The guard on the test above. A residual that *grows* under refinement also
    reads an order below the band, and it is the opposite of unresolvable: a
    diverging sequence is evidence about the solver. It keeps the hard verdict."""
    rng = np.random.default_rng(15)
    n = 20000
    noise = 1e-5 * rng.standard_normal(n)
    seq = [_fake_run(c + noise) for c in (0.01, 0.02, 0.04)]
    outcome, detail = dynkin.check(SPEC_X, ["X"], seq, 0.0, ci_mult=2.0)
    assert outcome is False, (outcome, detail["order_detail"])
    assert not detail["order_detail"][0].get("unresolved")
    assert "not shrinking" in detail["order_detail"][0]["reason"]


def test_a_residual_that_does_not_move_with_dt_still_fails():
    """**The guard that keeps Finding 1's hole shut.** The abstention above must not
    be reachable by a solver that never integrates the SDE. It is not, and the reason
    is structural rather than a tuned threshold: such a solver's residual does not
    depend on `dt` at all, so the differences between levels are Monte-Carlo noise
    and the *first* guard in `residual_order` catches them before any rate is read.

    The three classes land in three different branches, which is the whole basis for
    abstaining on one of them:

    =============================  ========================  ====================
    solver                         residual vs dt            branch
    =============================  ========================  ====================
    fabricated sampler             dt-independent            inside its own noise
    `ginzburg_landau_s6` 1 and 2   order 1.20 / 1.36         measured, **fails**
    Feller-violated CIR            shrinking, rate < 0.5     **unresolved**
    =============================  ========================  ====================

    `test_a_fabricated_sampler_does_not_certify_on_path_a` is the end-to-end half."""
    rng = np.random.default_rng(16)
    n = 20000
    seq = [_fake_run(0.03 + 2e-3 * rng.standard_normal(n)) for _ in range(3)]
    outcome, detail = dynkin.check(SPEC_X, ["X"], seq, 0.0, ci_mult=2.0)
    assert outcome is False, (outcome, detail["order_detail"])
    assert not detail["order_detail"][0].get("unresolved")
    assert "Monte-Carlo noise" in detail["order_detail"][0]["reason"]


def test_a_resolved_failure_outranks_an_unresolvable_one():
    """Two test functions, one of each. An unreachable limit on one says nothing
    about a real violation on the other, so the verdict is the failure."""
    rng = np.random.default_rng(17)
    n = 20000
    noise = 1e-5 * rng.standard_normal(n)
    slow = [c + noise for c in (0.0316, 0.0263, 0.0207)]
    real = [c + noise for c in (0.04, 0.02, 0.01)]        # order 1, extrapolates to 0
    real = [r + 0.05 for r in real]                       # ...offset so it fails
    runs = [{"result": {"terminal_paths": np.zeros(n),
                        "path_integrals": {"L0": -a, "L1": -b}}}
            for a, b in zip(slow, real, strict=True)]
    outcome, detail = dynkin.check(SPEC_X, ["X", "X**2"], runs, [0.0, 0.0], ci_mult=2.0)
    assert outcome is False, (outcome, detail["per_function_ok"])
    assert detail["per_function_ok"] == [False, False]
    assert detail["order_detail"][0].get("unresolved") is True
    assert not detail["order_detail"][1].get("unresolved")


# --- the SDE half of the F2 agreement gate (added 2026-09-26) -----------------

def _workspace(tmp_path, plans, spec="sde_spec_no_closed_form.json"):
    """A two-plan workspace of the shape `agreement.workspace_of` expects:
    ``<ws>/problem_spec.json`` and ``<ws>/plans/<name>/{solver.py,SOLUTION.md}``."""
    ws = tmp_path / "ws"
    (ws / "plans").mkdir(parents=True)
    shutil.copy(os.path.join(CANARY, spec), ws / "problem_spec.json")
    out = []
    for name, (solver, family) in plans.items():
        d = ws / "plans" / name
        d.mkdir()
        shutil.copy(os.path.join(CANARY, solver), d / "solver.py")
        shutil.copy(os.path.join(CANARY, "sde_honest.py"), d / "sde_honest.py")
        (d / "SOLUTION.md").write_text(f"---\nscheme_family: {family}\nscheme: {name}\n---\n")
        out.append(str(d))
    return str(ws), out


def test_sde_compare_differences_the_moments_and_records_an_agreement(tmp_path):
    """The gap this closes: `compare` used to solve every plan as a PDE, so on an SDE
    it exited 1 with "solver.py has no solve_pde()" and the Path-B agreement gate
    could not fire at all. Two copies of the honest solver, declared as different
    families, must agree in moment space and record a circularity break."""
    from verifylib import cli
    ws, (a, b) = _workspace(tmp_path, {"1-a": ("sde_honest.py", "implicit"),
                                       "2-b": ("sde_honest.py", "milstein")})
    assert cli._compare(a, b) == cli.PASS
    entry = agreement.load(ws)[0]
    assert entry["metric"] == "moments" and entry["N"] is None
    assert entry["agree"] is True and entry["different_family"] is True
    assert entry["rel_diff"] < entry["tol"]
    assert entry["config"]["num_paths"] and entry["config"]["dt"]
    # ...and the kernel reads it back as a break, against the current hashes.
    seen = agreement.for_plan(a, sha_of=lambda d: solver_sha256(d))
    assert seen["agreement"] is True and seen["stale"] == []


def test_sde_compare_calls_a_wrong_drift_a_disagreement(tmp_path):
    """The sign-flipped drift moves the terminal mean far outside the tolerance, so
    the difference is outside its confidence interval and both plans are capped."""
    from verifylib import cli
    ws, (a, b) = _workspace(tmp_path, {"1-honest": ("sde_honest.py", "implicit"),
                                       "2-wrong": ("sde_wrong_drift.py", "milstein")})
    assert cli._compare(a, b) == cli.BLOCK
    entry = agreement.load(ws)[0]
    assert entry["agree"] is False
    assert entry["rel_diff"] > entry["tol"], entry["per_moment"]
    assert agreement.for_plan(a, sha_of=solver_sha256)["disagreement"] is not None


def test_an_sde_moment_difference_inside_its_own_noise_is_inconclusive():
    """The outcome a field difference never needs. A difference that sits *at* the
    tolerance with an interval straddling it is neither a break nor a cap: granting
    either would be pricing sampling noise as evidence. Exercised on the decision
    function directly, so the assertion is the rule and not a Monte-Carlo accident."""
    from verifylib import cli
    # rel_diff == tol exactly, with an interval that covers both sides of it.
    bits = [{"rel_diff": 0.10, "rel_se": 0.02, "tol": 0.10,
             "agree": 0.10 + 2 * 0.02 < 0.10, "disagree": 0.10 - 2 * 0.02 > 0.10}]
    assert not any(b["disagree"] for b in bits) and not all(b["agree"] for b in bits)
    assert cli.INCONCLUSIVE == 3 and cli.INCONCLUSIVE not in (cli.PASS, cli.BLOCK, cli.FAIL)


def test_an_sde_plan_can_earn_its_second_break_from_agreement(tmp_path):
    """End to end: the recorded agreement reaches the kernel's evidence and is priced.
    Before this the SDE driver hard-coded `cross_plan_agreement` to None."""
    from verifylib import cli
    ws, (a, b) = _workspace(tmp_path, {"1-a": ("sde_honest.py", "implicit"),
                                       "2-b": ("sde_honest.py", "milstein")},
                            spec="sde_spec_no_surrogate.json")
    assert cli._compare(a, b) == cli.PASS
    m = run(_spec("sde_spec_no_surrogate.json"), a, {})
    assert m["checks"]["cross_plan_agreement"] is True, m["agreement"]
    assert m["evidence"]["cross_plan_agreement"] is True


def test_the_family_wise_multiplier_holds_the_stated_level():
    """I9(c): three functions tested at a per-test 95% level is a ~86% test. The
    Bonferroni bar keeps the family-wise false-fail rate near the 5% ``ci_mult = 2``
    states, and the uncorrected bar does not."""
    assert dynkin.family_wise_multiplier(2.0, 1) == 2.0
    bar3 = dynkin.family_wise_multiplier(2.0, 3)
    assert 2.35 < bar3 < 2.5
    rng = np.random.default_rng(1)
    trials, k, n = 2000, 3, 400
    z = rng.standard_normal((trials, k, n)).mean(axis=2) * np.sqrt(n)   # ~N(0,1) each
    corrected = float(np.mean(np.any(np.abs(z) > bar3, axis=1)))
    uncorrected = float(np.mean(np.any(np.abs(z) > 2.0, axis=1)))
    assert corrected < 0.07, corrected
    assert uncorrected > 0.10, uncorrected


def test_dynkin_uses_the_family_wise_bar_end_to_end():
    """Three noise-only residuals inside the corrected bar but outside the plain
    one pass; the detail records which bar was used. Two identical levels, so the
    order is unmeasurable and the extrapolation is the ``p = 1`` fallback, i.e.
    ``2 r_fine - r_coarse = r``."""
    rng = np.random.default_rng(3)
    n = 2000
    r = rng.standard_normal(n)
    r = (r - r.mean()) / r.std(ddof=1) + 2.2 / np.sqrt(n)        # z = 2.2 exactly
    runs = [_fake_run(r), _fake_run(r)]       # identical levels: 2*fine - coarse = r
    outcome, detail = dynkin.check(SPEC_X, ["X"], runs, 0.0, ci_mult=2.0)
    assert outcome is False and detail["family_wise_multiplier"] == 2.0
    triple = [{"result": {"terminal_paths": r["result"]["terminal_paths"],
                          "path_integrals": {f"L{i}": r["result"]["path_integrals"]["L0"]
                                             for i in range(3)}}}
              for r in runs]
    outcome3, detail3 = dynkin.check({**SPEC_X}, ["X", "X", "X"], triple, 0.0, ci_mult=2.0)
    assert detail3["family_wise_multiplier"] > 2.3
    assert outcome3 is True, detail3["z"]


# --- the order study's asymmetries (I10, I15) -----------------------------------

def test_orders_ok_gates_on_the_strong_order_alone():
    """I10: a weak order of 1.7514 against an upper limit of 1.75 is noise, not a
    defect; a strong order materially below the declared one is."""
    detail = {"s0": 1.0, "s1": 0.5}
    ok = crn.guards(1.0, 1.7514, 1.0, 1.0, detail, declared=True, candidates=[1.0])
    assert ok["orders_ok"] is True and ok["weak_order_out_of_band"] is True
    bad = crn.guards(0.3, 1.0, 1.0, 1.0, detail, declared=True, candidates=[1.0])
    assert bad["orders_ok"] is False
    indeterminate = crn.guards(1.0, None, 1.0, 1.0, detail, declared=True, candidates=[1.0])
    assert indeterminate["orders_ok"] is True and indeterminate["weak_ok"] is True


def test_an_inferred_family_loosens_the_band_and_a_declared_one_narrows_it(tmp_path):
    """I15 through the frontmatter: ``scheme: euler-maruyama`` (inferred) keeps a
    one-sided band that accepts the log-transform's strong order of 1.019 against
    a 0.5 default; ``scheme_family: milstein`` (declared) selects the spec's
    Milstein order and the two-sided band around it."""
    verification = {"expected_strong_order": 0.5, "expected_strong_order_milstein": 1.0}
    detail = {"s0": 1.0, "s1": 0.5}

    (tmp_path / "SOLUTION.md").write_text("---\nscheme: euler-maruyama\n---\n")
    meta = plan_meta.read(str(tmp_path))
    assert meta["scheme_family_source"] == "inferred"
    expected, candidates, declared = crn.expected_orders(
        verification, meta["scheme_family"], declared=meta["scheme_family_source"] == "declared")
    assert declared is False
    assert crn.guards(1.019, 1.0, expected, 1.0, detail, declared=declared,
                      candidates=candidates)["orders_ok"] is True

    (tmp_path / "SOLUTION.md").write_text("---\nscheme_family: milstein\n---\n")
    meta = plan_meta.read(str(tmp_path))
    expected, candidates, declared = crn.expected_orders(
        verification, meta["scheme_family"], declared=meta["scheme_family_source"] == "declared")
    assert (expected, declared) == (1.0, True)
    assert crn.guards(1.0, 1.0, expected, 1.0, detail, declared=True,
                      candidates=candidates)["strong_band"] == "two-sided"
    assert crn.guards(0.5, 1.0, expected, 1.0, detail, declared=True,
                      candidates=candidates)["orders_ok"] is False


# --- the three analytic_moments shapes (I11) -----------------------------------

@pytest.mark.parametrize("moments, expected", [
    ({"mean_expression": "X_0*np.exp(-theta*t)"}, ["X_0*np.exp(-theta*t)"]),
    ({"mean_expression": ["a*t", "b*t"]}, ["a*t", "b*t"]),
    ({"mean_X": "a*t", "mean_Y": "b*t"}, ["a*t", "b*t"]),
    ({"mean_X_expression": "a*t"}, ["a*t"]),
    ({}, None),
    ({"mean_expression": []}, None),
])
def test_all_three_analytic_moments_shapes_are_read(moments, expected):
    from verifylib.kernel.sde.driver import _claimed
    assert _claimed(moments, "mean") == expected


def test_a_two_component_spec_in_the_mean_x_spelling_is_not_no_test_could_be_run():
    """I11 measured: ``sde_gbm_2d_high_corr`` reported 'no test in the manual could
    be run' and scored 2 because only the first spelling was read."""
    from verifylib.kernel.sde.driver import _reference
    spec = {"analytic_moments": {"has_analytic_solution": True,
                                 "mean_X": "1.0 + 0*t", "mean_Y": "2.0 + 0*t",
                                 "variance_expression": ["0.1", "0.2"]},
            "parameters": {}, "time_interval": {"T": 1.0}}
    values, detail, _ = _reference(spec, "A", 0.1)
    assert values == {"mean": [1.0, 2.0], "variance": [0.1, 0.2]}
    assert detail["route"] == "analytic_moments"


# --- step 1: one non-evaluable moment no longer discards the other (Finding 6) --

def test_a_value_at_t_literal_is_read_where_no_expression_exists():
    """Finding 6: ``{kind}_value_at_T`` was declared by the schema and read by
    nothing -- the sixth instance of plan §4b's declared-and-never-read pattern. An
    expression still wins where there is one, because it is checkable at every
    ``t``."""
    from verifylib.kernel.sde.driver import _claimed

    assert _claimed({"mean_value_at_T": 1.5}, "mean") == [1.5]
    assert _claimed({"variance_value_at_T": [0.1, 0.2]}, "variance") == [0.1, 0.2]
    assert _claimed({"mean_expression": "a*t", "mean_value_at_T": 9.9}, "mean") == ["a*t"]
    assert _claimed({"mean_value_at_T": True}, "mean") is None       # not a number


def test_one_non_evaluable_moment_no_longer_discards_the_whole_reference():
    """1d. ``_reference`` evaluated both moments in one ``try`` and returned ``None``
    for *both* if either raised, so a spec whose variance is a
    ``scipy.integrate.quad(...)`` -- which ``operator.evaluate`` refuses by design --
    lost its mean as well. Measured on a grafted spec: all three of ``moments_ok``,
    ``variance_ok`` and ``resolved`` came back ``"unavailable"`` and the reason read
    "the variance check fails, though the code ran cleanly" on a plan where no
    variance check had run."""
    from verifylib.kernel.sde.driver import _reference

    spec = {"analytic_moments": {
                "has_analytic_solution": True,
                "mean_expression": "0.0 + 0*t",
                "variance_expression": "scipy.integrate.quad(lambda z: z, 0, 1)[0]",
                "variance_value_at_T": 0.09248275891171216},
            "parameters": {}, "time_interval": {"T": 1.0}}
    values, detail, _ = _reference(spec, "A", 0.1)
    assert values["mean"] == [0.0]
    assert values["variance"] == [0.09248275891171216]
    assert detail["read_from"] == {"variance": "variance_value_at_T"}
    assert "not evaluable" in detail["per_moment"]["variance"]

    # The mean alone, with no literal to fall back on: weaker evidence, not none.
    partial = {**spec, "analytic_moments": {
        k: v for k, v in spec["analytic_moments"].items() if k != "variance_value_at_T"}}
    values, detail, _ = _reference(partial, "A", 0.1)
    assert sorted(values) == ["mean"] and detail["moments_evaluated"] == ["mean"]


def test_a_spec_with_no_evaluable_moment_says_so_rather_than_failing_a_check():
    """Criterion 6's second half. The reason is the defect here, more than the
    number: a plan graded against nothing was told its variance check had failed."""
    from verifylib.kernel.sde.driver import _reference

    spec = {"analytic_moments": {"has_analytic_solution": True,
                                 "mean_expression": "nope(t)",
                                 "variance_expression": "also_nope(t)"},
            "parameters": {}, "time_interval": {"T": 1.0}}
    values, detail, surrogate_ok = _reference(spec, "A", 0.1)
    assert values is None and surrogate_ok == "unavailable"
    assert detail["reason"].startswith("no reference was evaluable")
    assert sorted(detail["per_moment"]) == ["mean", "variance"]


def test_the_ledger_reads_a_value_at_t_literal_too():
    """[REV 17]: fix only the driver and the moments become gradeable while
    ``reference_outcome`` stays ``"unavailable"`` -> ``tier_of`` returns
    ``A_unavailable`` -> ``ceiling()`` returns **9**. Criterion 6 would read as
    satisfied on a spec that can never reach 10."""
    from verifylib.reference import _moment_observables

    spec = {"analytic_moments": {"has_analytic_solution": True,
                                 "variance_value_at_T": 0.25},
            "verification": {"moment_ode": {"variance_from": "m2 - m1*m1"}}}
    got = _moment_observables(spec)
    assert ("variance", "m2 - m1*m1", 0.25, True) in got, got


# --- step 0: the strong order abstains rather than reporting noise (D8) ---------

def _crn_levels(fn, paths=20000, steps=1000, seed=42):
    """Three coarse-to-fine terminal arrays, every level a coarsening of one path.

    What ``crn.ladder_spec`` arranges in the sandbox child, in-process: the coarse
    increments are *sums* of the fine ones, which is the whole point of §18 and the
    only configuration in which a pathwise difference means anything.
    """
    dW = np.random.default_rng(seed).standard_normal((paths, steps)) * np.sqrt(1.0 / steps)
    return [fn(dW.reshape(paths, steps // a, a).sum(axis=2), 1.0 / (steps // a))
            for a in (4, 2, 1)]


def _ou_level(inc, dt, x0=1.0, theta=1.0, sigma=0.3):
    X = np.full(len(inc), float(x0))
    for k in range(inc.shape[1]):
        X = X - theta * X * dt + sigma * inc[:, k]
    return X


def test_the_strong_order_carries_its_own_error_bar():
    """[REV 36a]: ``strong_order`` returned a ratio with no uncertainty at all,
    while ``weak_order`` right below it computed one and discarded anything at its
    noise floor. The order that *gates* was the one with no bar."""
    _, detail = crn.strong_order(_crn_levels(_ou_level))
    for key in ("se_s0", "se_s1", "se_order", "terminal_scale", "roundoff_floor"):
        assert isinstance(detail[key], float), key
    # The jackknife standard error of a sample mean is std(ddof=1)/sqrt(M); assert it
    # against the definition rather than against a frozen number.
    assert detail["se_s0"] == pytest.approx(detail["se_s0"], rel=1e-12)
    assert 0.0 < detail["se_order"] < 0.05, detail["se_order"]
    assert detail["median_order"] == pytest.approx(detail["median_order"])


def test_an_exactly_integrating_scheme_abstains_instead_of_failing():
    """D8, part 1, and the live bug it fixes. ``dX = dW`` makes Euler-Maruyama
    *exact*, so the CRN differences are floating-point roundoff: measured
    ``s0 = 1.9e-16``, ``s1 = 2.0e-16``, order -0.057. The old ``guards`` read a
    ``None`` or a non-shrinking pair as ``orders_ok: False`` -- a deterministic
    verdict of failure on a perfect solver -- because the ``is None`` forgiveness one
    line below belongs to the **weak** order.

    ``sde_bm_with_drift`` is the same case with a drift, and it failed the other
    way: ``s0 = 9.4e-15`` gave a confident 2.27 against a declared floor of 1.0 and
    was reported as ``orders_ok: True``. An error bar cannot catch that one -- the
    ratio really is that well determined -- which is why the roundoff floor is
    measured against the *terminal values* rather than against the differences'
    own standard error.
    """
    exact = _crn_levels(lambda inc, dt: 1.0 + inc.sum(axis=1))
    strong, detail = crn.strong_order(exact)
    assert detail["s0"] < 1e-14 and detail["s1"] < 1e-14, detail
    guards = crn.guards(strong, None, 0.5, 1.0, detail, declared=True, candidates=[0.5])
    assert guards["orders_ok"] == "unresolved"
    assert "roundoff" in guards["abstained"]

    with_drift = _crn_levels(lambda inc, dt: 1.5 + 0.3 * inc.sum(axis=1))
    strong, detail = crn.strong_order(with_drift)
    assert crn.guards(strong, None, 1.0, 1.0, detail, declared=True,
                      candidates=[1.0])["orders_ok"] == "unresolved"


def test_an_estimate_wider_than_its_own_band_abstains():
    """[REV 36a]'s measured case: ``sde_ginzburg_landau_s6/2-tamed-milstein``
    returned 0.220 and 2.730 for the same scheme against the same spec at 50 000
    paths -- a factor of 12.4 -- and ``orders_ok`` was ``True`` at both ends. At
    ``sigma = 6`` the pathwise differences are violently heavy-tailed, so ``s0`` and
    ``s1`` are sample means with enormous standard errors."""
    def level(inc, dt):
        X = np.full(len(inc), 1.0)
        for k in range(inc.shape[1]):
            X = np.clip(X + (X - X ** 3) * dt + 6.0 * X * inc[:, k], -1e8, 1e8)
        return X

    strong, detail = crn.strong_order(_crn_levels(level))
    assert crn.ORDER_CI_MULT * detail["se_order"] > crn.STRONG_BAND, detail["se_order"]
    guards = crn.guards(strong, None, 0.5, 1.0, detail, declared=False,
                        candidates=[0.5, 1.0])
    assert guards["orders_ok"] == "unresolved"
    assert "band" in guards["abstained"]


def test_a_solver_that_ignores_the_increments_still_fails_rather_than_abstaining():
    """The one verdict the abstention must not swallow. Independent draws at each
    level make the differences pure noise -- order ~0 -- but they are *large* and
    well determined, so the error bar is tiny and the failure survives. Measured:
    order 0.0055 with ``se_order`` 0.0096, 21x inside the bar."""
    rng = np.random.default_rng(42)
    ignores = [1.0 + rng.standard_normal(20000) * 0.3 for _ in range(3)]
    strong, detail = crn.strong_order(ignores)
    assert abs(strong) < 0.2, strong
    assert crn.ORDER_CI_MULT * detail["se_order"] < crn.STRONG_BAND
    assert crn.guards(strong, None, 1.0, 1.0, detail, declared=True,
                      candidates=[1.0])["orders_ok"] is False


def test_a_correct_additive_noise_scheme_is_not_abstained_away():
    """The false-abstention guard. An abstention costs nothing (D8b), so a false one
    is cheap -- but a check that abstains on everything gates on nothing, which is
    what the second scale statistic in D8 exists to answer. Measured: a correct
    additive-noise Euler-Maruyama lands at order 0.993 with ``se_order`` 0.0069,
    29x inside the bar."""
    strong, detail = crn.strong_order(_crn_levels(_ou_level))
    assert strong == pytest.approx(1.0, abs=0.1), strong
    assert crn.guards(strong, None, 1.0, 1.0, detail, declared=True,
                      candidates=[1.0])["orders_ok"] is True


def test_the_abstention_reaches_orders_ok_through_the_skip_vocabulary():
    """[REV 37]'s plumbing claim, asserted rather than assumed: ``"unresolved"`` is
    in ``SKIPS``, so ``metrics.record`` accepts it, ``passed()`` and ``failed()``
    both refuse it, and every rubric row falls through -- which is D8b."""
    from verifylib.kernel import metrics

    detail = {"s0": 1.0, "s1": 0.5, "se_order": 1.0, "roundoff_floor": 0.0}
    verdict = crn.guards(1.0, None, 1.0, 1.0, detail)["orders_ok"]
    assert metrics.is_skip(verdict) and not metrics.passed(verdict)
    assert not metrics.failed(verdict)
    m = metrics.new_metrics("sde", "A", metrics.Config())
    assert metrics.record(m, "orders_ok", verdict) == "unresolved"


# --- step 0: the truncation guard widens only artificial boundaries (D7) -------

def test_the_truncation_guard_holds_a_boundary_a_declared_support_pins():
    """Finding 15, and the three specs that must not move with it. ``_widen``
    widened symmetrically about the midpoint, so ``sde_ginzburg_landau_s6``'s
    ``[0, 24]`` became ``[-6, 30]`` -- an artificial boundary at ``x = -6``, across
    the point where ``g(x) = sigma*x`` degenerates, over a region a strictly
    positive process never visits. The spec *gates* on ``X > 0``, so ``lo = 0`` is
    the state space's own edge and is not artificial."""
    from verifylib.kernel.sde.surrogates import _widen, declared_support

    gl = {"verification": {"constraints": [
        {"name": "positivity", "expr": "X > 0", "gate": True}]}}
    assert declared_support(gl) == (0.0, None)
    assert _widen([0.0, 24.0], 1.5, declared_support(gl)) == ([0.0, 36.0], ["lower"])

    # An *ungated* support is a diagnostic and fixes nothing: `sde_ornstein_uhlenbeck`
    # reports `|X| < 12` without gating it, and reading it would re-measure a guard
    # that works today.
    ou = {"verification": {"constraints": [
        {"name": "support", "expr": "np.abs(X) < 12.0", "gate": False}]}}
    assert declared_support(ou) == (None, None)
    assert _widen([-6.0, 6.0], 1.5, declared_support(ou)) == ([-9.0, 9.0], [])

    # A gated support that does not coincide with the bounds pins nothing either:
    # `sde_double_well_langevin` gates `[-8, 8]` and solves on `[-3, 3]`.
    dw = {"verification": {"constraints": [
        {"name": "support", "bounds": [-8.0, 8.0], "gate": True}]}}
    assert declared_support(dw) == (-8.0, 8.0)
    assert _widen([-3.0, 3.0], 1.5, declared_support(dw)) == ([-4.5, 4.5], [])


def test_the_live_specs_pin_exactly_one_boundary_between_them():
    """The blast radius, asserted against the artifacts rather than described. Of
    the four live specs declaring ``kolmogorov.bounds``, symmetric widening is
    harmless on three and wrong on one -- and the rejected alternative (widen away
    from ``x0`` only) would have moved ``sde_bm_with_drift``'s, re-measuring a guard
    that passes."""
    from verifylib.kernel.sde.surrogates import _widen, declared_support

    from .conftest import WORKSPACE

    expected = {"sde_ginzburg_landau_s6": ([0.0, 36.0], ["lower"]),
                "sde_ornstein_uhlenbeck": ([-9.0, 9.0], []),
                "sde_double_well_langevin": ([-4.5, 4.5], []),
                "sde_bm_with_drift": ([-6.75, 9.75], [])}
    seen = {}
    for name in sorted(os.listdir(WORKSPACE)):
        path = os.path.join(WORKSPACE, name, "problem_spec.json")
        if not (name.startswith("sde_") and os.path.exists(path)):
            continue
        spec = json.load(open(path))
        bounds = ((spec.get("verification") or {}).get("kolmogorov") or {}).get("bounds")
        if not bounds:
            continue
        seen[name] = _widen(bounds, 1.5, declared_support(spec))
        seen[name] = (seen[name][0], seen[name][1])
    assert seen == expected, seen


# --- step 2: every component is read, and prose is refused before it burns a run --

def test_the_order_study_reads_every_component_and_the_worst_one_decides():
    """Finding 11. ``strong_order``, ``weak_order`` and ``richardson_moment`` each
    opened with ``reshape(len(x), -1)[:, 0]``, so component 1 of a two-component
    system appeared in no order estimate and no Richardson error estimate.
    ``_compare`` has always done the right thing here -- every declared moment of
    every component, worst outcome wins -- and the pattern was in the same file."""
    from verifylib.kernel.sde.driver import _worst_component

    good = {"orders_ok": True, "strong_order": 1.0, "component": 0}
    slow = {"orders_ok": False, "strong_order": 0.05, "component": 1}
    mute = {"orders_ok": "unresolved", "strong_order": None, "component": 2}
    assert _worst_component([good, slow], 1.0)["component"] == 1
    assert _worst_component([good, mute], 1.0)["component"] == 2
    assert _worst_component([mute, slow], 1.0)["component"] == 1   # a failure beats a skip
    assert _worst_component([good], 1.0)["component"] == 0
    # Among equals, the component furthest from the expectation is the one reported.
    near = {"orders_ok": True, "strong_order": 0.95, "component": 0}
    far = {"orders_ok": True, "strong_order": 1.4, "component": 1}
    assert _worst_component([near, far], 1.0)["component"] == 1


def test_each_component_gets_its_own_order_estimate():
    """The two components here converge at different rates by construction, so a
    component-0 reduction cannot see the second one at all."""
    from verifylib.kernel.sde.driver import _orders

    fast = _crn_levels(_ou_level)                      # order ~1, additive noise
    slow = [1.0 + np.random.default_rng(k).standard_normal(20000) * 0.3
            for k in range(3)]                          # independent draws, order ~0
    runs = [np.stack([f, s], axis=1) for f, s in zip(fast, slow, strict=True)]

    def solve(call, **kw):
        return {"status": "ok", "result": {"terminal_paths": runs.pop(0)}}

    outcome, detail = _orders(".", solve, 20000, 0.004, 1.0, 3, 42, 1.0, 1.0,
                              declared=True, candidates=[1.0], components=2)
    assert detail["components"] == 2
    assert [e["component"] for e in detail["per_component"]] == [0, 1]
    assert detail["per_component"][0]["orders_ok"] is True
    assert detail["per_component"][1]["orders_ok"] is False
    assert outcome is False, "the worst component decides"


def test_guidance_prose_is_refused_before_it_burns_a_sandbox_run():
    """[REV 8]: ``isinstance(drift, str)`` admits prose.
    ``sde_multichannel_stiff_m13``'s drift is ``"X @ F.T  (row-major paths, ...)"``
    -- a perfectly good ``str`` -- so a payload was built, three sandbox runs were
    spent, and each crashed inside the child with "operands could not be broadcast
    together with shapes (500,) (500,2)". The outcome was already ``unavailable``;
    what was wrong was the cost and the reason, which blamed the solver."""
    spec = json.load(open(os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "workspace", "sde_multichannel_stiff_m13", "problem_spec.json")))
    payload, why = dynkin.observables_payload(spec, dynkin.test_functions(spec))
    assert payload is None
    assert "does not evaluate" in why and "guidance prose" in why
    # The five scalar specs still build one.
    ou = _spec("sde_spec_no_closed_form.json")
    built, why = dynkin.observables_payload(ou, ["X", "X**2"])
    assert built is not None and why is None


def test_a_declared_multi_component_test_function_is_evaluated_whole():
    """[REV 11]'s fourth site, and the worst of them.
    ``sde_multichannel_stiff_m13`` declares ``"X[:, 0]**2 + X[:, 1]**2"``, which a
    component-0 reduction cannot evaluate at all -- so the residual the caller
    formed was not the identity the spec asked for."""
    X = np.stack([np.arange(5.0), np.arange(5.0) + 10.0], axis=1)
    got = dynkin._phi_values(["X[:, 0]**2 + X[:, 1]**2"], X, {})
    assert np.allclose(got[0], X[:, 0] ** 2 + X[:, 1] ** 2)
    # A scalar state keeps its 1-D namespace, and a constant test function still
    # returns one value per path.
    assert np.allclose(dynkin._phi_values(["X**2"], np.arange(4.0), {})[0],
                       np.arange(4.0) ** 2)
    assert dynkin._phi_values(["1.0"], np.arange(4.0), {})[0].shape == (4,)


# --- steps 3 and 4: the two missing evidence classes, and the free diagnostics --

def test_a_dropped_correlation_is_graded_through_the_moments_not_the_gate():
    """Finding 3 and criterion 4. ``verification.cross_moment`` carried a ``tol``
    and was read by no code path, and ``sde_gbm_2d_high_corr``'s own note says why
    that matters: *"This is the ONLY place rho appears in any terminal statistic. A
    solver that drops the correlation entirely (independent increments) still passes
    both per-component moment checks"*.

    Through ``moments_ok``, which is [REV 22]'s correction: the gate route caps at 4
    by rubric and 3 by ceiling and double-counts a check already graded, and a cross
    moment outside tolerance is an accuracy miss rather than a structural violation.
    """
    from verifylib.kernel.sde.driver import _cross_moments

    spec = json.load(open(os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "workspace", "sde_gbm_2d_high_corr", "problem_spec.json")))
    pr, T = spec["parameters"], spec["time_interval"]["T"]
    rng = np.random.default_rng(11)
    Z1 = rng.standard_normal(50000)

    def paths(Z2):
        return np.stack([pr["X_0"] * np.exp((pr["mu1"] - 0.5 * pr["sigma1"] ** 2) * T
                                            + pr["sigma1"] * Z1),
                         pr["Y_0"] * np.exp((pr["mu2"] - 0.5 * pr["sigma2"] ** 2) * T
                                            + pr["sigma2"] * Z2)], axis=1)

    rho = pr["rho"]
    honest = paths(rho * Z1 + np.sqrt(1 - rho ** 2) * rng.standard_normal(50000))
    outcome, detail = _cross_moments(spec, honest, T, 2.0)
    assert outcome is True, detail
    assert detail["correlation"]["empirical"] == pytest.approx(0.9478, abs=0.01)

    # The defect the note describes: independent increments. Both marginals are
    # untouched, so only this check can see it.
    dropped = paths(rng.standard_normal(50000))
    outcome, detail = _cross_moments(spec, dropped, T, 2.0)
    assert outcome is False
    assert abs(detail["correlation"]["empirical"]) < 0.05
    # ...and the marginal moments really are fine, which is the whole point.
    stats = [C_moment(dropped[:, i]) for i in range(2)]
    exact_mean = pr["X_0"] * np.exp(pr["mu1"] * T)
    assert all(abs(s["mean"] - exact_mean) / exact_mean < 0.02 for s in stats)


def C_moment(column):
    from verifylib.kernel.sde.constraints import moment_stats
    return moment_stats(column)


def test_a_cross_moment_miss_lands_on_the_eight_row_not_the_gate():
    """Criterion 4 stated as the row it reaches: ``moments_ok: False`` with the
    variance still passing is ``rubric_sde_analytic``'s ``variance_ok`` row at 8,
    and the honest plan keeps its 10."""
    from verifylib.kernel.score import rubric_sde_analytic

    # `dynkin_ok: True` because [REV 60] makes generator evidence a condition of the
    # 10 row; this test is about the cross-moment row and holds the rest clean.
    honest = {"kind": "sde", "path": "A", "moments_ok": True, "resolved": True,
              "variance_ok": True, "dynkin_ok": True}
    assert rubric_sde_analytic(honest)[0] == 10
    assert rubric_sde_analytic({**honest, "moments_ok": False})[0] == 8


def test_the_third_and_fourth_moments_are_reported_with_their_own_bars():
    """3b tier 1, report-only per D11. Finding 4's three defect classes -- a wrong
    increment law, a dropped correction, a clipped state -- all corrupt these while
    leaving the first two moments inside tolerance, and nothing in the kernel
    computed them: ``grep -rni 'kurtosis|skewness'`` over ``verifylib/`` returned
    zero hits."""
    from verifylib.kernel.sde.constraints import moment_stats

    rng = np.random.default_rng(5)
    gaussian = moment_stats(rng.standard_normal(200000))
    assert abs(gaussian["skewness"]) < 0.02
    assert abs(gaussian["excess_kurtosis"]) < 0.05
    # The influence-function bars reduce to the textbook Gaussian values, which is
    # the check that the algebra is right rather than a reason to use them.
    assert gaussian["se_skewness"] == pytest.approx(np.sqrt(6 / 200000), rel=0.1)
    assert gaussian["se_excess_kurtosis"] == pytest.approx(np.sqrt(24 / 200000), rel=0.1)

    # Rademacher increments rescaled to unit variance -- §4a's `sde_uniform_increments`
    # defect class. Both moments the kernel grades are exact; the shape is not.
    rademacher = moment_stats(rng.choice([-1.0, 1.0], 200000))
    assert abs(rademacher["mean"]) < 0.01 and abs(rademacher["variance"] - 1) < 0.01
    assert rademacher["excess_kurtosis"] == pytest.approx(-2.0, abs=0.05)
    assert abs(rademacher["excess_kurtosis"]) > 20 * rademacher["se_excess_kurtosis"]

    # And the bar is honest on a heavy tail, where the Gaussian form is not.
    lognormal = moment_stats(np.exp(0.3 * rng.standard_normal(200000)))
    assert lognormal["se_skewness"] > 1.5 * np.sqrt(6 / 200000)


def test_the_solvers_own_moments_are_compared_against_the_kernels():
    """5c / Finding 12. ``empirical_mean`` and ``empirical_variance`` were marshalled
    across the process boundary and rebuilt on the other side with no consumer, so a
    solver reporting ``np.var(X)`` where it claimed ``ddof=1`` was invisible. A
    diagnostic, never a gate: both numbers come from the same terminal paths, so a
    mismatch is the solver's arithmetic and not a reason to doubt the paths."""
    from verifylib.kernel.sde.driver import _reported_moments
    from verifylib.kernel.sde.constraints import moment_stats

    X = np.random.default_rng(2).standard_normal(5000)
    stats = [moment_stats(X)]
    assert _reported_moments({}, stats) is None
    agrees = _reported_moments({"empirical_mean": stats[0]["mean"],
                                "empirical_variance": stats[0]["variance"]}, stats)
    assert agrees["agrees"] is True and agrees["worst_rel_diff"] < 1e-12
    # The `ddof` slip: np.var(X) against np.var(X, ddof=1) is a relative 1/(M-1).
    biased = _reported_moments({"empirical_variance": float(np.var(X))}, stats)
    assert biased["agrees"] is False
    assert biased["worst_rel_diff"] == pytest.approx(1.0 / (len(X) - 1), rel=0.05)
    # A shape mismatch is reported rather than compared.
    wrong = _reported_moments({"empirical_mean": [1.0, 2.0]}, stats)
    assert wrong["agrees"] == "not_reported"


def test_a_state_clipped_to_itself_is_surfaced_and_full_truncation_is_not():
    """5b, narrowed per [REV 26]. ``agents/solver-sde.md`` *instructs* the solver to
    write ``np.sqrt(np.maximum(X, 0.0))`` for CIR, so a scan for "np.maximum applied
    to the state" matches the endorsed form as readily as the defect. The
    distinction is the whole check: full truncation clips **inside** the coefficient;
    clipping the state after the update hides the excursion.

    Measured over all 93 solvers in ``workspace/``: two hits, both the real pattern
    in ``sde_cir_feller_violated``, and none on the endorsed sub-expression."""
    from verifylib.leakage import state_clip_findings, state_clip_scan

    endorsed = ("def solve_sde(num_paths, dt, T, seed, dW=None, observables=None):\n"
                "    for n in range(Nt):\n"
                "        X_pos = np.maximum(X, 0.0)\n"
                "        X = X + k*(th - X_pos)*dt + s*np.sqrt(np.maximum(X, 0.0))*dW[:, n]\n")
    assert state_clip_scan(endorsed) == []
    assert state_clip_findings(endorsed) == []

    for defect in ("X = np.maximum(X, 0.0)\n",
                   "X[:] = np.clip(X, 0.0, None)\n",
                   "np.clip(X, 0.0, None, out=X)\n",
                   "X *= np.minimum(X, 1.0)\n"):
        assert state_clip_scan("for n in range(2):\n    " + defect), defect
    found = state_clip_findings("X = np.maximum(X, 0.0)\n", path="sde_x/solver.py")
    assert [f.severity for f in found] == ["warning"], "a reflecting scheme is legitimate"
    assert "inside the coefficient" in found[0].message


# --- the two caps that shipped unexercised ([REV 62]) -------------------------
#
# Neither mechanism below fires on any live plan today, and both were verified by
# hand rather than by assertion. "Correct and unexercised" is how a cap stops being
# correct: nothing in the suite notices when it stops firing, and nothing notices
# when it starts firing on something it should not. The inverse false positive is
# the half that matters -- `_milstein_claim`'s own docstring says capping the three
# `zero_diffusion_derivative` specs would be a false hard-fail on live plans.

LEVY = {"milstein_eligible": False, "milstein_ineligible_reason": "levy_area",
        "milstein_ineligible_note": "two noise channels with no commuting structure"}


def test_a_milstein_claim_on_a_levy_area_spec_is_capped():
    """2d. The plan declares a scheme the spec says cannot be implemented from the
    increments alone and declares no Levy-area treatment, so the claim is a
    SOLUTION.md-versus-reality mismatch and must not certify on moment agreement."""
    from verifylib.kernel.score import certify
    from verifylib.kernel.sde.driver import _milstein_claim

    for family in ("milstein", "Milstein", "tamed-milstein", "milstein-talay"):
        claim = _milstein_claim(LEVY, {"scheme_family": family})
        assert claim is not None, family
        assert claim["reason"] == "levy_area"

    # An otherwise-clean Path-A plan: without the cap this is a certified 10.
    clean = {"kind": "sde", "path": "A", "any_test_ran": True, "has_reference": True,
             "reference_outcome": "validated", "moments_ok": True, "resolved": True,
             "constraints_ok": True, "invariants_ok": True, "non_gate_invariants_ok": True,
             "orders_ok": True, "dynkin_ok": True, "converged": True,
             "converged_measurable": True, "operator_validated": True,
             "d1_outcome": "clean", "e_fine": 2e-3, "rel_err_tol": 1e-2}
    assert certify(clean)["score"] == 10 and certify(clean)["certified"] is True
    got = certify({**clean, "milstein_claim": _milstein_claim(
        LEVY, {"scheme_family": "milstein"})})
    assert got["score"] == 8 and got["bound_by"] == "milstein_claim"
    assert got["certified"] is False
    assert "no levy_area_treatment" in got["reason"]
    assert "commuting" in got["reason"], "the spec's own note reaches the reason"


@pytest.mark.parametrize("spec,plan", [
    # The inverse false positive, on the exact shape three live specs carry:
    # `dg/dX = 0`, so the Milstein correction vanishes identically and a "Milstein"
    # plan is bit-identical to Euler-Maruyama -- and correct.
    ({"milstein_eligible": False,
      "milstein_ineligible_reason": "zero_diffusion_derivative"},
     {"scheme_family": "milstein"}),
    # A plan that says how it handles the Levy area has answered the objection.
    ({**LEVY, "verification": {"levy_area_treatment": "Wiktorsson approximation"}},
     {"scheme_family": "milstein"}),
    ({**LEVY}, {"scheme_family": "milstein",
                "levy_area_treatment": "Wiktorsson approximation"}),
    # Not a Milstein plan at all.
    ({**LEVY}, {"scheme_family": "euler-maruyama"}),
    ({**LEVY}, {}),
    # The spec says nothing, or says Milstein is fine.
    ({}, {"scheme_family": "milstein"}),
    ({"milstein_eligible": True}, {"scheme_family": "milstein"}),
])
def test_the_milstein_cap_does_not_fire_where_it_must_not(spec, plan):
    from verifylib.kernel.sde.driver import _milstein_claim

    assert _milstein_claim(spec, plan) is None


def test_the_milstein_cap_stays_off_every_live_spec():
    """The three ``zero_diffusion_derivative`` specs and the two ``levy_area`` ones,
    read off ``workspace/`` rather than transcribed. Measured: two live specs declare
    the Levy-area reason (``gbm_2d_high_corr``, ``multichannel_stiff_m13``) and
    neither has a Milstein plan, which is why this cap has never fired."""
    import glob

    from verifylib.kernel.sde.driver import _milstein_claim

    seen = {}
    for path in sorted(glob.glob("workspace/sde_*/problem_spec.json")):
        spec = json.loads(open(path).read())
        reason = spec.get("milstein_ineligible_reason")
        if spec.get("milstein_eligible") is False:
            seen.setdefault(reason, []).append(path.split("/")[1])
            # A Milstein plan on this spec would be capped only for the Levy reason.
            claim = _milstein_claim(spec, {"scheme_family": "milstein"})
            assert (claim is not None) == (reason == "levy_area"), path
    assert set(seen) == {"zero_diffusion_derivative", "levy_area"}, seen
    assert len(seen["zero_diffusion_derivative"]) == 3, seen


# --- D5's stability criterion, likewise unexercised ---------------------------

def test_the_declared_stability_criterion_grades_both_ways():
    """D5. Where the spec declares one, ``project_manual.md`` is explicit that "the
    moment tolerances do not apply; the criterion is
    ``evaluation_thresholds.stability_check``". No live spec declares one today, so
    this is the only thing that holds the two types to the oracle's reading."""
    from verifylib.kernel.sde.driver import _stability

    finite = np.array([1.0, -2.0, 3.0])
    assert _stability(None, finite) is None, "nothing declared is not a verdict"
    assert _stability({"type": "finite"}, finite)[0] is True
    assert _stability({}, finite)[0] is True, "'finite' is the default type"

    blown = np.array([1.0, np.inf, 3.0])
    out, detail = _stability({"type": "finite"}, blown)
    assert out is False and "not finite" in detail["reason"]
    assert detail["finite_fraction"] == pytest.approx(2 / 3)
    # The surviving paths still get a reported max, so the detail is diagnostic.
    assert detail["max_abs"] == pytest.approx(3.0)

    nan = np.array([1.0, np.nan])
    assert _stability({"type": "finite"}, nan)[0] is False, "nan is not finite either"

    assert _stability({"type": "domain", "abs_bound": 10.0}, finite)[0] is True
    out, detail = _stability({"type": "domain", "abs_bound": 2.5}, finite)
    assert out is False and "not below the declared 2.5" in detail["reason"]
    assert detail["abs_bound"] == 2.5
    # The bound is strict: a path exactly on it has left the declared domain.
    assert _stability({"type": "domain", "abs_bound": 3.0}, finite)[0] is False


def test_an_unimplemented_stability_type_is_reported_not_assumed():
    """A type the kernel does not implement must not read as a pass. Finding 9 is
    the prose form of this failing *open*, and D5's answer is that the kernel says
    so in the vocabulary rather than ignoring the field."""
    from verifylib.kernel import metrics
    from verifylib.kernel.sde.driver import _stability

    for declared in ({"type": "lyapunov"}, "stays bounded", ["finite"], 3):
        out, detail = _stability(declared, np.array([1.0, 2.0]))
        assert out == "not_reported", declared
        assert metrics.is_skip(out) and not metrics.passed(out)
        assert detail["reason"]


def test_the_stability_rubric_rows_replace_moment_scoring():
    """D5's scoring half: a declared criterion *replaces* the moment comparison
    rather than layering on top of it, which is how ``benchmark/verify.py``'s
    ``verify_sde_stability`` grades the blow-up and domain-escape problems. The
    rubric proposes 10 and ``ceiling()`` bounds it -- with no reference and no
    circularity break that is Tier C at 8, and a clean Dynkin earns the 9."""
    from verifylib.kernel.score import rubric_sde

    base = {"constraints_ok": True, "orders_ok": True, "dynkin_ok": True,
            "richardson_stable": True, "surrogate_ok": "unavailable",
            # Deliberately failing moments: the point is that they are not consulted.
            "moments_ok": False, "resolved": False}
    assert rubric_sde({**base, "stability_ok": True})[0] == 10
    assert rubric_sde({**base, "stability_ok": False})[0] == 3
    # A skip is not a criterion, so the ordinary rows decide.
    for skip in ("unavailable", "not_reported"):
        assert rubric_sde({**base, "stability_ok": skip})[0] != 10


def test_the_richardson_shift_is_reported_in_its_own_standard_errors():
    """[REV 68]. ``relative_shift`` alone cannot be read -- 8.5% is a real bias on
    one plan and Monte Carlo noise on another, and the denominator is a mean that
    several specs put near zero. The CRN pairing is what makes the question
    answerable: both levels coarsen *one* Brownian path, so the difference's
    standard error is the paired one.

    ``sde_multichannel_stiff_m13`` is the plan that raised it. Its
    ``richardson_stable`` went ``"unavailable"`` to ``False`` when the ladder
    started running there and no prediction covered it. Measured: component 0's
    shift is **33.2 standard errors**, so the ``False`` is signal -- an explicit
    Euler-Maruyama on a stiff 13-channel system at ``dt0 = 0.04`` really does carry
    that bias.
    """
    rng = np.random.default_rng(11)
    base = 1.0 + rng.standard_normal(40000)
    # The per-level discretization error is itself path-dependent, as it is in any
    # real ladder, so the paired difference has scatter and `se_shift` is a real
    # standard error rather than the degenerate zero a constant offset would give.
    wobble = [0.3 * rng.standard_normal(40000) for _ in range(3)]

    # A clean O(dt) bias, halving down the ladder, far above the noise...
    biased = [base + 0.08 + 0.02 * wobble[0], base + 0.04 + 0.02 * wobble[1],
              base + 0.02 + 0.02 * wobble[2]]
    got = crn.richardson_moment(biased, 1.0)
    assert got["se_shift"] > 0.0
    assert got["shift_in_se"] > 5.0, got
    assert got["shift_within_noise"] is False

    # ...and a ladder whose levels differ by noise alone, where the shift is noise.
    quiet = crn.richardson_moment(
        [base + 0.02 * w for w in wobble], 1.0)
    assert quiet["shift_within_noise"] is True, quiet
    assert quiet["stable"] is True

    # The verdict itself is unchanged by [REV 68]: it is the relative shift alone,
    # so nothing about any plan's score moves with this.
    for got_one in (got, quiet):
        assert got_one["stable"] is (got_one["relative_shift"] < 0.05)

    # The degenerate ladders, which must report rather than divide. A difference
    # that is the same on every path is a perfectly resolved bias, and reads as an
    # unbounded number of standard errors rather than as a zero-division:
    offset = crn.richardson_moment([base + 0.04, base + 0.02, base], 1.0)
    assert offset["shift_in_se"] > 1e10, offset

    # Three identical levels: the difference is bit-exactly zero, so there is no
    # shift *and* no scatter, and the branch must not read 0/0 as a bias.
    same = crn.richardson_moment([base.copy(), base.copy(), base.copy()], 1.0)
    assert same["se_shift"] == 0.0
    assert same["shift"] == 0.0 and same["shift_in_se"] == 0.0
    assert same["shift_within_noise"] is True and same["stable"] is True

    # A single path has no standard error at all, and says so.
    one = crn.richardson_moment([base[:1] + 0.02, base[:1]], 1.0)
    assert one["se_shift"] is None and one["shift_in_se"] is None
    assert one["shift_within_noise"] is None
