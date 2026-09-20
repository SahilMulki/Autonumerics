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

from verifylib.kernel import plan_meta, run, sandbox
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


@pytest.mark.xfail(strict=True, reason=(
    "manual §23 / rubric_sde give 9 for 'orders, Dynkin, constraints and a stable "
    "Richardson moment with no surrogate', but the SDE driver hardcodes "
    "operator_validated=False, so a clean Dynkin never counts as a circularity break "
    "and §4c's ceiling caps at 8 self_convergence. The 9 row is unreachable. Measured: "
    "the honest solver scores 8 on sde_spec_no_surrogate.json"))
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


@pytest.mark.xfail(strict=True, reason=(
    "rubric_sde tests failed(resolved) before dynkin_ok, so a wrong drift whose "
    "exploded variance widens the CI is scored 6 'MC-inconclusive: not a solver bug' "
    "while Dynkin sits at z ~ 1500. The message is false and the feedback it drives "
    "(raise num_paths) is wrong"))
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


@pytest.mark.xfail(strict=True, reason=(
    "rubric_sde's surrogate row (surrogate and moments_ok and resolved and "
    "constraints_ok -> 10) never consults orders_ok, so a solver that violates the "
    "CRN contract and whose order study therefore measured nothing still certifies "
    "at 10. Manual §7's rule for `override` -- a solver that ignores its inputs "
    "cannot be certified -- has no §18 counterpart in the rubric. Decide whether it "
    "should; this records the current behaviour either way"))
def test_a_solver_that_ignores_the_crn_increments_does_not_certify(tmp_path):
    m = _run(tmp_path, "sde_ignores_crn.py")
    assert m["score"] < 10, m["certification"]


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
    """I9(b): the error bar belongs to ``2 r_fine - r_coarse`` path by path, not
    to ``r_fine``. With anti-correlated levels the two differ by a factor of ~3."""
    rng = np.random.default_rng(0)
    fine = rng.standard_normal(4000)
    coarse = -fine + 0.01 * rng.standard_normal(4000)
    outcome, detail = dynkin.check(SPEC_X, ["X"], _fake_run(coarse), _fake_run(fine), 0.0)
    expected = float(np.std(2 * fine - coarse, ddof=1) / np.sqrt(fine.size))
    wrong = float(np.std(fine, ddof=1) / np.sqrt(fine.size))
    assert detail["se"][0] == pytest.approx(expected)
    assert detail["se"][0] > 2.5 * wrong
    assert detail["extrapolated"][0] == pytest.approx(float(np.mean(2 * fine - coarse)))


def test_dynkin_refuses_levels_that_cannot_share_a_brownian_path():
    """Different path counts at the two dt levels mean they were not driven by one
    path, and a Richardson extrapolation across them would be noise."""
    outcome, detail = dynkin.check(SPEC_X, ["X"], _fake_run(np.zeros(100)),
                                   _fake_run(np.zeros(200)), 0.0)
    assert outcome == "unavailable"
    assert "Brownian" in detail["reason"]


def test_common_random_numbers_shrink_the_extrapolation_noise(tmp_path):
    """I9(a), measured on the honest solver: under CRN the extrapolated residual's
    standard error is far smaller than with independent draws per level, because
    ``r_fine - r_coarse`` is then a pathwise difference rather than noise."""
    plan = _plan(tmp_path, "sde_honest.py")
    spec = _spec("sde_spec_no_closed_form.json")
    phis = ["X", "X**2"]
    payload = dynkin.observables_payload(spec, phis)
    call = lambda dt, seed: {"num_paths": 8000, "dt": dt, "T": 1.0, "seed": seed}  # noqa: E731

    levels = crn.ladder_spec(8000, 0.02, 1.0, levels=2, seed=7)
    crn_runs = [sandbox.run(plan, "sde", call(lv["dt"], 7), crn=lv, observables=payload)
                for lv in levels]
    indep_runs = [sandbox.run(plan, "sde", call(0.02, 7), observables=payload),
                  sandbox.run(plan, "sde", call(0.01, 8), observables=payload)]
    assert all(r["status"] == "ok" for r in crn_runs + indep_runs)

    _, with_crn = dynkin.check(spec, phis, crn_runs[0], crn_runs[1], 2.0)
    _, without = dynkin.check(spec, phis, indep_runs[0], indep_runs[1], 2.0)
    for a, b in zip(with_crn["se"], without["se"], strict=True):
        assert a < 0.5 * b, (with_crn["se"], without["se"])


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
    one pass; the detail records which bar was used."""
    rng = np.random.default_rng(3)
    n = 2000
    r = rng.standard_normal(n)
    r = (r - r.mean()) / r.std(ddof=1) + 2.2 / np.sqrt(n)        # z = 2.2 exactly
    runs = [_fake_run(r), _fake_run(r)]       # identical levels: 2*fine - coarse = r
    outcome, detail = dynkin.check(SPEC_X, ["X"], runs[0], runs[1], 0.0, ci_mult=2.0)
    assert outcome is False and detail["family_wise_multiplier"] == 2.0
    outcome3, detail3 = dynkin.check(
        {**SPEC_X}, ["X", "X", "X"],
        {"result": {"terminal_paths": runs[0]["result"]["terminal_paths"],
                    "path_integrals": {f"L{i}": runs[0]["result"]["path_integrals"]["L0"]
                                       for i in range(3)}}},
        {"result": {"terminal_paths": runs[1]["result"]["terminal_paths"],
                    "path_integrals": {f"L{i}": runs[1]["result"]["path_integrals"]["L0"]
                                       for i in range(3)}}},
        0.0, ci_mult=2.0)
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
