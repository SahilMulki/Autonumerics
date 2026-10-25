"""Harness-owned derivations of the BENCHOP-SLV reference values
(plan-benchop-slv.md step 6).

The claim this file holds is narrow and load-bearing: the 24 numbers the suite
will be anchored on are numbers this repo *reproduced*, not numbers it copied. So
the fast tests here re-derive every value that has a closed form, from the
published parameters, and check the transcription against the formula the source
says generated it.

  B1  the strikes are the formula's strikes -- a transcribed constant is checked
      against the rule that produced it, not read twice
  B2  the two exact families land on the published value at machine precision
  B3  the pieces each derivation rests on are right on their own: Fornberg
      weights, the killed heat kernel, the Hull-White moments, the Bessel density
  B4  the far-field convention that splits the QLSV call is reproducible, and the
      closed form says which side is the expectation
  B5  the stored ledger says what the module says, and the pipeline cannot read it
  B6  (slow) every derivation runs end to end and lands near its published value
  B7  the shifted twins are in the ledger, and their error rule was validated
      where a third party could adjudicate it
  B8  each of step 8's wrong schemes is wrong in the direction the plan claims
"""
import json
import os

import numpy as np
import pytest

from .conftest import BENCHMARK
from .conftest import BO as B

LEDGER = os.path.join(BENCHMARK, "references", "benchop_slv.json")


# --- B1: the transcription is checked against the rule, not re-read -------------

@pytest.mark.parametrize("which,want", [
    ("I", (0.434061722697, 0.5, 0.575954955084)),
    ("II", (0.051022538988, 0.07, 0.096035989138)),
])
def test_sabr_strikes_come_out_of_the_published_formula(which, want):
    """The formulations give K_i = S0 exp(0.1 sqrt(T) delta_i) *and* tabulate the
    three numbers. Agreeing to 5e-13 is how we know the spot, the maturity and the
    strikes were read off the same problem."""
    got = B.sabr_strikes(which)
    assert max(abs(a - b) for a, b in zip(got, want, strict=True)) < 1e-12


def test_every_published_problem_carries_three_values_and_a_stated_uncertainty():
    for key, pub in B.PUBLISHED.items():
        assert len(pub["values"]) == 3 == len(pub["labels"]), key
        assert all(np.isfinite(pub["values"])), key
        assert pub["abs_err"] >= 0.0, key
    # Exactly one value in the suite is not quoted to 15 digits, and its own
    # uncertainty is therefore a floor on any gate built from it.
    loose = [k for k, v in B.PUBLISHED.items() if v["abs_err"] > 0]
    assert loose == ["hhw_eucall1"]
    assert B.PUBLISHED["hhw_eucall1"]["abs_err"] == 5e-3


# --- B2: the two exact families ------------------------------------------------

def test_heston_characteristic_function_reproduces_the_published_call():
    p = B.LSV
    for S0, ref in zip(p["spots"], B.PUBLISHED["hst_eucall"]["values"], strict=True):
        got = B.heston_call(S0, p["K"], p["T"], p["kappa"], p["eta"], p["sigma"],
                            p["rho"], p["v0"])
        assert abs(got - ref) < 1e-12, (S0, got, ref)


def test_heston_hull_white_set_two_reproduces_the_published_call():
    """The collapsed integral, which is the whole content of :func:`hhw2_call`:
    folding the Gaussian rate into Lewis' integrand turns two quadratures into one
    and takes the agreement from 3e-5 relative to 1e-15."""
    for S0, ref in zip(B.HHW["spots"], B.PUBLISHED["hhw_eucall2"]["values"],
                       strict=True):
        assert abs(B.hhw2_call(S0, B.HHW) - ref) < 1e-11, S0


def test_the_heston_call_converges_to_black_scholes_as_the_vol_of_vol_vanishes():
    """An identity on the characteristic function rather than another comparison
    against the same answer key. With v0 = eta and rho = 0 the only thing a small
    vol-of-vol does is add kurtosis, whose leading term is O(sigma^2) -- so the
    *rate* is the test, not a single tolerance."""
    from scipy.stats import norm
    S0, K, T, v = 100.0, 90.0, 1.0, 0.09
    sd = np.sqrt(v * T)
    d1 = (np.log(S0 / K) + 0.5 * v * T) / sd
    want = S0 * norm.cdf(d1) - K * norm.cdf(d1 - sd)
    gaps = [abs(B.heston_call(S0, K, T, kappa=1.0, eta=v, sigma=s, rho=0.0, v0=v) - want)
            for s in (4e-3, 2e-3, 1e-3)]
    assert gaps[-1] < 1e-5
    for a, b in zip(gaps[:-1], gaps[1:], strict=True):
        assert 3.5 < a / b < 4.5, gaps


def test_the_heston_call_is_a_martingale_price():
    """K -> 0 must give back the spot: E[(S_T - K)^+] -> E[S_T] = S0."""
    p = B.LSV
    got = B.heston_call(100.0, 1e-8, p["T"], p["kappa"], p["eta"], p["sigma"],
                        p["rho"], p["v0"])
    assert abs(got - 100.0) < 1e-6


# --- B3: the pieces, each on its own -------------------------------------------

def test_fornberg_weights_reproduce_the_textbook_stencils():
    x = np.array([-1.0, 0.0, 1.0])
    assert np.allclose(B.fdw(x, 0.0, 1), [-0.5, 0.0, 0.5])
    assert np.allclose(B.fdw(x, 0.0, 2), [1.0, -2.0, 1.0])
    assert np.allclose(B.fdw(np.array([0.0, 1.0, 2.0]), 0.0, 1), [-1.5, 2.0, -0.5])
    assert np.allclose(B.fdw(x, 0.0, 0), [0.0, 1.0, 0.0])


def test_fornberg_weights_are_exact_on_a_stretched_grid():
    """The reason the V axis uses them: a stretched three-point second derivative
    is not the uniform one, and using the uniform one is a silent first-order error
    exactly where a Feller-violating CIR keeps its mass."""
    X = np.array([0.0, 0.1, 0.35])
    for m, fn, d in ((1, lambda z: z ** 2, 0.2), (2, lambda z: z ** 2, 2.0)):
        w = B.fdw(X, 0.1, m)
        assert abs(float(w @ fn(X)) - d) < 1e-12


def test_the_bridge_survival_reduces_to_the_one_barrier_formula():
    """With the far barrier out of reach the killed heat kernel must collapse onto
    the single-barrier reflection result 1 - exp(-2 (b-x0)(b-x1) / var)."""
    x0 = np.array([0.0, 0.0, 0.0])
    x1 = np.array([0.1, 0.4, 0.49])
    var, b = 0.04, 0.5
    got = B.bridge_survival(x0, x1, np.full(3, var), -50.0, b)
    want = 1.0 - np.exp(-2.0 * (b - x0) * (b - x1) / var)
    assert np.allclose(got, want, atol=1e-12)


def test_a_path_that_ends_outside_the_barriers_has_not_survived():
    got = B.bridge_survival(np.array([0.0]), np.array([0.6]), np.array([0.04]),
                            -0.5, 0.5)
    assert got[0] == 0.0


def test_the_hull_white_integrated_rate_matches_a_direct_computation():
    mu, var = B.hw_integrated_rate(B.HHW)
    a, s2, T, b, R0 = (B.HHW[k] for k in ("a", "sigma2", "T", "b", "R0"))
    # mean: R0 = b here, so E[R] is constant and E[M] = b T exactly
    assert b == R0
    assert abs(mu - b * T) < 1e-12
    # variance: double integral of the OU autocovariance, done numerically
    g = np.linspace(0.0, T, 2001)
    cov = (s2 ** 2 / (2 * a)) * (np.exp(-a * np.abs(g[:, None] - g[None, :]))
                                 - np.exp(-a * (g[:, None] + g[None, :])))
    assert abs(var - np.trapezoid(np.trapezoid(cov, g, axis=1), g)) < 1e-4


def test_the_bessel3_density_integrates_to_one_and_to_the_local_martingale_mean():
    """The density :func:`frozen_qlsv_call` integrates against, checked on its own
    two identities before it is used to settle anything."""
    from scipy.integrate import quad
    S0, theta = 125.0, B.LSV["v0"] * B.LSV["T"]
    r0 = 100.0 / S0

    def pdf(r):
        return (r / (r0 * np.sqrt(2 * np.pi * theta))
                * (np.exp(-(r - r0) ** 2 / (2 * theta))
                   - np.exp(-(r + r0) ** 2 / (2 * theta))))
    mass, _ = quad(pdf, 0.0, 60.0, limit=400)
    assert abs(mass - 1.0) < 1e-9
    mean, _ = quad(lambda r: (100.0 / r) * pdf(r), 0.0, 60.0, limit=400)
    assert abs(mean - B.frozen_qlsv_call(S0, 0.0, theta, use_erf=True)) < 1e-7


# --- B4: the convention that splits the QLSV call ------------------------------

def test_the_frozen_qlsv_asset_is_a_strict_local_martingale():
    """f(s) = 0.01 s^2 makes 100/S a Bessel(3) process, so E[S_T] < S0 -- strictly,
    and by an amount that is 2.2 at S0 = 125 rather than a rounding error."""
    theta = B.LSV["v0"] * B.LSV["T"]
    means = [B.frozen_qlsv_call(s, 0.0, theta, use_erf=True) for s in B.LSV["spots"]]
    for S0, m in zip(B.LSV["spots"], means, strict=True):
        assert m < S0
    assert abs(means[2] - 122.77285141) < 1e-6
    assert means[0] > 74.99           # the defect is concentrated at the high spot


def test_the_closed_form_says_the_grids_far_field_is_the_non_minimal_solution():
    """The arbiter. With the variance frozen the QLSV call has a closed form; the
    finite-difference solver with the usual linear far field returns something
    else, and it is the published convention."""
    theta, K = B.LSV["v0"] * B.LSV["T"], B.LSV["K"]
    exact = B.frozen_qlsv_call(125.0, K, theta)
    assert abs(exact - 29.98063226) < 1e-6
    conv = B.qlsv_convention_check("fast")
    assert abs(conv["closed_form"][2] - exact) < 1e-9
    # the grid's answer is 2.2 above it -- a different solution, not a coarse one
    assert conv["fd2d_linear_far_field"][2] - exact > 2.0


# --- B5: the ledger -------------------------------------------------------------

@pytest.mark.skipif(not os.path.exists(LEDGER), reason="ledger not built")
def test_the_ledger_agrees_with_the_published_values_in_the_module():
    with open(LEDGER) as fh:
        doc = json.load(fh)
    assert set(doc["problems"]) == set(B.PUBLISHED)
    for key, rec in doc["problems"].items():
        pub = B.PUBLISHED[key]
        assert rec["published"] == [float(x) for x in pub["values"]], key
        assert rec["reference_abs_err"] == float(pub["abs_err"]), key
        primary = next(r for r in rec["routes"] if r["name"] == rec["primary_route"])
        assert rec["confirmed_abs_err"] == pytest.approx(max(primary["abs_diff"])), key


# What each value was actually confirmed to, as measured on 2026-10-17. These are
# bars, not predictions: a change that makes any derivation worse than it already is
# should say so here rather than quietly lower the standard of the answer key.
CONFIRMED = {
    "hst_eucall": 1e-12, "hhw_eucall2": 1e-12,          # closed form, both families
    "hst_dntouch": 1e-5, "qlsv_dntouch": 1e-5, "qlsv_eucall": 1e-4,
    "sabr_set1": 1e-4, "sabr_set2": 2e-4,
    # The one exception, and it is the whole of what step 6 left unresolved: three
    # correlated factors leave only a sampler, and it sits 0.03-0.04 above a published
    # value that claims +-0.005 for itself. See the plan's section 8.
    "hhw_eucall1": 1e-3,
}


@pytest.mark.skipif(not os.path.exists(LEDGER), reason="ledger not built")
def test_every_published_value_was_independently_reproduced():
    """The point of the step: each of the 24 is a number this repo reproduced, to a
    recorded tolerance, by at least two routes."""
    with open(LEDGER) as fh:
        doc = json.load(fh)
    for key, rec in doc["problems"].items():
        assert rec["confirmed_rel_err"] < CONFIRMED[key], (key, rec["confirmed_rel_err"])
        assert len(rec["routes"]) >= 2, key


@pytest.mark.skipif(not os.path.exists(LEDGER), reason="ledger not built")
def test_the_recorded_arbiter_puts_the_sampler_on_the_closed_form_and_not_the_grid():
    """The stored measurement, not a fresh one. The sampler agrees with the closed
    form at every spot; the grid does not, and its disagreement *grows with the
    spot* -- 0.3 standard errors at 75 and 25 at 125 -- which is what a local
    martingale defect looks like and what a discretization error does not. That
    ordering is the argument for calling the published QLSV values a convention
    rather than an expectation."""
    with open(LEDGER) as fh:
        conv = json.load(fh)["problems"]["qlsv_eucall"]["convention"]
    exact, mc, se = (np.asarray(conv[k]) for k in ("closed_form", "mc", "mc_stderr"))
    grid = np.asarray(conv["fd2d_linear_far_field"])
    assert np.all(np.abs(mc - exact) / se < 3.0), (mc - exact) / se
    z = (grid - exact) / se
    assert z[2] > 10.0 and np.all(np.diff(z) > 0), z


@pytest.mark.skipif(not os.path.exists(LEDGER), reason="ledger not built")
def test_only_the_three_factor_problem_rests_on_a_single_method_family():
    """Two independent *families* -- not two runs of one sampler -- is what the plan
    asks for, and seven of the eight have it. The exception is named so that it
    cannot quietly become two."""
    with open(LEDGER) as fh:
        doc = json.load(fh)
    single = sorted(k for k, rec in doc["problems"].items()
                    if len({r["family"] for r in rec["routes"]}) < 2)
    assert single == ["hhw_eucall1"], single


def test_the_ledger_is_denied_to_the_pipeline():
    import fnmatch
    with open(os.path.join(BENCHMARK, "pipeline-settings.json")) as fh:
        deny = json.load(fh)["permissions"]["deny"]
    reads = [d[len("Read("):-1] for d in deny if d.startswith("Read(")]
    rel = "./benchmark/references/benchop_slv.json"
    assert any(fnmatch.fnmatch(rel, g) for g in reads), reads


# --- B6: the derivations themselves (slow) --------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize("key", list(B.PUBLISHED))
def test_a_coarse_derivation_still_lands_near_the_published_value(key):
    """Not a precision claim -- the coarse settings cannot make one. It is the
    claim that every route runs, returns three finite numbers, and is pointed at
    the problem it says it is."""
    routes, primary = B.derive(key, "fast")
    ref = np.asarray(B.PUBLISHED[key]["values"], dtype=float)
    hit = next(r for r in routes if r["name"] == primary)
    rel = np.abs(np.asarray(hit["values"]) - ref) / np.abs(ref)
    assert np.all(np.isfinite(rel)) and rel.max() < 2e-2, (key, rel)


# --- B7: the shifted twins (added in step 7) ------------------------------------

@pytest.mark.skipif(not os.path.exists(LEDGER), reason="ledger not built")
def test_every_twin_is_in_the_ledger_with_its_own_uncertainty():
    """A twin's answer key is this repo's, not anyone else's, so it carries an
    uncertainty by construction rather than by inheritance."""
    with open(LEDGER) as fh:
        twins = json.load(fh)["shifted"]
    assert set(twins) == set(B.SHIFTED)
    for key, rec in twins.items():
        assert rec["of"] in B.PUBLISHED, key
        assert len(rec["value"]) == 3 and all(np.isfinite(rec["value"])), key
        assert rec["reference_abs_err"] >= 0.0, key
        assert len(rec["shift"]) == 1, key      # exactly one parameter moves


@pytest.mark.skipif(not os.path.exists(LEDGER), reason="ledger not built")
def test_each_twins_error_rule_was_checked_where_a_third_party_could_adjudicate():
    """The twins have no published value to be held against, so the rule that sets
    their uncertainty is validated on the sibling that does: what it would have
    predicted there, against what the error turned out to be."""
    with open(LEDGER) as fh:
        twins = json.load(fh)["shifted"]
    for key, rec in twins.items():
        v = rec["rule_validated_on_sibling"]
        assert v and v["rule"] == rec["abs_err_rule"], key
        # within a factor of three of the error it was predicting, in either
        # direction -- an estimator that was out by an order of magnitude on the
        # sibling would not be worth declaring on the twin
        if v["actual"] > 0 and v["predicted"] > 0:
            ratio = v["predicted"] / v["actual"]
            assert 1 / 3 < ratio < 3, (key, v)


@pytest.mark.slow
@pytest.mark.parametrize("key", list(B.SHIFTED))
def test_a_twins_derivation_runs_and_its_routes_agree(key):
    routes, primary = B.derive_shifted(key, "fast")
    assert len(routes) >= 2
    hit = next(r for r in routes if r["name"] == primary)
    for other in routes:
        if other["name"] == primary:
            continue
        gap = max(abs(a - b) for a, b in zip(hit["values"], other["values"], strict=True))
        assert gap / max(abs(v) for v in hit["values"]) < 5e-3, (key, other["name"], gap)


@pytest.mark.skipif(not os.path.exists(LEDGER), reason="ledger not built")
def test_a_twin_is_a_long_way_from_its_sibling():
    """Not a restatement of the suite's gate test -- that one checks the shift
    against the tolerance, this one checks it is a different problem at all."""
    with open(LEDGER) as fh:
        doc = json.load(fh)
    for key, rec in doc["shifted"].items():
        base = doc["problems"][rec["of"]]["published"]
        moved = [abs(a - b) / abs(b) for a, b in zip(rec["value"], base, strict=True)]
        assert min(moved) > 1e-2, (key, moved)


# --- B8: the calibration's wrong schemes (added in step 8) ----------------------
#
# Step 8 pairs every problem with a plausible wrong scheme that differs from a
# correct one by a single decision. The measurement itself is expensive and lives
# in the ledger; what is checked here, cheaply, is that each wrong scheme is wrong
# in the *direction* the plan claims -- so that a gate separating them is
# separating the decision and not a sampling accident.

def test_every_published_problem_has_a_wrong_scheme_named():
    """Except the one the suite leaves out: QLSVeuCall is a grid convention, not
    an expectation, so no sampler of it is right or wrong in this sense."""
    assert set(B.CALIBRATION) == set(B.PUBLISHED) - {"qlsv_eucall"}
    for key, spec in B.CALIBRATION.items():
        assert spec["decision"], key
        assert spec["correct"] != spec["wrong"], key
        assert spec["correct"] not in spec.get("also", ()), key


def test_discrete_monitoring_lets_more_paths_survive_than_the_bridge_does():
    """The Broadie-Glasserman-Kou bias has a sign. Testing the barrier once per
    step misses the excursions that start and end inside it, so a double-no-touch
    priced that way is biased *high* -- on every spot, not on average."""
    kw = dict(p=B.LSV, kind="dntouch")
    bridge = B.lsv_mc([B.CHART_HESTON], 4000, 60, 7, monitor="bridge", **kw)[0]
    discrete = B.lsv_mc([B.CHART_HESTON], 4000, 60, 7, monitor="discrete", **kw)[0]
    assert (discrete.mean(axis=1) > bridge.mean(axis=1)).all()
    # and the same paths: the only difference is what was asked of them
    assert (discrete >= bridge - 1e-12).all()


def test_reflecting_the_forward_off_zero_destroys_the_martingale_upwards():
    """``E[S_T] = S_0`` exactly for this CEV forward. Absorbing loses a little of
    it downwards, which is the discretisation error a control variate can take
    back out; reflecting gains mass instead, which nothing takes back out."""
    kw = dict(seed=11, antithetic=False)
    absorbed = B.sabr_euler("I", 20_000, 200, boundary="absorb", **kw)
    reflected = B.sabr_euler("I", 20_000, 200, boundary="reflect", **kw)
    # payoffs are increasing in S, so the whole price block moves with the mean
    assert (reflected.mean(axis=1) > absorbed.mean(axis=1)).all()


def test_reflecting_the_variance_raises_it_where_truncation_holds_it_at_zero():
    """Both answer "what is sqrt of a negative variance". Truncation says the
    variance did not move; reflection says it moved the other way, which injects
    variance that the CIR dynamics never put there -- and with a Feller ratio of
    0.04 the variance is at zero often enough for that to compound."""
    kw = dict(seed=5, controls=False)
    trunc = B.hhw_mc("II", 4000, 250, variance="truncation", **kw)
    refl = B.hhw_mc("II", 4000, 250, variance="reflection", **kw)
    assert (refl.mean(axis=1) > trunc.mean(axis=1)).all()


@pytest.mark.skipif(not os.path.exists(LEDGER), reason="ledger not built")
def test_the_ledger_records_what_each_scheme_reached():
    with open(LEDGER) as fh:
        cal = json.load(fh).get("calibration", {})
    assert cal, "run: uv run python benchmark/benchop.py --calibrate"
    for k, rec in cal.items():
        roles = [s["role"] for s in rec["schemes"]]
        assert roles.count("correct") == 1 and roles.count("wrong") == 1, k
        for sch in rec["schemes"]:
            assert len(sch["rel_err"]) == len(rec["gate"]) == 3, k
            assert all(r >= 0 for r in sch["rel_err"]), k
            assert sch["wall_s"] > 0, k
