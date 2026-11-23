"""The pathwise accuracy axis: `crn.strong_error`, and `ranking.accuracy`'s use of it.

Why this file exists, in one case. `sde_clark_cameron` is the Clark-Cameron
counterexample: no scheme built from the Brownian increments alone beats strong order
1/2 on ``U = int S dW1``, and the pipeline's own control plan says in its SOLUTION.md
that it "must not win a tie". It won twice, as an OVERCLAIM, and the second time the
cause was that the kernel had no measurement of the quantity the problem is graded on.
The two order-1 plans report **identical** moment errors -- 0.0121 to four digits,
because they produce the same moments -- and differ 1.9x in pathwise error. Ranking
them on the moment ranked them on a quantity the problem does not ask about.

The numbers below are the real ones, off the 2026-11-22 run.
"""



import os

import pytest

from verifylib import ranking
from verifylib.kernel import score
from verifylib.kernel.sde import crn
from verifylib.tests.test_kernel_score import CLEAN_A

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#: N01's component 1 (``U``) ladder detail, per plan, and the error the *harness*
#: independently measured for that plan. The proxy is biased 2-4x high against these;
#: what the axis needs is the ordering, which is why they are pinned together.
_N01 = {
    "1-levy-area-milstein-drift-corrected": {
        "s0": 0.007632516531045533, "s1": 0.00399396581618486,
        "se_s0": 0.00013129502443128414, "se_s1": 6.795166882775044e-05,
        "se_order": 0.0351461468665263, "terminal_scale": 1.2513032855932555,
        "roundoff_floor": 1.2513032855932556e-12, "count": 2000,
        "_order": 0.9343368241243529, "_truth": 0.004467620540686654},
    "2-levy-area-milstein": {
        "s0": 0.011086268526091353, "s1": 0.005609832936952028,
        "se_s0": 0.00019069041342687233, "se_s1": 9.516687579478712e-05,
        "se_order": 0.030659329829001335, "terminal_scale": 1.247603209264436,
        "roundoff_floor": 1.247603209264436e-12, "count": 2000,
        "_order": 0.9827441448472477, "_truth": 0.008475212537234518},
    "3-euler-maruyama-control": {
        "s0": 0.07879500728926431, "s1": 0.05585190534004709,
        "se_s0": 0.001385222256358648, "se_s1": 0.0009509227785624559,
        "se_order": 0.034823239385763434, "terminal_scale": 1.2459134118193187,
        "roundoff_floor": 1.2459134118193186e-12, "count": 2000,
        "_order": 0.4964977208191784, "_truth": 0.052269202000353975},
}

N01_TOL = 0.02

#: Component 0 of the same run: ``S = 1 + mu t + W2`` is integrated exactly by every
#: scheme that consumes ``dW``, so both ladder differences sit at the roundoff floor.
_N01_EXACT = {"s0": 7.37e-16, "s1": 7.37e-16, "se_s0": 1e-18, "se_s1": 1e-18,
              "terminal_scale": 1.2513032855932555,
              "roundoff_floor": 1.2513032855932556e-12, "count": 2000}


def _detail(name):
    return {k: v for k, v in _N01[name].items() if not k.startswith("_")}


def _summary(name, tol=N01_TOL):
    """One plan's kernel summary with the pathwise axis populated, as the driver writes
    it. The moment fields are the run's real ones and are deliberately *identical*
    across the two order-1 plans -- that identity is the defect this axis exists for."""
    out = crn.strong_error(_detail(name), _N01[name]["_order"])
    # The control is an Euler-Maruyama plan and declares strong order 0.5: it is
    # measured against *its own* design rate, which is why it reads near-perfect on
    # the order axis while being an order worse on the problem.
    control = name.startswith("3-")
    return {"kind": "sde", "estimated_rel_error": 0.0121, "mc_se_rel": 0.0208,
            "ci_mult": 2.0,
            "scheme_family": "euler-maruyama" if control else "milstein",
            "observed_order": _N01[name]["_order"],
            "design_order": 0.5 if control else 1.0,
            "order_floor": 0.5 if control else 1.0,
            "invariant_worst_headroom": 0.4959,
            "strong_rel_err": out["rel_err"], "strong_rel_err_se": out["se_rel"],
            "strong_rel_err_tol": tol}


# --------------------------------------------------------------------------- crn


def test_the_strong_error_inverts_the_ladder_difference_with_the_fitted_order():
    """``e(dt) = s0 / (1 - 2**-p)``, relative to ``mean|X|``: Richardson's argument in
    pathwise space. Pinned on plan 1's real ladder."""
    out = crn.strong_error(_detail("1-levy-area-milstein-drift-corrected"),
                           _N01["1-levy-area-milstein-drift-corrected"]["_order"])
    expect = (0.007632516531045533 / 1.2513032855932555) / (1 - 2 ** -0.9343368241243529)
    assert out["rel_err"] == pytest.approx(expect, rel=1e-12)
    assert out["rel_err"] == pytest.approx(0.012795, abs=5e-6)
    assert out["inflation"] == pytest.approx(1 / (1 - 2 ** -0.9343368241243529))
    # Its own bar is tight -- CRN is what buys that -- which is what lets
    # `_strong_fraction` normalise against the bar instead of a fixed ceiling.
    assert out["se_rel"] / out["rel_err"] == pytest.approx(0.0172, abs=5e-4)


def test_the_estimate_orders_the_pool_the_way_the_harness_does():
    """The property the axis is for. The *level* is biased 2.1x-4.2x high, which is why
    this number may not gate; the *ordering* is what ranking needs and it is right."""
    got = {n: crn.strong_error(_detail(n), _N01[n]["_order"])["rel_err"] for n in _N01}
    assert sorted(got, key=got.get) == sorted(_N01, key=lambda n: _N01[n]["_truth"])
    for n, est in got.items():
        assert 2.0 < est / _N01[n]["_truth"] < 4.3, (n, est, _N01[n]["_truth"])
    # And the separation between the two order-1 plans is far outside the bar: 1.41x
    # against a relative standard error of 1.7%.
    a = got["1-levy-area-milstein-drift-corrected"]
    b = got["2-levy-area-milstein"]
    assert b / a == pytest.approx(1.406, abs=5e-3)


def test_a_component_integrated_exactly_abstains_instead_of_reporting_zero_error():
    """S is exact under every scheme here. An error of 0.0 would be a claim of perfect
    accuracy off arithmetic noise; the honest output is no estimate, and the driver
    takes the worst component so abstaining cannot hide an error elsewhere."""
    out = crn.strong_error(_N01_EXACT, -1.0772)
    assert out["rel_err"] is None
    assert "roundoff" in out["reason"]


def test_a_heavy_tailed_ladder_difference_abstains():
    """`strong_order`'s own note records `sde_ginzburg_landau_s6` returning 0.220 and
    2.730 for the same scheme -- a factor of 12.4 -- because at sigma = 6 the pathwise
    differences are violently heavy-tailed and ``s0`` is a sample mean. An error built
    on that mean inherits all of it."""
    heavy = {**_detail("2-levy-area-milstein"), "se_s0": 0.011086268526091353 * 0.4}
    out = crn.strong_error(heavy, 0.98)
    assert out["rel_err"] is None
    assert "heavy tails" in out["reason"]


def test_a_barely_converging_scheme_abstains_rather_than_reporting_a_huge_error():
    """Below `STRONG_ERROR_MIN_ORDER` the Richardson factor is large and badly
    conditioned -- it diverges at p = 0 -- so it measures the fit's wobble, not the
    error."""
    out = crn.strong_error(_detail("2-levy-area-milstein"),
                           crn.STRONG_ERROR_MIN_ORDER - 0.01)
    assert out["rel_err"] is None
    assert "conditioned" in out["reason"]
    # Just above it, the estimate exists -- the guard is a floor, not a wall.
    assert crn.strong_error(_detail("2-levy-area-milstein"),
                            crn.STRONG_ERROR_MIN_ORDER + 0.01)["rel_err"] is not None


def test_no_fitted_order_means_no_estimate():
    assert crn.strong_error(_detail("2-levy-area-milstein"), None)["rel_err"] is None
    assert crn.strong_error(_detail("2-levy-area-milstein"), float("nan"))["rel_err"] is None


# ----------------------------------------------------------------------- ranking


def test_accuracy_reads_the_strong_error_in_preference_to_the_moment():
    """The moment error is identical on these two plans. If `accuracy` still read it,
    the axis would tie and the ranking would fall through to fitted order -- which is
    exactly how the control won."""
    a = _summary("1-levy-area-milstein-drift-corrected")
    b = _summary("2-levy-area-milstein")
    assert a["estimated_rel_error"] == b["estimated_rel_error"]
    assert ranking.accuracy(a, N01_TOL) > ranking.accuracy(b, N01_TOL)
    # Without the pathwise fields the same pool is noise-limited and unresolvable:
    # mc_se_rel 0.0208 at ci_mult 2.0 is a bar of 0.0416 around an error of 0.0121.
    moments_only = {k: v for k, v in a.items() if not k.startswith("strong_rel_err")}
    assert ranking.noise_limited(moments_only)
    assert ranking.accuracy(moments_only, N01_TOL) is None


def test_the_n01_pool_ranks_the_plan_the_harness_grades_best_first():
    """The regression. Plan 1 is the plan whose graded error matches the one-shot Opus
    baseline to the last bit (0.004467620540686654); plan 2 is 1.9x worse and is what
    shipped; plan 3 is the control that must not win."""
    pool = [_summary(n) for n in _N01]
    assert "accuracy" in ranking.comparable_keys(N01_TOL, pool)
    # `rank_key` takes a **pool entry**, not a summary: it reads `score` and
    # `provenance` off the kernel block and the summary from inside it. Handed a bare
    # summary it finds no block, returns its "unscorable" sentinel for every plan, and
    # `sorted` then preserves the input order -- which looks exactly like a correct
    # ranking and proves nothing. An earlier version of this test did that.
    entries = [{"kernel": {"score": 9, "provenance": "analytic", "summary": s},
                "iter": 1} for s in pool]
    assert len({ranking.rank_key(e, pool) for e in entries}) == len(entries), \
        "every plan returned the same key: the entries are not being read"
    ranked = sorted(zip(_N01, entries, strict=True),
                    key=lambda p: ranking.rank_key(p[1], pool))
    assert [n for n, _ in ranked] == ["1-levy-area-milstein-drift-corrected",
                                      "2-levy-area-milstein",
                                      "3-euler-maruyama-control"]
    assert _N01[ranked[0][0]]["_truth"] == min(v["_truth"] for v in _N01.values())
    # And the axis is what separated them, rather than something upstream of it.
    assert ranking.decided_by(ranked[0][1], ranked[1][1], pool) == "sqi"
    assert ranking.decided_by(ranked[0][1], ranked[2][1], pool) == "sqi"


def test_the_control_is_pinned_to_the_floor_of_the_accuracy_axis():
    """Its estimate is 0.217 against a 0.02 gate: over ten times the tolerance, so the
    headroom is negative and clips. The control keeps a near-perfect *order* reading
    (0.496 against its own declared 0.5 floor), and that is precisely why the pool
    needed an axis measuring error rather than rate."""
    pool = [_summary(n) for n in _N01]
    ctl = _summary("3-euler-maruyama-control")
    assert ranking.accuracy(ctl, N01_TOL) == 0.0
    assert ranking.order(ctl, pool) > ranking.order(
        _summary("1-levy-area-milstein-drift-corrected"), pool)


def test_the_axis_is_inert_where_the_spec_declares_no_strong_gate():
    """Blast radius. `strong_rel_err_tol` is published only where the spec declares
    `strong_rel_err_max`, so every pool that does not gate on strong error ranks
    exactly as it did before."""
    s = _summary("2-levy-area-milstein")
    del s["strong_rel_err_tol"]
    assert ranking._strong_fraction(s) is None
    assert ranking.accuracy(s, N01_TOL) is None   # falls back, and is noise-limited


def test_a_strong_error_inside_its_own_bar_does_not_score_at_all():
    """F16, for this axis. An estimate the ladder could not resolve must come back
    None, not 1.0: handing a plan the top of the component for being imprecise is the
    inversion the moment path's `noise_limited` exists to refuse. CRN is what keeps
    this rare -- the real bar is 3.4% of the real estimate."""
    s = _summary("1-levy-area-milstein-drift-corrected")
    assert 0.0 < ranking._strong_fraction(s) < 1.0
    resolved = s["strong_rel_err"] / (s["strong_rel_err_se"] * s["ci_mult"])
    assert resolved > 25.0
    s["strong_rel_err_se"] = s["strong_rel_err"]          # bar = 2x the estimate
    assert ranking._strong_fraction(s) is None


def test_a_bar_wider_than_the_gate_places_no_plan_against_it():
    """Where the ladder's own bar exceeds the tolerance, nothing in the pool can be
    located against that tolerance at this budget. The axis says so by abstaining
    rather than by scoring every plan at its ceiling."""
    s = _summary("1-levy-area-milstein-drift-corrected")
    s["strong_rel_err"] = N01_TOL * 3.0
    s["strong_rel_err_se"] = N01_TOL          # bar = 2x the gate
    assert ranking._strong_fraction(s) is None


# ------------------------------------------------------------------- certification

#: `test_kernel_score`'s clean Path-A evidence, read as an SDE and carrying the three
#: moment-side gates that take it to 10. Imported rather than restated so that a
#: change to what "clean" means cannot pass here and fail there -- and it has to
#: reach 10 unaided, or these two cases would prove nothing about the cap.
CLEAN_SDE = dict(CLEAN_A, kind="sde", moments_ok=True, resolved=True,
                 dynkin_ok=True)


def test_a_ranking_grade_estimate_does_not_lift_the_certification_cap():
    """(A) is a ranking change and must not become a certification change by accident.
    The estimate runs 2-4x high; N01's shipped plan reads 0.0180 against a 0.02 gate, a
    10% margin on a 2x-biased statistic. Certifying on that would be the overclaim this
    cap exists to stop, one step later."""
    got = score.certify({**CLEAN_SDE, "strong_rel_err_tol": N01_TOL,
                         "strong_rel_err": 0.017989, "strong_rel_err_gated": False})
    assert got["score"] == 9 and got["certified"] is False
    assert got["bound_by"] == "strong_gate_not_measured"
    assert "calibrated for ranking only" in got["reason"]


def test_the_cap_still_names_an_unmeasured_gate_differently():
    got = score.certify({**CLEAN_SDE, "strong_rel_err_tol": N01_TOL,
                         "strong_rel_err": None, "strong_rel_err_gated": False})
    assert got["score"] == 9 and got["bound_by"] == "strong_gate_not_measured"
    assert "nothing measured a strong error" in got["reason"]


def test_the_baseline_certifies_at_ten_without_the_pathwise_keys():
    """The control for the two cases above: whatever they measure has to be the cap and
    not some other defect in the fixture."""
    got = score.certify(dict(CLEAN_SDE))
    assert (got["score"], got["certified"]) == (10, True)


def test_a_problem_with_no_strong_gate_is_untouched_by_either_branch():
    got = score.certify({**CLEAN_SDE, "strong_rel_err_tol": None,
                         "strong_rel_err": 0.0128, "strong_rel_err_gated": False})
    assert (got["score"], got["certified"]) == (10, True)
