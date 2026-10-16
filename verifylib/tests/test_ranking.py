"""The quality index and the ranking key.

The property that matters most here is the first one tested: **a component that
could not be measured never lowers a plan's standing.** Every other rule in this
repo says so -- manual §24/§25, `evaluator-pde.md`'s "a check skipped for cost never
counts against a plan", and the `not_reported` cap that already prices a missing
invariant once. An index that scored an absent component as zero would charge the
same defect twice and would make a cheap evaluation look like a bad solver.
"""

import json

import pytest

from verifylib import ranking as RK

TOL = 1e-2

# A plan with every component measurable and all of them good.
FULL = {"estimated_rel_error": 1e-8, "observed_order": 4.0, "design_order": 4,
        "scheme_family": "fd", "invariant_worst_headroom": 6.0,
        "d1_outcome": "clean", "d1_slope": 4.0, "solve_time_s": 1.0}


def sqi(summary, pool=()):
    return RK.sqi(summary, TOL, pool)[0]


# --- the renormalisation rule -------------------------------------------------

DROPS = {"order": ("observed_order", "design_order"),
         "structure": ("invariant_worst_headroom",),
         "consistency": ("d1_outcome", "d1_slope")}


def _without(name):
    return {k: v for k, v in FULL.items() if k not in DROPS[name]}


@pytest.mark.parametrize("drop", sorted(DROPS))
def test_an_unmeasurable_component_is_never_scored_zero(drop):
    """The index is the mean of what *was* measured, renormalised over those
    weights. What it must never be is the mean with a zero standing in for the
    missing one -- that charges a skipped check as a failed one, which every rule in
    this repo forbids."""
    thin = _without(drop)
    assert RK.components(thin, TOL)[drop] is None
    zeroed = sum(RK.WEIGHTS[k] * (v or 0.0)
                 for k, v in RK.components(thin, TOL).items()
                 if k != "cost") / sum(RK.WEIGHTS[k] for k in RK.CERT_COMPONENTS)
    assert sqi(thin) > zeroed


@pytest.mark.parametrize("drop", sorted(DROPS))
def test_a_missing_component_leaves_the_mean_of_the_measured_ones(drop):
    """Stated exactly, because the weaker-sounding version is the true one: the
    index is a weighted *mean*, so dropping a component moves it toward the mean of
    the rest -- up if the missing one was below average, down if above. It is
    neither credited nor penalised against a fixed bar."""
    thin = _without(drop)
    parts = {k: v for k, v in RK.components(thin, TOL).items() if v is not None}
    expected = (sum(RK.WEIGHTS[k] * v for k, v in parts.items())
                / sum(RK.WEIGHTS[k] for k in parts))
    assert sqi(thin) == pytest.approx(expected)


def test_measured_and_poor_is_not_the_same_as_not_measured():
    poor = {**FULL, "invariant_worst_headroom": 0.0}      # passed exactly at its bar
    assert sqi(poor) < sqi(_without("structure"))
    assert RK.structure(poor) == 0.0 and RK.structure(_without("structure")) is None


def test_skipping_a_weak_check_cannot_buy_a_better_rank_than_the_score_allows():
    """The one place renormalisation could be gamed: `FULL`'s D1 slope is its
    weakest component, so an evaluation where D1 came back `unavailable` scores a
    *higher* index. That is harmless only because the architecture already prices it
    one level up -- a D1 that did not run forfeits a circularity break, so
    `score.ceiling` caps the plan at 9 where the reporting plan reaches 10, and the
    key reads `score` before it reads quality. This test is what keeps those two
    facts tied together."""
    reported = {**FULL, "d1_outcome": "slow"}     # fell, but under the design rate
    skipped = {**FULL, "d1_outcome": "unavailable"}
    assert sqi(skipped) > sqi(reported)          # the index alone prefers the skip...
    assert RK.rank_key({"kernel": {"score": 10, "provenance": "manufactured",
                                   "summary": reported}}) < \
           RK.rank_key({"kernel": {"score": 9, "provenance": "manufactured_partial",
                                   "summary": skipped}})   # ...the key does not


def test_an_index_with_nothing_measurable_is_none_not_zero():
    index, parts = RK.sqi({}, TOL)
    assert index is None and set(parts) == set(RK.WEIGHTS)


def test_weights_are_a_partition_of_one():
    assert sum(RK.WEIGHTS.values()) == pytest.approx(1.0)
    assert set(RK.CERT_COMPONENTS) < set(RK.WEIGHTS)
    assert "cost" not in RK.CERT_COMPONENTS      # a certification cannot read the pool


# --- accuracy -----------------------------------------------------------------

def test_the_heat_2d_spread_is_not_a_difference():
    """The case the rework exists for. Three second-order plans certified at
    1.010e-04, 1.019e-04 and 1.023e-04 against a 1e-2 tolerance, and the winner was
    picked on the 0.9% gap between the first two. On a log scale that gap is
    0.0005 of a point, and the plans tie."""
    errs = [1.010e-4, 1.019e-4, 1.023e-4]
    got = [RK.accuracy({"estimated_rel_error": e}, TOL) for e in errs]
    assert max(got) - min(got) < 0.001
    # ...while the fourth-order plan that never ran would separate by a third of the
    # whole scale.
    assert RK.accuracy({"estimated_rel_error": 1e-8}, TOL) - max(got) > 0.3


def test_accuracy_saturates_at_the_floor_and_bottoms_out_at_the_tolerance():
    assert RK.accuracy({"estimated_rel_error": TOL}, TOL) == pytest.approx(0.0)
    assert RK.accuracy({"estimated_rel_error": 1e-12}, TOL) == pytest.approx(1.0)
    assert RK.accuracy({"estimated_rel_error": 10 * TOL}, TOL) == pytest.approx(0.0)


def test_the_worse_horizon_and_graded_errors_win():
    """`estimated_rel_error` on a chaotic problem is the error at the reference
    horizon; the problem asks about T. The accuracy component must read the one the
    problem is graded on, never the flattering one."""
    summary = {"estimated_rel_error": 1e-9, "estimated_rel_error_T": 5e-3}
    assert RK.accuracy(summary, TOL) < RK.accuracy({"estimated_rel_error": 1e-9}, TOL)


def test_an_error_of_exactly_zero_is_unmeasured_not_perfect():
    """`baseline_select.looks_n_independent`'s shape: a solver computing on its own
    internal grid differs between N and 2N only by interpolation round-off, so the
    ladder reports 0.0. Under the old key that sorted *ahead* of every honestly
    measured plan."""
    assert RK.accuracy({"estimated_rel_error": 0.0}, TOL) is None


def test_an_sde_error_snaps_up_to_its_confidence_half_width():
    """Two plans whose moment errors differ by less than the Monte-Carlo noise are
    tied, and ranking them apart reads the seed. §14 makes every comparison at
    `ci_mult` standard errors; so does this."""
    a = {"estimated_rel_error": 2e-3, "mc_se_rel": 5e-3, "ci_mult": 2.0}
    b = {"estimated_rel_error": 8e-3, "mc_se_rel": 5e-3, "ci_mult": 2.0}
    assert RK.accuracy(a, 0.1) == RK.accuracy(b, 0.1)      # both snap to 1e-2
    assert RK.uncertainty(a) == pytest.approx(1e-2)
    # With no sampling error there is nothing to snap to, and the two separate.
    bare = {k: v for k, v in a.items() if k != "mc_se_rel"}
    assert RK.uncertainty(bare) is None


# --- order fidelity -----------------------------------------------------------

def test_order_is_two_sided_for_an_algebraic_scheme():
    """A 2nd-order scheme measuring 4.0 is a ladder outside its asymptotic range,
    which `p_sane` already refuses to call asymptotic. Overshoot is not a bonus."""
    at = {"observed_order": 2.0, "design_order": 2, "scheme_family": "fd"}
    over = {**at, "observed_order": 3.0}      # one order above
    under = {**at, "observed_order": 1.0}     # one order below
    assert RK.order(at) == pytest.approx(1.0)
    assert RK.order(over) < RK.order(at)
    assert RK.order(under) == pytest.approx(RK.order(over))   # equal distance, equal credit
    assert RK.order({**at, "observed_order": 4.0}) == pytest.approx(0.0)


def test_a_spectral_scheme_has_no_order_to_be_faithful_to():
    """Its error is not C*h^p, and its declared `spatial_order` is a D1 stencil hint
    rather than a rate -- `pde_kdv_zabusky_kruskal`'s plan 1 declares 8 and says the
    observed ladder order will be ~4. Comparing the two marked both spectral plans
    below the FD plans while their errors were five decades better."""
    spec = {"observed_order": 5.45, "design_order": 8, "scheme_family": "spectral"}
    assert RK.order(spec) is None
    assert RK.order({**spec, "observed_order": 12.0}) is None


def test_the_sde_strong_order_band_is_two_sided_around_the_expectation():
    """The old key's sign error. `order_floor` on an SDE is the *expected* strong
    order and `strong_ok` is |g - g_exp| < 0.4, so a margin of +0.39 was sorting
    ahead of 0.00 -- ranking by distance from the correct answer."""
    em = {"kind": "sde", "observed_order": 0.5, "order_floor": 0.5, "design_order": 0.5}
    high = {**em, "observed_order": 0.89}
    assert RK.order(em) == pytest.approx(1.0)
    assert RK.order(high) < RK.order(em)


def test_a_waived_order_check_is_not_measurable():
    waived = {"observed_order": 0.1, "design_order": 4, "order_check": "waived_chaotic"}
    assert RK.order(waived) is None


def test_an_undeclared_design_order_is_not_guessed():
    """Inferring a design order from a plan's slug would let a plan rename itself
    into a better rank. Measured across `workspace/`, 82 of 104 plans declare none."""
    assert RK.order({"observed_order": 4.0, "scheme_family": "fd"}) is None


# --- structure, consistency, cost ---------------------------------------------

def test_structure_reads_headroom_not_raw_drift():
    """Raw drifts of different invariants are not on one scale, which is why the
    proposed `max invariant drift` key could not have worked."""
    assert RK.structure({"invariant_worst_headroom": 3.0}) == pytest.approx(1.0)
    assert RK.structure({"invariant_worst_headroom": 0.0}) == pytest.approx(0.0)
    assert RK.structure({"invariant_max_drift": 1e-16}) is None   # not the comparable form


def test_consistency_is_a_slope_never_a_magnitude():
    """Manual §25 puts a residual reported as a magnitude on the never-happen list;
    §26 explains there can be no calibrated threshold for one."""
    fast = {"d1_outcome": "clean", "d1_slope": 4.0, "design_order": 4}
    slow = {"d1_outcome": "slow", "d1_slope": 1.0, "design_order": 4}
    assert RK.consistency(fast) > RK.consistency(slow)
    for skipped in ("unresolved", "unavailable"):
        assert RK.consistency({"d1_outcome": skipped, "d1_slope": 0.1}) is None
    # `clean` with no finite slope is the round-off floor: the best outcome.
    assert RK.consistency({"d1_outcome": "clean", "design_order": 4}) == 1.0


def test_the_sde_consistency_component_inverts_the_dynkin_distance():
    near = {"dynkin_z": 0.2, "dynkin_bar": 2.43}
    far = {"dynkin_z": 2.4, "dynkin_bar": 2.43}
    assert RK.consistency(near) > RK.consistency(far)
    assert RK.consistency(far) == pytest.approx(0.0123, abs=1e-3)


def test_cost_is_pool_relative_and_undefined_for_a_singleton():
    pool = [{"solve_time_s": 0.1}, {"solve_time_s": 1.0}, {"solve_time_s": 10.0}]
    assert RK.cost(pool[0], pool) == pytest.approx(1.0)
    assert RK.cost(pool[-1], pool) == pytest.approx(0.0)
    assert RK.cost(pool[1], pool) == pytest.approx(0.5)      # log-scaled midpoint
    assert RK.cost(pool[0], pool[:1]) is None


def test_cost_reads_the_solver_not_the_evaluation():
    """`wall_time_s` bundles the ladder, MMS, D1 and the temporal probe, which §24
    prices between 1.0x and 35x by evidence path. Measured on
    `pde_heat_1d/3-fd4-rk4`: solve 0.28 s against evaluation 1.20 s."""
    pool = [{"wall_time_s": 1.0}, {"wall_time_s": 30.0}]
    assert RK.cost(pool[0], pool) is None


# --- the key ------------------------------------------------------------------

def _entry(score, provenance, **summary):
    return {"kernel": {"score": score, "provenance": provenance,
                       "summary": {"solve_time_s": 1.0, **summary}}}


def test_score_and_provenance_outrank_quality():
    """They are claims about what was *established*, not measures of quality. A
    weighted sum must never trade a validated-analytic 9 against a
    self-convergence 10 -- averaging that away is exactly what the provenance
    vocabulary exists to prevent."""
    weak_ten = _entry(10, "self_convergence", estimated_rel_error=TOL)
    strong_nine = _entry(9, "analytic", estimated_rel_error=1e-12)
    assert RK.rank_key(weak_ten) < RK.rank_key(strong_nine)
    # ...and within one score, provenance still leads.
    assert RK.rank_key(_entry(10, "analytic", estimated_rel_error=TOL)) < \
           RK.rank_key(_entry(10, "manufactured", estimated_rel_error=1e-12))


def test_quality_decides_a_tie_that_used_to_be_decided_on_noise():
    tied = [_entry(10, "analytic", estimated_rel_error=e) for e in (1.010e-4, 1.023e-4)]
    better = _entry(10, "analytic", estimated_rel_error=1e-8)
    pool = [e["kernel"]["summary"] for e in (*tied, better)]
    ranked = sorted([*tied, better], key=lambda e: RK.rank_key(e, pool))
    assert ranked[0] is better


def test_the_pool_breaks_a_genuine_tie_on_cost():
    """The sentence the index exists to express: equally accurate within the error
    bars, so take the cheaper one."""
    fast = _entry(10, "analytic", estimated_rel_error=1.010e-4, solve_time_s=0.1)
    slow = _entry(10, "analytic", estimated_rel_error=1.010e-4, solve_time_s=10.0)
    pool = [fast["kernel"]["summary"], slow["kernel"]["summary"]]
    assert RK.rank_key(fast, pool) < RK.rank_key(slow, pool)


def test_an_unscored_entry_sorts_last():
    assert RK.rank_key({}) > RK.rank_key(_entry(1, "none"))


def test_every_component_stays_inside_the_unit_interval():
    """Whatever a driver hands it -- including values no real run produces."""
    wild = {"estimated_rel_error": 1e-300, "observed_order": 1e6, "design_order": 1,
            "scheme_family": "fd", "invariant_worst_headroom": 1e6,
            "d1_outcome": "clean", "d1_slope": -50.0, "solve_time_s": 1e-9,
            "dynkin_z": float("inf")}
    pool = [wild, {"solve_time_s": 1e9}]
    for name, value in RK.components(wild, TOL, pool).items():
        assert value is None or 0.0 <= value <= 1.0, (name, value)
    index, _ = RK.sqi(wild, TOL, pool)
    assert index is None or 0.0 <= index <= 1.0


# --- the `rank` command -------------------------------------------------------

def _plan(tmp_path, name, block):
    plan = tmp_path / "plans" / name
    plan.mkdir(parents=True)
    lines = "\n".join(f"{k}: {v}" for k, v in block.items())
    (plan / "SOLUTION.md").write_text(f"<metrics>\n{lines}\n</metrics>\n")
    return plan


def test_rank_command_orders_a_workspace_and_names_its_missing_inputs(tmp_path, capsys):
    """It reads blocks; it never re-runs a solver. A block written before a field
    existed has to be visible as such, or the table prints a confident order over
    columns of dashes."""
    from verifylib import cli
    _plan(tmp_path, "1-slow", {"score": 10, "provenance": "analytic",
                               "estimated_rel_error": "1.010e-04", "solve_time_s": 10.0})
    _plan(tmp_path, "2-fast", {"score": 10, "provenance": "analytic",
                               "estimated_rel_error": "1.023e-04", "solve_time_s": 0.1})
    assert cli.main(["rank", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    # Tied on accuracy inside the noise, so the cheaper plan wins -- which is the
    # whole point, and the opposite of what the old `estimated_rel_error` key did.
    assert "best: 2-fast" in out
    assert "order, structure" in out          # neither block carries those inputs


def test_rank_command_reports_json_and_refuses_an_empty_workspace(tmp_path, capsys):
    from verifylib import cli
    assert cli.main(["rank", str(tmp_path)]) == 1
    _plan(tmp_path, "1-only", {"score": 9, "provenance": "manufactured_partial",
                               "estimated_rel_error": "1e-6"})
    assert cli.main(["rank", str(tmp_path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ranking"][0]["plan"] == "1-only"
    assert payload["ranking"][0]["components"]["cost"] is None    # a pool of one


def test_the_sde_order_band_follows_the_band_the_guard_applied():
    """[REV 34] / [REV 43] / D12. `order_margin`'s SDE branch carried a two-sided
    distance from the floor, the same shape `crn.guards` had -- so a correct
    additive-noise Euler-Maruyama, whose strong order is 1 against a defaulted floor
    of 0.5, scored `1 - |0.986 - 0.5|/0.5 = 0.028`: last in its pool, on a right
    answer. The branch is unreachable today (`design_order` is set only by the PDE
    driver), which is why no score-based acceptance criterion can see it and why
    this is asserted by calling `order` directly.

    The band is not re-derived here. `crn.guards` computes it once and the SDE
    driver copies it into `summary["order_band"]`, so there is one rule and three
    call sites.
    """
    em = {"kind": "sde", "observed_order": 0.986, "order_floor": 0.5, "design_order": 0.5}
    assert RK.order(em) == pytest.approx(0.028, abs=0.01)          # two-sided, the default
    one_sided = {**em, "order_band": "one-sided (scheme family undeclared in the plan)"}
    assert RK.order(one_sided) > 0.9
    # One-sided still penalises a genuine shortfall, which is the defect the band exists
    # for: an order near zero is a solver that ignores its increments.
    assert RK.order({**one_sided, "observed_order": 0.1}) == pytest.approx(0.2, abs=0.01)


def test_an_abstained_sde_order_is_not_ranked_on():
    """D8's abstention, one layer up. `crn.guards` returns `orders_ok: "unresolved"`
    when the estimate's own error bar admits more than the band; ranking a plan on a
    number the kernel just declined to gate on would reintroduce the defect where no
    criterion looks."""
    abstained = {"kind": "sde", "observed_order": 2.73, "order_floor": 1.0,
                 "design_order": 1.0, "order_check": "unresolved"}
    assert RK.order(abstained) is None
    assert RK.order({**abstained, "order_check": "true"}) is not None
