"""The BENCHOP-SLV fixes, each against the measurement that motivated it.

``docs/improving-autonumerics/`` is the write-up: a ten-problem option-pricing run
in which the pipeline shipped answers a geometric mean 2.6x further from the
reference than a single untooled model completion's median attempt, while its own
plan pools already held a better plan on six of ten problems. The fixes are
numbered F0-F19 there and the numbers are kept here, because what each test pins is
a specific defect with a specific measurement behind it and the two should not drift
apart.

The four that changed the priority order of the whole thing, measured by running the
shipped ranker on the shipped metrics blocks:

* ``accuracy`` -- 0.35 of the quality index -- read **1.000 on all eleven** plans of
  the four pools whose artefacts survive, because it snaps the error up to twice its
  own Monte-Carlo noise and then divides by the same half-width (F16).
* ``ranking.order`` -- 0.25 -- returned ``None`` on **all 88** metrics blocks in
  ``workspace/``, because ``design_order`` was set by the PDE driver alone (F17).
* ``consistency`` -- 0.10 -- was ``None`` throughout, so the live index was two
  thirds structure and one third **cost**, and in four pools of four the most
  accurate plan was the slowest and ranked last (F4).
* The price the solver computed never crossed the SDE sandbox boundary at all (F0).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

from verifylib import ranking
from verifylib.kernel import plan_meta, sandbox
from verifylib.kernel.sde import crn

# --- F0 / F12: the graded quantity crosses the boundary -----------------------

SOLVER = '''
import numpy as np

def solve_sde(num_paths, dt, T, seed=42, dW=None, observables=None):
    rng = np.random.default_rng(seed)
    Nt = max(1, round(T / dt))
    if dW is None:
        dW = np.sqrt(T / Nt) * rng.standard_normal((num_paths, Nt))
    X = 1.0 + dW.sum(axis=1)
    payoff = np.maximum(X - 1.0, 0.0)
    return {"terminal_paths": X,
            "empirical_mean": float(X.mean()),
            "empirical_variance": float(X.var(ddof=1)),
            "functionals": {"price_K1": float(payoff.mean()), "bad": float("nan")},
            "functional_se": {"price_K1": float(payoff.std(ddof=1) / np.sqrt(num_paths)),
                              "orphan": 1.0},
            "num_paths": num_paths, "dt": T / Nt, "T": T}
'''


@pytest.fixture
def priced_plan(tmp_path):
    plan = tmp_path / "1-em"
    plan.mkdir()
    (plan / "solver.py").write_text(SOLVER)
    return str(plan)


def test_the_sde_sandbox_carries_the_price_the_solver_computed(priced_plan):
    """F0. ``_dump_sde`` captured four keys and ``functionals`` was not one of them,
    and ``_reconstruct``'s sde branch returned before the line that would have
    restored it -- so on **every** SDE problem in the repo the quantity of interest
    was discarded at the process boundary before the kernel could see it. The
    benchmark harness hit the identical bug on its own boundary and fixed it; this
    one never got the same fix, and every downstream claim about selection rests on
    the omission."""
    got = sandbox.run(priced_plan, "sde", {"num_paths": 4000, "dt": 0.05, "T": 1.0,
                                           "seed": 7})
    assert got["status"] == "ok", got
    assert "price_K1" in got["result"]["functionals"]
    assert got["result"]["functionals"]["price_K1"] > 0.0


def test_a_functional_standard_error_crosses_with_it(priced_plan):
    """F12. A price is one float, so its error bar cannot be recovered on the kernel
    side at all -- it has to be returned. It is also the only number in the contract
    that prices variance reduction, which is the axis two plans running the same
    scheme at the same dt differ on."""
    got = sandbox.run(priced_plan, "sde", {"num_paths": 4000, "dt": 0.05, "T": 1.0,
                                           "seed": 7})
    se = got["result"]["functional_se"]
    assert 0.0 < se["price_K1"] < got["result"]["functionals"]["price_K1"]


def test_an_unusable_functional_is_dropped_rather_than_crashing_the_solve(priced_plan):
    """A non-finite value and a standard error for a name that did not come back are
    both dropped at the boundary. The check that consumes them then reports
    ``not_reported``, which is a measurement; a raised exception would report
    ``crashed`` and blame the solver for the wrong thing."""
    got = sandbox.run(priced_plan, "sde", {"num_paths": 1000, "dt": 0.1, "T": 1.0})
    assert "bad" not in got["result"]["functionals"]
    assert "orphan" not in got["result"]["functional_se"]


def test_the_sandbox_reports_cpu_time_as_well_as_wall_time(priced_plan):
    """F4. Wall time is divided by a forking solver's worker count and is
    contaminated by the contention between a pool's own plans at ``parallelism:
    3``; the log-scale pool-relative form in ``ranking.cost`` cancels a *uniform*
    slowdown and contention is not uniform. CPU time is immune to both."""
    got = sandbox.run(priced_plan, "sde", {"num_paths": 1000, "dt": 0.1, "T": 1.0})
    assert isinstance(got["cpu_s"], float) and got["cpu_s"] > 0.0


# --- F16: accuracy stops declaring a tie at its own ceiling -------------------

def _noise_limited(err=1e-3, se=1e-3):
    return {"estimated_rel_error": err, "mc_se_rel": se, "ci_mult": 2.0}


def test_an_error_inside_its_own_noise_is_unmeasured_not_perfect():
    """F16, and the measurement that found it: all eleven B01-B04 plans scored
    accuracy **1.000**, every one of them because its ``estimated_rel_error`` was
    below twice its ``mc_se_rel`` and ``achievable_decades`` then divided by the
    same half-width. A tie at the component's maximum is not a tie, it is a claim --
    and the claim is false, because what the kernel knows is that the error was not
    resolved at this budget."""
    assert ranking.noise_limited(_noise_limited()) is True
    assert ranking.accuracy(_noise_limited(), 0.1) is None
    # ...and the three worst real cases, where the noise is 23x to 38x the estimate.
    for est, se in ((7.521e-3, 0.2635), (7.912e-3, 0.1811), (7.371e-3, 0.2837)):
        assert ranking.accuracy({"estimated_rel_error": est, "mc_se_rel": se,
                                 "ci_mult": 2.0}, 0.1) is None


def test_a_resolved_error_still_scores_and_still_orders():
    """The fix must not take accuracy away from a plan that measured one. An error
    comfortably outside its own bar is a measurement and keeps ordering."""
    good = {"estimated_rel_error": 1e-4, "mc_se_rel": 1e-6, "ci_mult": 2.0}
    worse = {"estimated_rel_error": 1e-2, "mc_se_rel": 1e-6, "ci_mult": 2.0}
    assert ranking.accuracy(good, 0.1) > ranking.accuracy(worse, 0.1)
    # A PDE Path A summary has no error bar at all and is untouched by the change.
    assert ranking.accuracy({"estimated_rel_error": 1e-4}, 0.1) is not None


# --- F3: the component reads the quantity the problem is scored on ------------

def _priced(err, tol=0.002, **kw):
    return {"functional_rel_error": err, "functional_rel_tol": tol,
            "functional_name": "price_K1", **kw}


def test_accuracy_prefers_the_declared_functional_over_the_moment_error():
    """F3. ``estimated_rel_error`` on an SDE is the worse of the mean and variance
    errors, judged against ``variance_rel_err_max``; neither the price nor its own
    ``rel_tol`` was ever consulted, on a suite scored entirely on prices."""
    assert ranking.accuracy(_priced(1e-4), 0.1) > ranking.accuracy(_priced(1.5e-3), 0.1)


def test_a_functional_estimate_is_never_noise_limited():
    """The point of F12. ``functional_rel_error`` is a dt-bias term **plus** the
    reported sampling term, so the noise is inside the estimate rather than a floor
    underneath it -- which is exactly why it can order a pool the moment error
    cannot."""
    summary = _priced(1.2e-3, **_noise_limited())
    assert ranking.noise_limited(summary) is True      # the moment error is not resolved
    assert ranking.accuracy(summary, 0.1) is not None  # the price still is


def test_the_functional_ceiling_does_not_reward_a_noisier_plan():
    """Why :data:`ranking.FUNCTIONAL_DECADES` is a constant and not the plan's own
    ``achievable_decades``. A ceiling computed from each plan's own error bar
    divides out the quantity that separates plans: with it, a plan whose error is
    twice its sibling's scores *higher* because its bar is wider."""
    tight = _priced(1e-3)
    loose = _priced(2e-3)
    assert ranking.accuracy(tight, 0.1) > ranking.accuracy(loose, 0.1)


# --- F4 / F7: cost stops deciding, and says so --------------------------------

def _plan(name, seconds, **summary):
    return {"plan": name, "iter": 1,
            "kernel": {"score": 9.0, "provenance": "analytic",
                       "summary": {"solve_time_s": seconds,
                                   "invariant_worst_headroom": 0.5, **summary}}}


def test_cost_does_not_decide_a_pool_whose_accuracy_was_never_measured():
    """F4, and the cleanest measured case. On ``sde_heston_dnt`` three plans tie at
    score 9, provenance ``analytic`` and structure 1.000; the index spread
    1.000/0.976/0.846 is **entirely** cost, over 7 seconds out of 65, and the plan
    it puts last is the most accurate of the three. With accuracy's 0.35
    renormalised away, a component declared at a tenth of the index decides a third
    of it -- which is not breaking a tie between equals, because nothing
    established that they were equals."""
    pool = [_plan("a", 57.9, **_noise_limited())["kernel"]["summary"],
            _plan("b", 64.9, **_noise_limited())["kernel"]["summary"]]
    assert ranking.cost_admissible(0.1, pool) is False
    assert ranking.components(pool[0], 0.1, pool)["cost"] is None
    # ...and admissible again the moment every plan has a measured accuracy.
    resolved = [dict(s, estimated_rel_error=1e-2, mc_se_rel=1e-6) for s in pool]
    assert ranking.cost_admissible(0.1, resolved) is True
    assert ranking.components(resolved[0], 0.1, resolved)["cost"] is not None


def test_the_winner_does_not_change_when_the_solve_times_are_swapped():
    """F4's acceptance test, **exercising ``rank_key`` and not ``sqi`` alone** --
    the first version of the fix pointed only at the weighted component, and
    ``solve_time_s`` enters twice. On ``sde_dupire_local_vol_dnt`` the index was
    identical and the fifth lexicographic key decided outright, choosing the plan
    that is 3.4-4.5 sigma worse on the graded price."""
    fast = _plan("fast", 10.0, **_noise_limited())
    slow = _plan("slow", 60.0, **_noise_limited())
    pool = [e["kernel"]["summary"] for e in (fast, slow)]
    before = min((fast, slow), key=lambda e: ranking.rank_key(e, pool))["plan"]
    fast["kernel"]["summary"]["solve_time_s"] = 60.0
    slow["kernel"]["summary"]["solve_time_s"] = 10.0
    after = min((fast, slow), key=lambda e: ranking.rank_key(e, pool))["plan"]
    assert before == after == "fast"       # i.e. nothing was decided by the clock


def test_a_pool_that_mixes_cpu_and_wall_time_is_not_ranked_on_cost():
    """They are not the same quantity, so a pool holding one of each would be ranked
    on which measurement it happened to get."""
    pool = [dict(_plan("a", 10.0)["kernel"]["summary"], cost_source="cpu",
                 estimated_rel_error=1e-2, mc_se_rel=1e-6),
            dict(_plan("b", 60.0)["kernel"]["summary"], cost_source="wall",
                 estimated_rel_error=1e-2, mc_se_rel=1e-6)]
    assert ranking.cost(pool[0], pool) is None


def test_the_ranker_names_the_key_that_actually_decided():
    """F7, demoted from a fix to a reporting change because plan position, scheme
    family and solver cost are perfectly collinear on this suite and the positional
    bias the original fix was written against is not separable from them. What is
    real is the silence: an exhausted ranking key is the ranker reporting that the
    pool is indistinguishable, and three of the ten shipped winners were chosen that
    way with nothing in the record saying so."""
    a = _plan("a", 10.0, estimated_rel_error=1e-4, mc_se_rel=1e-6)
    b = _plan("b", 60.0, estimated_rel_error=1e-2, mc_se_rel=1e-6)
    pool = [e["kernel"]["summary"] for e in (a, b)]
    assert ranking.decided_by(a, b, pool) == "sqi"
    # Identical on everything measurable: the honest answer is "tie", not "cost".
    c, d = _plan("c", 10.0, **_noise_limited()), _plan("d", 60.0, **_noise_limited())
    pool = [e["kernel"]["summary"] for e in (c, d)]
    assert ranking.decided_by(c, d, pool) == "tie"


# --- F17: the order component exists on an SDE --------------------------------

def test_the_order_component_fires_once_a_design_order_is_published():
    """F17. ``ranking.order`` needs ``design_order``; the only assignment in the repo
    was ``kernel/driver.py``'s, from the plan's declared ``spatial_order``, on the
    **PDE** path. Across all 88 metrics blocks in ``workspace/`` -- 23 SDE and 65
    PDE -- ``design_order`` appeared in zero and the component had never once
    fired."""
    base = {"kind": "sde", "order_check": "true", "observed_order": 0.49,
            "order_floor": 0.5, "order_band": "two-sided"}
    assert ranking.order(base) is None
    assert ranking.order(dict(base, design_order=0.5)) is not None
    # An abstention still yields None: ranking a plan on an estimate the kernel just
    # declined to gate on reintroduces the defect one layer up.
    assert ranking.order(dict(base, design_order=0.5,
                              order_check="unresolved")) is None


# --- F18: an exactly sampled step is excused, not failed ----------------------

def test_a_declared_exactly_sampled_component_abstains_instead_of_failing():
    """F18, and it is the sibling of a case the repo already fixed. ``abstention``
    excuses a scheme whose CRN differences collapse to **roundoff** -- exact
    *integration* -- and detects it because the differences vanish. Exact
    *sampling* is invisible to that test for the opposite reason: the draw uses
    auxiliary randomness, the levels decorrelate, and the differences stay O(1).

    Measured on ``sde_sabr_cev_call_i``: an exact Poisson-mixed BESQ(0) step reads
    ``-0.0251 +/- 0.0126``, a tight interval around a meaningless number, so neither
    the standard-error hatch nor the band-width hatch fires either, and
    ``score.rubric_sde`` drops a converged, invariant-clean plan from 9 to 7. The
    plan had written the prediction into its own SOLUTION.md before implementation
    and nothing downstream could read it."""
    detail = {"s0": 1.0, "s1": 1.02, "se_s0": 0.01, "se_s1": 0.01,
              "se_order": 0.0126, "roundoff_floor": 1e-13}
    assert crn.abstention(-0.0251, detail) is None          # no hatch fires today
    hard = crn.guards(-0.0251, 1.0, 0.5, 1.0, detail, declared=True)
    assert hard["orders_ok"] is False
    excused = crn.guards(-0.0251, 1.0, 0.5, 1.0, detail, declared=True,
                         exact_sampling=True)
    assert excused["orders_ok"] == "unresolved"
    assert "sampled exactly" in excused["abstained"]
    assert excused["exact_sampling"] is True


def test_the_exemption_is_declared_and_never_inferred():
    """Like ``scheme_family``, and for the same reason: a kernel that inferred it
    would let a plan rename itself out of the only check that catches a solver
    ignoring ``dW``. What keeps the declaration honest is that an abstention is not
    a pass -- it costs the plan the order check's evidence rather than earning it."""
    assert plan_meta.parse_components("0, 1") == (0, 1)
    assert plan_meta.parse_components("[1]") == (1,)
    assert plan_meta.parse_components(None) == ()
    assert plan_meta.parse_components("all") == ()      # unparseable voids it whole
    assert plan_meta.parse_components("-1") == ()


def test_a_plan_declares_the_exemption_in_its_frontmatter(tmp_path):
    plan = tmp_path / "2-exact-besq0"
    plan.mkdir()
    (plan / "SOLUTION.md").write_text(
        "---\nid: 2\nscheme_family: exact-transition\n"
        "exact_sampling_components: 0\n---\n\nbody\n")
    meta = plan_meta.read(str(plan))
    assert meta["exact_sampling_components"] == (0,)
    assert meta["scheme_family"] == "exact-transition"
    assert meta["scheme_family_source"] == "declared"


def test_the_new_families_exist_so_a_cross_family_agreement_can_count():
    """F6, argued from the code rather than from the 104x the write-up first quoted,
    which the seed study did not support -- the largest three-seed mean ratio
    anywhere on the suite is 2.91x. The mechanism stands on its own: an agreement
    between two plans counts as a circularity break only when their families
    differ, and before these values existed an exact chi-squared scheme and
    Andersen QE both had to declare ``other``.

    They carry **no** expected strong order, deliberately: the CRN study reads
    0.23-0.31 against declared floors of 0.5 on six plans across three families on
    this suite, so it is in no state to have more requirements hung off it."""
    assert "exact-transition" in plan_meta.SDE_FAMILIES
    assert "quadratic-exponential" in plan_meta.SDE_FAMILIES
    assert "exact-transition" not in crn.FAMILY_ORDER_KEY
    assert "quadratic-exponential" not in crn.FAMILY_ORDER_KEY
    assert plan_meta.infer_family("andersen-qe-exact-variance") == "quadratic-exponential"
    assert plan_meta.infer_family("broadie-kaya-exact") == "exact-transition"


# --- F15: an abstaining component's noise is not an order ---------------------

def test_an_abstaining_component_does_not_publish_its_noise_as_the_order():
    """The second half of B02's diagnosis. The order study is per component and the
    verdict binds to the worst, which is right -- a component the estimator could
    not read is not evidence that the others are fine. The **scalar** carried into
    the metrics block then came from that same component: on
    ``sde_sabr_cev_call_ii`` the exact lognormal volatility leg abstained at the
    roundoff floor and the block published its ``-1.0772``, while the S component's
    real ``0.235 +/- 0.1305`` lived only in the review prose. A second plan with a
    different scheme reports the identical ``-1.0772``, which is what gives it
    away: two schemes cannot share an order to four decimals."""
    from verifylib.kernel.sde.driver import _orders

    # Component 0 is a clean strong-order-0.5 ladder (differences shrinking by
    # sqrt(2) per halving); component 1 is an exactly sampled leg the plan declared.
    base = np.linspace(1.0, 2.0, 4000)
    offsets = [0.02 + 0.02 / np.sqrt(2), 0.02 / np.sqrt(2), 0.0]   # coarse to fine
    runs = [np.stack([base + d, np.ones_like(base)], axis=1) for d in offsets]

    def solve(call, **kw):
        return {"status": "ok", "result": {"terminal_paths": runs.pop(0)}}

    _ok, detail, ladder = _orders(".", solve, 4000, 0.004, 1.0, 3, 42, 0.5, 1.0,
                                  declared=True, candidates=[0.5], components=2,
                                  exact_sampling=(1,))
    assert detail["abstained"]
    # The binding component abstained, so the scalar comes from a measured one --
    # or is None, which is honest, and never the abstaining component's number.
    assert detail["strong_order"] != detail["abstaining_component_order"]
    assert len(ladder) == 3


# --- F8: a forking solver can be imported at all ------------------------------

def test_the_benchmark_runner_registers_the_solver_module_and_leaves_it(tmp_path):
    """F8. A module executed from a file and never registered has no importable
    name, so a top-level function handed to ``multiprocessing.Pool`` raises
    ``PicklingError`` -- which is what voided B07, whose solver was forking workers.
    And it must **stay** registered: the originally quoted ``finally`` would have
    reinstated the bug one call later, when the pool forks and the child looks the
    name up again."""
    from benchmark.runner import _import_solver

    path = tmp_path / "solver.py"
    path.write_text("def solve_sde(**kw):\n    return {}\n")
    sys.modules.pop("_sandbox_solver", None)
    try:
        mod = _import_solver(str(path))
        assert sys.modules.get("_sandbox_solver") is mod
    finally:
        sys.modules.pop("_sandbox_solver", None)


def test_a_solver_that_fails_to_import_does_not_leave_a_half_built_module(tmp_path):
    from benchmark.runner import _import_solver

    path = tmp_path / "solver.py"
    path.write_text("raise ValueError('boom')\n")
    sys.modules.pop("_sandbox_solver", None)
    with pytest.raises(ValueError):
        _import_solver(str(path))
    assert "_sandbox_solver" not in sys.modules


# --- F10: the report can state the harness error at all -----------------------

def test_the_harness_error_is_reported_on_a_functional_graded_problem():
    """F10. The per-plan table read ``rel_l2_err``, which does not exist on a
    problem scored on declared prices -- so the entire BENCHOP suite printed ``-``
    in its error column, and reconstructing those numbers by hand is what made the
    first analysis of the run expensive. One number for every grading shape, as a
    multiple of the gate, because the per-value tolerances on this suite span 0.002
    to 0.015 and two raw errors are not comparable."""
    from benchmark.report import harness_error

    value, label = harness_error({
        "functional_errs": {"price_K1": 1.2e-3, "price_K2": 9.0e-3},
        "functional_tol_used": {"price_K1": 2e-3, "price_K2": 1.5e-2},
        "functional_tol": 2e-3})
    assert label == "functional"
    assert value == pytest.approx(0.6)        # the worst of 0.60 and 0.60... K1 ties
    # The widened gate is read per value, not averaged: a value held to a looser
    # bar because the reference itself is uncertain is not failing at 9e-3.
    assert harness_error({"functional_errs": {"a": 1.0e-2},
                          "functional_tol_used": {"a": 5.0e-3},
                          "functional_tol": 5.0e-3})[0] == pytest.approx(2.0)
    assert harness_error({})[0] is None


# --- the measured artefacts, where they still exist ---------------------------

WORKSPACE = os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "workspace")
POOLS = ("sde_sabr_cev_call_i", "sde_sabr_cev_call_ii", "sde_heston_call_feller",
         "sde_heston_dnt")


def _pool(slug):
    from verifylib.review import parse_metrics

    root = os.path.join(WORKSPACE, slug, "plans")
    if not os.path.isdir(root):
        return []
    out = []
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name, "SOLUTION.md")
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as fh:
            block = parse_metrics(fh.read())
        if block:
            out.append(block)
    return out


@pytest.mark.parametrize("slug", POOLS)
def test_solve_time_cannot_change_the_ranking_of_a_benchop_pool(slug):
    """The regression this whole set exists to prevent, asserted directly.

    Before the fixes: in four pools of four the most accurate plan is the slowest
    and the index ranks it last or near-last, and within-pool pairs where the higher
    index is the more accurate plan run 8 of 30 pooled over three seeds -- binomial
    p = 0.008 against a coin flip. The index was not failing to measure quality; it
    was measuring something that ran the other way.

    This used to assert ``components(...)["cost"] is None`` -- a *proxy*: cost can
    only distort a ranking where it is admissible, and it is admissible only where
    every plan's accuracy is measurable, so "accuracy is unmeasurable here" stood
    in for "cost cannot decide". The proxy stopped holding on 2026-11-01, and for a
    good reason: getting `evaluation_thresholds.functionals` into the B03 spec made
    `_functional_fraction` return a value, which made accuracy measurable, which
    made cost admissible. A successful repair tripped the guard.

    So the property itself is asserted instead, and it needs no proxy: **rank the
    pool, then rank it again with every plan's ``solve_time_s`` reversed, and the
    order must be identical.** Reversing rather than perturbing is deliberate --
    it turns the fastest plan into the slowest and back, the largest change the
    pool admits, so a surviving order cannot be luck.

    Measured 2026-11-01, the two halves of this guard divide the work: the
    structural asserts pin the invariant on all four pools, and the reversal
    assert *bites* on ``sde_heston_call_feller``, whose order goes ``[1, 0, 2] ->
    [1, 2, 0]`` under the old index-plus-stopwatch and is unmoved under the new
    one. The other three pools are unmoved either way -- their solve times are too
    close, or their plans separate on score first -- so the reversal assert alone
    would be a guard with one tooth. Both halves stay.

    This holds on every pool whether or not accuracy discriminates, which is what
    the accuracy-gap formulation could not manage: measured 2026-11-01, the
    accuracy component is flat across all four BENCHOP pools (``None`` on
    ``sde_heston_dnt``, ``0.0`` on the other three), so any assertion phrased in
    terms of an accuracy *gap* skips everywhere and guards nothing. That flatness
    is itself worth knowing and is recorded in
    ``ranking.CERT_COMPONENTS``: at the kernel's diagnostic path count the graded
    quantity is not resolvable, so the one component that matters contributes
    nothing and the index rests on the others.
    """
    pool = _pool(slug)
    if len(pool) < 2:
        pytest.skip(f"{slug} has no plan directories on this checkout")

    # Cost is not in the index, and the stopwatch is not in the key.
    assert "cost" not in ranking.CERT_COMPONENTS
    assert "cost" not in ranking.WEIGHTS
    assert "solve_time_s" not in ranking.RANK_KEYS

    def order(blocks):
        entries = [{"kernel": {"summary": s, **s}, "plan": i}
                   for i, s in enumerate(blocks)]
        entries.sort(key=lambda e: ranking.rank_key(e, blocks))
        return [e["plan"] for e in entries]

    times = [s.get("solve_time_s") for s in pool]
    flipped = [{**s, "solve_time_s": t} for s, t in zip(pool, reversed(times),
                                                        strict=True)]
    assert order(pool) == order(flipped), (
        f"{slug}: reversing the pool's solve times reordered it -- "
        f"{times} -> {list(reversed(times))}")

    # ...and where accuracy *does* separate two plans tied on score and
    # provenance, the less accurate one must not be the winner.
    entries = [{"kernel": {"summary": s, **s}} for s in pool]
    entries.sort(key=lambda e: ranking.rank_key(e, pool))
    top = entries[0]["kernel"]
    band = [e["kernel"] for e in entries
            if (e["kernel"].get("score"), e["kernel"].get("provenance"))
            == (top.get("score"), top.get("provenance"))]

    def acc(k):
        s = k["summary"]
        return ranking.accuracy(s, ranking._tol_of(s, {}))

    graded = [acc(k) for k in band if acc(k) is not None]
    if len(set(graded)) > 1:
        assert acc(top) == max(graded), (
            f"{slug}: the winner's accuracy component is {acc(top)} but a plan "
            f"tied with it on score and provenance measured {max(graded)}")


REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# --- the B03 Sonnet run (2026-10-31): a correct plan graded 3 and not selected ----
#
# One chain, four links, each of which had to hold for the run to fail the way it
# did. The pipeline produced an Andersen-QE solver whose three prices were all inside
# their gates -- the harness passed it 10/10 -- scored it 3, and shipped a plain Euler
# plan that could not allocate at grading scale and returned `crashed`. The four tests
# below pin each link open.


def test_spec_vocabulary_reports_the_structured_threshold_keys():
    """Link 1: `functionals` was absent from the vocabulary, so it was never declared.

    `evaluation_thresholds.functionals` is read by both drivers through
    `functionals.run_declared`, but it is read as `(thresholds or {}).get(...)` and
    the first cut of `cli.py spec-vocabulary` matched only the `thresholds.get(...)`
    shape. The formulator ran the command, did not find a per-functional tolerance
    key, and recorded B03's three price gates as `ambiguous` R23/R24 with the note
    "No `evaluation_thresholds` key exists for a per-functional price tolerance" --
    so the kernel never measured the only quantity the problem is scored on, and the
    two surviving plans tied at 3 with no accuracy signal between them.

    A vocabulary that under-reports is worse than no vocabulary: the agent trusts it
    and stops looking."""
    import json as _json
    import subprocess
    import sys as _sys

    out = subprocess.run([_sys.executable, "verifylib/cli.py", "spec-vocabulary",
                          "--json"], capture_output=True, text=True, check=True,
                         cwd=REPO)
    vocab = _json.loads(out.stdout)
    assert "functionals" in vocab["evaluation_threshold_keys"]
    assert "functionals" in vocab["structured_threshold_keys"]
    # The shape has to be in the output too. "functionals exists" sends an agent
    # looking for a float; it takes a nested object.
    assert "rel_tol" in vocab["structured_threshold_keys"]["functionals"]
    assert "gate" in vocab["structured_threshold_keys"]["functionals"]
    # And the frontmatter field from link 2, for the same reason.
    assert "exact_sampling_components" in vocab["plan_frontmatter"]


def test_an_exact_sampling_family_with_no_declaration_is_told_which_field_to_declare():
    """Link 2: the abstention existed, was opt-in, and was undiscoverable.

    `crn.abstention` excuses a near-zero strong order only for the components a plan
    names in `exact_sampling_components`, and that is deliberate -- "ignores the
    supplied increments" is a real defect on every other family, and the kernel cannot
    tell the two apart by measurement. What was missing was any way for a plan to
    *learn* the field exists: B03's QE plan declared `scheme_family:
    quadratic-exponential`, read a strong order of -0.0021, and got the bare
    "erratic in dt" with no mention of the one line that would have fixed it.

    The hint never changes a verdict. It is checked here against the four cases where
    it must stay silent, because a hint that fires on a plain Euler plan reading a
    near-zero order would be telling it to hide a genuine defect."""
    from verifylib.kernel.sde import crn

    fires = crn.undeclared_exact_sampling("quadratic-exponential", -0.0021, ())
    assert fires is not None
    assert "exact_sampling_components" in fires
    # ...and it says what it costs, so the declaration is not read as free.
    assert "not a pass" in fires
    assert crn.undeclared_exact_sampling("exact-transition", 0.01, ()) is not None
    # Silent: already declared, wrong family, order plainly measured, no family.
    assert crn.undeclared_exact_sampling("quadratic-exponential", -0.0021, (0, 1)) is None
    assert crn.undeclared_exact_sampling("euler-maruyama", -0.0021, ()) is None
    assert crn.undeclared_exact_sampling("quadratic-exponential", 0.97, ()) is None
    assert crn.undeclared_exact_sampling(None, -0.0021, ()) is None


def test_the_kernel_is_never_more_permissive_than_the_grader_about_memory():
    """Link 3: the kernel allowed 8 GB where the harness allows 4.

    B03's winning plan pre-allocated its noise block as `(num_paths, Nt, 2)`: 4.0 GB
    at the spec's declared 500k paths, which fit inside the kernel's 8192 and scored
    3 with `solve_time_s: 49.7`, and 16 GB at the 2M paths the problem is graded on,
    where the harness spent its whole 300 s budget thrashing and returned `crashed`.
    The plan's own `__main__` ran at 500k, so nothing in the pipeline ever saw it.

    Pinned against `verify.DEFAULT_SOLVER_MEM_MB` rather than the literal 4096, so the
    two cannot drift apart again. Lowering the kernel's ceiling cannot fail a solver
    the grader would accept -- that is the direction that makes it safe."""
    import importlib.util

    from verifylib.kernel import sandbox

    spec = importlib.util.spec_from_file_location(
        "_bench_verify", os.path.join(REPO, "benchmark", "verify.py"))
    verify = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verify)
    assert sandbox.DEFAULT_MEM_MB <= verify.DEFAULT_SOLVER_MEM_MB


def test_peak_memory_over_the_grading_ceiling_caps_the_score():
    """Link 4: the overrun was invisible, and on macOS it cannot be prevented.

    `RLIMIT_AS` is advisory on Darwin and ignored for most allocations, so the aligned
    ceiling is measured and not enforced: plan 1 re-evaluated at the 4096 limit still
    ran, and peaked at 7853 MB -- 1.92x. The measurement is therefore what has to
    bind. Capped at 4 rather than failed, because the plan may be numerically correct
    and what is missing is a demonstration at the graded scale; 4 sits below the 6 an
    MC-inconclusive run keeps, since that one needs more paths and this one cannot be
    given them."""
    from verifylib.kernel import score

    base = {"kind": "sde", "path": "B", "crashed": False, "any_test_ran": True,
            "surrogate_ok": True, "moments_ok": True, "resolved": True,
            "constraints_ok": True, "orders_ok": True, "invariants_ok": True,
            "richardson_stable": True}
    clean = score.certify(dict(base, mem_fraction_used=0.16, peak_mem_mb=654.5,
                               mem_mb_limit=4096.0))
    assert not any(c["source"] == "memory" for c in clean["caps"])

    over = score.certify(dict(base, mem_fraction_used=1.917, peak_mem_mb=7853.3,
                              mem_mb_limit=4096.0))
    memory = [c for c in over["caps"] if c["source"] == "memory"]
    assert len(memory) == 1 and memory[0]["score"] == 4
    assert over["score"] <= 4
    # The reason has to carry the fix, not just the finding.
    assert "per step" in memory[0]["reason"]


def test_the_memory_cap_reads_the_graded_call_and_not_the_kernels_own_ladder():
    """The correction to the test above, and it cost a frozen canary 4 points.

    The first cut capped on a max over every solve in the plan. But the CRN ladder
    refines to ``2**(levels-1)`` times as many steps, so its finest level needs
    several times the graded call's memory -- the kernel's own cost choice, not
    something the grader asks of the solver. Measured: `sde_ignores_crn`, whose graded
    call peaks at 797 MB (0.19 of the ceiling), reaches 4611 MB on the finest ladder
    level and was dropped from 8 to 4 by a memory finding unrelated to what it tests.

    So `mem_fraction_used` -- the field the cap reads -- is the graded call's, and the
    ladder's peak is reported beside it as `ladder_mem_fraction` without capping
    anything. B03's plan 1 is unaffected: its *graded call* alone peaks at 7851 MB
    against the 4096 ceiling, because `sqrt(dt) * standard_normal(shape)` holds two
    copies of the block at once."""
    from verifylib.kernel import score

    base = {"kind": "sde", "path": "B", "crashed": False, "any_test_ran": True,
            "surrogate_ok": True, "moments_ok": True, "resolved": True,
            "constraints_ok": True, "orders_ok": True, "invariants_ok": True,
            "richardson_stable": True}
    # A comfortable graded call whose ladder goes over: no finding.
    ladder_only = score.certify(dict(base, mem_fraction_used=0.195,
                                     peak_mem_mb=4610.9, mem_mb_limit=4096.0))
    assert not any(c["source"] == "memory" for c in ladder_only["caps"])
    # `certify` must read `mem_fraction_used` and nothing else for this decision, so
    # a large `peak_mem_mb` alone can never cap -- pinned against the same evidence
    # with no memory fields at all rather than a literal, since this fixture's score
    # is set by its generator evidence and is not the subject here.
    assert ladder_only["score"] == score.certify(dict(base))["score"]


# --- the component-pairing defect, measured on B03 2026-11-01 ----------------


def _heston_spec():
    """B03's shape: a 2-component state whose only reference covers component 1."""
    return {
        "state_dimension": 2,
        "parameters": {"kappa": 2.58, "theta": 0.043, "sigma": 1.0, "V_0": 0.114},
        "time_interval": {"T": 1.0},
        "verification": {
            "moment_ode": {
                "state": ["m1", "m2"],
                "rhs": ["kappa*(theta - m1)",
                        "2*kappa*theta*m1 - 2*kappa*m2 + sigma**2*m1"],
                "initial": ["V_0", "V_0**2"],
                "closes_exactly": True,
                "mean_from": "m1",
                "variance_from": "m2 - m1**2",
                "applies_to_component": 1,
                "applies_to_name": "V",
            },
        },
    }


def test_a_reference_scoped_to_one_component_is_not_compared_against_another():
    """The whole of the 2026-11-01 B03 failure, in one assertion.

    The spec says its moment ODE covers component 1 (``V``) and nothing read the
    field, so ``V``'s reference ``(0.0484, 0.0102)`` was compared positionally
    against component 0 (``S``, empirical ``(100.17, 603.59)``) and reported
    ``mean_rel_err: 2069`` on a solver whose ``V`` moments were inside tolerance.
    """
    from verifylib.kernel.sde.driver import _compare, _reference

    reference, detail, surrogate_ok = _reference(_heston_spec(), "B", 0.1)
    assert surrogate_ok is True
    assert detail["route"] == "moment_ode"
    assert detail["applies_to_component"] == 1
    assert detail["applies_to_name"] == "V"
    # The pairing travels in the detail, parallel to the value lists -- out of
    # `reference` itself, which callers iterate and float. It is keyed by moment
    # kind, because the 2026-11-01 re-run showed the two kinds need not cover the
    # same components: that spec's mean covered both and its variance covered only
    # `V`, and one shared list gave the variance the mean's pairing.
    assert detail["components"] == {"mean": [1], "variance": [1]}
    assert reference["mean"] == [pytest.approx(0.04837995, rel=1e-6)]

    # S is component 0 and nothing in the spec says anything about its moments.
    stats = [{"mean": 100.17, "variance": 603.59, "se_mean": 0.35, "se_variance": 13.5},
             {"mean": 0.0481, "variance": 0.00998, "se_mean": 2.1e-4,
              "se_variance": 1.1e-4}]
    outcome, got = _compare(reference, stats, 0.05, 0.1, 0.01, 2.0,
                            components=detail["components"])

    assert [e["component"] for e in got["mean"]] == [1], \
        "the only reference is component 1's, so component 1 is the only comparison"
    assert got["mean"][0]["empirical"] == 0.0481
    # The number that capped three plans at 6 and burned a usage window.
    assert got["mean_rel_err"] < 0.05, got["mean_rel_err"]
    assert outcome is True, got


def test_the_resolution_check_pairs_by_component_too():
    """`resolved` read the same mispairing, and it is the field the conductor's
    unresolved-retry rule fires on -- so a mispaired reference did not merely
    misreport, it dispatched three extra solver cycles."""
    import verifylib.kernel.metrics as metrics
    from verifylib.kernel.sde import constraints as C
    from verifylib.kernel.sde.driver import _paired

    reference, components = {"variance": [0.0102]}, [1]
    stats = [{"mean": 100.17, "variance": 603.59, "se_mean": 0.35, "se_variance": 13.5},
             {"mean": 0.0481, "variance": 0.00998, "se_mean": 2.1e-4,
              "se_variance": 1.1e-4}]
    per = [{"component": i, **C.resolution(st, ref, 0.1, 2.0)}
           for i, st, ref in _paired(stats, reference, "variance", components)]
    assert [p["component"] for p in per] == [1]
    assert all(p["resolved"] for p in per), per
    assert metrics.passed(all(p["resolved"] for p in per))

    # Positionally -- the old behaviour -- component 0's CI is 2*13.5/0.0102 wide
    # against V's reference, which no path count on earth resolves.
    old = C.resolution(stats[0], reference["variance"][0], 0.1, 2.0)
    assert not old["resolved"]


def test_an_out_of_range_component_demotes_the_route_instead_of_clamping():
    """Clamping an invalid index to 0 is the mispairing, not a recovery from it."""
    from verifylib.kernel.sde import surrogates

    spec = _heston_spec()
    spec["verification"]["moment_ode"]["applies_to_component"] = 7
    got, detail = surrogates.best(spec, trust_tol=0.1)
    assert got is None, "a reference for a component the state does not have is no reference"
    assert "applies_to_component 7" in detail["attempted"]["moment_ode"]


def test_a_scalar_only_surrogate_declines_on_a_multi_component_spec():
    """`stationary` integrates `drift_expression` in the single variable `X`. On a
    2-D spec that is the system's drift, not the component's, and B03 survived it
    only because its drift happens to be prose."""
    from verifylib.kernel.sde import surrogates

    spec = _heston_spec()
    spec["verification"]["stationary_density"] = {
        "ergodic": True, "support": [1e-6, 2.0], "applies_to_component": 1}
    spec["drift_expression"] = "kappa*(theta - X)"
    spec["diffusion_expression"] = "sigma*sqrt(X)"
    got, detail = surrogates.stationary(spec)
    assert got is None
    assert "scalar" in detail["reason"] and "2 state components" in detail["reason"]


# --- the 2026-11-01 re-run: the same defect class, reached by Path A ------------

def _feller_moments():
    """``analytic_moments`` as ``sde_heston_call_feller``'s formulator wrote it on
    the third run, verbatim in shape: an exact CIR mean and variance for ``V``, and
    for ``S`` an exact mean but **no** variance, because ``Var[S_T]`` has no closed
    form under Feller violation. The hole is the point."""
    return {"has_analytic_solution": True, "state_dimension": 2,
            "mean_X": "S_0", "variance_X": None,
            "mean_Y": "theta + (V_0 - theta) * np.exp(-kappa * t)",
            "variance_Y": "(sigma_v**2 / kappa) * ( V_0 * (np.exp(-kappa*t) "
                          "- np.exp(-2*kappa*t)) + 0.5*theta*(1 - np.exp(-kappa*t))**2 )"}


def _feller_spec():
    return {"state_dimension": 2, "analytic_moments": _feller_moments(),
            "time_interval": {"T": 1.0},
            "parameters": {"S_0": 100.0, "V_0": 0.114, "kappa": 2.58, "theta": 0.043,
                           "sigma_v": 1.0, "rho": -0.36, "r": 0.0, "K": 100.0}}


def test_a_null_variance_slot_does_not_hand_its_neighbour_reference_to_component_0():
    """The 2026-11-01 failure, which is *not* the 2026-10-31 one.

    The October fix taught the Path-B surrogate to read ``applies_to_component``.
    This spec declares no such field and never reaches that code: its reference is
    Path A's ``analytic_moments``. ``variance_X`` is ``null`` and ``variance_Y`` is
    exact, and the reader compacted ``[None, variance_Y]`` to ``[variance_Y]`` --
    losing the slot -- while ``_reference`` computed one ``components`` list for
    both kinds as ``range(max(len(v)))``. So the one-entry variance inherited the
    two-entry mean's ``[0, 1]`` and V's CIR variance 0.010213 was read against S's
    empirical 606.84.

    Measured: ``variance_rel_err: 5.942e+04`` on all three plans, ``resolved``
    false, every plan capped at 6 -- on a problem the harness then graded a pass at
    0.408 of tolerance (verdict ``UNDERCLAIM``). All three evaluators diagnosed it
    correctly, including the arithmetic ``2 * 2.0906 / 0.0102127 = 409.4`` behind
    ``ci_width_rel``, and were overruled by their own grader. Paired by component
    the same numbers give 0.0012.
    """
    from verifylib.kernel.sde.driver import _paired, _reference

    reference, detail, _ = _reference(_feller_spec(), "A", 0.10)
    # The mean covers both components; the variance covers only V. The two kinds
    # carry their own pairing precisely because they disagree here.
    assert detail["components"] == {"mean": [0, 1], "variance": [1]}
    assert reference["variance"] == [pytest.approx(0.010212731, rel=1e-6)]

    stats = [{"mean": 100.05, "variance": 606.84}, {"mean": 0.0484, "variance": 0.010225}]
    pairs = _paired(stats, reference, "variance", detail["components"])
    assert [c for c, _, _ in pairs] == [1], "V's variance belongs to V"
    _, st, ref = pairs[0]
    assert abs(st["variance"] - ref) / ref < 0.01            # 0.0012, not 5.942e+04


def test_the_widest_spelling_wins_so_a_bare_string_cannot_shadow_a_component_pair():
    """Found while fixing the above, on a different spec.

    ``sde_multichannel_stiff_m13`` declares a bare ``mean_expression`` **and**
    ``mean_X`` / ``mean_Y``, the bare string being a duplicate of ``mean_X``.
    ``{kind}_expression`` was checked first, so the pair was shadowed: component 1's
    reference was discarded and only half of that 2-D spec was ever graded. Nothing
    reported a problem, because grading one component of two is indistinguishable
    from a correct scalar check unless you count.
    """
    from verifylib.kernel.sde.driver import _claimed_pairs

    moments = {"mean_expression": "a*t", "mean_X": "a*t", "mean_Y": "b*t"}
    assert _claimed_pairs(moments, "mean") == [(0, "a*t"), (1, "b*t")]
    # An expression still beats a literal at equal coverage -- it is checkable at
    # every `t` -- but a literal covering two components beats a string covering one.
    assert _claimed_pairs({"mean_expression": "a*t", "mean_value_at_T": 9.9},
                          "mean") == [(0, "a*t")]
    assert _claimed_pairs({"mean_expression": "a*t", "mean_value_at_T": [1.0, 2.0]},
                          "mean") == [(0, 1.0), (1, 2.0)]


def test_an_unowned_scalar_reference_on_a_multi_component_spec_abstains():
    """Fix 3, the guard rather than a repair.

    Two runs produced two different spec shapes and the kernel guessed the identity
    pairing both times. Where a spelling carries no slot information -- a bare
    string or a scalar literal -- and the state has several components, there is no
    fact of the matter about which component is meant, and a guess does not read as
    a missing check: it reads as a confident four-figure relative error. Abstaining
    turns this class of defect into a visible ``unavailable`` with a reason naming
    the fix. The abstention is per kind, so the other moment and every structural
    check still stand.
    """
    from verifylib.kernel.sde.driver import _reference

    spec = {"state_dimension": 2, "time_interval": {"T": 1.0}, "parameters": {},
            "analytic_moments": {"mean_X": "1.0 + 0*t", "mean_Y": "2.0 + 0*t",
                                 "variance_expression": "0.5 + 0*t"}}
    reference, detail, _ = _reference(spec, "A", 0.10)
    assert reference == {"mean": [1.0, 2.0]}, "the mean is slotted and still graded"
    assert "variance" not in detail["components"]
    assert "which component it describes" in detail["per_moment"]["variance"]

    # A one-component spec has nothing to be ambiguous about.
    scalar = {"state_dimension": 1, "time_interval": {"T": 1.0}, "parameters": {},
              "analytic_moments": {"mean_expression": "1.0 + 0*t"}}
    _, scalar_detail, _ = _reference(scalar, "A", 0.10)
    assert scalar_detail["components"] == {"mean": [0]}


def test_a_martingale_reference_naming_a_parameter_is_resolved_not_floated():
    """``reference_value_at_T: "S_0"`` -- a parameter *name*, which is the natural
    way to write ``E[S_T] = S_0``.

    It was handed straight to ``float()``. Measured on ``sde_heston_call_feller``
    across all three runs (2026-10-30, 10-31, 11-01): ``ValueError: could not
    convert string to float: 'S_0'``, the gate reported ``not_reported`` on every
    plan every time, and ``problem.md`` headlines that very check -- "the asset is a
    martingale: E[S(T)] = S0 exactly". The one invariant the problem statement leads
    with had never run.
    """
    import numpy as np

    from verifylib.kernel import invariants as I

    entry = {"name": "martingale", "expr": "X[:, 0]", "reference_value_at_T": "S_0",
             "gate": True, "ci_mult": 2.0}
    rng = np.random.default_rng(42)
    X = np.column_stack([100.0 + rng.normal(0, 24.6, 20000),
                         np.abs(0.0484 + rng.normal(0, 0.06, 20000))])
    ctx = I.SdeContext(X, params={"S_0": 100.0}, num_paths=len(X), T=1.0)

    stats, reason = I._martingale_stats(entry, ctx)
    assert reason is None, reason
    assert stats is not None, "the gate ran at all"
    drift, _ = stats[0]
    assert drift < 0.5, drift                      # ~0.01 against a bar of ~0.22
    assert I._martingale(entry, ctx)[0] is True

    # An unresolvable name is still refused -- with a reason, not a stack trace.
    bad = {**entry, "reference_value_at_T": "not_a_parameter"}
    stats, reason = I._martingale_stats(bad, ctx)
    assert stats is None and "neither a number nor evaluable" in reason
