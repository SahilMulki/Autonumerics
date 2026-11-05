"""Ranking: one quality index, and the lexicographic key the conductor sorts on.

`commands/conductor.md` Phase 3 used to rank on
``score -> provenance -> estimated_rel_error -> (observed_order - order_floor) ->
wall_time_s -> iterations``. Three of those keys are noise, and one has the wrong
sign:

* **`estimated_rel_error` compared as a point estimate.** Measured on
  ``pde_heat_2d``: three second-order plans certified at 1.010e-04, 1.019e-04 and
  1.023e-04 -- a 1.3% spread -- and the winner was decided on a 0.9% margin, while
  the one plan in the pool with a different asymptotic rate never ran. A difference
  smaller than the estimate's own error bar is not a difference. ``run.py``'s
  ``plan_harness_error`` already applies exactly this discipline on the oracle side
  (its ``margin``), and ``cli.py compare`` applies it between plans; the ranking key
  was the one place it was missing.
* **`observed_order - order_floor`, descending.** ``order_floor`` is a pass bar --
  1.0, and 0.0 on ``pde_kuramoto_sivashinsky`` -- not a target, so the margin is not
  a quality margin. On an algebraic scheme ``richardson.guards``' ``p_sane`` already
  treats ``p >= theoretical_order + 1`` as evidence the ladder is *not* asymptotic,
  so a large margin is a red flag. And on the SDE side ``order_floor`` carries the
  expected **strong** order, whose gate is the two-sided ``|g - g_exp| < 0.4``: there
  the key sorts by distance from the correct answer, in the wrong direction.
* **`wall_time_s` as cost.** It times the whole evaluation -- the ladder, MMS, D1,
  the temporal probe -- which ``verification_manual.md`` §24 prices between 1.0x and
  35x depending on evidence path and which Stage-2 checks ran. Two plans on one
  problem differ in it by an order of magnitude for reasons that have nothing to do
  with their solvers. ``solve_time_s`` is the number a work-precision comparison
  needs.

So the three collapse into one weighted index, and the key becomes
``score -> provenance -> sqi -> solve_time_s -> iterations``. What the index is for
is the sentence a numerical analyst would say and no previous key could express:
**when two methods are equally accurate within their own error bars, take the
cheaper one.**
"""

from __future__ import annotations

import math

from .kernel.invariants import MAX_HEADROOM
from .kernel.metrics import is_skip
from .kernel.score import provenance_rank

#: Decades below the tolerance at which a **PDE**'s accuracy saturates. 1e-2 -> 1e-8
#: is where solvers on these grids actually bottom out; past it the ladder is measuring
#: round-off rather than the scheme. An SDE's ceiling is its sampling noise instead and
#: is computed per run -- see :func:`achievable_decades`.
ACCURACY_DECADES = 6.0

#: Decades of structural margin at which component C saturates, below
#: :data:`~verifylib.kernel.invariants.MAX_HEADROOM` so the cap never binds first.
STRUCTURE_DECADES = 3.0

#: The weights. They are asserted, not fitted -- `benchmark/rank_study.py` exists to
#: fit them against P90 regret once a sweep populates per-plan oracle errors across
#: the suite. The ordering is the argument:
#:
#: * **accuracy** is what the problem asks for;
#: * **order fidelity** is what makes the accuracy predictable rather than lucky,
#:   and it is the primary object of code verification (Roache; Oberkampf & Roy);
#: * **structure** is what practitioners will not trade -- a symplectic scheme with
#:   larger local error beats a higher-order one over long horizons because the
#:   energy error stays bounded instead of drifting;
#: * **consistency** is real but already priced into the *score*, where a clean D1 is
#:   worth a full point of certification, so weighting it heavily double-counts.
#:
#: **Cost is not among them.** It was, at 0.10, "subordinate to correctness and
#: decisive between equals", and it is now measured and reported but never ranked
#: on -- the pool's `cost` column in `cli.py rank` is a diagnostic and nothing
#: more. The weights below are unchanged in value; dropping the fifth means
#: :func:`sqi` renormalises over 0.90 instead of 1.00, which is exactly what
#: ``sqi_cert`` already did, so a certification's index is unchanged by this.
WEIGHTS = {"accuracy": 0.35, "order": 0.25, "structure": 0.20, "consistency": 0.10}

#: The index's components -- all of them. Named separately from :data:`WEIGHTS`
#: because :func:`components` still *computes* ``cost`` for the report, and this
#: tuple is what :func:`sqi` is allowed to read.
#:
#: Cost is excluded for two independent reasons, and the second is the one that
#: settles it. First, cost is a property of the *pool*, not of a plan, and a
#: certification must not depend on how fast a sibling ran. Second, and
#: decisively: **cost is not a measure of whether a plan is right, so it has no
#: business choosing between plans at all.** A slower scheme that resolves the
#: problem is the better plan, full stop; speed is something the reader weighs
#: afterwards against their own budget, and the ranking's job is to say which
#: answer to trust.
#:
#: Measured on `sde_heston_call_feller`, 2026-11-01, after the component-pairing
#: fix. Accuracy came back 0.000 for all three plans -- at the kernel's diagnostic
#: 50k paths the Monte-Carlo noise (`mc_se_rel` 0.022) is an order of magnitude
#: wider than the tightest price gate (0.002), so the graded quantity is not
#: measurable there and the one component that matters is structurally flat. The
#: index then fell through to order fidelity and `cost`, and `cost` ran 0.980 /
#: 0.000 / 1.000 across the pool: a full-range discriminator, decided entirely by
#: a 0.5s-vs-2.6s solve time, standing in for an accuracy nobody could measure.
#: The winner happened to be right that day. It was not right *for* that reason.
CERT_COMPONENTS = ("accuracy", "order", "structure", "consistency")


def _clip01(x):
    if x is None:
        return None
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else float(x))


def _num(value):
    """``value`` as a finite float, or None. Skip markers are strings and bools are
    not measurements, so both fall out here rather than at every call site."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if value == value and math.isfinite(value) else None


# --- the components ----------------------------------------------------------
#
# Each returns a value in [0, 1], or None for "not measurable here". None is not
# zero and must never be: manual §24/§25 and `evaluator-pde.md`'s Key Rules all say
# a check skipped for cost never counts against a plan, and an invariant the solver
# did not report already caps the score at 9 -- charging it again in the ranking
# punishes one defect twice. :func:`sqi` renormalises over whatever was measurable.


def accuracy(summary, tol):
    """Headroom below the tolerance as a share of what is achievable here.

    The error is first **snapped up to its own uncertainty** -- a plan cannot claim
    accuracy finer than its own error bar, and on an SDE that snap is what turns two
    moment errors inside the same Monte-Carlo noise into a genuine tie instead of one
    more tiebreaker to read the seed off.

    Normalising by :func:`achievable_decades` rather than by a fixed number of decades
    is the same correction :data:`verifylib.kernel.score.PLACEMENT_FRACTION` records,
    and it has to be made in both places. Against a flat 6 decades an SDE's accuracy
    could not exceed ~0.2, because Monte Carlo at the declared path count cannot get
    further inside the tolerance than its own sampling noise: measured on
    `sde_double_well_langevin`, three plans whose quality indices were 0.177, 0.178 and
    0.177 -- accuracy contributing almost nothing and the ranking falling to the other
    components -- are 0.69 to 0.73 of the way to their ceiling.
    """
    # F3. Where the problem declares a gated quantity of interest and the kernel
    # measured it, *that* is what the problem is scored on, and the moment error is
    # a proxy for it at best. `functional_rel_error` is the sum of a dt-bias term
    # and the solver's reported sampling term, so unlike the moment error it is
    # never below its own noise -- the noise is inside it rather than under it.
    fn = _functional_fraction(summary)
    if fn is not None:
        return _clip01(fn)
    # On a chaotic problem the graded horizon is the one the problem asks about, and
    # where `graded_N` is declared the graded grid is what it is scored on. Both are
    # worse-case in `headroom_decades`, never better: an error quoted at a reference
    # horizon must not stand in for the error at T. An error of exactly zero comes
    # back None there -- it is the signature of a solver whose output did not respond
    # to the grid at all (`baseline_select.looks_n_independent`), so it is unmeasured
    # rather than perfect.
    #
    # F16. The snap below turns an error inside its own error bar into a tie, which
    # is right, and `achievable_decades` then divides by *the same half-width*, which
    # places that tie at exactly 1.0 -- the component's ceiling. Measured across the
    # eleven BENCHOP B01-B04 plans, every one of them snaps and every one scores
    # 1.000, so 0.35 of the index orders nothing and is renormalised onto structure
    # and cost. A tie at the ceiling is not a tie, it is a claim; what the kernel
    # actually knows here is that the error was **not resolved** at this budget, and
    # the vocabulary for that is None.
    if noise_limited(summary):
        return None
    return _clip01(headroom_fraction(summary, tol))


#: Decades below the declared gate that a Monte-Carlo functional could plausibly
#: reach. Two, because a hundredfold improvement on a sampling-limited estimate
#: costs ten thousand times the paths.
#:
#: Fixed, and deliberately **not** the plan's own ``achievable_decades``. A ceiling
#: computed from each plan's own error bar divides out the quantity that separates
#: plans: a noisy plan gets a low ceiling and scores the same fraction as a tight one
#: with the same bias, and a plan that bought its accuracy with a control variate --
#: the axis C0 beat the pipeline on -- is rewarded with a *higher* bar. The ceiling
#: has to be a property of the problem for the fraction to order anything.
FUNCTIONAL_DECADES = 2.0


def _functional_fraction(summary):
    """Accuracy headroom on the declared quantity of interest, or None.

    ``functional_rel_error`` and ``functional_rel_tol`` are written by the kernel
    where the spec declares a gated functional and the solver returned a value for
    it; both are relative and in the same units, so the headroom is the ratio's log.
    """
    err = _num(summary.get("functional_rel_error"))
    tol = _num(summary.get("functional_rel_tol"))
    if err is None or tol is None or tol <= 0.0 or err <= 0.0:
        return None
    return math.log10(tol / err) / FUNCTIONAL_DECADES


def noise_limited(summary):
    """Is the reported error smaller than the plan's own error bar?

    True where the kernel's accuracy statistic sits inside the sampling noise that
    produced it, so the plan's real error is somewhere in ``[0, half_width]`` and the
    estimate does not locate it. False where the error is resolved, and False where
    there is no error bar to compare against (every PDE Path A plan).
    """
    err = _num(summary.get("estimated_rel_error"))
    half_width = uncertainty(summary)
    return err is not None and half_width is not None and err < half_width


def headroom_decades(summary, tol):
    """Decades of accuracy headroom below the tolerance, unclipped, or None.

    :func:`accuracy` is this divided by :data:`ACCURACY_DECADES` and clipped, for use
    as one component of a weighted index. This is the raw quantity, which is what the
    certification's band placement reads -- see
    :data:`verifylib.kernel.score.PLACEMENT_DECADES` for why placement must not read
    the composite.
    """
    err = _num(summary.get("estimated_rel_error"))
    if err is None or not _num(tol):
        return None
    # Worst case over the horizon and grid the problem is graded on: an error quoted
    # at a reference horizon or the top ladder level must not stand in for the error
    # where the problem asks.
    keys = ["estimated_rel_error_graded"]
    # ...except on a chaotic problem. `error_horizon` is set only there, and manual
    # §28 waives pointwise accuracy at full T precisely because trajectories separate
    # past the Lyapunov time: the graded statistic is the one at `chaotic_T_ref`.
    # `converged` still *gates* on `estimated_rel_error_T`, so a solver wrong at T
    # does not certify -- but grading quality on it re-imposes the penalty the waiver
    # exists to remove, and measured on the frozen Kuramoto-Sivashinsky plans it moved
    # correct runs down by one and two points.
    if summary.get("error_horizon") is None:
        keys.append("estimated_rel_error_T")
    for key in keys:
        other = _num(summary.get(key))
        if other is not None:
            err = max(err, other)
    err = max(err, uncertainty(summary) or 0.0)
    # Divide Roache's factor of safety back out. It decides whether the run *passes*
    # -- `converged` tests `gci < tol`, which is what §4 says the GCI is for -- not how
    # good the solver is. Leaving it in made §7a's two-level saving cost a point of
    # score, because an imported probe order carries fs = 3.0 where a measured one
    # carries 1.25.
    fs = _num(summary.get("error_fs"))
    if fs and fs > 0.0:
        err /= fs
    if err <= 0.0:
        return None
    return math.log10(float(tol) / err)


def achievable_decades(summary, tol, *, ceiling=ACCURACY_DECADES):
    """The most headroom any solver could reach on this problem, in decades.

    For a **PDE** that is set by the round-off floor: ``tol`` down to ~1e-8 is where
    solvers on these grids bottom out, so it is :data:`ACCURACY_DECADES`.

    For an **SDE** it is set by the sampling noise, not the scheme. A Monte Carlo
    estimate's accuracy is bounded by ``1/sqrt(M)``, so the floor is the confidence
    half-width and the ceiling is ``log10(tol / (ci_mult * se))``. Measured on the
    canary suite's honest SDE solver: at 50 000 paths against a 10% variance
    tolerance the ceiling is **0.90 decades**, and the honest solver sits at 0.896 --
    it is already as accurate as the path count permits. Judging it against a
    PDE-style two-decade bar asks for ~200 000 paths, which is a budget the problem
    set, not a property of the solver.
    """
    half_width = uncertainty(summary)
    if half_width is None or not _num(tol) or half_width <= 0.0:
        return float(ceiling)
    return max(math.log10(float(tol) / half_width), 0.0)


def headroom_fraction(summary, tol):
    """Headroom as a share of what is achievable, or None.

    One quantity for both kinds. It is what the certification's band placement reads:
    *did this run get a given fraction of the way from the tolerance to the best its
    method and budget allow.*
    """
    decades = headroom_decades(summary, tol)
    if decades is None:
        return None
    ceiling = achievable_decades(summary, tol)
    return decades / ceiling if ceiling > 0.0 else 1.0


def uncertainty(summary):
    """The error bar on ``estimated_rel_error``, or None.

    **SDE** is where this does work: ``mc_se_rel`` is the relative standard error of
    the checked moment, §14 makes every comparison at ``ci_mult`` of it, and two
    plans whose errors differ by less than the combined half-width differ by
    sampling noise. Ranking them apart is reading the seed.

    **PDE Path B needs no snap** -- ``estimated_rel_error`` *is* the GCI there
    (`driver.py`, ``e_fine = richardson["gci"]``), Roache's factor of safety already
    applied, so the reported number is the error bar. **Path A has no bar the kernel
    can see**: the error is measured against the spec's closed form, and what bounds
    it is the reference's own accuracy, which only the harness knows
    (``problem["reference_error"]``). Returning None there is correct rather than
    conservative -- and the log scale in :func:`accuracy` already refuses to read
    the ``pde_heat_2d`` spread as a difference without it, which is the case that
    motivated the snap in the first place.
    """
    se = _num(summary.get("mc_se_rel"))
    if se is None:
        return None
    return se * (_num(summary.get("ci_mult")) or 2.0)


def order(summary):
    """Order fidelity against the scheme's **design** rate, one-sided.

    **Only a shortfall counts.** This was two-sided, on the argument that overshoot is
    not a bonus: a 2nd-order scheme measuring 4.0 has a ladder outside its asymptotic
    range or a problem whose symmetry flattered it. That argument is sound and belongs
    somewhere else. ``richardson.guards``' ``p_sane`` already refuses to call such a
    ladder asymptotic and the *score* already pays for it, so charging it again here
    punishes one reading twice -- and does it in the rank key, where no acceptance
    criterion can see the result, because it moves ranks rather than scores.

    What the two-sided distance actually measured was distance from a **pass bar**.
    ``order_floor`` is a floor -- 1.0, 0.0 on ``pde_kuramoto_sivashinsky``, and a
    kernel-defaulted 0.5 on most SDEs -- and a scheme is not better for landing on it
    exactly. Measured on ``sde_hhw_call_corr``: three Euler-Maruyama schemes read
    ``observed_order`` 1.0078, 0.9924 and 1.0054 against a declared floor of 0.5, and
    the two-sided branch ranked them 0.0000, 0.0152 and 0.0000 -- the plan furthest
    below 1.0 winning, on a pool whose true errors were 0.0474, 0.7445 and 0.5017. The
    component was anti-correlated with quality, which it is on any problem whose floor
    sits below the scheme's real rate: the common case for additive and near-additive
    noise. The module's own §19 rule already said so -- *an order above the theoretical
    one is noise rather than a bonus* -- and noise is not a penalty either.

    This also retires the ``order_band`` branch, and with it the last reader of a field
    the ``<metrics>`` block did not carry (11-sde-accuracy-fixes.md §1, §3). The band
    is still reported, because it is what ``crn.guards`` applied in deciding whether
    the order *passed*; it just no longer decides a rank. One rule, and now one side.

    A spectral scheme stays the one case that is not measurable at all -- not generous,
    None: it converges faster than any algebraic rate, so a ladder order fitted off it
    has no design target to be a fraction of.

    Returns None where the check was waived (`order_check: waived_spec` /
    `waived_chaotic`), where an SDE order study **abstained**
    (`order_check: unavailable`), where the order is non-finite (a run at the
    round-off floor, which `richardson` reports as `inf` and which the score already
    credits), where the scheme is spectral (see below), or where no design order was
    declared. That last case is most plans today: measured
    across `workspace/`, 22 of 104 declare `spatial_order` and 25 declare
    `scheme_family` -- both landed with the no-closed-form work, so the fields are
    present on the newer pools only. Inferring a design order from a plan's *slug*
    would let a plan rename itself into a better rank, which is the one thing the
    declared-never-inferred rule across this repo exists to stop.
    """
    # `order_check` is a three-valued string, not a bool: `driver._order_verdict`
    # returns "true", "waived_spec" or "waived_chaotic". Testing it for
    # stringhood reads a *passing* check as a waived one.
    state = str(summary.get("order_check") or "")
    if state.startswith("waived") or is_skip(state):
        # A skip token is the SDE abstention and its neighbours: `"unresolved"` is
        # `crn.abstention` (the estimator's own error bar admits more than the band,
        # or both pathwise differences are at the roundoff scale), `"unavailable"` is
        # a ladder that crashed, `"not_run"` is §24's cost skip. Ranking a plan on an
        # estimate the kernel just declined to gate on would reintroduce the defect
        # one layer up, where no score-based acceptance criterion can see it.
        #
        # [REV 59]: matched against the vocabulary rather than against the one
        # literal `"unavailable"`. The abstention's token changed in that revision,
        # and an exact-string test here would have gone on ranking plans on the very
        # estimate `guards` had declined to read -- silently, because this moves
        # ranks and not scores.
        return None
    observed = _num(summary.get("observed_order"))
    design = _num(summary.get("design_order"))
    if observed is None or design is None or design <= 0.0:
        return None
    if summary.get("kind") == "sde" or summary.get("scheme_family") in _SDE_FAMILIES:
        # `order_floor` carries the expected strong order on an SDE, which is the bar
        # `crn.guards` judged the plan against, so it is the bar a shortfall is
        # measured from. `max(target, 0.5)` keeps the scale from collapsing on a
        # floor of zero.
        target = _num(summary.get("order_floor")) or design
        return _clip01(1.0 - _shortfall(observed, target) / max(target, 0.5))
    if summary.get("scheme_family") == "spectral":
        # Not measurable, and not for want of numbers. A spectral method's error is
        # not C*h^p, so an order fitted off the ladder is a local slope with no
        # design target to compare against -- and a spectral plan's declared
        # `spatial_order` is not a rate at all. It is the hint §26's D1 uses to pick
        # a higher-order comparison stencil, which `pde_kdv_zabusky_kruskal`'s own
        # plan 1 says in as many words: "a declared floor, not a claim of algebraic
        # 8th-order convergence [...] the observed ladder order will be ~ 4, not 8".
        # Scoring 5.45 against 8 marked the two spectral plans down to 0.68 and 0.58
        # against the FD plans' 0.98 -- while their errors were 1.5e-10 and 2.4e-9
        # against 1.7e-05.
        #
        # A spectral plan that is genuinely failing to converge is caught a level up:
        # `richardson.guards` applies its own spectral branch (`p >= theoretical_
        # order`), and a failed `asymptotic` caps the score at 4 (manual §4).
        return None
    return _clip01(1.0 - _shortfall(observed, design) / design)


def _shortfall(observed, target):
    """How far ``observed`` falls short of ``target``, never how far it exceeds it."""
    return max(0.0, target - observed)


_SDE_FAMILIES = frozenset({"euler-maruyama", "milstein", "tamed-euler-maruyama",
                           "tamed-milstein", "implicit", "exact-transition",
                           "quadratic-exponential"})

#: §26's outcomes as a graded component. ``clean`` is the residual falling at or past
#: the design rate, or sitting at the round-off floor; ``slow`` falls but
#: under-performs and §26 says it "reports, never gates"; ``stalled`` does not fall.
#: The skips are None -- ``unresolved`` means the check measured the *kernel*, not the
#: solver, and must not count against a plan.
D1_OUTCOME = {"clean": 1.0, "slow": 0.5, "stalled": 0.0}


def structure(summary, *, decades=STRUCTURE_DECADES):
    """The worst declared invariant's margin below its **own** tolerance.

    Worst-case rather than an average: a plan is as structurally sound as its
    weakest conserved quantity. Normalised per invariant because raw drifts are not
    comparable -- a mass drift of 1e-8 against a 1e-8 tolerance is marginal and a
    symmetry drift of 1e-8 against 1e-4 is not, and `invariant_max_drift` cannot tell
    them apart. That is why the plan's proposed "max invariant drift" key would not
    have worked even once the field existed.
    """
    worst = _num(summary.get("invariant_worst_headroom"))
    if worst is None:
        return None
    return _clip01(worst / min(decades, MAX_HEADROOM))


def consistency(summary):
    """D1's slope margin, or the SDE's Dynkin distance. **Never a magnitude.**

    ``verification_manual.md`` §25 puts "a residual reported as a magnitude" on the
    never-happen list and §26 explains why there can be no calibrated threshold: a
    solver's residual is dominated by its own truncation error, so any constant bar
    either fails every correct coarse-grid solver or passes everything. The rate is
    the measurement. ``slope_test`` passes at ``p_design - 1``; this reports how far
    past that bar the slope got.

    It reads D1's **verdict**, not a margin computed here. ``residual.slope_test``
    already compared the slope against ``p_design - 1`` using
    ``verification.theoretical_order``, which §26 is explicit is *not* the plan's
    declared ``spatial_order``: recomputing the margin from ``design_order`` scored
    `pde_kdv_zabusky_kruskal`'s two spectral plans at 0.000 -- the most accurate plans
    in the pool -- because their declared 8 is a stencil hint and no real residual
    slope clears ``(slope - 7)/8``. The outcome vocabulary is the graded quantity §26
    defines, so it is the one to use.

    On the SDE side the counterpart is how far the extrapolated generator residual
    sits from zero in its own standard errors, against the family-wise bar -- small
    is good, so it inverts.
    """
    z = _num(summary.get("dynkin_z"))
    if z is not None:
        bar = _num(summary.get("dynkin_bar")) or 2.43
        return _clip01(1.0 - abs(z) / bar)
    return D1_OUTCOME.get(summary.get("d1_outcome"))


def cost(summary, pool):
    """Pool-relative solver cost, log-scaled. Absolute seconds are not comparable
    across problems, and within one problem the spread is what matters.

    Reads ``solve_time_s`` -- the solver at the grid the error is quoted on -- never
    ``wall_time_s``. Returns None for a single-plan pool, where there is nothing to
    be relative to.

    **And None for a pool that mixes cost sources** (F4). ``solve_time_s`` is CPU
    time where the sandbox could measure it and wall time where it could not, and
    those are not the same quantity: a forking solver's wall time is divided by its
    worker count. A pool holding one of each would be ranked on which measurement
    it got, so it is not ranked on cost at all. ``cost_source`` is absent on every
    metrics block written before this landed, and a pool of those is uniform.
    """
    mine = _num(summary.get("solve_time_s"))
    sources = {s.get("cost_source") for s in pool if isinstance(s, dict)}
    if len({s for s in sources if s}) > 1 or "mixed" in sources:
        return None
    times = sorted(t for t in (_num(s.get("solve_time_s")) for s in pool)
                   if t is not None and t > 0.0)
    if mine is None or mine <= 0.0 or len(times) < 2 or times[0] == times[-1]:
        return None
    return _clip01(math.log10(times[-1] / mine) / math.log10(times[-1] / times[0]))


# --- the index ---------------------------------------------------------------


def cost_admissible(tol, pool):
    """May cost contribute to the index at all in this pool? (F4)

    Only where **every** plan in the pool had its accuracy measured. Cost is
    subordinate to correctness and decisive between equals, and that second clause
    is the whole justification for the weight: the plans have to be equals, which
    means the comparison that would have separated them has to have been made.

    Where it has not been -- where the errors all sit inside their own Monte-Carlo
    noise, which is all eleven BENCHOP B01-B04 plans -- renormalisation hands
    accuracy's 0.35 to the components that remain, and cost goes from a tenth of the
    index to a third of it. Measured there, in four pools of four, the most accurate
    plan is the slowest and the index ranks it last. A component promoted by the
    absence of the one it is subordinate to is not breaking a tie, it is deciding.

    A single-plan pool is admissible and `cost` returns None for it anyway.
    """
    summaries = [s for s in pool if isinstance(s, dict)]
    if len(summaries) < 2:
        return True
    return all(accuracy(s, tol) is not None for s in summaries)


def components(summary, tol, pool=()):
    return {"accuracy": accuracy(summary, tol), "order": order(summary),
            "structure": structure(summary), "consistency": consistency(summary),
            "cost": cost(summary, pool) if cost_admissible(tol, pool) else None}


def sqi(summary, tol, pool=(), *, keys=None):
    """``(index, components)``. ``index`` is None when nothing was measurable.

    Renormalises over the measurable components rather than scoring an absent one as
    zero. Substituting zero charges a skipped check as a failed one, which every rule
    in this repo forbids; substituting a pool median makes one plan's index depend on
    its siblings', which nothing a certification reads may do. Renormalisation does
    neither.

    Stated precisely, because the weaker version is the true one: the index is a
    weighted **mean**, so dropping a component moves it toward the mean of the rest
    -- up if the missing one was below average, down if above. It is not invariant
    under dropping, and it cannot be: a mean over fewer terms is a different mean.
    What it guarantees is that nothing is scored against a bar it was never measured
    against.

    That leaves one gap, and it is closed elsewhere rather than here: a plan whose
    *weakest* component is the one that did not run scores a higher index than one
    that reported it. The architecture already prices that a level up -- a D1 that
    did not run forfeits a circularity break, so ``score.ceiling`` caps the plan at 9
    where the reporting plan reaches 10 -- and :func:`rank_key` reads ``score``
    before it reads quality. Skipping a check can improve the index; it cannot
    improve the rank.
    """
    parts = components(summary, tol, pool)
    # `parts` is returned whole -- `cli.py rank` prints every component, `cost`
    # included, and removing it from the *report* would hide a measurement rather
    # than decline to rank on it. Only the weighted sum is restricted.
    ranked = CERT_COMPONENTS if keys is None else keys
    measured = {k: v for k, v in parts.items() if v is not None and k in ranked}
    if not measured:
        return None, parts
    total = sum(WEIGHTS[k] for k in measured)
    return sum(WEIGHTS[k] * v for k, v in measured.items()) / total, parts


def sqi_cert(summary, tol):
    """The index the certification reads.

    Identical to :func:`sqi` since 2026-11-01, when cost left the index entirely
    (see :data:`CERT_COMPONENTS`). Kept as its own name because the callers mean
    different things by it -- `kernel/driver.py` wants "the index a certification
    may depend on", and that it now coincides with the ranking index is a fact
    about the weights, not something those callers should have to know.
    """
    return sqi(summary, tol, keys=CERT_COMPONENTS)[0]


# --- the key -----------------------------------------------------------------


def rank_key(entry, pool=()):
    """`commands/conductor.md` Phase 3 step 2, as a sort key. Lower sorts first.

    ``score`` and ``provenance`` stay lexicographic and stay first. They are claims
    about *what was established*, not measures of quality, and a weighted sum must
    never trade a validated-analytic 9 against a self-convergence 10 -- the
    distinction the provenance vocabulary exists to carry would be the first thing
    it averaged away. Everything below them is quality, and that is one number.

    An entry the kernel could not score sorts last.
    """
    m = entry.get("kernel") or entry.get("metrics") or {}
    if not m:
        return (1, 0, 0, 0.0, float("inf"), 0)
    summary = m.get("summary") or m
    tol = _tol_of(summary, m)
    index, _ = sqi(summary, tol, pool)
    # `solve_time_s` used to sit here, below the index, breaking exact ties
    # outright -- which is how three of the ten BENCHOP winners were chosen. It is
    # gone for the reason :data:`CERT_COMPONENTS` gives: cost does not measure
    # whether a plan is right, so it does not get to choose between plans, and a
    # tiebreak is the most decisive form of choosing there is. Where score,
    # provenance and the index all tie, `iter` decides; where that ties too the
    # plans are genuinely indistinguishable on everything this kernel measures,
    # and `decided_by` says so rather than reaching for the stopwatch (F7).
    return (
        0,
        -(_num(m.get("score")) or 0.0),            # score descending
        provenance_rank(m.get("provenance")),      # provenance order
        -(index if index is not None else -1.0),   # quality descending; unscored last
        _num(entry.get("iter")) or 0.0,            # fewest iterations
    )


#: The lexicographic keys :func:`rank_key` returns, in order, after the leading
#: "was this scorable at all" flag. :func:`decided_by` names the first one that
#: separated two entries.
RANK_KEYS = ("score", "provenance", "sqi", "iter")


def decided_by(winner, runner_up, pool=()):
    """Which ranking key actually chose the winner over the runner-up. (F7)

    An exhausted key is not a ranking. Where score, provenance, the index and
    ``iter`` all tie, the ranker did not rank -- the pool is indistinguishable on
    everything it can measure, and saying so is the whole point of this function.
    It used to reach ``solve_time_s`` at that position and ship the fastest plan,
    which was a defensible default and an indefensible silence; its absence from
    the record is what made the BENCHOP analysis expensive. Since 2026-11-01 the
    stopwatch is not consulted at all and the answer there is ``"tie"``.

    Returns the key's name, or ``"tie"`` where every key is equal.
    """
    a, b = rank_key(winner, pool), rank_key(runner_up, pool)
    for name, left, right in zip(RANK_KEYS, a[1:], b[1:], strict=True):
        if left != right:
            return name
    return "tie"


def _tol_of(summary, metrics):
    """The tolerance the accuracy component is measured against: whatever the kernel
    was configured with, else the manual's default."""
    config = (metrics.get("config") or {})
    for key in ("rel_l2_err_max", "variance_rel_err_max"):
        value = config.get(key)
        value = value.get("value") if isinstance(value, dict) else value
        if _num(value):
            return float(value)
    return 0.1 if summary.get("mc_se_rel") is not None else 0.01
