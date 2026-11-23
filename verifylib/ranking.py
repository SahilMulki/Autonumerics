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
from .kernel.sde.crn import STRONG_BAND

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
    # The same rule, one route earlier. Where the problem is graded pathwise and the
    # ladder produced a strong error, *that* is the quantity the problem is scored on
    # and the moment error is not even a proxy for it -- on `sde_clark_cameron` the
    # two order-1 plans report moment errors identical to four digits (0.0121 each)
    # because they produce the same moments, and differ 1.9x in the pathwise error
    # the harness grades. Ranking them on the moment is ranking them on a quantity
    # the problem does not ask about.
    sf = _strong_fraction(summary)
    if sf is not None:
        return _clip01(sf)
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
        # A seed study changes what this condition means, so it changes the answer.
        # On one run, "the error is inside its own bar" is ignorance: the estimate
        # conflates bias with noise and nobody knows which. Over ``k`` seeds the
        # sampling term has been divided by ``sqrt(k)`` and the bias has not, so the
        # same inequality is a **measurement** -- this plan's bias is below what the
        # study can resolve, i.e. it is at the floor its path count allows, which is
        # the definition of this component's ceiling.
        #
        # This is the one place the study moves an outcome rather than sharpening a
        # number, and it is deliberate. F16's note that "a tie at the ceiling is not
        # a tie, it is a claim" is right about a single run, where the claim has no
        # evidence; 16 seeds are the evidence. Measured on `sde_sis_epidemic`'s
        # three-plan pool (§8i): the two plans with resolved biases score 0.6875 and
        # 0.5383 and the plan at the floor scores 1.0, which is the ordering the
        # harness independently measures and the one the pool failed to make.
        if seed_resolved(summary):
            return 1.0
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


def _strong_fraction(summary):
    """Accuracy headroom on the pathwise strong error, or None.

    ``strong_rel_err`` is written by the SDE kernel wherever the CRN ladder could
    estimate one (``crn.strong_error``), and ``strong_rel_err_tol`` wherever the spec
    declares the gate. Both are relative and in the same units.

    Shaped like :func:`headroom_fraction` rather than like
    :func:`_functional_fraction`, and the difference matters. A Monte-Carlo
    functional gets a *fixed* two-decade ceiling because a tighter plan would buy its
    headroom with paths. A strong error is not sampling-limited in the same way:
    common random numbers make ``X(dt) - X(dt/2)`` a pathwise difference whose mean
    carries very little variance -- 1.7% of itself on all three N01 plans -- so the
    statistic really is resolved, and the ceiling its own bar implies is the honest
    one. Normalising against a fixed two decades instead would compress a 1.4x
    separation to 0.097 against 0.023 and leave the axis barely able to outvote a
    0.05 difference in fitted order.

    **The snap is kept.** A plan may not claim a strong error finer than the bar on
    its own ladder difference, exactly as :func:`accuracy` does not let one claim a
    moment error finer than its Monte-Carlo bar.
    """
    err = _num(summary.get("strong_rel_err"))
    tol = _num(summary.get("strong_rel_err_tol"))
    if err is None or tol is None or tol <= 0.0 or err <= 0.0:
        return None
    ci = _num(summary.get("ci_mult")) or 2.0
    se = _num(summary.get("strong_rel_err_se"))
    half_width = se * ci if se is not None and se > 0.0 else None
    if half_width is not None:
        # F16, for this axis. An estimate inside its own bar does not locate the
        # error, and the vocabulary for that is None -- not the ceiling. Returning
        # 1.0 here would hand a plan the top of the component for being imprecise,
        # which is the exact inversion the moment path's `noise_limited` exists to
        # refuse. In practice CRN makes this rare: on all three N01 plans the bar is
        # 3.4% of the estimate.
        if err < half_width:
            return None
        # ...and if the bar is wider than the gate itself, no plan can be placed
        # against that gate at this budget, so the axis has nothing to say about any
        # of them rather than something flattering about all of them.
        ceiling = math.log10(tol / half_width)
        if ceiling <= 0.0:
            return None
    else:
        ceiling = FUNCTIONAL_DECADES
    return math.log10(tol / err) / ceiling


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

    **The comparison is made on one statistic, not two.** Where
    :func:`seed_resolved` has a median, that median is the estimate and its
    between-seed bar is the bar; where it does not, the single graded run is both.
    Mixing them -- a single-seed error against a ``sqrt(k)``-tighter bar -- would
    declare plans resolved on the strength of their noise, which is the opposite of
    what this predicate is for.
    """
    seeded = seed_resolved(summary)
    err = seeded[0] if seeded else _num(summary.get("estimated_rel_error"))
    half_width = uncertainty(summary)
    return err is not None and half_width is not None and err < half_width


def seed_resolved(summary):
    """``(median_rel_error, se_of_that_median)`` from the kernel's accuracy seed
    study, or ``None`` where it was not run.

    **Why a seed median resolves what one seed cannot.** A single graded run reports
    ``bias + noise`` and an error bar that measures the noise alone, so on an SDE the
    two are routinely the same size and :func:`noise_limited` is True -- the honest
    verdict that the estimate does not locate the error. Averaging over ``k``
    independent seeds leaves the bias untouched and shrinks the sampling term as
    ``1/sqrt(k)``, so the median *is* an estimate of the quantity the ranker wants to
    order plans on, with an error bar that can be smaller than it.

    Measured on `sde_sis_epidemic` (`docs/plan-notable-sde.md` §8h): a graded variance
    error of 0.73% against a within-run standard error of 0.50% is not resolved, while
    the median over 16 seeds is 0.45% against a between-seed standard error of about
    0.12%, which is. The same pool is the one that shipped under
    ``decided_by: sqi (accuracy did not resolve pool-wide)``, with accuracy's 0.35 of
    the index renormalised onto the other components -- i.e. the most important
    component ordered nothing, on a question the compute could have answered.
    """
    err = _num(summary.get("accuracy_seed_median_rel_error"))
    bar = _num(summary.get("accuracy_seed_median_se_rel"))
    k = _num(summary.get("accuracy_seeds"))
    if err is None or bar is None or k is None or k < 2:
        return None
    return err, bar


def headroom_decades(summary, tol):
    """Decades of accuracy headroom below the tolerance, unclipped, or None.

    :func:`accuracy` is this divided by :data:`ACCURACY_DECADES` and clipped, for use
    as one component of a weighted index. This is the raw quantity, which is what the
    certification's band placement reads -- see
    :data:`verifylib.kernel.score.PLACEMENT_DECADES` for why placement must not read
    the composite.
    """
    seeded = seed_resolved(summary)
    # The seed median where there is one: it estimates the plan's bias, which is the
    # quantity that distinguishes two schemes, where the single graded run reports
    # bias and noise together. The snap below still applies, against the median's own
    # (smaller) bar rather than the single run's.
    err = seeded[0] if seeded else _num(summary.get("estimated_rel_error"))
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
    ci = _num(summary.get("ci_mult")) or 2.0
    seeded = seed_resolved(summary)
    if seeded:
        # The between-seed standard error of the median, at the same ``ci_mult`` the
        # rest of §14's comparisons use. It is the bar on the statistic actually being
        # ranked, so substituting it here is what lets `accuracy` stop returning None
        # on a pool whose single-seed errors all sat inside their own noise.
        return seeded[1] * ci
    se = _num(summary.get("mc_se_rel"))
    if se is None:
        return None
    return se * ci


def order(summary, pool=()):
    """Order fidelity against the scheme's **design** rate, one-sided.

    With a ``pool``, and only on the SDE branch, the rate is read against the best
    rate anyone demonstrated *on this problem* wherever the floor has stopped
    discriminating -- see :func:`_order_against_pool` for the two guards and the
    separation floor that gate that. Without one the reading is the absolute,
    floor-relative one and nothing below changes; :func:`sqi_cert` passes no pool, so
    **a certification never depends on a sibling plan's order study.** That is not a
    convenience of the signature, it is the rule: an index a certification reads may
    not move because another plan was added to the pool.

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
        absolute = _clip01(1.0 - _shortfall(observed, target) / max(target, 0.5))
        return _order_against_pool(summary, pool, absolute)
    if _is_sde(summary) or any(_is_sde(s) for s in pool if isinstance(s, dict)):
        # An SDE plan that declared **no** `scheme_family` and no `kind`, in a pool
        # where some sibling did. It reaches neither branch above, so before 10o it
        # could not reach the pool-relative reading either -- and that is not a corner
        # case: on `sde_sciml_additive` **two of three plans** are in exactly this
        # position, including the pool's best, so shipping the rule without this line
        # left it inert on one of the two pools it was shipped for. The `absolute`
        # value is computed by the generic branch below and is unchanged; only the
        # pool comparison is reached, and only because a *pool* is one problem. This
        # is the §10d vocabulary gap in a fourth place (10o).
        return _order_against_pool(
            summary, pool, _clip01(1.0 - _shortfall(observed, design) / design))
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


def _is_sde(summary):
    """Whether this summary is an SDE plan's, by declaration only.

    The two tests :func:`order` already used, named once so the pool test and the
    plan test cannot drift apart. Declared, never inferred: a plan's slug is not
    evidence, which is the rule the whole module keeps.
    """
    return bool(summary.get("kind") == "sde"
                or summary.get("scheme_family") in _SDE_FAMILIES)


#: Combined standard errors of separation that the pool's two best measured orders
#: must show before :func:`_order_against_pool` will read one against the other, and
#: the unit the resulting shortfall is scaled in. One constant does both jobs: it is
#: the resolution the order study is credited with, so the bar for *acting* on a
#: separation and the scale a separation is *measured* in are the same number by
#: construction.
#:
#: **5.0, and what the corpus can and cannot say about that** (plan-sciml-benchmark.md
#: 10o). The nine sciml pools separate their top two orders by 0.2, 0.3, 0.6, 1.9,
#: 3.6, 13.4, 15.6, 31.5 and 3166 sigma. The arm scores identically -- 8 of 9 pools,
#: regret geomean 1.004, 0.379x against the C0 one-shot -- for **every value in
#: ``[2, 13]``**, and falls to 7 of 9 at 14, where it stops firing on
#: ``sde_sciml_nltest``: the 13.4-sigma, 622x pool this whole line of work exists to
#: fix. So the corpus pins the ceiling and says nothing at all about the floor.
#:
#: 5.0 is therefore a **choice inside a measured window**, and the argument for it is
#: not the score. At 2.0 the rule also fires on ``sde_sciml_stiffquad_mild``, whose
#: top two are 3.6 sigma apart with all three plans within 0.25 of order 1.0 -- the
#: thinnest evidence anywhere in the corpus, and a pool the ranker already gets right
#: *without* this rule. Declining it costs nothing measurable and is the difference
#: between a rule that acts on resolved separations and one that acts on any
#: separation at all. 5.0 is the conventional "clearly resolved" figure, 2.7x below
#: the pool that binds the ceiling and 1.4x above the pool it declines.
#:
#: It is deliberately **not** the corpus's own optimum, which would be the one thing
#: 10f, 10h, 10m and 10o each say not to ship: a constant fitted to nine pools, on a
#: holdout that cannot test it.
MIN_ORDER_SEPARATION = 5.0


def _order_against_pool(summary, pool, absolute):
    """``absolute``, or the pool-relative reading where the floor carries no signal.

    :func:`order` measures a shortfall against ``order_floor``, and on an SDE pool
    where every plan clears its own floor that returns 1.0 for all of them. Measured
    on ``sde_sciml_nltest`` (plan-sciml-benchmark.md 10l): three plans read strong
    orders **0.9650 / 0.9981 / 1.5851**, ranking the pool exactly as the harness does
    in all three positions, and the component mapped them to 1.000 / 0.998 / 1.000 --
    tying the pool's best plan with its worst, while ``structure`` outweighed the one
    component pointing the right way by 7.4x. The pool shipped a plan **622x** worse
    than one it already held.

    So where the floor has stopped discriminating, read the rate against the best rate
    anyone demonstrated on the problem instead. Three things gate that, and each was
    measured before it was written:

    1. **No plan short of its floor by more than the gate's own band.** A real
       shortfall is a finding about that plan and is not overruled by what the pool
       managed. This never fires on any of the 18 recorded pools (largest shortfall
       0.2296 against a band of 0.4) and is kept because the alternative is a rule
       that silently stops being about floors at all.
    2. **The top two separated by more than** :data:`MIN_ORDER_SEPARATION`
       **combined standard errors.** Without this the rule is "higher order is
       better", which §19 is explicit is wrong -- *an order far above the theoretical
       one is noise, not a bonus*. One pool proves it load-bearing:
       ``sde_sciml_linear_ito``'s top two read 0.98647 +/- 0.0096 and 0.98312 +/-
       0.0150, 0.0033 apart, and the **lower** of them is that pool's best plan by
       the harness's own measurement; with the guard disabled the pool flips and
       undoes 10k.
    3. **The shortfall scaled in those same standard errors**, not in the orders'
       own magnitude. 10o found this the hard way: scaled by ``max(hi, 0.5)``, the
       rule repairs ``sde_sciml_nltest`` only because its orders happen to differ in
       the first decimal, and loses ``sde_sciml_additive``, whose top two are
       2.0003835 and 1.9999994 -- about 3200 standard errors and 2.46x of harness
       error apart, mapped to 1.0 and 0.999808, which ``structure`` then outvotes by
       far more than the signal. A reading in units of the orders themselves is not
       reading the order study.

    What this is **not**: a claim that the pool's top rate is the right one. It is a
    claim that a separation the ladder resolves is worth ranking on, and that an
    unresolved one is not. Where either guard stands down the absolute reading is
    returned unchanged, which is 4 of the 9 sciml pools and all 9 of the BENCHOP
    holdout's -- every pool there holds an exact sampler with no order to measure, so
    this function is structurally unreachable on that corpus and 10o is explicit that
    it therefore carries no evidence either way from it.
    """
    summaries = [s for s in pool if isinstance(s, dict)]
    if len(summaries) < 2:
        return absolute
    rows = []
    for other in summaries:
        # `order(other)` with no pool: the absolute reading, and None where the check
        # was waived or abstained. A pool in which any plan has no order at all is one
        # where there is no "best rate demonstrated here" to read against.
        if order(other) is None:
            return absolute
        value = _num(other.get("observed_order"))
        err = _num(other.get("observed_order_se"))
        floor = _num(other.get("order_floor"))
        if value is None or err is None or floor is None:
            return absolute
        rows.append((value, err, floor))
    mine = _num(summary.get("observed_order"))
    my_err = _num(summary.get("observed_order_se"))
    if mine is None or my_err is None:
        return absolute
    if any(_shortfall(value, floor) > STRONG_BAND for value, _, floor in rows):
        return absolute                                   # guard 1
    top = sorted(rows, key=lambda row: -row[0])
    (best, best_err, _), (second, second_err, _) = top[0], top[1]
    gap_err = math.sqrt(best_err ** 2 + second_err ** 2)
    if gap_err <= 0.0 or (best - second) <= MIN_ORDER_SEPARATION * gap_err:
        return absolute                                   # guard 2
    pair_err = math.sqrt(best_err ** 2 + my_err ** 2)
    if pair_err <= 0.0:
        # A zero-width ladder cannot express a significance, and dividing by it would
        # make every plan but the top one score zero on no evidence at all.
        return absolute
    sigmas = _shortfall(mine, best) / pair_err
    return _clip01(1.0 - sigmas / MIN_ORDER_SEPARATION)


_SDE_FAMILIES = frozenset({"euler-maruyama", "milstein", "tamed-euler-maruyama",
                           "tamed-milstein", "implicit", "exact-transition",
                           "quadratic-exponential", "strang-splitting"})

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


def comparable_keys(tol, pool, keys=CERT_COMPONENTS):
    """The components **every** plan in the pool measured, for ranking them against
    each other. (sciml 10a, M06)

    :func:`sqi` renormalises over whichever components a plan measured, which is
    right for a plan on its own and wrong for a comparison: a mean over four terms
    and a mean over three are different means, so two indices built from different
    component sets are not comparable as numbers. The docstring there names the half
    of this that ``score`` already covers -- a plan that *skipped* a check forfeits a
    circularity break and is capped, so skipping cannot improve its rank. The other
    half has no such backstop.

    Measured on ``sde_sciml_stiffquad_mild`` (plan-sciml-benchmark.md 10a): three
    plans, all certified at 10, same provenance. The semi-implicit plan's
    ``estimated_rel_error`` of 2.545e-3 was **below its own error bar** of 8.82e-3,
    so :func:`accuracy` returned None under F16 -- correctly, because an error inside
    its own noise is not located. Its index was then renormalised over the remaining
    three components and came out 0.708 against 0.722 for a Euler-Maruyama whose
    error was five times larger and resolved. The ranker reported
    ``decided_by: sqi``, on a 2% difference between two means over different terms,
    and shipped the plan the harness measures as **5.1x less accurate**. Nothing
    capped the winner, because nothing was skipped: the losing component was missing
    for being *too accurate to resolve*.

    So a component may contribute to a comparison only where every plan in the pool
    measured it. That is not a new rule -- :func:`cost_admissible` is this same rule
    stated for one component ("cost is subordinate to correctness and decisive
    between equals, and the plans have to be equals") -- and this is it applied to
    all of them. Where a comparison was not made it does not decide, and the keys
    below the index do.

    A pool of fewer than two is no comparison, so every key stands.
    """
    summaries = [s for s in pool if isinstance(s, dict)]
    if len(summaries) < 2:
        return tuple(keys)
    shared = tuple(k for k in keys
                   if all(components(s, tol, pool).get(k) is not None
                          for s in summaries))
    # Every plan missing the same component is not a reason to rank on nothing.
    return shared or tuple(keys)


def error_upper_bound(summary):
    """Upper confidence limit on the plan's relative error, or None.

    ``estimated_rel_error + uncertainty``, on the one statistic
    :func:`noise_limited` compares (a seed median with its between-seed bar where the
    study ran, the single graded run otherwise).

    **Why a bound where :func:`accuracy` has a point.** The point estimate is what
    goes missing: F16 returns None exactly when the error sits inside its own bar, so
    the estimate does not locate it. The *bound* is unaffected -- it is located for a
    noise-limited plan and a resolved one alike, which is what lets a comparison keep
    the accuracy axis instead of deleting it for the whole pool.
    """
    seeded = seed_resolved(summary)
    err = seeded[0] if seeded else _num(summary.get("estimated_rel_error"))
    half_width = uncertainty(summary)
    if err is None or half_width is None:
        return None
    return float(err) + float(half_width)


def accuracy_bound(summary, tol):
    """:func:`accuracy` read off :func:`error_upper_bound` instead of the estimate.

    Same ceiling (:func:`achievable_decades`) and the same clip, so a pool ranked on
    this is ranked on one quantity throughout -- the worst case each plan's own
    evidence permits. Ordering by it is ordering by how much a plan has *shown*,
    which is the question a comparison asks; it is deliberately not a certification
    input, where what matters is where the error actually is.
    """
    hi = error_upper_bound(summary)
    gate = _num(tol)
    if hi is None or hi <= 0.0 or gate is None or gate <= 0.0:
        return None
    ceiling = achievable_decades(summary, tol)
    decades = math.log10(gate / hi)
    return _clip01(decades / ceiling if ceiling and ceiling > 0.0 else 1.0)


def accuracy_on_bounds(pool, tol):
    """Should this pool's accuracy component be read off the bounds?

    Yes where some plan's estimate did not resolve *and* every plan has a bound --
    the configuration in which :func:`comparable_keys` would otherwise drop accuracy
    for all of them.

    This is the repair of a failure the rule that replaced it merely relocated.
    ``comparable_keys`` exists because on ``sde_sciml_stiffquad_mild`` an unresolved
    plan renormalised over three components and lost to a Euler-Maruyama **5.1x less
    accurate**; dropping the axis pool-wide then shipped, on
    ``sde_clark_cameron`` (2026-11-22), a Euler-Maruyama control **11.7x less
    accurate** than a sibling in the same pool -- a plan whose own ``SOLUTION.md``
    says "it is not the candidate to ship, and it must not win a tie". Both times the
    accurate plan lost for the same reason: accuracy could not be used at all. The
    bounds are usable in both.

    Measured on that pool, tol 0.1: bounds 2.2313e-2 / 4.2816e-2 / 5.0258e-2 give
    0.837 / 0.474 / 0.383, and the index orders the plans the way the harness does
    (4.468e-3 < 8.475e-3 < 5.227e-2) where the restricted index had the control first.
    """
    summaries = [s for s in pool if isinstance(s, dict)]
    if len(summaries) < 2:
        return False
    unresolved = [s for s in summaries if accuracy(s, tol) is None
                  and noise_limited(s) and not seed_resolved(s)]
    if not unresolved:
        return False
    # **At least one plan's error has to be located, or the bounds rank on noise.**
    # Measured on `sde_sciml_linear_ito`, tol 0.1: all three plans noise-limited,
    # estimates 5.190e-3 / 5.299e-3 / 3.842e-3 inside bars of ~3.1e-2, so the bounds
    # come out 3.667e-2 / 3.670e-2 / 3.504e-2 -- a 4% spread that is entirely the
    # sampling noise in the estimates. Ranking on it put a Euler-Maruyama the harness
    # measures at 2.058e-2 ahead of a Platen at 2.925e-3, a real 7.0x inversion
    # bought with the luckiest draw; `sde_sciml_cubic` inverted 31.1x the same way.
    # Where nothing is resolved there is no accuracy evidence at any path count the
    # run bought, and the honest reading is the one `comparable_keys` already makes:
    # the axis decides nothing. The bounds are for the **mixed** pool, where a
    # located error anchors the comparison the unresolved plan is bounded against --
    # `sde_clark_cameron` (two resolved, one not) and `sde_sciml_stiffquad_det`.
    if len(unresolved) == len(summaries):
        return False
    return all(accuracy_bound(s, tol) is not None for s in summaries)


def components(summary, tol, pool=()):
    # Accuracy is read off the bounds for the whole pool or for none of it: a mean
    # mixing a point-estimate headroom with a bound headroom is the "means over
    # different quantities" error `comparable_keys` and `sqi` both exist to refuse.
    acc = (accuracy_bound(summary, tol) if accuracy_on_bounds(pool, tol)
           else accuracy(summary, tol))
    return {"accuracy": acc, "order": order(summary, pool),
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

    **That argument covers skipping and not unmeasurability.** A component can also
    be absent because the kernel *could not resolve* it -- F16's noise-limited
    accuracy -- and then no ceiling compensates, because nothing was skipped. Two
    indices renormalised over different component sets are not comparable as point
    values at all, so :func:`rank_key` restricts a comparison to
    :func:`comparable_keys`; this function keeps renormalising, which is what a
    single plan's index has to do.
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
    # Ranked over the components the whole pool measured, so the indices being
    # compared are means over the same terms -- see :func:`comparable_keys`.
    index, _ = sqi(summary, tol, pool, keys=comparable_keys(tol, pool))
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
