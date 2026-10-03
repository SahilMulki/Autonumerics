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
#:   worth a full point of certification, so weighting it heavily double-counts;
#: * **cost** is subordinate to correctness and decisive between equals.
WEIGHTS = {"accuracy": 0.35, "order": 0.25, "structure": 0.20,
           "consistency": 0.10, "cost": 0.10}

#: Cost is a property of the *pool*, not of a plan, and a certification must not
#: depend on how fast a sibling ran. ``SQI_cert`` drops it and renormalises.
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
    # On a chaotic problem the graded horizon is the one the problem asks about, and
    # where `graded_N` is declared the graded grid is what it is scored on. Both are
    # worse-case in `headroom_decades`, never better: an error quoted at a reference
    # horizon must not stand in for the error at T. An error of exactly zero comes
    # back None there -- it is the signature of a solver whose output did not respond
    # to the grid at all (`baseline_select.looks_n_independent`), so it is unmeasured
    # rather than perfect.
    return _clip01(headroom_fraction(summary, tol))


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
    """Order fidelity against the scheme's **design** rate, two-sided.

    Two-sided because overshoot is not a bonus. A 2nd-order scheme measuring 4.0 has
    a ladder outside its asymptotic range or a problem whose symmetry flattered it;
    ``p_sane`` already refuses to call that asymptotic. A spectral scheme is the one
    exception -- it converges faster than any algebraic rate, so exceeding the
    declared order is expected and only falling short counts.

    Returns None where the check was waived (`order_check: waived_spec` /
    `waived_chaotic`), where the order is non-finite (a run at the round-off floor,
    which `richardson` reports as `inf` and which the score already credits), where
    the scheme is spectral (see below), or where no design order was declared. That last case is most plans today: measured
    across `workspace/`, 22 of 104 declare `spatial_order` and 25 declare
    `scheme_family` -- both landed with the no-closed-form work, so the fields are
    present on the newer pools only. Inferring a design order from a plan's *slug*
    would let a plan rename itself into a better rank, which is the one thing the
    declared-never-inferred rule across this repo exists to stop.
    """
    # `order_check` is a three-valued string, not a bool: `driver._order_verdict`
    # returns "true", "waived_spec" or "waived_chaotic". Testing it for
    # stringhood reads a *passing* check as a waived one.
    if str(summary.get("order_check") or "").startswith("waived"):
        return None
    observed = _num(summary.get("observed_order"))
    design = _num(summary.get("design_order"))
    if observed is None or design is None or design <= 0.0:
        return None
    if summary.get("kind") == "sde" or summary.get("scheme_family") in _SDE_FAMILIES:
        # `order_floor` carries the expected strong order, a target with the
        # two-sided band `strong_ok` applies. Distance from it is the measurement.
        target = _num(summary.get("order_floor")) or design
        return _clip01(1.0 - abs(observed - target) / max(target, 0.5))
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
    return _clip01(1.0 - abs(observed - design) / design)


_SDE_FAMILIES = frozenset({"euler-maruyama", "milstein", "tamed-euler-maruyama",
                           "tamed-milstein", "implicit"})

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
    """
    mine = _num(summary.get("solve_time_s"))
    times = sorted(t for t in (_num(s.get("solve_time_s")) for s in pool)
                   if t is not None and t > 0.0)
    if mine is None or mine <= 0.0 or len(times) < 2 or times[0] == times[-1]:
        return None
    return _clip01(math.log10(times[-1] / mine) / math.log10(times[-1] / times[0]))


# --- the index ---------------------------------------------------------------


def components(summary, tol, pool=()):
    return {"accuracy": accuracy(summary, tol), "order": order(summary),
            "structure": structure(summary), "consistency": consistency(summary),
            "cost": cost(summary, pool)}


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
    if keys is not None:
        parts = {k: v for k, v in parts.items() if k in keys}
    measured = {k: v for k, v in parts.items() if v is not None}
    if not measured:
        return None, parts
    total = sum(WEIGHTS[k] for k in measured)
    return sum(WEIGHTS[k] * v for k, v in measured.items()) / total, parts


def sqi_cert(summary, tol):
    """The index the certification reads: no cost component (see
    :data:`CERT_COMPONENTS`)."""
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
    index, _ = sqi(summary, _tol_of(summary, m), pool)
    return (
        0,
        -(_num(m.get("score")) or 0.0),            # score descending
        provenance_rank(m.get("provenance")),      # provenance order
        -(index if index is not None else -1.0),   # quality descending; unscored last
        _num(summary.get("solve_time_s")) or float("inf"),   # cost ascending
        _num(entry.get("iter")) or 0.0,            # fewest iterations
    )


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
