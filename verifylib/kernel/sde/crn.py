"""Common random numbers, and the strong/weak order study built on them.

``verification_manual.md`` §18-19. Self-convergence in ``dt`` is only meaningful
when every refinement level is driven by the **same** Brownian path; the ``seed``
argument cannot deliver that, because ``standard_normal((M, Nt))`` and
``standard_normal((M, 2Nt))`` from one seed are unrelated draws and the coarse
increments are not sums of the fine ones. The contract therefore takes the
increments explicitly.

Two asymmetries from §19 are enforced here rather than left to the caller, because
treating the two orders as equally reliable fails correct solvers:

* **The strong order gates.** CRN makes ``X^dt - X^{dt/2}`` a pathwise difference,
  so its mean is measured with very little variance. Measured on GBM at 20 000
  paths: 0.488 for Euler-Maruyama, 1.006 for Milstein.
* **The weak order is a diagnostic.** It is a difference *of means*, so the MC
  error does not cancel, and for some (scheme, phi) pairs the leading weak-error
  term vanishes outright and ``log2(w0/w1)`` returns garbage -- measured, 4.18 on
  GBM with Euler-Maruyama and ``phi = tanh``, purely because ``w1`` had fallen
  below its own standard error. So the estimate is taken across several ``phi``,
  anything at the noise floor is discarded, and an indeterminate result is **not**
  a failure.
"""

from __future__ import annotations

import numpy as np


def ladder_spec(num_paths, dt0, T, levels=3, seed=42, m=None):
    """``[{crn payload}, ...]`` coarse to fine, all coarsenings of one path.

    The payloads are seeds and shapes rather than arrays: the sandbox child
    rebuilds the increments deterministically, so the finest level's 160 MB never
    crosses a process boundary (manual §18's memory note).
    """
    Nt0 = max(1, round(float(T) / float(dt0)))
    fine = Nt0 * 2 ** (levels - 1)
    out = []
    for k in range(levels):
        out.append({"seed": int(seed), "num_paths": int(num_paths), "Nt_fine": int(fine),
                    "T": float(T), "aggregate": 2 ** (levels - 1 - k), "m": m,
                    "dt": float(T) / (Nt0 * 2 ** k)})
    return out


#: The half-width of the band :func:`guards` judges the strong order against --
#: two-sided and one-sided alike. Named because the abstention rule is stated
#: *against the band*: an estimate whose own uncertainty admits more than the band
#: cannot decide the question the band asks, whichever side it is on.
STRONG_BAND = 0.4

#: A pathwise difference below this fraction of the terminal values is the
#: arithmetic, not the scheme.
#:
#: This is the floor the *strong* order needs, and it is not the one ``weak_order``
#: uses. ``s0`` and ``s1`` are means of **absolute** differences, so they are
#: strictly positive and their own standard error is ~``1/sqrt(M)`` of themselves
#: however tiny they are: on ``sde_bm_with_drift``, where Euler-Maruyama is pathwise
#: exact, ``s0 = 9.4e-15`` with ``se_s0/s0 ~ 0.007``, and ``log2(s0/s1)`` came back
#: 2.27 with an error bar of 0.014 -- a confident measurement of floating-point
#: noise, reported as ``orders_ok: True`` against a declared floor of 1.0. No error
#: bar can catch that, because the quantity really is that well determined; what it
#: is not is a discretization error. The scale it has to be compared against is the
#: terminal values themselves.
#:
#: 1e-12 rather than a few multiples of eps because the accumulation over ``Nt``
#: steps is itself eps-scaled and ``strong_order`` does not see ``Nt``. The
#: asymmetry licenses the slack: per plan §11 D8b an abstention costs nothing, so a
#: false abstention is free and a false *verdict* on roundoff is the bug being
#: fixed. No SDE scheme in this repo reaches a relative 1e-12 pathwise difference at
#: a practical ``dt`` without being exact.
ROUNDOFF_REL = 1e-12


def strong_order(terminals, component=0):
    """``(order, detail)`` from three coarse-to-fine terminal-path arrays.

    The detail carries the estimate's **own uncertainty**, which is what makes an
    abstention possible one level up. ``s0`` and ``s1`` are sample means, so the
    jackknife standard error of each *is* ``std(ddof=1)/sqrt(M)`` -- one pass, no
    resampling loop -- and the order is ``log2(s0) - log2(s1)``, so the delta method
    propagates them::

        var(order) = [var(d0)/(M s0^2) + var(d1)/(M s1^2) - 2 cov(d0,d1)/(M s0 s1)] / ln(2)^2

    The covariance term is not optional: ``d0`` and ``d1`` share the middle level's
    paths, so dropping it overstates the bar.

    Why this is needed, measured: ``sde_ginzburg_landau_s6/2-tamed-milstein``
    returns 0.220 and 2.730 for the same scheme against the same spec at 50 000
    paths -- a factor of 12.4 -- and ``orders_ok`` was ``True`` at both ends. At
    ``sigma = 6`` the pathwise differences are violently heavy-tailed, so ``s0`` and
    ``s1`` are sample means with enormous standard errors and ``log2(s0/s1)``
    inherits all of it. ``weak_order`` has had this discipline since it was written;
    the strong order, which is the one that *gates*, had none.

    A **median** absolute difference ships beside the mean as a diagnostic (plan
    §11 D8, part 3). It has bounded variance under the same tails and yields the
    same exponent whenever the tail index does not drift across the ladder, so it is
    the candidate for gating where the mean-based estimate can only abstain. It does
    not gate yet: promotion is conditional on it tracking the mean-based estimate on
    ``3-log-transform-euler``, where the log transform makes the noise additive and
    the mean-based estimate is already tight to 1.1x across 11 runs.
    """
    X = [_component(x, component) for x in terminals]
    d0, d1 = np.abs(X[0] - X[1]), np.abs(X[1] - X[2])
    count = int(len(d0))
    s0, s1 = float(np.mean(d0)), float(np.mean(d1))
    # The scale the differences are roundoff *relative to*: the terminal values at
    # the finest level, which is what any of these differences is a difference of.
    scale = float(np.mean(np.abs(X[-1])))
    detail = {"s0": s0, "s1": s1, "terminal_scale": scale,
              "roundoff_floor": ROUNDOFF_REL * scale, "count": count,
              **_median_order(d0, d1)}
    if count > 1:
        detail["se_s0"] = float(np.std(d0, ddof=1) / np.sqrt(count))
        detail["se_s1"] = float(np.std(d1, ddof=1) / np.sqrt(count))
    if s0 <= 0.0 or s1 <= 0.0:
        return None, {**detail, "reason": "a pathwise difference of exactly zero"}
    if count > 1:
        cov = float(np.cov(d0, d1, ddof=1)[0, 1]) / count
        variance = (detail["se_s0"] ** 2 / s0 ** 2 + detail["se_s1"] ** 2 / s1 ** 2
                    - 2.0 * cov / (s0 * s1)) / np.log(2.0) ** 2
        detail["se_order"] = float(np.sqrt(max(variance, 0.0)))
    return float(np.log2(s0 / s1)), detail


def _component(x, component=0):
    """One state component of a terminal-path array, as a flat ``(M,)`` vector."""
    flat = np.asarray(x, dtype=float).reshape(len(x), -1)
    return flat[:, min(int(component), flat.shape[1] - 1)]


def _median_order(d0, d1):
    """The same exponent off the **median** absolute difference. Diagnostic only."""
    m0, m1 = float(np.median(d0)), float(np.median(d1))
    out = {"median_s0": m0, "median_s1": m1, "median_order": None}
    if m0 > 0.0 and m1 > 0.0:
        out["median_order"] = float(np.log2(m0 / m1))
    return out


#: How many standard errors wide the interval is that has to fit inside the band.
#: 2.0, which is this repo's ``ci_mult`` everywhere else, so "the error bar admits
#: more than the band" is the same sentence in the same units as every other SDE
#: comparison: the estimate's two-standard-error interval must fit inside the band
#: it would be judged against, or it cannot decide what the band asks.
#:
#: Measured, on three configurations that must come out differently: a correct
#: additive-noise Euler-Maruyama at order 0.993 has ``se_order = 0.0069``, 29x
#: inside the bar; a solver that ignores the supplied increments has 0.0096 and
#: stays a **failure**, which is the one verdict that must survive; a multiplicative
#: scheme at ``sigma = 6`` has 0.265 and abstains.
ORDER_CI_MULT = 2.0


def abstention(strong, detail, *, floor_mult=2.0, band=STRONG_BAND,
               ci_mult=ORDER_CI_MULT):
    """Why the strong order carries no verdict, or ``None`` when it does.

    Three reasons, and all three are "the estimator could not measure this", never
    "the solver is wrong". Plan §11 D8: ``"unavailable"`` is in
    :data:`..metrics.SKIPS`, so ``passed()`` and ``failed()`` both refuse it and the
    plan keeps the score it has -- an unmeasurable strong order on heavy-tailed
    multiplicative noise is a property of the kernel's estimator, and charging it to
    the solver is the false hard-fail on correct work this repo keeps rediscovering.
    """
    if strong is None or not np.isfinite(strong):
        return (detail.get("reason")
                or "the strong-order estimate is not a finite number")
    floor = float(detail.get("roundoff_floor") or 0.0)
    smallest = min(float(detail.get("s0", 0.0)), float(detail.get("s1", 0.0)))
    if floor > 0.0 and smallest <= floor:
        return (f"both pathwise differences are at the roundoff scale of the terminal "
                f"values ({smallest:.3g} against a floor of {floor:.3g}): the scheme "
                f"integrates this problem exactly, or near enough that the ratio "
                f"measures arithmetic rather than discretization")
    for key, se_key in (("s0", "se_s0"), ("s1", "se_s1")):
        value, se = detail.get(key), detail.get(se_key)
        if se is not None and float(value) <= floor_mult * float(se):
            return (f"{key} = {float(value):.3g} is within {floor_mult:g} standard "
                    f"errors of zero ({float(se):.3g}), so the ratio is noise")
    se_order = detail.get("se_order")
    if se_order is not None and float(ci_mult) * float(se_order) > float(band):
        return (f"the estimate is {strong:.3g} +/- {float(se_order):.3g}, so its "
                f"{ci_mult:g}-sigma interval is wider than the {float(band):g} band "
                f"it would be judged against and cannot decide whether the order "
                f"clears its floor")
    return None


#: Test functions the weak order is estimated across. One polynomial, one
#: quadratic, one bounded nonlinear -- §20's rationale, reused: a scheme right on
#: polynomials and wrong on a bounded function has a tail problem worth reporting.
WEAK_PHIS = (("x", lambda x: x), ("x2", lambda x: x ** 2),
             ("tanh", lambda x: np.tanh(x)))


def weak_order(terminals, phis=WEAK_PHIS, floor_mult=2.0, component=0):
    """``(median order or None, per-phi detail)``. ``None`` means indeterminate."""
    X = [_component(x, component) for x in terminals]
    orders, detail = [], {}
    for name, phi in phis:
        a, b, c = (phi(x) for x in X)
        w0, w1 = abs(float(np.mean(a) - np.mean(b))), abs(float(np.mean(b) - np.mean(c)))
        se = float(np.std(c - b, ddof=1) / np.sqrt(len(c)))
        if w1 < floor_mult * se or w1 <= 0.0 or w0 <= w1:
            detail[name] = {"w0": w0, "w1": w1, "se": se, "used": False,
                            "reason": "at the noise floor, or not shrinking"}
            continue
        order = float(np.log2(w0 / w1))
        detail[name] = {"w0": w0, "w1": w1, "se": se, "used": True, "order": order}
        orders.append(order)
    return (float(np.median(orders)) if orders else None), detail


#: The declared expected order this SDE family maps to. The **spec** declares the
#: values; the plan declares only which of them applies to it, so a plan cannot
#: lower its own bar -- the same rule §5a settles for ``p_design`` on the PDE side.
FAMILY_ORDER_KEY = {
    "milstein": "expected_strong_order_milstein",
    "tamed-milstein": "expected_strong_order_milstein",
}


def expected_orders(verification, scheme_family, *, declared=True):
    """``(expected_strong, candidates, declared)``.

    ``declared`` says whether ``scheme_family`` was *stated* by the plan-creator or
    *inferred* by the kernel from the free-text ``scheme:``. Only a stated one
    narrows the band to two-sided, because an inference must never make a test
    stricter -- a false hard-fail on a correct solver is invisible as a false
    positive, which is the rule this repo applies everywhere else.

    ``sde_ginzburg_landau_s6/3-log-transform-euler`` is why. Its ``scheme:`` reads
    ``euler-maruyama``, which infers strong order 0.5 -- but it is Euler-Maruyama on
    the *log transform*, where the noise becomes additive and the strong order is
    therefore **1**. It measures 1.019, correctly, and the inferred two-sided band
    at 0.5 +/- 0.4 failed it. The free-text field cannot express that; a declared
    ``scheme_family`` can.

    ``sde_ginzburg_landau_s6`` is why this is not a single field. Its spec declares
    ``expected_strong_order: 0.5`` *and* ``expected_strong_order_milstein: 1.0``,
    and its own note says in as many words: "do not fail a Milstein plan for
    measuring 1.0 against the 0.5 default". Which one applies is a property of the
    plan's scheme, and until the plan declares it the kernel cannot know.
    """
    candidates = []
    for key, value in (verification or {}).items():
        if str(key).startswith("expected_strong_order") and isinstance(value, (int, float)):
            candidates.append(float(value))
    default = float((verification or {}).get("expected_strong_order", 0.5))
    if not candidates:
        candidates = [default]
    key = FAMILY_ORDER_KEY.get((scheme_family or "").strip().lower())
    floor_declared = isinstance((verification or {}).get("expected_strong_order"),
                                (int, float))
    if key and isinstance((verification or {}).get(key), (int, float)):
        # Selecting *among the spec's declared values* is safe even on an inference:
        # every candidate came from the formulator, and the band stays two-sided
        # only when the plan actually stated its family.
        return float(verification[key]), candidates, bool(declared)
    if scheme_family and declared and floor_declared:
        return default, candidates, True
    # D2: **one-sided whenever the floor's source is `kernel_default`**, not only
    # when the family is undeclared. The default is 0.5, which is Euler-Maruyama's
    # order under *multiplicative* noise and wrong by a factor of two under additive
    # noise -- six of the stale specs are additive or noise-free, where the order is
    # 1. Measured on the OU canary with the family declared and the floor defaulted:
    # `orders_ok: False` at observed 0.9863 against a floor of 0.5, on a correct
    # solver. `expected_orders`' own rule is that an inference must never make a test
    # stricter, and a floor nobody declared is an inference.
    return default, candidates, False


def guards(strong, weak, expected_strong, expected_weak, detail, *,
           declared=True, candidates=()):
    """§19's guards.

    ``orders_ok`` is **three-valued**: ``True``, ``False``, or the literal
    ``"unresolved"`` when :func:`abstention` says the estimator could not measure
    this. The third value is the fix for a guard that reported the wrong outcome --
    see the comment at the abstention call.

    [REV 59]: that third value was ``"unavailable"``, which ``_orders`` also
    returns when **a CRN level crashed**. One token for both put a property of
    the estimator and a property of the code under the same label, and
    ``score.ceiling`` credited the union of them: ``is_skip(order_ok)`` was true
    for a crashed ladder, so a plan whose ladder never produced a number could
    collect D8b's abstention credit and reach a circularity-break 9. ``m13``'s
    ``IndexError`` was recorded under exactly this token. ``"unresolved"`` is the
    word the rest of the kernel already uses for "the check, not the solver, is
    the limit" (``residual.OUTCOMES``, ``constraints.py``, ``dynkin.check``), and
    it is what earns the credit; ``"unavailable"`` keeps the crash and earns
    nothing.

    ``weak_ok`` is **true when the estimate is unavailable**: an indeterminate weak
    order is not evidence of a defect.

    When the plan does not declare its scheme family, **or the floor it would be
    judged against came from the kernel default rather than the spec**, the band
    widens and becomes one-sided. Two-sided is only meaningful against the
    expectation for the scheme actually implemented, and §19 already establishes the
    asymmetry -- "an order far *above* the theoretical one is noise, not a bonus" --
    so an undeclared plan is failed only for measuring materially *below* the lowest
    order any of the spec's
    declarations would allow. Measured: all three Ginzburg-Landau plans land between
    0.95 and 1.52 against a 0.5 default, and failing them for a missing declaration
    would be a false positive on three correct solvers.
    """
    candidates = [float(c) for c in (candidates or [expected_strong])]
    shrinking = detail.get("s1", 1.0) < detail.get("s0", 0.0)
    # The abstention comes first, and it is the one thing `guards` could not reach
    # before. `strong_order` returning `None` landed here as `finite = False` ->
    # `s_ok = False` -> `orders_ok: False`, a verdict of *failure*: the `is None`
    # forgiveness one line below belongs to the **weak** order and nothing mirrored
    # it for the strong one. So an exactly-integrating scheme was failed rather than
    # excused, which is live today on `dX = dW`.
    abstained = abstention(strong, detail)
    if declared:
        s_ok = shrinking and abs(strong - float(expected_strong)) < STRONG_BAND
    else:
        s_ok = shrinking and strong >= min(candidates) - STRONG_BAND
    s_ok = "unresolved" if abstained else bool(s_ok)
    w_ok = weak is None or (0.5 * float(expected_weak) <= weak
                            <= float(expected_weak) + 0.75)
    # §19's own heading: "The strong order is the gate; the weak order is a
    # diagnostic." An order *above* the theoretical one is noise, not a bonus --
    # the section says so -- and an indeterminate one is explicitly not evidence of
    # a defect, so an above-band weak order is the same noise the `is None` branch
    # already forgives. Measured: a correct tamed Euler-Maruyama plan on
    # sde_ginzburg_landau_s6 lands at 1.7514 against an upper limit of 1.75 and was
    # failed for 0.0014. `orders_ok` therefore gates on the strong order alone; the
    # weak order is reported, and an out-of-band one is flagged rather than fatal.
    out = {"strong_ok": s_ok, "weak_ok": bool(w_ok),
           "weak_order_out_of_band": bool(weak is not None and not w_ok),
           "strong_order": strong, "weak_order": weak,
           "expected_strong_order": float(expected_strong),
           "expected_strong_candidates": candidates,
           "strong_band": "two-sided" if declared else "one-sided (scheme family "
                                                       "undeclared in the plan)",
           "band_half_width": float(STRONG_BAND),
           "expected_weak_order": float(expected_weak),
           "orders_ok": s_ok}
    if abstained:
        out["abstained"] = abstained
    return out


def richardson_moment(terminals, weak, component=0, *, ci_mult=2.0):
    """The weak-order Richardson extrapolation of the mean (§19).

    ``stable`` is the verdict: the extrapolated mean sits within 5% of the finest
    level's, so the ladder is somewhere the extrapolation means something.

    ``shift_in_se`` is [REV 68], and it is **report-only**. A relative shift on its
    own cannot be read: 8.5% is a real bias on one plan and pure Monte Carlo noise
    on another, and the denominator is a mean that several specs put near zero.
    The two levels are a coarsening of *one* Brownian path, so the difference is
    paired and its standard error is the paired one -- an order of magnitude below
    either mean's own, which is the whole reason the ladder uses common random
    numbers. Dividing by it says how many standard errors the shift is.

    Measured on ``sde_multichannel_stiff_m13``, which is the plan that raised the
    question (its ``richardson_stable`` went ``"unavailable"`` to ``False`` when
    the ladder started running there, and nothing said whether that was signal):

        component 0   shift 1.07e-2 (9.07% relative)   se_shift 3.21e-4   **33.2 se**
        component 1   shift 2.30e-3 (1.55% relative)   se_shift 2.39e-4      9.6 se

    So the ``False`` is correct and the mean really does move: an explicit
    Euler-Maruyama on a stiff 13-channel system at ``dt0 = 0.04`` carries an
    ``O(dt)`` bias 33 standard errors wide, not a resolution problem. The
    verdict is deliberately left on the relative shift alone -- ``stable`` is a
    statement about the *ladder*, and at 50000 paths almost any real shift clears
    the noise, so gating on standard errors would make it fire on correct plans.
    """
    X = [_component(x, component) for x in terminals]
    p = max(float(weak) if weak is not None else 0.5, 0.5)
    m1, m2 = float(np.mean(X[-2])), float(np.mean(X[-1]))
    star = m2 + (m2 - m1) / (2.0 ** p - 1.0)
    shift = abs(star - m2)
    spread = shift / (abs(star) + 1e-14)
    paired = (np.asarray(X[-1], dtype=float).ravel()
              - np.asarray(X[-2], dtype=float).ravel())
    if len(paired) > 1:
        se_shift = float(np.std(paired, ddof=1) / np.sqrt(len(paired))) / (2.0 ** p - 1.0)
    else:
        se_shift = None
    if se_shift is None:
        in_se, within = None, None
    elif se_shift == 0.0:
        # A pathwise-constant difference: every path moved by the same amount, so
        # there is no scatter for noise to hide a bias in. Zero standard errors
        # when the shift is zero too, and unbounded when it is not -- which is the
        # honest reading of a perfectly resolved difference, not a division to dodge.
        in_se = 0.0 if shift == 0.0 else float("inf")
        within = shift == 0.0
    else:
        in_se, within = shift / se_shift, bool(shift <= float(ci_mult) * se_shift)
    return {"mean_star": star, "mean_finest": m2, "relative_shift": spread,
            "shift": shift, "se_shift": se_shift, "shift_in_se": in_se,
            "shift_within_noise": within,
            "stable": bool(spread < 0.05), "p_used": p}
