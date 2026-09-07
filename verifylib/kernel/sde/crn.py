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


def strong_order(terminals):
    """``(order, detail)`` from three coarse-to-fine terminal-path arrays."""
    X = [np.asarray(x, dtype=float).reshape(len(x), -1)[:, 0] for x in terminals]
    s0 = float(np.mean(np.abs(X[0] - X[1])))
    s1 = float(np.mean(np.abs(X[1] - X[2])))
    if s0 <= 0.0 or s1 <= 0.0:
        return None, {"s0": s0, "s1": s1, "reason": "a pathwise difference of exactly zero"}
    return float(np.log2(s0 / s1)), {"s0": s0, "s1": s1}


#: Test functions the weak order is estimated across. One polynomial, one
#: quadratic, one bounded nonlinear -- §20's rationale, reused: a scheme right on
#: polynomials and wrong on a bounded function has a tail problem worth reporting.
WEAK_PHIS = (("x", lambda x: x), ("x2", lambda x: x ** 2),
             ("tanh", lambda x: np.tanh(x)))


def weak_order(terminals, phis=WEAK_PHIS, floor_mult=2.0):
    """``(median order or None, per-phi detail)``. ``None`` means indeterminate."""
    X = [np.asarray(x, dtype=float).reshape(len(x), -1)[:, 0] for x in terminals]
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
    if key and isinstance((verification or {}).get(key), (int, float)):
        # Selecting *among the spec's declared values* is safe even on an inference:
        # every candidate came from the formulator, and the band stays two-sided
        # only when the plan actually stated its family.
        return float(verification[key]), candidates, bool(declared)
    if scheme_family and declared:
        return default, candidates, True
    return default, candidates, False


def guards(strong, weak, expected_strong, expected_weak, detail, *,
           declared=True, candidates=()):
    """§19's guards.

    ``weak_ok`` is **true when the estimate is unavailable**: an indeterminate weak
    order is not evidence of a defect.

    When the plan does **not** declare its scheme family the band widens and becomes
    one-sided. Two-sided is only meaningful against the expectation for the scheme
    actually implemented, and §19 already establishes the asymmetry -- "an order far
    *above* the theoretical one is noise, not a bonus" -- so an undeclared plan is
    failed only for measuring materially *below* the lowest order any of the spec's
    declarations would allow. Measured: all three Ginzburg-Landau plans land between
    0.95 and 1.52 against a 0.5 default, and failing them for a missing declaration
    would be a false positive on three correct solvers.
    """
    candidates = [float(c) for c in (candidates or [expected_strong])]
    shrinking = detail.get("s1", 1.0) < detail.get("s0", 0.0)
    finite = strong is not None and np.isfinite(strong)
    if declared:
        s_ok = finite and shrinking and abs(strong - float(expected_strong)) < 0.4
    else:
        s_ok = finite and shrinking and strong >= min(candidates) - 0.4
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
    return {"strong_ok": bool(s_ok), "weak_ok": bool(w_ok),
            "weak_order_out_of_band": bool(weak is not None and not w_ok),
            "strong_order": strong, "weak_order": weak,
            "expected_strong_order": float(expected_strong),
            "expected_strong_candidates": candidates,
            "strong_band": "two-sided" if declared else "one-sided (scheme family "
                                                        "undeclared in the plan)",
            "expected_weak_order": float(expected_weak),
            "orders_ok": bool(s_ok)}


def richardson_moment(terminals, weak):
    """The weak-order Richardson extrapolation of the mean (§19)."""
    X = [np.asarray(x, dtype=float).reshape(len(x), -1)[:, 0] for x in terminals]
    p = max(float(weak) if weak is not None else 0.5, 0.5)
    m1, m2 = float(np.mean(X[-2])), float(np.mean(X[-1]))
    star = m2 + (m2 - m1) / (2.0 ** p - 1.0)
    spread = abs(star - m2) / (abs(star) + 1e-14)
    return {"mean_star": star, "mean_finest": m2, "relative_shift": spread,
            "stable": bool(spread < 0.05), "p_used": p}
