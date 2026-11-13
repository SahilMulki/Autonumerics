"""The declared terminal law, and the shape test built on it.

``verification_manual.md`` §14's three-way outcome applied to a *distribution*
rather than to a moment. Finding 4: there is no distributional evidence of any kind
on the SDE side, so three defect classes pass everything -- a wrong increment law
(Rademacher increments rescaled to unit variance), a dropped Milstein correction
whose first two moments happen to land, and a clipped state that moves mass to a
point and leaves the mean inside tolerance.

**Why the sample is standardized first** (plan §10 D4). A one-sample KS test against
the *raw* terminal law at 50 000 paths resolves the scheme's own ``O(dt)`` bias and
rejects correct solvers. Measured, correct Euler-Maruyama, n = 50 000:

===========  =======  =========  =====================  ========
process      dt       KS D       crit (alpha = 0.05)    verdict
===========  =======  =========  =====================  ========
OU           1e-3     0.00477    0.00607                pass, by a hair
OU           **1e-2** 0.00663    0.00607                **reject**
GBM          1e-3     0.00580    0.00607                pass, by a hair
GBM          **1e-2** 0.00678    0.00607                **reject**
===========  =======  =========  =====================  ========

The live specs run at ``dt = 0.01`` and ``0.005`` with 50 000 paths -- exactly this
regime. The bar is 0.006 in sup-norm while a weak-order-1 scheme's CDF bias is
``O(dt)``, so the test measures *resolution* rather than correctness. Standardizing
both sides kills the location and scale bias, which is what the ``O(dt)`` error
mostly is, and leaves the test measuring the thing the first two moments cannot
see: **shape**. The mean and the variance keep their own checks, in ``_compare``.

**Why the critical value is bootstrapped** (plan-sciml-benchmark.md SS10a, SS10g). The
table above is the half of the problem standardizing fixed. The other half is that
``KS_C95 / sqrt(n)`` is the wrong bar for this statistic: standardizing the sample by
its own mean and standard deviation estimates two parameters from the data being
tested, which is the Lilliefors situation and has no closed-form null. Here the
standardization is also *asymmetric* -- the sample is divided by its own standard
deviation, the reference by the law's exact one -- so on a heavy-tailed law the
divisor's own noise dominates the shape signal the test is for. The honest bar is
**wider**, and :func:`bootstrap_critical` simulates it. What the asymptotic bar cost,
measured on M01's lognormal (``n = 50 000``, ``dt = 0.0016``, excess kurtosis 47.65):

==================  ========  =================  ====================
solver              KS D      vs asymptotic      vs bootstrap
==================  ========  =================  ====================
**exact** (0 error) 0.007133  **FAIL, 1.17x**    pass, 0.39x
euler-maruyama      0.004602  pass, 0.76x        pass, 0.25x
milstein            0.006141  **FAIL, 1.01x**    pass, 0.34x
heun                0.006754  **FAIL, 1.11x**    pass, 0.37x
rademacher dW       0.017254  FAIL, 2.84x        pass, 0.95x  *(see below)*
clipped state       0.064974  FAIL, 10.70x       **FAIL, 3.58x**
==================  ========  =================  ====================

A pathwise-**exact** solver failed the old bar, and failed it worse than the
Euler-Maruyama that passed; over nine seeds it failed on five. Those were not
detections, so the power the old bar appeared to have was a bias. At the coarse steps
finding 4's defect classes were calibrated at (``dt = 0.01``, 100 steps) the bootstrap
bar keeps every one of them: correct solvers land at 0.16-0.98x and Rademacher
increments at **2.52-3.01x**, clipping at **3.28-3.79x**, so the ~2.6x separation that
made gating safe survives.

**The one power loss, recorded rather than tuned away.** At ``dt = 0.0016`` -- 625
steps -- Rademacher increments sit at 0.95-1.09x of the bootstrap bar and are no
longer reliably caught. That is the CLT, not the bar: summing 625 iid +/-sqrt(dt)
increments gives a terminal law very close to the lognormal, so there is little shape
error left to detect. The old bar "caught" it only in the same sense that it caught
the exact solver. The fix for this is a change to the *statistic* rather than its bar
-- standardizing the reference by the sample's moments too, which removes the divisor
mismatch and narrows the null -- and that is a cross-suite recalibration, recorded in
SS10g and not done here.

Each law also states its own exact standardized third and fourth moments, which is
what lets 3b's cheap tier grade rather than only report (plan §11 D11): declaring
the law once serves both tiers, and separate ``skewness_value_at_T`` fields would be
a second source of truth for the same fact.
"""

from __future__ import annotations

import numpy as np

from ...operator import evaluate

#: Asymptotic KS critical value at alpha = 0.05, as ``c / sqrt(n)``. 1.3581 is the
#: standard two-sided constant.
#:
#: **No longer the grading bar** -- see :func:`bootstrap_critical`. It is kept because
#: it is still reported beside the bootstrap value as ``ks_critical_asymptotic``, and
#: because it is the fallback for a law with no sampler. The comment this replaces
#: said the sample sizes here "are far past where the asymptotic form matters", which
#: is true of the Kolmogorov limit and was never the issue: the statistic is not
#: distribution-free at *any* n once its divisor is estimated from the sample.
KS_C95 = 1.3581


def _number(value, params):
    return float(evaluate(value, params)) if isinstance(value, str) else float(value)


class Law:
    """One terminal law: its first four moments and its CDF.

    ``standardized_cdf`` is the CDF of ``(X - mean)/sd``, which is what the test
    compares the standardized sample against.
    """

    __slots__ = ("name", "mean", "sd", "skewness", "excess_kurtosis", "_cdf",
                 "_rvs", "shape_key")

    def __init__(self, name, mean, sd, skewness, excess_kurtosis, cdf,
                 rvs=None, shape_key=()):
        self.name, self.mean, self.sd = name, float(mean), float(sd)
        self.skewness, self.excess_kurtosis = float(skewness), float(excess_kurtosis)
        self._cdf = cdf
        #: ``rvs(n, rng) -> ndarray``, iid draws from this law. Needed by the
        #: bootstrap bar, which has to sample the *declared* law rather than
        #: resample the solver's output: resampling would carry the solver's own
        #: shape error into its own critical value.
        self._rvs = rvs
        #: What the statistic's null distribution depends on, which is the law's
        #: **shape** and not its location or scale -- both sides of the comparison
        #: are standardized, so ``mean`` and ``sd`` divide out. Two laws with the
        #: same key share a bar, and that is what makes caching it sound.
        self.shape_key = (name, *(round(float(v), 12) for v in shape_key))

    def standardized_cdf(self, z):
        return self._cdf(self.mean + self.sd * np.asarray(z, dtype=float))

    def sample(self, n, rng):
        """``n`` iid draws, or None where this law has no sampler."""
        return None if self._rvs is None else self._rvs(int(n), rng)


def _normal(entry, params):
    mean = _number(entry.get("mean", 0.0), params)
    variance = _number(entry["variance"], params)
    sd = float(np.sqrt(variance))
    from scipy.stats import norm
    # One shape, so one bar per `n` for every normal law there will ever be.
    return Law("normal", mean, sd, 0.0, 0.0,
               lambda x: norm.cdf(x, loc=mean, scale=sd),
               rvs=lambda n, rng: rng.normal(mean, sd, n))


def _lognormal(entry, params):
    """``log X ~ N(mu, s^2)``. GBM, Black-Scholes and Exp-OU terminal laws."""
    mu = _number(entry["log_mean"], params)
    s2 = _number(entry["log_variance"], params)
    s = float(np.sqrt(s2))
    mean = float(np.exp(mu + 0.5 * s2))
    variance = float((np.exp(s2) - 1.0) * np.exp(2 * mu + s2))
    e = float(np.exp(s2))
    from scipy.stats import norm
    return Law("lognormal", mean, float(np.sqrt(variance)),
               (e + 2.0) * np.sqrt(e - 1.0),
               e ** 4 + 2 * e ** 3 + 3 * e ** 2 - 6.0,
               lambda x: np.where(np.asarray(x) > 0.0,
                                  norm.cdf(np.log(np.maximum(np.asarray(x), 1e-300)),
                                           loc=mu, scale=s), 0.0),
               rvs=lambda n, rng: np.exp(rng.normal(mu, s, n)),
               # `mu` is pure location in log space and scales the law, so the
               # standardized shape depends on `s` alone.
               shape_key=(s,))


def _noncentral_chi2(entry, params):
    """``X = scale * chi2'(df, nc)`` -- the exact CIR transition law."""
    df = _number(entry["df"], params)
    nc = _number(entry["nc"], params)
    scale = _number(entry.get("scale", 1.0), params)
    mean = scale * (df + nc)
    variance = 2.0 * scale ** 2 * (df + 2 * nc)
    third = 8.0 * scale ** 3 * (df + 3 * nc)
    fourth = 48.0 * scale ** 4 * (df + 4 * nc)
    sd = float(np.sqrt(variance))
    from scipy.stats import ncx2
    return Law("noncentral_chi2", mean, sd,
               third / sd ** 3, fourth / variance ** 2,
               lambda x: ncx2.cdf(np.maximum(np.asarray(x), 0.0) / scale, df, nc),
               rvs=lambda n, rng: scale * rng.noncentral_chisquare(df, nc, n),
               # `scale` is a pure scale factor; the shape is (df, nc).
               shape_key=(df, nc))


BUILDERS = {"normal": _normal, "lognormal": _lognormal,
            "noncentral_chi2": _noncentral_chi2}


def build(entry, params):
    """``(Law, None)`` or ``(None, reason)`` for one declared law."""
    if not isinstance(entry, dict):
        return None, "terminal_law is not an object"
    name = entry.get("law")
    builder = BUILDERS.get(name)
    if builder is None:
        return None, (f"terminal_law.law {name!r} is not one the kernel can build "
                      f"(known: {sorted(BUILDERS)})")
    try:
        return builder(entry, params), None
    except KeyError as exc:
        return None, f"terminal_law is missing {exc.args[0]!r} for law {name!r}"
    except Exception as exc:  # noqa: BLE001 -- a formulator defect, reported
        return None, f"terminal_law did not evaluate: {type(exc).__name__}: {exc}"


#: Bootstrap replications behind each critical value. 1000, because the bar is a
#: sample quantile and carries its own Monte-Carlo error: measured on M01's
#: lognormal at ``n = 50 000``, the 95th percentile's relative standard deviation
#: across independent bootstrap seeds is 6.0% at B = 200, 3.0% at B = 600 and 2.3%
#: at B = 1000. The separation the bar has to resolve is a factor of ~2.4 (see the
#: table in this module's docstring), so 2.3% is comfortably inside it, and the cost
#: is ~3 s for a lognormal and ~12 s for a noncentral chi-squared against a kernel
#: that already spends ~1500 s on the CRN ladder.
KS_BOOTSTRAP_DRAWS = 1000

#: Fixed, so a grading bar is a function of the law and the path count and nothing
#: else. A bar reseeded per run would move the verdict between two gradings of the
#: same solver, which is worse than a bar 2% off: every plan on a problem is judged
#: against the *same* number, so the comparison between plans stays exact.
KS_BOOTSTRAP_SEED = 20261113

#: ``{(law.shape_key, n): crit}``. The null distribution depends on the law's shape
#: and the sample size alone, so every component and every plan of a problem shares
#: one bar and pays for it once per process.
_BARS = {}


def bootstrap_critical(law, n, draws=KS_BOOTSTRAP_DRAWS):
    """The 95th percentile of :func:`ks_standardized`'s statistic **under the
    declared law**, or None where the law cannot be sampled.

    Why this replaces ``KS_C95 / sqrt(n)``: that constant is the Kolmogorov value,
    which holds for a statistic comparing a sample against a *fully specified*
    distribution. This statistic standardizes the sample by its own mean and
    standard deviation before comparing it against the law's standardized CDF, so
    two of its parameters are estimated from the data it is testing -- the Lilliefors
    situation, whose null distribution has no closed form and must be simulated.

    The direction of the error is the part that mattered. Estimating location and
    scale usually makes ``D`` *smaller*, so one might expect a narrower bar; here the
    standardization is asymmetric -- the sample is divided by its own standard
    deviation while the reference is divided by the law's exact one -- and on a
    heavy-tailed law that divisor is the noisy quantity. The relative standard error
    of a sample standard deviation is ``sqrt((kappa - 1) / 4n)``, and M01's terminal
    lognormal has excess kurtosis 47.65, so at ``n = 50 000`` the divisor itself
    carries 1.58% error. The honest bar is therefore **wider**: measured 0.01663
    against the asymptotic 0.006074, a factor of 2.74.

    What that cost, before it was fixed (plan-sciml-benchmark.md SS10a): on M01's own
    draw a **pathwise-exact** solver -- zero error, same increments -- failed this
    gate at 1.17x while the Euler-Maruyama that passed sat at 0.76x, and the two
    better of three plans were failed at 1.01x and 1.11x. The gate was reading its
    own divisor's noise. Across nine seeds the exact sampler failed it on five.
    """
    if n < 2 or law.sample(1, np.random.default_rng(0)) is None:
        return None
    key = (law.shape_key, int(n))
    if key not in _BARS:
        rng = np.random.default_rng(KS_BOOTSTRAP_SEED)
        ds = np.empty(int(draws), dtype=float)
        for i in range(int(draws)):
            ds[i] = _statistic(law.sample(n, rng), law)
        _BARS[key] = float(np.quantile(ds, 0.95))
    return _BARS[key]


def _statistic(sample, law):
    """The sup-norm distance between the standardized sample and the standardized
    law, or None for a sample with no shape to test."""
    x = np.asarray(sample, dtype=float).ravel()
    n = x.size
    sd = float(np.std(x, ddof=1))
    if n < 2 or sd <= 0.0:
        return None
    z = np.sort((x - float(np.mean(x))) / sd)
    reference = np.asarray(law.standardized_cdf(z), dtype=float)
    empirical = np.arange(1, n + 1, dtype=float) / n
    return float(max(np.max(np.abs(empirical - reference)),
                     np.max(np.abs(reference - (empirical - 1.0 / n)))))


def ks_standardized(sample, law):
    """``(D, crit, detail)`` -- the sup-norm distance in **shape** only.

    The sample is standardized by its own mean and standard deviation and the law by
    its exact ones, so a location or scale error -- which is what an ``O(dt)`` weak
    error mostly is -- cancels on both sides and the statistic measures the
    remaining shape mismatch.

    ``crit`` is :func:`bootstrap_critical`, not ``KS_C95 / sqrt(n)``; the asymptotic
    value is still reported beside it as ``ks_critical_asymptotic`` so a reader can
    see which bar a verdict came from and what the other one would have said. A law
    with no sampler falls back to the asymptotic bar, flagged, because an ungradable
    check is worse than a conservative one and the fallback is now visible.
    """
    x = np.asarray(sample, dtype=float).ravel()
    n = x.size
    d = _statistic(x, law)
    if d is None:
        return None, None, {"reason": "a degenerate sample has no shape to test"}
    asymptotic = float(KS_C95 / np.sqrt(n))
    crit = bootstrap_critical(law, n)
    detail = {"num_paths": int(n), "law": law.name,
              "ks_critical_asymptotic": asymptotic}
    if crit is None:
        return d, asymptotic, {**detail, "ks_critical_kind": "asymptotic",
                               "ks_critical_note":
                                   f"law {law.name!r} has no sampler, so no "
                                   f"parametric-bootstrap bar could be built; the "
                                   f"asymptotic Kolmogorov value is used and is "
                                   f"known to reject correct solvers on "
                                   f"heavy-tailed laws"}
    return d, crit, {**detail, "ks_critical_kind": "parametric_bootstrap",
                     "ks_bootstrap_draws": int(KS_BOOTSTRAP_DRAWS)}
