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

Each law also states its own exact standardized third and fourth moments, which is
what lets 3b's cheap tier grade rather than only report (plan §11 D11): declaring
the law once serves both tiers, and separate ``skewness_value_at_T`` fields would be
a second source of truth for the same fact.
"""

from __future__ import annotations

import numpy as np

from ...operator import evaluate

#: Asymptotic KS critical value at alpha = 0.05, as ``c / sqrt(n)``. 1.3581 is the
#: standard two-sided constant; the sample sizes here (20 000-200 000) are far past
#: where the asymptotic form matters.
KS_C95 = 1.3581


def _number(value, params):
    return float(evaluate(value, params)) if isinstance(value, str) else float(value)


class Law:
    """One terminal law: its first four moments and its CDF.

    ``standardized_cdf`` is the CDF of ``(X - mean)/sd``, which is what the test
    compares the standardized sample against.
    """

    __slots__ = ("name", "mean", "sd", "skewness", "excess_kurtosis", "_cdf")

    def __init__(self, name, mean, sd, skewness, excess_kurtosis, cdf):
        self.name, self.mean, self.sd = name, float(mean), float(sd)
        self.skewness, self.excess_kurtosis = float(skewness), float(excess_kurtosis)
        self._cdf = cdf

    def standardized_cdf(self, z):
        return self._cdf(self.mean + self.sd * np.asarray(z, dtype=float))


def _normal(entry, params):
    mean = _number(entry.get("mean", 0.0), params)
    variance = _number(entry["variance"], params)
    sd = float(np.sqrt(variance))
    from scipy.stats import norm
    return Law("normal", mean, sd, 0.0, 0.0, lambda x: norm.cdf(x, loc=mean, scale=sd))


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
                                           loc=mu, scale=s), 0.0))


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
               lambda x: ncx2.cdf(np.maximum(np.asarray(x), 0.0) / scale, df, nc))


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


def ks_standardized(sample, law):
    """``(D, crit, detail)`` -- the sup-norm distance in **shape** only.

    The sample is standardized by its own mean and standard deviation and the law by
    its exact ones, so a location or scale error -- which is what an ``O(dt)`` weak
    error mostly is -- cancels on both sides and the statistic measures the
    remaining shape mismatch.
    """
    x = np.asarray(sample, dtype=float).ravel()
    n = x.size
    sd = float(np.std(x, ddof=1))
    if n < 2 or sd <= 0.0:
        return None, None, {"reason": "a degenerate sample has no shape to test"}
    z = np.sort((x - float(np.mean(x))) / sd)
    reference = np.asarray(law.standardized_cdf(z), dtype=float)
    empirical = np.arange(1, n + 1, dtype=float) / n
    d = float(max(np.max(np.abs(empirical - reference)),
                  np.max(np.abs(reference - (empirical - 1.0 / n)))))
    crit = float(KS_C95 / np.sqrt(n))
    return d, crit, {"num_paths": int(n), "law": law.name}
