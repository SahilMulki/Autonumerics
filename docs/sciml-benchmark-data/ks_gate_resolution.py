"""§10a of plan-sciml-benchmark.md: can the terminal-law KS gate rank SDE schemes?

The first step-10 pipeline run on M01 turned on this one statistic. Three plans were
built -- Euler-Maruyama, Milstein and a Stratonovich Heun -- and the two *better* ones
were the two that failed `terminal_law_ok`, so the conductor shipped the worst of its
own three. This script asks whether that outcome carries information about the schemes
or only about the draw.

It reproduces the kernel's own test exactly: `verifylib/kernel/sde/laws.ks_standardized`
standardizes the sample by its own mean and sd and the lognormal law by its exact ones,
then compares in sup-norm against `KS_C95 / sqrt(n)` with `KS_C95 = 1.3581`. The
standardization is deliberate and documented there -- it removes the location and scale
bias that an O(dt) weak error mostly *is*, leaving shape. The consequence measured here
is that what remains is dominated by the sample, not by the scheme.

Four solvers share one Brownian path per seed (common random numbers), so every
difference below is the scheme's:

    exact       X_T = X_0 exp((mu - sigma^2/2) T + sigma W_T)   -- zero pathwise error
    em          Euler-Maruyama,                     strong order 0.5
    milstein    + 0.5 sigma^2 X (dW^2 - dt),        strong order 1.0
    heun        Stratonovich drift + trapezoid,     strong order 1.0

Reported per seed: each solver's terminal relative strong error (the quantity M01's
problem.md actually grades) beside its KS D. The two orderings are then compared.

M01's grading configuration, from workspace/.../problem_spec.json: n = 50 000 paths,
dt = 0.0016 (625 steps), T = 1, mu = 1.01, sigma = 0.87, X_0 = 0.5.

numpy + scipy only, ~5 minutes. Seeded, so the printed numbers are the ones in §10a,
and they are also written to `measurements/ks_gate_resolution.json` so
`audit_plan_claims.py` can hold the prose to them.
"""

import json
import pathlib

import numpy as np
from scipy.stats import norm

X0, MU, SIGMA, T = 0.5, 1.01, 0.87, 1.0
DT, NSTEPS = 0.0016, 625
NPATHS = 50_000
KS_C95 = 1.3581
SEEDS = (42, 1, 2, 3, 4, 5, 6, 7, 8)

LOG_MEAN = np.log(X0) + (MU - 0.5 * SIGMA**2) * T
LOG_SD = SIGMA * np.sqrt(T)
LAW_MEAN = float(np.exp(LOG_MEAN + 0.5 * LOG_SD**2))
LAW_SD = float(np.sqrt((np.exp(LOG_SD**2) - 1.0) * np.exp(2 * LOG_MEAN + LOG_SD**2)))


def ks_standardized(x):
    """The kernel's statistic, verbatim in effect: shape only, both sides standardized."""
    x = np.asarray(x, dtype=float)
    n = x.size
    z = np.sort((x - x.mean()) / x.std(ddof=1))
    reference = norm.cdf(np.log(np.maximum(LAW_MEAN + LAW_SD * z, 1e-300)),
                         loc=LOG_MEAN, scale=LOG_SD)
    reference = np.where(LAW_MEAN + LAW_SD * z > 0.0, reference, 0.0)
    empirical = np.arange(1, n + 1, dtype=float) / n
    d = float(max(np.max(np.abs(empirical - reference)),
                  np.max(np.abs(reference - (empirical - 1.0 / n)))))
    return d, float(KS_C95 / np.sqrt(n))


def run_one_seed(seed):
    # One (paths, steps) block, which is how every graded solver in this repository
    # draws its own increments -- `rng.standard_normal((num_paths, Nt))`. Drawing
    # per step instead consumes the same stream in a different order and realizes
    # different paths, so the seed-42 row would no longer correspond to the run.
    rng = np.random.default_rng(seed)
    dW = rng.standard_normal((NPATHS, NSTEPS)) * np.sqrt(DT)
    em = np.full(NPATHS, X0)
    mil = np.full(NPATHS, X0)
    heun = np.full(NPATHS, X0)
    w = np.zeros(NPATHS)
    strat_drift = MU - 0.5 * SIGMA**2          # Stratonovich-corrected drift
    for n in range(NSTEPS):
        dw = dW[:, n]
        w += dw
        em = em + MU * em * DT + SIGMA * em * dw
        mil = (mil + MU * mil * DT + SIGMA * mil * dw
               + 0.5 * SIGMA**2 * mil * (dw * dw - DT))
        pred = heun + strat_drift * heun * DT + SIGMA * heun * dw
        heun = (heun + 0.5 * strat_drift * (heun + pred) * DT
                + 0.5 * SIGMA * (heun + pred) * dw)
    exact = X0 * np.exp((MU - 0.5 * SIGMA**2) * T + SIGMA * w)
    del dW

    out = {}
    denom = float(np.sqrt(np.mean(exact**2)))
    for name, x in (("exact", exact), ("em", em), ("milstein", mil), ("heun", heun)):
        d, crit = ks_standardized(x)
        err = float(np.sqrt(np.mean((x - exact) ** 2))) / denom
        out[name] = {"ks": d, "crit": crit, "ks_ok": d <= crit, "rel_err": err,
                     # The divisor the standardization uses, against the law's exact
                     # one. This is the term the test cannot remove: see main().
                     "sd_ratio": float(np.std(x, ddof=1)) / LAW_SD - 1.0}
    return out


def main():
    print(f"n = {NPATHS}, dt = {DT} ({NSTEPS} steps), T = {T}, "
          f"KS critical = {KS_C95 / np.sqrt(NPATHS):.6f}\n")
    names = ("exact", "em", "milstein", "heun")
    rows = []
    for seed in SEEDS:
        r = run_one_seed(seed)
        rows.append((seed, r))
        tag = "  <-- the pipeline's own seed" if seed == SEEDS[0] else ""
        print(f"--- seed {seed}{tag}")
        for n_ in names:
            v = r[n_]
            print(f"   {n_:<9} rel_err={v['rel_err']:.3e}  KS={v['ks']:.6f}  "
                  f"{'pass' if v['ks_ok'] else 'FAIL':<4} ({v['ks'] / v['crit']:.3f}x crit)")

    # Does KS rank the schemes the way their pathwise error does?
    graded = ("em", "milstein", "heun")
    monotone = 0
    inverted = 0
    em_passes_better_fail = 0
    exact_fails = []
    for seed, r in rows:
        by_err = sorted(graded, key=lambda k: r[k]["rel_err"])
        by_ks = sorted(graded, key=lambda k: r[k]["ks"])
        if by_err == by_ks:
            monotone += 1
        if by_ks == list(reversed(by_err)):
            inverted += 1
        if r["em"]["ks_ok"] and not r["milstein"]["ks_ok"] and not r["heun"]["ks_ok"]:
            em_passes_better_fail += 1
        if not r["exact"]["ks_ok"]:
            exact_fails.append(seed)

    k = len(rows)
    print(f"\nover {k} seeds:")
    print(f"   KS order == pathwise-error order        : {monotone}/{k}")
    print(f"   KS order == exactly reversed            : {inverted}/{k}")
    print(f"   EM passes while both better schemes fail: {em_passes_better_fail}/{k}"
          "   <-- the configuration the run produced")
    ks_exact = [r["exact"]["ks"] for _, r in rows]
    print(f"   the *exact* sampler's own KS            : {min(ks_exact):.4f}-{max(ks_exact):.4f}"
          f" against {KS_C95 / np.sqrt(NPATHS):.4f}")
    print(f"   seeds where a pathwise-exact solver FAILS the gate: "
          f"{exact_fails if exact_fails else 'none'}")
    spread = [abs(r["milstein"]["ks"] - r["em"]["ks"]) for _, r in rows]
    print(f"   |KS(milstein) - KS(em)|                 : {min(spread):.5f}-{max(spread):.5f}"
          f"  (critical value {KS_C95 / np.sqrt(NPATHS):.5f})")

    # Why. Standardizing removes a location and scale *bias*, but it divides by an
    # estimate, and on this law that estimate is the noisy quantity. The relative
    # standard error of a sample sd is sqrt((kurtosis - 1) / 4n); M01's terminal
    # lognormal has excess kurtosis 47.65, i.e. kurtosis 50.65, so at n = 50 000 the
    # divisor itself carries ~1.6% error. A random scale error of that size is not
    # removed by the standardization -- it *is* the standardization -- and it moves the
    # standardized CDF by the same order as the bar the test compares against.
    kurt = (np.exp(LOG_SD**2) ** 4 + 2 * np.exp(LOG_SD**2) ** 3
            + 3 * np.exp(LOG_SD**2) ** 2 - 6.0) + 3.0
    predicted = float(np.sqrt((kurt - 1.0) / (4 * NPATHS)))
    observed = [abs(r["exact"]["sd_ratio"]) for _, r in rows]
    print("\nthe mechanism -- the divisor, not the scheme:")
    print(f"   kurtosis of the terminal law            : {kurt:.2f}")
    print(f"   predicted rel. SE of the sample sd      : {predicted:.4%}"
          "   = sqrt((kurt - 1) / 4n)")
    print(f"   observed |sample sd / law sd - 1|, exact: "
          f"{min(observed):.4%}-{max(observed):.4%}")
    corr = float(np.corrcoef([r["exact"]["ks"] for _, r in rows],
                             [abs(r["exact"]["sd_ratio"]) for _, r in rows])[0, 1])
    print(f"   corr(KS, |sd error|) over the exact rows : {corr:+.3f}"
          "   <-- the statistic is reading its own divisor")

    out = pathlib.Path(__file__).resolve().parent / "measurements" / "ks_gate_resolution.json"
    out.write_text(json.dumps({
        "config": {"num_paths": NPATHS, "dt": DT, "nsteps": NSTEPS, "T": T,
                   "X0": X0, "mu": MU, "sigma": SIGMA, "ks_c95": KS_C95,
                   "ks_critical": KS_C95 / np.sqrt(NPATHS), "seeds": list(SEEDS),
                   "increments": "one (paths, steps) block per seed, as graded solvers draw"},
        "per_seed": {str(seed): r for seed, r in rows},
        "readings": {
            "ks_order_matches_error_order": monotone,
            "ks_order_reversed": inverted,
            "em_passes_while_better_fail": em_passes_better_fail,
            "seeds": k,
            "exact_solver_gate_failures": exact_fails,
            "exact_ks_min": min(ks_exact), "exact_ks_max": max(ks_exact),
            "between_scheme_spread_min": min(spread),
            "between_scheme_spread_max": max(spread),
            "law_kurtosis": kurt,
            "predicted_sd_rel_se": predicted,
            "corr_ks_vs_sd_error": corr,
        },
    }, indent=1, sort_keys=True) + "\n")
    print(f"\nwritten: {out.relative_to(pathlib.Path(__file__).resolve().parents[2])}")


if __name__ == "__main__":
    main()
