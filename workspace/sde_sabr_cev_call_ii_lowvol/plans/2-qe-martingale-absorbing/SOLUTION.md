---
id: 2
plan_slug: qe-martingale-absorbing
scheme: quadratic-exponential
scheme_family: other
strategy: Andersen-style quadratic-exponential (QE) moment-matched step for S that is an exact conditional martingale and puts a probability atom at the absorbing zero, driven through a Gaussian copula by the correlated increment, with an exact log-step for sigma and put-call-parity pricing.
---

## SDE Reference

Two-factor SABR, Ito, beta = 1/2, zero drift. State X = (S, sigma); column 0 = S (forward), column 1 = sigma (volatility).

```
dS     = sigma * S^(1/2) dW_S,          S(0)     = X_0 = 0.07
dsigma = nu * sigma dW_sigma,           sigma(0) = Y_0 = 0.34
d<W_S, W_sigma> = rho dt
```

| Parameter | Value | Meaning |
|---|---|---|
| X_0 | 0.07 | S(0), initial forward |
| Y_0 | 0.34 | sigma(0), initial volatility |
| beta | 0.5 | CEV exponent |
| nu | 0.8 | vol-of-vol |
| rho | -0.6 | corr(dW_S, dW_sigma) |
| K1 | 0.051022538988 | strike (ITM) |
| K2 | 0.07 | strike (ATM, = X_0) |
| K3 | 0.096035989138 | strike (OTM) |
| T | 10.0 | horizon, interval [0, 10] |

- Drift: `[0, 0]`. S is a forward: no drift, no discounting anywhere.
- Diffusion against INDEPENDENT channels `dW[..., 0]`, `dW[..., 1]`:
  `G = [[sigma*sqrt(max(S,0)), 0], [nu*rho*sigma, nu*sqrt(1-rho^2)*sigma]]`,
  i.e. `dW_S = dW[...,0]`, `dW_sigma = rho*dW[...,0] + sqrt(1-rho^2)*dW[...,1]`.
- Boundary: S = 0 is attainable and ABSORBING (about 58% of paths are absorbed by T = 10). Never negative, never reflected. sigma is a driftless GBM, strictly positive.
- Quantities of interest: undiscounted `price_Ki = E[(S(T) - Ki)^+]`, absorbed paths contribute 0. Graded to 1% relative against an undisclosed reference.

## Numerical Scheme

- **Method**: quadratic-exponential (QE) moment-matched step for S, frozen sigma within the step; exact log-normal step for sigma. Not Euler-Maruyama and not Milstein.
- **dt**: 0.004 (given by the contract; Nt = max(1, round(T/dt)) = 2500, then dt = T/Nt). Do not hardcode a different step.
- **num_paths**: 1000000 (graded call), plus a second call at 250000 with the same dt, T, seed
- **seed**: 42, via `np.random.default_rng(seed)` only
- **T**: 10.0
- **Milstein correction**: not applicable (`milstein_eligible: false`, `levy_area`).
- **Expected strong order**: 0.5 (the spec's only declared floor; `scheme_family: other` selects it). QE is a distributional scheme: on the quadratic branch it is a smooth monotone map of the same Gaussian increment and differs from the EM step by O(sigma^2 dt), so pathwise CRN convergence at ~0.5 is expected; the exponential (near-zero) branch is not pathwise, so the estimator may well report `"unresolved"`, which costs nothing.
- **Expected weak order**: 1.0 (nominal; in practice the dt-bias of the prices is small, see probe)
- **Reference for scoring**: none -- self-convergence only (`verification.expected_provenance: self_convergence`). No closed form, no closing moment ODE, the Kolmogorov route is scalar-only and this problem is 2-D, not ergodic, Dynkin refused by the memory budget at (1e6, 2500). The kernel measures only the CRN order ladder, the gated constraints and the invariants. Ceiling is the self-convergence band (6-8) unless a cross-family agreement with a sibling plan of a different `scheme_family` is recorded -- this plan declares `other` precisely so that such an agreement counts.

### Why this scheme

Conditional on sigma being frozen over [t_n, t_n + dt], dS = sigma_n sqrt(S) dB is a zero-drift CIR (kappa = 0), i.e. a scaled squared Bessel process of dimension 0, whose exact transition is `c * chi'^2_0(lambda)` with `c = sigma_n^2 dt / 4`, `lambda = 4 S_n / (sigma_n^2 dt)`: mean exactly S_n, variance `sigma_n^2 S_n dt`, and an atom at 0 of mass `exp(-lambda/2)`. That is exactly the absorbing behaviour problem.md demands. QE matches the first two conditional moments of that law exactly with a cheap closed-form sampler that can be driven by a single Gaussian, so the correlation with the sigma increment is kept.

### Step (per n = 0..Nt-1), all paths vectorised

```
Z0, Z1   = dW[:, n, 0]/sqrt(dt), dW[:, n, 1]/sqrt(dt)        (or fresh N(0,1) draws)
Z_S      = Z0                                                 (drives S)
Z_sig    = rho*Z0 + sqrt(1-rho^2)*Z1                          (drives sigma)

alive    = S > 0                    (dead paths stay exactly 0.0 forever)
m        = S                        (conditional mean -> exact martingale)
s2       = sigma^2 * S * dt         (conditional variance)
psi      = s2 / m^2 = sigma^2*dt / S

if psi <= psi_c (= 1.5):            quadratic branch
    b2 = 2/psi - 1 + sqrt(2/psi) * sqrt(2/psi - 1)
    a  = m / (1 + b2)
    S_new = a * (sqrt(b2) + Z_S)^2
else:                               exponential branch (mass at 0 = absorption)
    p    = (psi - 1)/(psi + 1)
    beta = (1 - p)/m
    U    = Phi(Z_S)
    S_new = 0                          if U <= p
          = log((1 - p)/(1 - U))/beta  otherwise

sigma_new = sigma * exp(nu*sqrt(dt)*Z_sig - 0.5*nu^2*dt)      (exact; uses sigma_n)
```

All of m, psi, the branch choice and sigma in the S-step use the START-of-step state (S_n, sigma_n). Compute branches only on the masked subsets (`alive & psi<=1.5`, `alive & psi>1.5`) to avoid divide-by-zero on dead paths; dead paths are assigned 0.0.

`Phi` (standard normal CDF) must be numpy-only (no scipy): use the Abramowitz-Stegun 7.1.26 rational approximation of erf (|error| < 1.5e-7), applied as `0.5*(1 + sign(x)*erf(|x|/sqrt 2))`. Clamp `1 - U` away from 0 (e.g. `np.maximum(1-U, 1e-300)`) before the log. Using `U = Phi(Z_S)` (not an independent uniform) is what transmits rho to S on the near-zero branch and keeps the scheme a deterministic function of the supplied increments, which the CRN order study and the two-path-count check both require.

### Price estimator (put-call parity control variate)

Report

```
price_Ki = (X_0 - Ki) + mean(max(Ki - S_T, 0))
```

instead of the raw `mean(max(S_T - Ki, 0))`. It uses the exact martingale identity E[S(T)] = X_0 (stated in problem.md), i.e. S_T as a control variate at coefficient 1, and is still a genuine MC estimate. The bounded put payoff cuts the standard error ~4x versus the tail-dominated raw call (K3 at 1e6 paths: ~0.12% vs ~0.3%). This scheme is already a martingale step by step, so the parity correction here mostly removes MC noise from the heavy S tail rather than scheme bias. Also record the raw estimator under side keys (e.g. `raw_price_K1..K3`) so the evaluator can see both.

### Planner's probe (exploratory, NOT a reference value)

2e5 paths, seed 1, parity estimator, this exact scheme:

| dt | price_K1 | price_K2 | price_K3 | E[S(T)] | absorbed |
|---|---|---|---|---|---|
| 0.016 | 0.049941 | 0.043350 | 0.035297 | 0.07005 | 0.585 |
| 0.004 | 0.049899 | 0.043288 | 0.035226 | 0.07013 | 0.584 |
| 0.001 | 0.049888 | 0.043272 | 0.035190 | 0.07046 | 0.584 |

Prices move by < 0.2% from dt = 0.016 down to 0.001 (approaching from above), and agree with an independent truncated-EM probe extrapolated to dt -> 0 to within ~0.1%. MC s.e. per price ~0.11-0.28% at 2e5. Sanity: K1 > K2 > K3, all within [max(X_0-K, 0), X_0].

## Implementation Notes

- **Contract**: `solve_sde(num_paths, dt, T, seed=42, dW=None, observables=None) -> dict`. Return `terminal_paths` shape `(num_paths, 2)` in (S, sigma) order, `empirical_mean` and `empirical_variance` (ddof=1) as lists of 2 floats, `functionals` with float `price_K1`, `price_K2`, `price_K3`, plus `num_paths`, `dt`, `T`. numpy + stdlib only; no work at import.
- **Memory**: never materialise a `(1e6, 2500, 2)` increment array (40 GB). When `dW is None`, draw `rng.standard_normal((num_paths, 2))` once per step from one generator. When the evaluator supplies `dW` (shape `(num_paths, Nt, 2)`, already scaled by sqrt(dt)), use it exactly, take Nt from its shape, set dt = T/Nt, and draw no randomness of your own (in particular, no separate uniform for the exponential branch).
- **Observables hook**: accumulate `0.5*(phi(X_prev) + phi(X_new))*dt` per path with X of shape `(num_paths, 2)`; return under `path_integrals`.
- **Absorption**: once S = 0.0 it is exactly 0.0 for the rest of the path (explicit `alive` mask). Never reflect, never floor at eps. Gated constraints: all finite, `S_T >= 0` (guaranteed by construction: both branches are non-negative), `sigma_T > 0`, shape `(num_paths, 2)`.
- **Large-S edge**: for very small psi (large S, small sigma) `b2` is large and the quadratic branch is a near-Gaussian step; no special handling needed. For psi -> infinity (S tiny, sigma huge) p -> 1, so the path is absorbed with probability -> 1; also fine. Guard only the divide by S on dead paths.
- **sigma**: exact log step; do not use an additive EM step.
- **Invariants (ungated)**: `mean(S_T)` vs 0.07 (tol 0.00035) should pass up to MC noise, since every step preserves the conditional mean exactly -- but S_T is heavy-tailed, so a ~0.5% deviation at 2e5 paths is noise (probe saw 0.07046 at dt = 0.001). `mean(sigma_T)` vs 0.34 runs ~1% low from lognormal tail sampling.
- **Two path counts**: 1e6 and 2.5e5 at the same seed give different draws (per-step draw shape scales with num_paths), so the estimates differ naturally. No caching.
- **Runtime**: roughly 2x plain EM per step (branch masks + Phi); expect ~4-5 minutes per 1e6-path call. Precompute masks once per step and reuse; work on index subsets.
- **Self-convergence evidence** the solver should report in Results: prices at dt = 0.016, 0.008, 0.004 (and 0.002 if time permits) at a common seed with the parity estimator, plus MC s.e. per price at 1e6 paths and `mean(S_T)`, so the dt-bias and the MC error can each be shown to sit well below 1%.

## Results

Graded call: `solve_sde(1000000, 0.004, 10.0, seed=42)` (Nt = 2500, about 191 s).

- empirical_mean = [0.070062, 0.332781]  (E[S_T] vs 0.07: +0.09%, within tol 0.00035; E[sigma_T] ~2% low, lognormal tail noise)
- empirical_variance (ddof=1) = [0.018479, 21.2552]
- absorbed fraction = 0.5851; S_T >= 0 and sigma_T > 0 on every path; absorption via explicit alive mask

Prices (parity estimator), 1e6 paths, dt = 0.004:

| | K1 | K2 | K3 |
|---|---|---|---|
| price (parity) | 0.049941 | 0.043346 | 0.035297 |
| raw E[(S-K)^+] | 0.050002 | 0.043408 | 0.035358 |
| MC s.e. (parity) | 2.4e-5 (0.05%) | 3.3e-5 (0.08%) | 4.4e-5 (0.12%) |

Path-count check (same dt, T, seed), 2.5e5 paths: 0.049915 / 0.043304 / 0.035243. Differences from 1e6 are 0.5-1.2 combined-s.e.-ish scale (<0.16%), so they differ and agree within noise.

Self-convergence in dt (1e6 paths, seed 42, parity estimator):

| dt | K1 | K2 | K3 | mean(S_T) |
|---|---|---|---|---|
| 0.016 | 0.049942 | 0.043350 | 0.035308 | 0.07020 |
| 0.008 | 0.049932 | 0.043333 | 0.035279 | 0.07011 |
| 0.004 | 0.049941 | 0.043346 | 0.035297 | 0.07006 |

Across dt = 0.016 -> 0.004 the prices move by < 0.04% (K1, K2) and < 0.1% (K3), at or below MC noise (the ladders use distinct per-step draw shapes only via dt, so differences include MC noise); both dt-bias and MC error sit well below the 1% tolerance. Hooks verified: explicit `dW` (num_paths, Nt, 2) uses Nt = dW.shape[1]; `observables` returns `path_integrals` of shape (num_paths,).

## Evaluation

(Filled by evaluator)

<metrics>
score: 2
certified: false
provenance: none
estimated_rel_error: 2.341e-14
error_is_estimate: true
observed_order: -0.8375
order_floor: 0.5000
scheme_family: other
invariants_ok: true
constraints_ok: true
invariant_max_drift: 7.219e-03
invariant_worst_headroom: 0.1063
order_check: unresolved
mc_se_rel: 0.1977
resolved: false
ci_mult: 2.0000
resolution_evidence: not_used_as_accuracy_evidence
solver_sha256: b1f753c9b635cdd4ab29ac1d0f042cb1206228418a7a9f297416d027dbe94b9f
solve_time_s: 198.8020
wall_time_s: 231.1400
</metrics>

<review score=2>

**Score: 2/10**

The 2 is a statement about the evidence, not about the solver. The kernel could run no test that measures accuracy against a reference. The solver's constraints and invariants are clean.

### Numerical Accuracy
- provenance:              none  (reference route: none. moment_ode, kolmogorov and stationary_density were all attempted and all unavailable)
- mean_rel_err:            not computed, no reference. Empirical mean = [0.070062, 0.332781], se = [1.36e-4, 4.61e-3] (ci_mult 2.0)
- variance_rel_err:        not computed, no reference. Empirical variance = [0.018479, 21.2552], se = [1.18e-3, 4.20] (ci_mult 2.0)
- mc_se_rel:               0.1977 (the sigma-column variance, 2 x se / var is about 40%)   resolved: false. This is the `unavailable` form (no reference variance to resolve against), not an MC-inconclusive verdict
- observed_order:          -0.84 +/- 0.094 (expected 0.5, two-sided band). Aggregate is `unresolved` and is a property of the estimator, see below. The S column alone reads 0.453 +/- 0.049 (2 x se = 0.10) against 0.5, inside the 0.4 band. The sigma column reads -0.84 at roundoff scale
- weak_order:              indeterminate, at the noise floor (all three test functions rejected)
- dynkin:                  unavailable. The ladder's finest level needs a 160 GB increment array against the 12 GB budget (1e6 paths x 10000 steps x 2 channels), as the spec predicted. Replication not applicable (one seed, no verdict to replicate)
- terminal_law:            not declared
- shape:                   S: skewness 23.8 +/- 12.5, excess kurtosis 4.1e3 +/- 3.0e3. sigma: skewness 134.9 +/- 40.1, excess kurtosis 3.9e4 +/- 1.9e4. Extremely heavy-tailed, so the sample variance of sigma is unreliable (se about 20% of the value)
- constraints:             finite pass, shape pass, positivity (S >= 0) pass with `neg_fraction` 0, vol_positivity pass. Gates clean
- invariants:              the kernel's single reported `martingale` entry (drift 7.219e-3, bar 9.221e-3 = 2 x se, headroom 0.106) matches the sigma column: |0.332781 - 0.34| = 7.22e-3 against 2 x 4.61e-3. It passes inside its interval. The two same-named entries appear to collapse into one key, so the S martingale is not separately itemised. By my own arithmetic from the reported moments, E[S_T] = 0.070062 vs 0.07 is +6.2e-5, which is 0.45 se (se 1.36e-4) and inside the spec's tol of 3.5e-4, but that is not a kernel verdict
- reported_moments_agree:  true (worst relative difference 4.8e-12)
- provenance_as_expected:  false (expected self_convergence, achieved none). Diagnostic only, never scored on
- resolution_evidence:     not_used_as_accuracy_evidence
- bound_by:                certification, "no test in the manual could be run". The rubric alone would have allowed 7 ("the order study reported unresolved and one of Dynkin, the constraints or the Richardson stability did not hold")

### What the 2 does and does not say
- The graded quantities are the three call prices, graded to 1% against an undisclosed reference. The kernel does not evaluate `functionals` on the SDE path, so none of price_K1..K3 was checked. Nothing here says the prices are wrong.
- The solver's own self-convergence table in SOLUTION.md (prices moving under 0.1% across dt = 0.016 to 0.004, standard errors 0.05 to 0.12%) is the plan's own evidence. It is not kernel-measured and does not enter the score.
- No agent_cap was applied. `solver.py` implements the QE scheme the plan describes. Absorption is an explicit `alive` mask (idx = flatnonzero(S > 0)), dead paths are written as exactly 0.0, and there is no reflection and no `X = np.maximum(X, 0)` statement. `dW` is consumed exactly (Nt from `dW.shape[1]`, Z = dW / sqrt(dt)), and the exponential branch uses `Phi(Z_S)` instead of an independent uniform. I saw no defect that justifies a cap.

### Why the order study abstained (property of the estimator, not of the solver)
`detail.orders_ok.abstained` reports that both pathwise differences are at the roundoff scale of the terminal values. This is the sigma component. The scheme advances sigma by the exact log-step `sigma * exp(nu*sqrt(dt)*Z_sig - 0.5*nu^2*dt)` and sigma(T) is a deterministic function of the summed increments, so refining dt changes nothing beyond arithmetic (s0 = 1.7e-15 against a floor of 3.2e-13). The kernel takes the aggregate order from the worst column, so the clean S result (0.453, `orders_ok: true` per component) is not what `orders_ok` shows. This is not a request for a different scheme, and the abstention costs the plan nothing by design. It is reported because it hides a real S-column measurement, and the kernel reports the aggregate headline as `unresolved` / `order_check: unresolved`.

### Feedback for solver
- None of the score-relevant findings here are solver defects, and I have no code change to request. `solver.py` consumes `dW`, absorbs correctly and passes every gated constraint.
- Nothing can lift the plan above 2 on this spec as it stands. Lifting needs one of: (a) a Dynkin verdict, which the 160 GB ladder at (1e6, 0.004) prevents; (b) a cross-family agreement with a sibling plan of a different `scheme_family` on the independent-seed comparison (`cli.py compare`), which this plan was designed to allow by declaring `other`; (c) a spec change by the formulator. Run `compare` against a plan using a different family (for example truncated EM) once one exists.
- The reported martingale headroom of 0.106 belongs to the sigma column (E[sigma_T] = 0.3328 vs 0.34), which is lognormal tail noise as the spec's own note says. E[S_T] = 0.070062 (+0.09%, 0.45 se) shows no bias. No action needed. If the plan wants to show this, report E[S_T] at several seeds in Results.
- Keep the parity estimator, but note it is not scored. Report both `price_Ki` and the raw estimators, as the code already does.

### Formulator defects to report back
- **A usable reference was not declared, or the kernel cannot use it.** `moment_ode`, `kolmogorov` and `stationary_density` are all absent or disabled, so the problem is Path B with no surrogate. The spec's notes say that Kolmogorov is 2-D and not wired, and that the moment hierarchy does not close, and I agree with both. I found no reference that was available and merely undeclared. However, two model-free facts are exact and could give a real reference with no Dynkin run: `E[S_T] = X_0` and the lognormal `sigma_T` law. The `functional_sanity_bounds` (price_K1 > K2 > K3, convexity in K, and the bound max(X_0 - K, 0) <= price <= X_0) are also checkable, but the kernel does not evaluate them. This is a kernel and spec limitation, not something the solver can fix.
- **`dynkin_note` is correct, and the ceiling is set by the kernel.** The spec says that Dynkin will be refused at the contract's settings, which is what happened. As a result, every correct plan on this problem is bounded at 2 to 7 regardless of quality. The formulator or the kernel owner may want a reduced-path Dynkin (it widens the interval, which is why the kernel refuses) or a price-functional evaluator wired into the SDE path.
- **Kernel aggregation quirk.** An exactly-integrated column (sigma) forces the aggregate `orders_ok` to `unresolved` even when another column measures cleanly. Not a formulator defect, noted for the kernel owner.

</review>
