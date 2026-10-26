---
id: 1
plan_slug: euler-maruyama-absorbing
scheme: euler-maruyama
scheme_family: euler-maruyama
strategy: Euler-Maruyama on S with absorption at zero (frozen at exactly 0 once a step would cross it), exact log-step for sigma, and call prices reported through the put-call-parity control variate E[S(T)] = S(0).
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

- **Method**: Euler-Maruyama (S) with absorbing truncation; exact log-normal step (sigma)
- **dt**: 0.004 (given by the contract; Nt = max(1, round(T/dt)) = 2500, then dt = T/Nt). Do not hardcode a different step.
- **num_paths**: 1000000 (graded call), plus a second call at 250000 with the same dt, T, seed
- **seed**: 42, via `np.random.default_rng(seed)` only
- **T**: 10.0
- **Milstein correction**: not applicable. Spec has `milstein_eligible: false` (`levy_area`): the two channels do not commute in the S row, and dg_S/dS = sigma/(2 sqrt S) is singular at the absorbing boundary.
- **Expected strong order**: 0.5 (nominal EM; the non-Lipschitz sqrt diffusion and absorbing boundary may degrade it, and an `"unresolved"` order study is expected and costs nothing)
- **Expected weak order**: 1.0 (nominal)
- **Reference for scoring**: none -- self-convergence only (`verification.expected_provenance: self_convergence`). There is no closed form, no closing moment ODE, the Kolmogorov route is scalar-only and this problem is 2-D, the process is not ergodic, and Dynkin is refused by the memory budget at (1e6 paths, 2500 steps). The kernel can measure only the CRN strong-order ladder, the gated constraints and the martingale invariants. Ceiling is the self-convergence band (6-8) unless a cross-family agreement with a sibling plan of a different `scheme_family` is recorded.

### Step (per n = 0..Nt-1), all paths vectorised

```
Z0, Z1      = dW[:, n, 0]/sqrt(dt), dW[:, n, 1]/sqrt(dt)     (or fresh N(0,1) draws)
dW_S        = sqrt(dt)*Z0
dW_sig      = sqrt(dt)*(rho*Z0 + sqrt(1-rho^2)*Z1)
S_trial     = S + sigma*sqrt(S)*dW_S                         (S >= 0 by invariant)
S_new       = where(S > 0 and S_trial > 0, S_trial, 0.0)     (absorb: once 0, stays 0)
sigma_new   = sigma*exp(nu*dW_sig - 0.5*nu^2*dt)             (exact, uses sigma_n)
```

Both updates use the state at the START of the step (S_n, sigma_n). Absorption is permanent: a path with S = 0 is never revived, which is automatic here because the diffusion vanishes at S = 0, but keep an explicit `alive` mask anyway so a future edit cannot introduce reflection.

### Price estimator (put-call parity control variate)

Report

```
price_Ki = (X_0 - Ki) + mean(max(Ki - S_T, 0))
```

not the raw `mean(max(S_T - Ki, 0))`. This uses the exact martingale identity E[S(T)] = X_0 (problem.md states it), so it is the raw estimator with S_T as a control variate at coefficient 1. It is still a genuine Monte-Carlo estimate (it varies with num_paths and seed). Two reasons, both measured by the planner on a 2e5-path probe:

1. **Variance.** S(T) is extremely heavy-tailed (sigma(T) has log-sd 2.53), so the raw call estimator's standard error is ~0.3% of price at 1e6 paths for K3 and the sample is tail-dominated; the bounded put payoff cuts the s.e. by ~4x (to ~0.12% for K3 at 1e6).
2. **Bias.** The truncated EM step is not a martingale -- clipping at 0 pushes E[S] up (probe: E[S(T)] = 0.07045 at dt = 0.004 vs exact 0.07). Through parity that bias would land one-for-one in every call price; the parity estimator removes it.

Also record the raw estimator in the result (e.g. `functionals["raw_price_K1"]` etc. or a side dict) so the evaluator can see the gap, but the graded keys `price_K1..K3` carry the parity values.

### Planner's probe (exploratory, NOT a reference value)

2e5 paths, seed 1, parity estimator, this exact scheme:

| dt | price_K1 | price_K2 | price_K3 | absorbed |
|---|---|---|---|---|
| 0.016 | 0.049643 | 0.042948 | 0.034765 | 0.578 |
| 0.004 | 0.049818 | 0.043180 | 0.035082 | 0.583 |
| 0.001 | 0.049864 | 0.043241 | 0.035146 | 0.583 |

Successive differences shrink ~5x per 4x refinement (roughly first order in dt), so the dt = 0.004 bias is about -0.1% (K1), -0.15% (K2), -0.25% (K3), comfortably inside the 1% tolerance, with MC s.e. per price ~0.11-0.28% at 2e5. Sanity: K1 > K2 > K3, all within [max(X_0-K, 0), X_0].

## Implementation Notes

- **Contract**: `solve_sde(num_paths, dt, T, seed=42, dW=None, observables=None) -> dict`. Return `terminal_paths` shape `(num_paths, 2)` in (S, sigma) order, `empirical_mean` and `empirical_variance` (ddof=1) as lists of 2 floats, `functionals` with float `price_K1`, `price_K2`, `price_K3`, plus `num_paths`, `dt`, `T`. numpy + stdlib only; no work at import.
- **Memory**: never materialise a `(1e6, 2500, 2)` increment array (40 GB). When `dW is None`, draw `rng.standard_normal((num_paths, 2))` once per step from one generator. When the evaluator supplies `dW` (shape `(num_paths, Nt, 2)`, already scaled by sqrt(dt)), use it exactly, take Nt from its shape, set dt = T/Nt, and draw no randomness of your own.
- **Observables hook**: accumulate `0.5*(phi(X_prev) + phi(X_new))*dt` per path with X of shape `(num_paths, 2)`; return under `path_integrals`. (Dynkin is expected to be refused by the budget at graded settings, but the hook must work at the ladder's small settings.)
- **Positivity guard**: compute `sqrt(np.maximum(S, 0.0))`; S must be exactly 0.0 (not a tiny negative or tiny positive) once absorbed. Gated constraints: all finite, `S_T >= 0`, `sigma_T > 0`, shape `(num_paths, 2)`.
- **No reflection**: never use `abs(S_trial)` or `max(S_trial, eps)`; the absorbed state is 0.0 for the rest of the path. The positivity constraint cannot detect reflection, so the reviewer will check `solver.py` for it.
- **sigma**: exact log-Euler step above; do not use an additive EM step for sigma (it can go negative and is far less accurate at log-sd 2.53).
- **Invariants (ungated)**: `mean(S_T)` vs 0.07 (tol 0.00035) -- the truncated EM step will likely miss this by ~0.5% because the scheme is not a martingale; that is a documented scheme property, not a defect, and the parity estimator is precisely what keeps it out of the prices. `mean(sigma_T)` vs 0.34 runs ~1% low from lognormal tail sampling; tail noise.
- **Two path counts**: the 1e6 and 2.5e5 runs share seed 42 and so share the first 2.5e5 path streams only if the per-step draw shape is the same -- it is not (it scales with num_paths), so the estimates naturally differ. Do not cache or constant-fold anything.
- **Runtime**: ~2500 vectorised steps on 1e6 paths; budget roughly 2-3 minutes per call. Use in-place numpy ops where convenient (`np.multiply(..., out=)`), but correctness first.
- **Self-convergence evidence** the solver should report in Results: prices at dt = 0.016, 0.008, 0.004 (and 0.002 if time permits) with the parity estimator at a common seed, plus MC s.e. per price at 1e6 paths, so the dt-bias and the MC error can each be shown to sit well below 1%.

## Results

Run: `solve_sde(1000000, 0.004, 10.0, seed=42)`, about 171 s. Nt = 2500, absorbed fraction 0.5832.

- empirical_mean (S, sigma): [0.070426, 0.332781]
- empirical_variance (S, sigma): [0.018506, 21.255]

| Quantity | 1e6 paths (graded, parity) | 2.5e5 paths (parity) | 1e6 raw estimator |
|---|---|---|---|
| price_K1 | 0.049853 | 0.049832 | 0.050280 |
| price_K2 | 0.043231 | 0.043192 | 0.043657 |
| price_K3 | 0.035146 | 0.035091 | 0.035573 |

- E[S(T)] = 0.070426 vs exact 0.07 (+0.61%): the truncated EM step is not a martingale, as documented. The parity estimator keeps this bias out of the prices; the raw estimator is about 0.9-1.2% higher.
- E[sigma(T)] = 0.3328 vs 0.34 (about 2% low, lognormal tail noise).
- Prices ordered K1 > K2 > K3 and inside [max(X_0-K,0), X_0]. The 1e6 and 2.5e5 estimates differ and agree within sampling noise.
- The dW and observables hooks were exercised (dt = T/Nt from dW.shape[1]; path_integrals shape (1000,)).
- dt self-convergence (parity estimator, common Brownian path via solve_sde(dW=...) built by summing one fine 5000-step increment block; 40 000 paths, so s.e. shown is for 4e4 paths and scales to about 1/5 of that at 1e6):

| dt | price_K1 | price_K2 | price_K3 | absorbed |
|---|---|---|---|---|
| 0.016 | 0.049604 +- 0.000123 | 0.042922 +- 0.000165 | 0.034778 +- 0.000220 | 0.5784 |
| 0.008 | 0.049760 | 0.043119 | 0.035042 | 0.5817 |
| 0.004 | 0.049829 | 0.043203 | 0.035159 | 0.5832 |
| 0.002 | 0.049880 | 0.043270 | 0.035236 | 0.5846 |

  Successive differences shrink roughly 1.5-2x per halving (about first order or slightly slower). The 0.004 -> 0.002 step moves K1/K2/K3 by +0.10%/+0.16%/+0.22%, so the dt = 0.004 bias is about -0.1% to -0.25% (somewhat larger if the rate is below first order). MC s.e. at 1e6 paths is about 0.05% (K1), 0.08% (K2), 0.12% (K3). Both sit inside the 1% tolerance.

## Evaluation

(See review below.)

<metrics>
score: 2
certified: false
provenance: none
estimated_rel_error: 2.22e-14
error_is_estimate: true
observed_order: -0.73            # component 1 (sigma), abstained; component 0 (S) read 0.513 +/- 0.058
order_floor: 0.5
scheme_family: euler-maruyama
invariants_ok: true
constraints_ok: true
invariant_max_drift: 7.22e-03
d1_outcome: unavailable
operator_validated: false
reference_outcome: unavailable
order_check: unresolved
mc_se_rel: 0.1977
resolved: false
dynkin_z: unavailable
resolution_evidence: not_used_as_accuracy_evidence
solve_time_s: 171.3
wall_time_s: 207.1
</metrics>

<review score=2>

**Score: 2/10**

### Numerical Accuracy
- provenance:              none  (reference route: none; moment_ode does not close, kolmogorov has no bounds declared (2-D), stationary_density not ergodic)
- mean_rel_err:            not measured (moments_ok "unavailable": no reference). Reported E[S(T)] = 0.070426 +/- 2 x 0.000136 (exact martingale value 0.07 is inside +0.6%, 3.1 se); E[sigma(T)] = 0.33278 +/- 2 x 0.00461 (exact 0.34, inside the interval)
- variance_rel_err:        not measured (no reference). Var S = 0.018506 +/- 2 x 0.001069; Var sigma = 21.26 +/- 2 x 4.20
- mc_se_rel:               0.1977 (sigma variance, the widest)   resolved: false ("no reference variance to resolve against" -- a statement about the absent reference, not about path count)
- observed_order:          S (component 0): 0.513 +/- 0.058 (expected 0.5, two-sided, orders_ok true). sigma (component 1): abstained, unresolved
- weak_order:              S: 0.745 (used on `x` and `tanh`, not on `x2`); sigma: indeterminate, at the noise floor
- dynkin:                  unavailable (memory budget: 160 GB finest increment array vs 12 GB cap, as the spec predicted). Not a failure, not a pass; it is why the score cannot reach 9
- terminal_law:            not declared
- shape:                   S: skewness 21.4 +/- 10.6, excess kurtosis 3333 +/- 2426; sigma: skewness 134.9 +/- 40.1, excess kurtosis 39098 +/- 18700 (extremely heavy-tailed, as the lognormal with log-sd 2.53 requires)
- constraints:             finite true, shape true, positivity true (drift -0.0), vol_positivity true; martingale (non-gate) true, drift 7.2e-3 against bar 9.2e-3 (headroom 0.106)
- resolution_evidence:     not_used_as_accuracy_evidence
- bound_by:                certification -- "no test in the manual could be run". The rubric cap was 7, and the certification cap of 2 is what binds

### Reading the result
- **This 2 is not a solver defect.** Every check that could run came back clean: no gate violated, S >= 0 and sigma > 0 on every path, the martingale invariant holds, the S-component strong order is 0.513 +/- 0.058 against the nominal 0.5, and `reported_moments_agree` is true (worst relative difference 4.3e-12).
- The kernel recorded `any_test_ran: false` because the only checks that produce evidence of a reference kind (surrogate, Dynkin, manufactured) were all unavailable. The constraints, invariants and order ladder do not count toward that. The score is therefore a property of the spec and kernel, not of `solver.py`.
- **The graded quantities were not measured at all.** `price_K1..K3` are the 1%-tolerance gated quantities, and the kernel has no functional check on the SDE path. The prices below are the solver's own reported values and are *not* verified here: price_K1 0.049853, price_K2 0.043231, price_K3 0.035146 at 1e6 paths (parity estimator), against 0.049832, 0.043192, 0.035091 at 2.5e5. The differences (0.04%, 0.09%, 0.16%) are small, but I have not attached an MC standard error to the prices because the kernel did not report one; treat them as unconfirmed.
- Source review: absorption is genuine. `alive = (S > 0) & (S_trial > 0)` then `np.where(alive, S_trial, 0.0)` sets absorbed paths to exactly 0 with no `abs`, no reflection and no `max(S_trial, eps)`. The `np.maximum(S, 0.0)` is inside the sqrt coefficient (full truncation, endorsed), not a post-step clip of the state. sigma uses the exact log step. The `dW` path consumes `dW[:, n, 0/1]` and takes `Nt = dW.shape[1]`. SOLUTION.md matches solver.py, so no `agent_cap` is warranted.
- Disclosure on the price estimator: `price_Ki = (X_0 - Ki) + mean(max(Ki - S_T, 0))` imposes E[S(T)] = X_0 from the problem statement rather than the simulation's own mean. It is a legitimate control variate, and the raw estimator is also returned (raw prices 0.050280, 0.043657, 0.035573, about 0.9-1.2% higher). That 1% gap is as large as the tolerance, so the choice of estimator decides pass or fail on the graded numbers. It is defensible against the true process (E[S] = 0.07 exactly), but a reviewer should know the grader's reference could be either.

### Formulator findings
- A mean-only moment ODE was deliberately not declared (E[S(T)] = X_0 and the closed E[sigma^k]). That is a reasoned choice and I agree that declaring it would mark `surrogate` on evidence that does not touch the graded prices. No missed reference route found for the prices; the 2-D Kolmogorov solve is the only real reference and the kernel cannot run it.
- `provenance_as_expected: false` (expected self_convergence, reached none) is because the self-convergence route requires a reference or Richardson on a measured quantity, and the Richardson here is on component moments of a tail-dominated variable rather than the prices. Not scored on.

### Feedback for solver
- Nothing in `solver.py` needs to change for correctness. The absorbing step, log-sigma step and `dW` handling are right.
- sigma's order study abstained because the log-Euler step is exact in sigma, so both pathwise differences sit at roundoff (1.7e-15 against a 3.2e-13 floor). That is a property of the estimator, not of the scheme; do not change scheme for it.
- The `dt` self-convergence table from the SOLUTION.md Results section ("not rerun here") is the only discretization evidence for the prices, and it rests on a 2e5-path probe with a different seed. If you want the graded prices to carry evidence, rerun the table at dt = 0.016, 0.008, 0.004, 0.002 on a common `dW` and report each price with its MC s.e.; the kernel cannot do it for you.
- E[S(T)] = 0.070426 is +0.6% off the martingale value (3.1 se). Within the kernel's bar, but it is a known bias of the truncated EM step (about +0.0004). The parity estimator hides it in the prices; it would not be hidden in a reference computed by a non-martingale-corrected scheme.

</review>
