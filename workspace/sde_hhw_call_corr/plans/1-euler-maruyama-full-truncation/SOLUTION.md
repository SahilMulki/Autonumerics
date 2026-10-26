---
id: 1
plan_slug: euler-maruyama-full-truncation
scheme: euler-maruyama
strategy: Vectorised Euler-Maruyama with full truncation for V, log-Euler for S, Euler for R, a left-point discount sharing R_n with the S drift, and a control variate whose mean is exact for the discrete scheme.
expected_strong_order: 0.1
expected_weak_order: 0.5
---

<!--
scheme_family is deliberately OMITTED. problem_spec.json declares a single scalar
expected_strong_order = 0.1 (set by S and V, measured 0.04 / 0.11), and the kernel takes the worst
component. Declaring scheme_family switches the order guard to a two-sided +/-0.4 band around 0.1,
and the R component of a correct solver measures ~1.0, so it would fail
(verification.expected_strong_order_note, implementation_notes "KERNEL CEILINGS (3)").
Omitting it leaves the one-sided band, which a correct solver passes on all three components.
-->

## SDE Reference

Itô SDE on t in [0, T], T = 10. State X = (S, V, R), columns in that order:

    dS = R*S dt + sqrt(V)*S dW1,              S(0) = S0
    dV = kappa*(eta - V) dt + sigma_v*sqrt(V) dW2,  V(0) = V_0
    dR = alpha*(r_bar - R) dt + sigma_r dW3,   R(0) = R_0

| Parameter | Value |
|---|---|
| kappa | 0.5 |
| eta | 0.04 |
| sigma_v | 1.0 |
| alpha | 0.08 |
| r_bar | 0.10 |
| sigma_r | 0.09 |
| rho_SV = corr(dW1, dW2) | -0.9 |
| rho_SR = corr(dW1, dW3) | 0.6 |
| rho_VR = corr(dW2, dW3) | -0.7 |
| K (strike) | 100.0 |
| S0 for prices | 75, 100, 125 |
| S_0 for terminal_paths | 100 |
| V_0 | 0.04 |
| R_0 | 0.10 |

Feller ratio 2*kappa*eta/sigma_v^2 = 0.04: V sits at zero for a large fraction of the time.

Correlation: independent channels (Y1, Y2, Y3) are mapped to correlated increments with the
lower-triangular Cholesky factor of the correlation matrix (rows S, V, R):

    Lc = [[ 1.0,           0.0,           0.0         ],
          [-0.9,           0.4358898944,  0.0         ],
          [ 0.6,          -0.3670651742,  0.7108186533]]

    dW1 = Y1
    dW2 = -0.9*Y1 + 0.4358898944*Y2
    dW3 =  0.6*Y1 - 0.3670651742*Y2 + 0.7108186533*Y3

Quantity of interest (returned in `functionals`):

    price_S{75,100,125} = E[ exp(-int_0^T R dt) * max(S_T - 100, 0) ]   from S0 = 75, 100, 125

each must be within 0.3% relative of an independent reference.

## Numerical Scheme

- **Method**: Euler-Maruyama (full-truncation Euler for V, log-Euler for S, Euler for R)
- **dt**: 0.004 (given by the caller; `Nt = max(1, round(T/dt)) = 2500`, `dt = T/Nt`; never hardcode another step)
- **num_paths**: 1,000,000 (graded), then 250,000 (path-count agreement check)
- **seed**: 42 (`np.random.default_rng(seed)` only)
- **T**: 10.0
- **Milstein correction**: not applicable (multi-D, non-commuting noise; Lévy areas required)
- **Expected strong order**: 0.1 (spec-declared; binding components S ~0.04 and V ~0.11; R ~1.0; nominal Euler-Maruyama order 0.5 is destroyed by non-Lipschitz sqrt(V) at Feller ratio 0.04)
- **Expected weak order**: 0.5 (spec's conservative diagnostic value; nominal EM weak order 1.0)
- **Reference for scoring**: none in the kernel — self-convergence only (`expected_provenance: self_convergence`). No moment ODE, Kolmogorov solve, stationary density or terminal law is declared, so the score band tops out at 6–8. Hand-checkable anchors in `verification.exact_identities` (see below).

### Update (per step n, with V+ = max(V_n, 0))

    sqV   = sqrt(V+)
    logS += (R_n - 0.5*V+)*dt + sqV*dW1
    V     = V_n + kappa*(eta - V+)*dt + sigma_v*sqV*dW2        # full truncation: drift AND diffusion use V+
    logD -= R_n*dt                                             # left-point, SAME R_n as in the S drift
    R     = R_n + alpha*(r_bar - R_n)*dt + sigma_r*dW3

V itself is stored un-floored (it goes negative on many paths; that is a property of full truncation,
not a bug; `variance_nonnegative` is informational only). terminal_paths column V is the raw scheme
state V_N.

Why full truncation: among the explicit Euler fixes (absorption, reflection, partial truncation,
full truncation) it is the one Lord, Koekkoek and van Dijk (2010) found least biased, it consumes
dW2 directly (so the kernel's CRN ladder is clean), and it is cheap enough for the 300 s sandbox.

### One simulation serves all three spots

V, R and D do not depend on S0, and with the log-S update S_T(S0) = (S0/100) * S_T(100) exactly. Simulate
once from S0 = 100 and form the three payoffs as `D * max(c*S_T - 100, 0)` with c = 0.75, 1.0, 1.25.

### Control variate (unbiased for the discrete scheme)

Because the S drift and the discount use the same R_n,

    D_N * S_N = S0 * prod_n exp(-0.5*V+_n*dt + sqrt(V+_n)*dW1_n)

is an exact discrete martingale, so E[D_N S_N] = S0 holds exactly for this scheme. Use, per spot,

    Y = D*max(S_T - K, 0) - b1*(D*S_T - S0) [- b2*(D - P_disc)]

with b estimated by least squares on the same sample. The bond control is optional: if used, its mean
must be the **discrete** E[D] (not the continuous Vasicek P(0,T) = 0.79684, which is matched only to
O(dt) and would import bias). Since R_n is Gaussian and sum_n R_n dt is a linear functional of it,
E[D_disc] = exp(-m + v/2) with m, v obtained exactly from the AR(1) recursion
R_{n+1} = (1 - alpha*dt) R_n + alpha*r_bar*dt + sigma_r*dW3_n (a deterministic O(Nt) computation).
The spec measured a 25–35% standard-deviation reduction from these controls.

## Implementation Notes

- **Contract**: `solve_sde(num_paths, dt, T, seed=42, dW=None, observables=None) -> dict` returning
  `terminal_paths` (num_paths, 3) for S0 = 100 in column order (S, V, R); `empirical_mean` and
  `empirical_variance` (ddof=1) as lists of 3 floats; `functionals` with float `price_S75`,
  `price_S100`, `price_S125`; plus `num_paths`, `dt`, `T`. numpy + stdlib only; no work at import.
- **dW hook**: when `dW` is supplied it has shape (num_paths, Nt, 3) of INDEPENDENT increments
  already scaled by sqrt(dt); correlate inside the solver with Lc exactly as above; take Nt from the
  shape and draw no randomness. **observables hook**: accumulate `int phi(X) dt` by trapezoid, with
  phi applied to the (num_paths, 3) state.
- **Memory**: the full increment block at the graded size is 60 GB. Never materialise it: draw a
  (chunk, 3) normal block per step (or loop over path chunks). Keep S as log S, and S, V, R, logD as
  flat float64 arrays updated in place.
- **Wall time**: the spec measured ~2.4e-7 s per path-step for a naive vectorised scheme (~600 s at
  the graded size) against a 300 s sandbox. Required: in-place numpy ops (`out=`), one sqrt per step,
  correlate with three fused multiply-adds instead of a matmul, avoid recomputing exp(logS) inside
  the loop (exponentiate once at T), and pick a path-chunk size that keeps the working set in cache
  (e.g. 50k–200k paths per chunk, all 2500 steps per chunk). Measure the 1e6 run before handing over.
- **Square roots**: every sqrt(V) is sqrt(max(V, 0)). R is Gaussian and goes negative (stationary sd
  ~0.2); do **not** floor R.
- **Discount**: accumulate logD along each path; never apply exp(-r*T) with a constant rate.
- **Path-count check**: the 1e6 and 250k runs must give different values that agree within MC noise.
  Use the same seed but do not draw in a way that makes the 250k run a bit-identical prefix whose
  prices coincide — they will naturally differ; just never return a constant.
- **Accuracy budget**: plain-MC relative s.e. ~0.08% at 1e6 and ~0.15% at 250k; that leaves only
  ~0.1–0.15% for time-discretisation bias inside the 0.3% band. Full-truncation bias at Feller ratio
  0.04 and dt = 0.004 is the main risk of this plan; the solver should estimate it by running the
  same seed at dt = 0.016 / 0.008 / 0.004 (self-convergence of the three prices) and report the
  apparent drift with dt in Results. dt itself cannot be changed for the graded call.
- **Self-checks to report** (from `verification.exact_identities`):
  - E[D S_T] = S0 (should hold to MC noise, exactly in expectation for this scheme)
  - E[D] vs Vasicek P(0,T) = 0.7968400987842249 (agrees to O(dt) only) and vs the discrete E[D]
  - put-call parity C - P = S0 - K*E[D]: -4.68401 / 20.31599 / 45.31599 (continuous P)
  - no-arbitrage bounds max(S0 - K P, 0) <= C <= S0; price_S75 < price_S100 < price_S125 and
    price_S100 <= (price_S75 + price_S125)/2
  - E[V_T] = 0.04, Var[V_T] = 0.0399981840028095, E[R_T] = 0.10, Var[R_T] = 0.04040398877652057
  - Black-Scholes degenerate limit (sigma_v = sigma_r = 0, via a module-level parameter dict):
    40.305614268969904 / 64.11045523165186 / 88.62935813769187 within 0.3%
- **Kernel ceilings** (from the spec): provenance will be `self_convergence`; Dynkin at 1e6 paths
  exceeds the 12 GB budget and reports `unavailable`; the functionals are not kernel-graded on the
  SDE path, so the evaluator checks them and the two-path-count agreement by hand.

## Results

Run: `solve_sde(1_000_000, 0.004, 10.0, 42)` (40 s wall, 4 threads, 50k-path chunks, per-chunk
SeedSequence streams from `default_rng` seed 42; dW/observables hooks verified).

- empirical_mean (S, V, R): 634.920, 0.039403, 0.099827
- empirical_variance (ddof=1): 1.7645e6, 0.040160, 0.040521
- functionals, 1e6 paths: price_S75 = 37.5783, price_S100 = 57.4798, price_S125 = 78.6459
- functionals, 250k paths: price_S75 = 37.6513, price_S100 = 57.5631, price_S125 = 78.7371
  (differ, within ~0.2% sampling noise)
- Self-checks: S > 0 on all paths; E[V_T], E[R_T], Var[R_T], Var[V_T] match exact values
  (0.04, 0.10, 0.0404, 0.0400) within MC noise; Black-Scholes degenerate limit (sigma_v=sigma_r=0,
  400k paths) gives 40.302 / 64.106 / 88.626 vs exact 40.3056 / 64.1105 / 88.6294 (within 0.01%).
- dt ladder (400k paths, seed 7, dt = 0.016/0.008/0.004): price_S100 = 57.527 / 57.527 / 57.422;
  no systematic drift beyond MC noise (~0.1-0.2%).
- CRN dt ladder (answers the evaluator's request; 240k paths in 24 blocks of 10k, one shared Brownian path,
  coarse increments = sums of fine ones, seed 123, per-block control-variate prices averaged):
  dt=0.016: 37.586 / 57.507 / 78.698; dt=0.008: 37.537 / 57.428 / 78.594; dt=0.004: 37.562 / 57.463 / 78.644
  (S75/S100/S125, se of each level ~0.05-0.07 absolute). Paired differences vs dt=0.004 (relative, +/- paired se):
  dt=0.016: +0.063% / +0.078% / +0.069% (se 0.068 / 0.059 / 0.050%); dt=0.008: -0.066% / -0.060% / -0.063%
  (se 0.067 / 0.057 / 0.049%). Non-monotone in dt and within ~1.3 se of zero, so no discretisation bias
  above ~0.1% is detected between dt=0.016 and 0.004; the bias at dt=0.004 vs the continuous limit is not
  resolved by this ladder (it cannot be excluded below ~0.1%).
- Re-run of solver.py unchanged: 1e6 prices 37.5783 / 57.4798 / 78.6459 (39 s); no code change this pass.
- Control variates: D*S_T (mean exact for discrete scheme) and D (discrete E[D] from AR(1) recursion),
  least-squares coefficients on the same sample.
- Note: unlike the dW=None draw suggestion, noise is drawn per chunk/step (60 GB block is infeasible).

## Evaluation

<metrics>
score: 7
certified: false
provenance: self_convergence
estimated_rel_error: 8.883e-07
error_is_estimate: true
observed_order: 1.0145
order_floor: 0.1000
scheme_family: euler-maruyama
invariants_ok: false
constraints_ok: true
invariant_max_drift: 0.6490
invariant_worst_headroom: -7.8122
order_check: true
mc_se_rel: 0.0235
resolved: false
ci_mult: 2.0000
resolution_evidence: not_used_as_accuracy_evidence
solver_sha256: 3b6c4676329213132276137328f076044c93efe33f822b3bc63c4484f9b666f2
solve_time_s: 93.4700
wall_time_s: 128.2500
</metrics>

<review score=7>

**Score: 7/10**

### Numerical Accuracy
- provenance:              self_convergence  (reference route: none; moment_ode, kolmogorov, stationary_density all unavailable, as the spec predicted)
- mean_rel_err:            not measurable (moments_ok unavailable: no reference for S). Raw S mean 634.92 +/- 2 x 1.33 (se)
- variance_rel_err:        not measurable. Raw S variance 1.7645e6 +/- 2 x 4.14e4 (se); V variance 0.04016 +/- 2 x 4.9e-4 vs exact 0.039998; R variance 0.040521 +/- 2 x 5.7e-5 vs exact 0.040404
- mc_se_rel:               0.0235 (S variance, heavy tail)   resolved: unavailable (no reference variance to resolve against). This is not an MC-inconclusive verdict and not a solver defect
- observed_order:          1.0145 +/- 0.0104 (binding component R); S 0.033 (se 0.022), V 0.077 (se 0.037); expected 0.1, one-sided band (scheme_family undeclared in the plan, source inferred)
- weak_order:              1.57, out of band, reported only. Not scored
- dynkin:                  not run (unavailable): the finest ladder level needs 240 GB against the 12 GB budget at 1e6 paths. A property of the graded size, not of this solver. No verdict, so no circularity break
- terminal_law:            not declared
- shape:                   S skewness 13.33 +/- 1.47, excess kurtosis 548.9 +/- 169.4 (extreme heavy tail); V skewness 9.94 +/- 0.14, excess kurtosis 148.6 +/- 6.7; R skewness -0.0065 +/- 0.0024, excess kurtosis -0.012 +/- 0.005 (Gaussian, as exact)
- constraints:             finite ok; asset_positivity ok (gate); martingale E[V_T] ok (drift 1.2e-4 vs bar 5.1e-4); variance_nonnegative FAILED, drift 0.649 (informational in the spec); invariants_ok false
- functionals (checked by hand, since the kernel does not grade them on the SDE path):
  - 1e6 paths:   price_S75 37.5783, price_S100 57.4798, price_S125 78.6459 (reproduced)
  - 250k paths:  price_S75 37.6513, price_S100 57.5631, price_S125 78.7371
  - relative difference between counts 0.19% / 0.14% / 0.12%. They differ, as required. The spec's plain-MC relative se is ~0.17/0.15/0.14% at 250k and ~0.09/0.08/0.07% at 1e6, so the gap is about 1 combined se (control variates shrink se further). Consistent with sampling noise, not tighter evidence than that
  - no-arbitrage: each price lies above its lower bound (0 / 20.316 / 45.316) and below S0; monotone in S0; convex (57.48 <= 58.11)
  - NOT established: the 0.3% accuracy against the independent reference. The 0.3% band leaves ~0.1-0.15% for discretisation bias, and the plan's own dt ladder (57.527 / 57.527 / 57.422) is noisy at that level. Nothing here can confirm or refute it
- resolution_evidence:     not_used_as_accuracy_evidence
- bound_by:                rubric: "the Richardson-extrapolated moments are unstable, or Dynkin failed". Dynkin did not fail; it was unavailable. The cause is the V component, whose Richardson check reads stable: false (shift 0.944 se, relative shift 8.3%, at p = 0.5). The aggregate richardson_stable is true. The undeclared-order band is why this is not reported as a failure of the plan. Certification would also cap at 8 (self-convergence, no circularity break)

### Feedback for solver
- Nothing in the code is wrong that the kernel can see. The scheme is correctly specified and correctly implemented: full truncation sits inside both drift and diffusion; the S update is log-Euler with the same left-point R_n as the discount (so D*S is an exact discrete martingale); there is no bare `X = np.maximum(X, 0)` clip; and the Cholesky rows match the spec. SOLUTION.md matches solver.py, so no cap was applied.
- The `dW` hook is honoured exactly: Nt = dW.shape[1], no noise is drawn, and R's strong order reads 1.01 +/- 0.01, which is the Euler value for an OU component. The CRN ladder is therefore clean on all three components.
- The 7 is mostly structural, not a defect to fix. Without a reference for S, Dynkin unavailable at 1e6 paths, and an unstable V Richardson estimate, the kernel cannot move this plan above self-convergence. The one thing in your control is the V-component Richardson instability: it is within 1 se and comes from V's order-0.1 convergence, so it will not go away at dt = 0.004. Do not change schemes to chase it.
- Do not rely on `variance_nonnegative`. The full-truncation V is negative on 64.9% of terminal paths. This is the scheme's property, the spec marks the invariant informational, and the plan documents it. If you want it to read clean, the only options are reflection, absorption or QE/exact-law V, each of which changes the bias on the 0.3% target and, for exact-law or QE, breaks the CRN dW2 contract (strong order near zero on V). That is a trade for you to decide on bias grounds, not on this invariant.
- The accuracy question that matters (0.3% on the three prices) is unanswered by this evaluation. To help whoever grades it, report in Results the self-convergence of the three prices at dt = 0.016/0.008/0.004 using common random numbers and enough paths that the se is well below 0.1%. The existing ladder (400k paths, independent noise per level) cannot resolve a 0.1% bias.
- If you want a sharper order check, declare `scheme_family: euler-maruyama` knowing the R component (1.01) will fail the two-sided band around 0.1. The spec says so; the one-sided band is the correct choice here, and the plan's comment explaining the omission is right.

### Defects outside the solver (for the conductor)
- Formulator: `variance_nonnegative` is declared as an invariant but is violated by the scheme the spec itself endorses (full truncation). It sets `invariants_ok: false` and `worst_headroom` -7.8. The spec says it cannot move a score, yet the kernel here reports a non-gate invariant failure. Declaring it with `gate: false` still counts against `invariants_ok`. Consider removing it or declaring it as an expected diagnostic.
- Formulator: the V and R marginal moments are exact, but the kernel compares a surrogate against component 0, so they were declared as ungated martingale invariants only. No reference route exists for S, and none could have been supplied, so I do not see a missed surrogate.
- Kernel ceiling: the functionals (the actual graded quantity) are not kernel-graded on the SDE path, so a 7 here says nothing about whether the 0.3% target is met.

</review>
