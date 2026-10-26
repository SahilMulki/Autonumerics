---
id: 1
plan_slug: euler-maruyama
scheme: euler-maruyama
scheme_family: euler-maruyama
strategy: Full-truncation Euler for V, log-Euler for S on one shared set of increments, with a per-step Brownian-bridge crossing correction in log S (frozen V+_n) giving conditional survival weights for continuous double-barrier monitoring at all three spots.
---

## SDE Reference

Heston, Ito, zero rate, on t in [0, T] with T = 1.0:

    dS = sqrt(V) * S dW1                        S(0) = S0
    dV = kappa*(eta - V) dt + sigma*sqrt(V) dW2  V(0) = Y_0
    d<W1, W2>_t = rho dt

| Parameter | Value |
|---|---|
| kappa | 2.58 |
| eta | 0.043 |
| sigma (vol of variance) | 1.0 |
| rho | -0.36 |
| r | 0.0 (no S drift, no discounting) |
| Y_0 = V(0) | 0.164 |
| X_0 = S0 for terminal_paths | 100.0 |
| barrier_lo, barrier_hi | 50.0, 150.0 (corridor is the open interval (50, 150)) |
| spots for functionals | 75.0, 100.0, 125.0 |

Feller ratio 2*kappa*eta/sigma^2 = 0.22188 < 1: V reaches 0, and the scheme must state its treatment there (below).

State layout: `X[:, 0] = S`, `X[:, 1] = V`.

Generator (independent channels Z1, Z2):

    drift     = [0, kappa*(eta - V+)]
    G         = [[sqrt(V+)*S, 0], [sigma*rho*sqrt(V+), sigma*sqrt(1 - rho^2)*sqrt(V+)]],  V+ = max(V, 0)

Functionals (the graded quantity):

    no_touch_S{75,100,125} = P(50 < S_t < 150 for ALL t in [0, 1] | S(0) = S0, V(0) = 0.164)

Monitoring is **continuous**. A path that leaves the corridor between grid points and returns has knocked out.

Exact terminal moments at T = 1 (supporting evidence only, for S0 = 100): E[S] = 100, Var(S) = 768.581, E[V] = 0.0521687, Var(V) = 0.0115699.

## Numerical Scheme

- **Method**: Euler-Maruyama family. Full-truncation Euler for V and log-Euler for S, with Brownian-bridge barrier correction
- **dt**: 0.002 (as given; `Nt = max(1, round(T/dt))`, then `dt = T/Nt`, so Nt = 500). Do not hardcode a different step.
- **num_paths**: 400000 for the graded call, then 100000 for the second call (same dt, T, seed)
- **seed**: 42, via `np.random.default_rng(seed)` only
- **T**: 1.0
- **Milstein correction**: none. The spec marks this problem Milstein-ineligible (`levy_area`).
- **Expected strong order**: 0.5
- **Expected weak order**: 1.0
- **Reference for scoring**:
  - Terminal moments use the analytic moments (`analytic_moments`, validated against `verification.moment_ode` to about 1e-14).
  - The three functionals have no closed form. The evaluator checks them itself, following `verification.qoi_verification_plan`: dt-flatness under CRN, the two-path-count check, the optional `functional_reference_pde` Kolmogorov solve, and cross-plan agreement.

### Update, per step n (vectorised over paths, streamed: never store the full increment array)

1. Draw `Z = rng.standard_normal((num_paths, 2))`, which gives two independent standard normals per path. Set `dW1 = sqrt(dt)*Z[:,0]` and `dW2 = rho*dW1 + sqrt(1 - rho^2)*sqrt(dt)*Z[:,1]`.
2. `Vp = max(V_n, 0)` (full truncation).
3. Log return, shared by all spots: `R_{n+1} = R_n - 0.5*Vp*dt + sqrt(Vp)*dW1`, with `R_0 = 0`. Then `ln S^{(k)}_n = ln S0_k + R_n`. This keeps S > 0, and the scheme is exactly martingale-preserving given V_n.
4. `V_{n+1} = V_n + kappa*(eta - Vp)*dt + sigma*sqrt(Vp)*dW2`. V_n uses the untruncated value.
5. Barrier correction, applied per spot k with log-distances `a = ln(50/S0_k)` and `c = ln(150/S0_k)` measured in R:
   - Grid breach: if `R_{n+1} <= a` or `R_{n+1} >= c`, set the weight to 0. The corridor is open, so touching the barrier counts as a knock-out.
   - Otherwise, if both endpoints are inside the corridor:
     - `p_lo = exp(-2*(R_n - a)*(R_{n+1} - a)/(Vp*dt))`
     - `p_hi = exp(-2*(c - R_n)*(c - R_{n+1})/(Vp*dt))`
     - `w *= (1 - p_lo)*(1 - p_hi)`
   - If `Vp == 0`, there is no diffusion this step. The crossing probability is 0, so guard the division and do not divide by zero.
   - The bridge in ln S carries drift -Vp/2, which does not change the conditional crossing probability, because bridges are drift-free.
   - The two-barrier product is the standard first-order approximation. The exact double-barrier bridge series differs by terms of order exp(-2*(c-a)^2/(Vp*dt)), which are negligible here. A solver may use the exact series if it wishes.
6. `no_touch_S{k} = mean(w_k)` over paths. This is conditional Monte Carlo: the bridge weights are integrated, not sampled. It is a genuine estimate that changes with num_paths, and it has lower variance than a kill-draw.

### Return dict

- `terminal_paths`: shape (num_paths, 2), holding `[S_T, V_T]` for S0 = 100, where `S_T = 100*exp(R_Nt)`.
  - S_T is unconditional on knock-out.
  - The V column reports `max(V_hat_T, 0)`. State this in Results. It differs from raw V_hat by less than 0.3% in the moments.
- `empirical_mean`, `empirical_variance` (ddof=1): lists of 2 floats.
- `functionals`: `{"no_touch_S75", "no_touch_S100", "no_touch_S125"}`, each a Python float.
- Also `num_paths`, `dt`, `T`.

### Extended-contract hooks (optional args, default None)

Signature: `solve_sde(num_paths, dt, T, seed=42, dW=None, observables=None)`.

- `dW`: shape (num_paths, Nt, 2) of INDEPENDENT Brownian increments, already scaled by sqrt(dt). Channel 0 drives W1 and channel 1 drives Z. Apply the correlation inside the solver. When `dW` is given, take Nt from its shape and draw no randomness of your own.
- `observables`: accumulates the trapezoid path integrals of phi(X), with X = [S, V] for S0 = 100. Return them under `path_integrals`.

## Implementation Notes

- **V = 0 treatment, required by R6**: full truncation. `V+ = max(V, 0)` is used in the drift, in the V diffusion, in the S diffusion and in the bridge variance. The raw V_hat may go negative (about 18% of paths at dt = 0.002 is expected and correct). Do not reflect.
- **Continuous monitoring is the whole problem**. Grid-only monitoring biases survival HIGH by a barrier shift of about 0.5826*sqrt(V*dt), which here means several percent, against a 0.6% tolerance. The bridge weight is not optional.
- **Spot S0 = 125 is the tightest case**, being nearest the upper barrier with the lowest survival. Conditional weights matter most there.
- **Memory**: keep R, V and three weight vectors of length num_paths, and stream draws per step. Do not allocate 400000 x 500 x 2 increments (3.2 GB) on the default path.
- **Two-path-count check**: the 400000 and 100000 calls must give different values that agree within sampling noise. Do not cache or round the results.
- **Constraints gated by the kernel**: every terminal state must be finite, and S > 0. Log-Euler guarantees S > 0. The martingale check E[S_T] = 100 is diagnostic.
- **Dynkin**: `verification.dynkin_budget_note` expects the kernel to refuse the Dynkin ladder at 400000 paths (12.8 GB > 12 GB budget). In that case `dynkin_ok` gets no verdict, and the analytic rubric caps the score at **9**. That is expected and is not a solver defect.
- **Strong order**: the Feller violation and the sqrt at V = 0 can degrade the measured strong order. The order study may return `"unresolved"`, which costs nothing.
- **Self-check before handing to the evaluator**: under CRN, compare the functionals at dt and at dt/4. Bridge-corrected values should move far less than 0.6%. For contrast, a grid-only estimate should sit visibly higher.
- Use numpy and stdlib only, and do no work at import time.

## Results

Run: `uv run python workspace/sde_heston_dnt_hivar/plans/1-euler-maruyama/solver.py` (dt = 0.002, Nt = 500, T = 1, seed 42). Full-truncation Euler for V, log-Euler for S, Brownian-bridge double-barrier survival weights (conditional MC). Increments are drawn path-major in blocks of 20000 paths (shape (m, Nt, 2)) to bound memory; the dW and observables hooks run through the same code path. The V column of `terminal_paths` is `max(V_hat_T, 0)`.

| num_paths | mean [S, V] | variance [S, V] (ddof=1) | no_touch_S75 | no_touch_S100 | no_touch_S125 |
|---|---|---|---|---|---|
| 400000 | [100.0511, 0.051973] | [769.887, 0.011471] | 0.790853 | 0.859465 | 0.596900 |
| 100000 | [100.1536, 0.051807] | [771.662, 0.011344] | 0.790232 | 0.858773 | 0.596064 |

Exact (S0=100): mean [100, 0.0521687], variance [768.581, 0.0115699]. Runtime about 76 s at 400000 paths. Both hooks (explicit `dW`, one-entry `observables`) confirmed working.

dt-flatness check under CRN (60000 paths, one shared fine increment array of 2000 steps summed to 500 / 1000 / 2000 steps; `solver.py` unchanged, run via the `dW` hook):

| level | bridge S75 | bridge S100 | bridge S125 | grid-only S75 | grid-only S100 | grid-only S125 |
|---|---|---|---|---|---|---|
| dt = 0.002 | 0.79098 | 0.85925 | 0.60004 | 0.80047 | 0.86567 | 0.61732 |
| dt/2 | 0.79086 | 0.85895 | 0.59999 | 0.79723 | 0.86332 | 0.61203 |
| dt/4 | 0.78988 | 0.85922 | 0.60003 | 0.79477 | 0.86267 | 0.60870 |

Bridge-corrected values move by at most 0.14% relative (S75), 0.04% (S100), 0.01% (S125) between dt and dt/4, well inside the 0.6% tolerance. Grid-only values sit higher at every level (by 1.2%, 0.75% and 2.9% at dt = 0.002 for S75, S100, S125) and drift down as dt shrinks, as expected. No `functional_reference_pde` solve was run. The 100000-path call reuses the first chunks of the 400000-path noise (same seed, chunked draws), so the two-path-count agreement is weak evidence.

## Evaluation

(Filled by evaluator)

<metrics>
score: 9
certified: false
provenance: analytic
estimated_rel_error: 8.533e-03
error_is_estimate: false
observed_order: 0.2866
order_floor: 0.5000
scheme_family: euler-maruyama
invariants_ok: true
constraints_ok: true
invariant_max_drift: 0.0511
invariant_worst_headroom: 0.2349
invariants_not_reported: ['variance_nonnegative_report', 'no_touch_monotone_structure']
reference_outcome: validated
mean_rel_err: 3.759e-03
variance_rel_err: 8.533e-03
order_check: true
mc_se_rel: 7.892e-03
resolved: true
ci_mult: 2.0000
resolution_evidence: not_used_as_accuracy_evidence
solver_sha256: cfe3b17496d5e5c986286042bff41c2ecb03383fd2555e85481146e83bd5f0ad
solve_time_s: 77.2690
wall_time_s: 88.2700
</metrics>

<review score=9>

**Score: 9/10**

### Numerical Accuracy
- provenance:              analytic  (reference route: analytic_moments, reference_outcome validated; the spec reports moment_ode agreement to ~1e-14)
- mean_rel_err:            S: 5.1e-04 (100.051 +/- 2 x 0.0439 vs 100); V: 3.76e-03 (0.051973 +/- 2 x 1.69e-04 vs 0.052169). Tolerance 5%, passed
- variance_rel_err:        S: 1.70e-03 (769.89 +/- 2 x 2.90 vs 768.58); V: 8.53e-03 (0.011471 +/- 2 x 9.05e-05 vs 0.011570). Tolerance 10%, passed
- mc_se_rel:               7.89e-03 (relative CI width 7.5e-03 for S, 1.56e-02 for V, against a bar of 0.05)   resolved: true
- observed_order:          0.287 +/- 0.019 (median-based 0.302), expected 0.5, two-sided band (half-width 0.4), inside the band. Worst component is V; S reads 0.336 +/- 0.013
- weak_order:              indeterminate, at the noise floor (all three test functions "at the noise floor, or not shrinking"). Not scored on
- dynkin:                  not run, status "unavailable": budget refusal (400000 paths x 2000 steps x 2 channels = 12.8 GB, above the 12 GB cap). One seed, nothing to replicate
- terminal_law:            not declared
- shape:                   S: skewness 0.526 +/- 0.023, excess kurtosis 3.66 +/- 0.26. V: skewness 3.97 +/- 0.032, excess kurtosis 22.9 +/- 0.55. Reported only, not gated
- constraints:             finite: true, spot_positive: true, martingale: true (|E[S_T] - 100| = 0.051 against a bar of 0.088). variance_nonnegative_report and no_touch_monotone_structure: not_reported (declared without an expr, by design)
- resolution_evidence:     not_used_as_accuracy_evidence
- bound_by:                rubric. Every measurable check passed, but `dynkin_ok` never reached a verdict, so the 10 row is closed. Uncertified, not suspect

Other checks:
- reported_moments_agree: true (worst relative difference 1.3e-12), so the solver's own `empirical_variance` does use ddof=1.
- The supplied `dW` is consumed. The CRN ladder measured a real order (0.29, not near zero).
- cross_plan_agreement: unavailable (no sibling plan).

**The graded quantity is not scored by this kernel.** The kernel scored the terminal moments only. It does not read `evaluation_thresholds.functionals`, which the spec says is the real grade (three no-touch probabilities, gate: true, 0.6% relative). The 9 says nothing about those three numbers, and I did not recompute them. The only evidence is the plan's own Results table, which I did not verify:
- no_touch_S75: 0.790853 at 400000 paths, 0.790232 at 100000
- no_touch_S100: 0.859465 at 400000, 0.858773 at 100000
- no_touch_S125: 0.596900 at 400000, 0.596064 at 100000

They differ between the two path counts, as required, by 6.2e-4, 6.9e-4 and 8.4e-4. The binomial s.e. at 100000 paths alone is about 1.3e-3 at p = 0.79, and the conditional-weight estimator should be tighter. Both calls use the same seed with chunked path-major draws, so the 100000-path run reuses the first quarter of the 400000-path run's noise, and their agreement is weaker evidence than it looks. No dt-flatness check under CRN and no `functional_reference_pde` solve was run. A bridge error or residual grid bias would not show up in anything above.

### Feedback for solver
- The terminal moments are clean and nothing needs changing for the 9. The cap to 9 is the Dynkin budget refusal that `verification.dynkin_budget_note` predicted. It is not caused by anything in `solver.py`, and only a change to the spec's `num_paths` or `dt` could clear it.
- The strong order on V (0.29 against a nominal 0.5) is inside the band only because the half-width is 0.4. This is plausible full-truncation behaviour with Feller ratio 0.22 and sqrt(V) at zero. No scheme change is requested. A QE scheme for V (Andersen 2008) would be the route to a higher reading, but it is optional and does not change the score.
- Run the functional check the kernel cannot. Under one shared `dW`, compare the three no-touch values at dt, dt/2 and dt/4 and confirm they move by well under 0.6% relative. Repeat with the bridge weight removed to confirm the grid-only value sits visibly higher. SOLUTION.md's Results table does not contain this.
- If you want the two-path-count check to be a stronger test, the 100000-path call would need noise independent of the 400000-path call. The contract only requires the same seed, so this is optional.

Spec-side note for the formulator (not a solver defect): `verification.functional_reference_pde` is marked `consumed_by_kernel: false`, and two declared invariants have no `expr`, so the three graded functionals have no kernel-side reference or check. This is a pipeline coverage gap, not a missed surrogate. The moment-ODE route was used for the moments, and the Kolmogorov route is scalar-only by the spec's own note.

</review>
