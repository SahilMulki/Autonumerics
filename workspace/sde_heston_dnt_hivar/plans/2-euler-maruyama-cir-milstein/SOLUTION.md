---
id: 2
plan_slug: euler-maruyama-cir-milstein
scheme: euler-maruyama
scheme_family: euler-maruyama
strategy: Log-Euler for S with a scalar Milstein-corrected, full-truncation update for the CIR variance on its own combined noise channel, plus a per-step Brownian-bridge crossing correction in log S (frozen V+_n) giving conditional survival weights for continuous double-barrier monitoring at all three spots.
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

- **Method**: Euler-Maruyama family. Log-Euler for S; full-truncation Euler for V plus the scalar CIR Milstein term on V's own channel; Brownian-bridge barrier correction.
- **Why the family is still euler-maruyama**: per `problem_spec.json → milstein_ineligible_note`, the system as a whole is Milstein-ineligible (`levy_area`). The S-equation cross term would need the Levy area of (W1, W2), which is not computed. A scalar Milstein correction applied to the V equation alone is legitimate, because V is driven by the single combined channel W2. The spec directs that such a plan declare `scheme_family: euler-maruyama`, and its joint strong order stays at the EM value.
- **dt**: 0.002 (as given; `Nt = max(1, round(T/dt))`, then `dt = T/Nt`, so Nt = 500). Do not hardcode a different step.
- **num_paths**: 400000 for the graded call, then 100000 for the second call (same dt, T, seed)
- **seed**: 42, via `np.random.default_rng(seed)` only
- **T**: 1.0
- **Milstein correction (V only)**: `+ (sigma^2/4)*(dW2^2 - dt)`. This is g*g'/2 with g = sigma*sqrt(V) and g' = sigma/(2*sqrt(V)), so g*g' = sigma^2/2, independent of V.
- **Expected strong order**: 0.5. The S channel is EM without a Levy area.
- **Expected weak order**: 1.0
- **Reference for scoring**:
  - Terminal moments use the analytic moments (`analytic_moments`, validated against `verification.moment_ode` to about 1e-14).
  - The three functionals have no closed form. The evaluator checks them itself, following `verification.qoi_verification_plan`: dt-flatness under CRN, the two-path-count check, the optional `functional_reference_pde` Kolmogorov solve, and cross-plan agreement.

### Update, per step n (vectorised over paths, streamed: never store the full increment array)

1. Draw `Z = rng.standard_normal((num_paths, 2))`, which gives two independent standard normals per path. Set `dW1 = sqrt(dt)*Z[:,0]` and `dW2 = rho*dW1 + sqrt(1 - rho^2)*sqrt(dt)*Z[:,1]`. dW2 has variance dt.
2. `Vp = max(V_n, 0)` (full truncation).
3. Log return, shared by all spots: `R_{n+1} = R_n - 0.5*Vp*dt + sqrt(Vp)*dW1`, with `R_0 = 0`. Then `ln S^{(k)}_n = ln S0_k + R_n`. This keeps S > 0, and the scheme is exactly martingale-preserving given V_n.
4. `V_{n+1} = V_n + kappa*(eta - Vp)*dt + sigma*sqrt(Vp)*dW2 + 0.25*sigma^2*(dW2^2 - dt)`.
   - All terms use step-n quantities. The correction must not use V_{n+1}.
   - The correction is NOT multiplied by any V-dependent factor.
   - It may push V below 0. The truncation at the next step handles that.
5. Barrier correction, applied per spot k with log-distances `a = ln(50/S0_k)` and `c = ln(150/S0_k)` measured in R:
   - Grid breach: if `R_{n+1} <= a` or `R_{n+1} >= c`, set the weight to 0. The corridor is open, so touching the barrier counts as a knock-out.
   - Otherwise, if both endpoints are inside the corridor:
     - `p_lo = exp(-2*(R_n - a)*(R_{n+1} - a)/(Vp*dt))`
     - `p_hi = exp(-2*(c - R_n)*(c - R_{n+1})/(Vp*dt))`
     - `w *= (1 - p_lo)*(1 - p_hi)`
   - If `Vp == 0`, there is no diffusion in S this step. The crossing probability is 0, so guard the division.
   - Given V_n, the log-Euler S increment is exactly Gaussian with variance Vp*dt, so the frozen-V bridge is the exact conditional crossing law of the discretised S path.
   - The drift -Vp/2 does not change the bridge crossing probability.
   - The two-barrier product neglects only terms of order exp(-2*(c-a)^2/(Vp*dt)).
6. `no_touch_S{k} = mean(w_k)` over paths. This is conditional Monte Carlo: a genuine estimate that changes with num_paths, with lower variance than a kill-draw.

### Return dict

- `terminal_paths`: shape (num_paths, 2), holding `[S_T, V_T]` for S0 = 100, where `S_T = 100*exp(R_Nt)`.
  - S_T is unconditional on knock-out.
  - The V column reports `max(V_hat_T, 0)`. State this in Results.
- `empirical_mean`, `empirical_variance` (ddof=1): lists of 2 floats.
- `functionals`: `{"no_touch_S75", "no_touch_S100", "no_touch_S125"}`, each a Python float.
- Also `num_paths`, `dt`, `T`.

### Extended-contract hooks (optional args, default None)

Signature: `solve_sde(num_paths, dt, T, seed=42, dW=None, observables=None)`.

- `dW`: shape (num_paths, Nt, 2) of INDEPENDENT Brownian increments, already scaled by sqrt(dt). Channel 0 drives W1 and channel 1 drives Z. Apply the correlation inside the solver. When `dW` is given, take Nt from its shape and draw no randomness of your own.
- `observables`: accumulates the trapezoid path integrals of phi(X), with X = [S, V] for S0 = 100. Return them under `path_integrals`.

## Implementation Notes

- **V = 0 treatment, required by R6**: full truncation plus the CIR Milstein term. `V+ = max(V, 0)` is used in the drift, in both diffusions and in the bridge variance. The raw V_hat may be negative.
  - The Milstein term reduces V's discretisation error near the boundary compared with plain truncated Euler, but it still adds a kick of `(sigma^2/4)*(dW2^2 - dt)` even at V = 0. That kick has mean zero, so it does not bias E[V].
  - Write in Results what fraction of terminal V_hat is negative.
- **Continuous monitoring is the whole problem**. Grid-only monitoring biases survival HIGH by a barrier shift of about 0.5826*sqrt(V*dt), which here means several percent, against a 0.6% tolerance. The bridge weight is not optional.
- **Spot S0 = 125 is the tightest case**, being nearest the upper barrier with the lowest survival.
- **Memory**: keep R, V and three weight vectors, and stream draws per step. Do not allocate 400000 x 500 x 2 increments on the default path.
- **Two-path-count check**: the 400000 and 100000 calls must give different values that agree within sampling noise. Do not cache or round the results.
- **Constraints gated by the kernel**: every terminal state must be finite, and S > 0. The martingale check E[S_T] = 100 is diagnostic.
- **Dynkin**: `verification.dynkin_budget_note` expects the kernel to refuse the Dynkin ladder at 400000 paths. In that case `dynkin_ok` gets no verdict, and the analytic rubric caps the score at **9**. That is expected.
- **Strong order**: declared 0.5. The Feller violation may make the study report `"unresolved"`, which costs nothing.
  - A measured order near 1.0 would not be a failure of the scheme, but the declared value stays 0.5 because the S channel lacks a Levy area.
- **Self-check before handing to the evaluator**: under CRN, compare the functionals at dt and dt/4. They should move far less than 0.6%.
  - Also check E[V_T] against 0.0521687 and Var(V_T) against 0.0115699.
- Use numpy and stdlib only, and do no work at import time.

## Results

Run: `solve_sde(num_paths, dt=0.002, T=1.0, seed=42)` (Nt = 500), streamed per-step draws. Runtime about 30 s per 400000-path call.

| num_paths | E[S_T] | Var(S_T) | E[V_T] | Var(V_T) |
|---|---|---|---|---|
| 400000 | 100.0715 | 769.969 | 0.052239 | 0.011581 |
| 100000 | 100.0977 | 767.841 | 0.051899 | 0.011440 |

Exact: E[S] = 100, Var(S) = 768.581, E[V] = 0.0521687, Var(V) = 0.0115699.

Functionals (conditional bridge-weight estimates):

| num_paths | no_touch_S75 | no_touch_S100 | no_touch_S125 |
|---|---|---|---|
| 400000 | 0.791387 | 0.860261 | 0.596882 |
| 100000 | 0.790126 | 0.859927 | 0.597636 |

The two counts differ and agree within sampling noise.

Self-check, CRN dt-flatness (50000 paths, shared Brownian path, dt = 1/250 vs dt/4 = 1/1000): no_touch_S75 0.78926 vs 0.78992 (0.08% rel), S100 0.85885 vs 0.85950 (0.07%), S125 0.59273 vs 0.59364 (0.15%). All far below the 0.6% tolerance, so the bridge correction removes the O(sqrt(dt)) monitoring bias. Re-run of `solver.py` reproduces the table above exactly.

Zero-boundary treatment: full truncation plus the scalar CIR Milstein term. The raw V_hat is negative at T on about 22.9% of paths (400000 run) and 22.8% (100000 run). `terminal_paths[:, 1]` reports `max(V_hat_T, 0)`. `terminal_paths[:, 0]` is `100*exp(R_T)`, unconditional on knock-out. The result dict also carries `frac_negative_V_raw`.

Hooks: `dW` of shape (num_paths, Nt, 2) is used as given (Nt from its shape, no own randomness, correlation applied inside); `observables` phi receives X = [S, max(V,0)] for S0 = 100 and its trapezoid integrals are returned in `path_integrals`. Both were exercised in `__main__`.

## Evaluation

(Filled by evaluator)

<metrics>
score: 9
certified: false
provenance: analytic
estimated_rel_error: 1.805e-03
error_is_estimate: false
observed_order: 0.2420
order_floor: 0.5000
scheme_family: euler-maruyama
invariants_ok: true
constraints_ok: true
invariant_max_drift: 0.0715
invariant_worst_headroom: 0.0891
invariants_not_reported: ['variance_nonnegative_report', 'no_touch_monotone_structure']
reference_outcome: validated
mean_rel_err: 1.349e-03
variance_rel_err: 1.805e-03
order_check: true
mc_se_rel: 7.939e-03
resolved: true
ci_mult: 2.0000
resolution_evidence: not_used_as_accuracy_evidence
solver_sha256: 8beec07daa93faa16405d436393485aa36e8d54f65e2277bc9d6a2ab9277ba12
solve_time_s: 51.0480
wall_time_s: 60.8700
</metrics>

<review score=9>

**Score: 9/10**

### Numerical Accuracy
- provenance:              analytic  (reference route: analytic_moments, validated; moment ODE agrees to ~1e-14 per the spec)
- mean_rel_err:            S 7.1e-4 (100.071 +/- 2.0 x 0.0439 vs 100), V 1.35e-3 (0.052239 +/- 2.0 x 1.70e-4 vs 0.052169); tol 5%
- variance_rel_err:        S 1.80e-3 (769.97 +/- 2.0 x 3.12 vs 768.58), V 9.9e-4 (0.011581 +/- 2.0 x 9.2e-5 vs 0.011570); tol 10%
- mc_se_rel:               7.9e-3 (CI width 0.81% on S, 1.59% on V against a 5% bar)   resolved: true
- observed_order:          0.242 +/- 0.020 (SE; median-based 0.385; expected 0.5, two-sided band 0.4, so 0.1 to 0.9). Per component: S 0.274, V 0.242. Inside the band
- weak_order:              indeterminate: all three test functions at the noise floor or not shrinking (reported, not scored)
- dynkin:                  unavailable (budget refusal): the finest level needs 12.8 GB of increments against a 12 GB cap (400000 paths x 2000 steps x 2 channels). Not run, no z
- terminal_law:            not declared
- shape:                   S skewness 0.554 +/- 0.047, excess kurtosis 4.58 +/- 1.13; V skewness 3.96 +/- 0.040, excess kurtosis 23.2 +/- 0.86 (reported max(V,0), as declared)
- constraints:             finite pass (gate), spot_positive pass (gate), martingale pass (E[S_T] drift 0.0715 against bar 0.0877, headroom 0.089, thin but inside); variance_nonnegative_report and no_touch_monotone_structure not_reported (diagnostic only, no expr)
- resolution_evidence:     not_used_as_accuracy_evidence
- bound_by:                rubric. Everything measurable passed, but the Dynkin check did not reach a verdict, so the 10 row is closed. Uncertified, not suspect

reported_moments_agree: true (worst rel diff 6e-13), so the solver's own ddof=1 arithmetic is correct. surrogate_ok is unavailable only because the exact analytic moments were used instead, and provenance_as_expected is true. cross_plan_agreement is unavailable (no sibling plans compared).

**What the score does not cover.** The graded quantity in `problem.md` is the three no-touch probabilities (gated at 0.6% relative). The kernel does not read `evaluation_thresholds.functionals` on this route, so this score is on terminal moments only. The solver reports no_touch_S75/S100/S125 = 0.791387 / 0.860261 / 0.596882 at 400000 paths and 0.790126 / 0.859927 / 0.597636 at 100000. I did not verify them against any reference. The two counts differ and are consistent with binomial noise (S125 gap 7.5e-4, s.e. at most about 8e-4 per count). The `functional_reference_pde` Kolmogorov solve or a dt/4 CRN probe is the route that would check them.

### Feedback for solver
- Nothing in the moments, constraints, MC resolution or reported-moment arithmetic needs changing. The remaining gap from 9 to 10 is the Dynkin ladder exceeding its memory budget at the stated 400000 paths. That is the spec's expected refusal (`dynkin_budget_note`), not a defect in `solver.py`. Do not reduce `num_paths` to get around it.
- Strong order 0.242 +/- 0.020 is inside the two-sided band around 0.5, but it sits near the lower edge (floor 0.1). This is consistent with a Feller-violated CIR under full truncation, where the pathwise error converges slowly. The median-based order is 0.385. Not a defect. Do not change the scheme on this account.
- Your SOLUTION.md says the scalar CIR Milstein term on V is "legitimate" and it declares `scheme_family: euler-maruyama`, which matches the spec's instruction, so no `milstein_claim` cap applies. The solver implements what the plan describes (checked: the correction uses step-n quantities, `0.25*sigma^2*(dW2^2 - dt)`).
- `martingale` headroom is only 0.089 (E[S_T] - 100 = 0.0715 against a 0.0877 bar, about 1.6 standard errors). This is within noise at this seed, but the 100000-path run gives 100.098, so do not tighten anything on the strength of it.
- `solver.py` uses `np.maximum(V, 0.0)` only inside the coefficients and in the reported terminal state. The terminal clip is declared in SOLUTION.md, so it is not a hidden excursion. Raw V_hat is negative on about 22.9% of paths at T.
- Optional, to help a later evaluator: the functional values above are unverified here. A self-check of `no_touch_*` at dt versus dt/4 under the shared `dW` hook would show the bridge correction is working, and would be useful for REPORT.md.

</review>
